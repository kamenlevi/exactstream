"""tgl16v.py (Breakthrough, Tue 17 Nov 2026 10:2x; EXPERT REWRITE, variants for the quiet bench):
  tgl16g  kernel A with ONE simdgroup per K chunk doing BOTH projections (gate and up) on one staged activation tile: the
          x staging (device loads + threadgroup stores) halves, 4 simdgroups / 128 threads per threadgroup, an 18.7 KB
          slice set. Per projection the decode, the two MMAs, the epilogue and the chunk order are tgl16's, so the bits
          should not move (checked, not assumed). Kernel B = tgl16's.
  a8b16   C1's int8 kernel A (w4a8) + C2's bf16 kernel B (tgl16, rowsum64(h)): is the int8 error mostly h's? (FAST)
Pool reads stay coherent(system): the codes / LSB pointers are declared coherent here (new names, outside coh1cq's list),
the p11 scale / bias pools go straight into P11's coherent helpers (coh1cq._ren)."""
import os
import sys

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import w4a8 as W                                            # noqa: E402
import tgl16 as G                                           # noqa: E402

SGWG = 1168                                                # words per simdgroup: codes 2 x 256 + x 512 + sb 2 x 64 + rx 16
_K = {}

_PRE_AG1 = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // K chunk kc = sg; both projections per simdgroup
    const uint kc = sg;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier0 = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
"""
_CORE_G = r"""
    const int sr = lane >> 1, shf = lane & 1, rr = lane & 15;
    const uint xr_s = (uint)pidx[q0 + (sr < nq ? sr : 0)] / NR;
    const uint xr_r = (uint)pidx[q0 + (rr < nq ? rr : 0)] / NR;
    const device T* xb_s = x + (size_t)xr_s * K + 32 * shf;
    const device float* rx_src = xs + (size_t)xr_r * (K / 64);
    threadgroup uint32_t tg_all[KS_C * SGWG_C];                 // per-simdgroup slices; later the chunk partials
    constexpr int KG = K / 64; constexpr uint NB11 = N * KG / 32, PW11 = PWM_C, PWL11 = PWL_C;
    const size_t wo = ((size_t)e * (N / 32) + n0 / 32) * (KG * 32 * 6) + lane * 6;
    coherent(system) const device uint32_t* wr0 = (coherent(system) const device uint32_t*)w3g + wo;
    coherent(system) const device uint32_t* wr1 = (coherent(system) const device uint32_t*)w3u + wo;
    const uint so = (n0 / 32) * (KG * 32) + lane;
    const P11R sr0 = p11r((const device uint*)s3g, e, PW11, NB11, so);
    const P11R sr1 = p11r((const device uint*)s3u, e, PW11, NB11, so);
    P11R br0 = {}, br1 = {};
    coherent(system) const device uint32_t* lr0 = (coherent(system) const device uint32_t*)l1g;
    coherent(system) const device uint32_t* lr1 = (coherent(system) const device uint32_t*)l1u;
    if (tier == 4) {
        const size_t lo = ((size_t)el * (N / 32) + n0 / 32) * (KG * 32 * 2) + lane * 2;
        lr0 = (coherent(system) const device uint32_t*)l1g + lo;
        lr1 = (coherent(system) const device uint32_t*)l1u + lo;
        br0 = p11r((const device uint*)bg, el, PWL11, NB11, so);
        br1 = p11r((const device uint*)bu, el, PWL11, NB11, so);
    } else {
        br0 = p11r((const device uint*)(tier == 2 ? b2g : b3g), e, PW11, NB11, so);
        br1 = p11r((const device uint*)(tier == 2 ? b2u : b3u), e, PW11, NB11, so);
    }
    const float smul = tier == 4 ? 0.5f : 1.0f;
    threadgroup uint32_t* tb0w = tg_all + sg * SGWG_C;              // gate codes (256 words)
    threadgroup uint32_t* tb1w = tb0w + 256;                        // up codes (256 words)
    threadgroup T* taw = (threadgroup T*)(tb0w + 512);              // x: 16 rows x 64 bf16 (512 words)
    threadgroup float* sb0 = (threadgroup float*)(tb0w + 1024);     // gate: 32 scales s', 32 biases b
    threadgroup float* sb1 = (threadgroup float*)(tb0w + 1088);     // up
    threadgroup float* rxw = (threadgroup float*)(tb0w + 1152);     // 16 row sums
    constexpr int GPC = KG / KS_C;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TA = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    TA ta0(taw, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TA ta1(taw + 32, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB g0((threadgroup uchar*)tb0w, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB g1((threadgroup uchar*)tb0w + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB u0((threadgroup uchar*)tb1w, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB u1((threadgroup uchar*)tb1w + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, float>();
    float accg[16], accu[16];
    short ecol[16], erow[16];
    for (int i = 0; i < 16; ++i) {
        accg[i] = 0.0f; accu[i] = 0.0f;
        const auto ix = cg.get_multidimensional_index(i);
        ecol[i] = (short)ix[0]; erow[i] = (short)ix[1];
    }
    for (int g = kc * GPC; g < (kc + 1) * GPC; ++g) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint a0 = wr0[192 * g], a1 = wr0[192 * g + 1], a2 = wr0[192 * g + 2], a3 = wr0[192 * g + 3],
                   a4 = wr0[192 * g + 4], a5 = wr0[192 * g + 5];
        const uint c0 = wr1[192 * g], c1 = wr1[192 * g + 1], c2 = wr1[192 * g + 2], c3 = wr1[192 * g + 3],
                   c4 = wr1[192 * g + 4], c5 = wr1[192 * g + 5];
        const device uint4* xsrc = (const device uint4*)(xb_s + 64 * g);
        const uint4 xa = xsrc[0], xb = xsrc[1], xc = xsrc[2], xd4 = xsrc[3];
        const float rxv = rx_src[g];
        uint ch[8] = {a0 & 0xFFFFFFu, (a0 >> 24) | ((a1 & 0xFFFFu) << 8), (a1 >> 16) | ((a2 & 0xFFu) << 16), a2 >> 8,
                      a3 & 0xFFFFFFu, (a3 >> 24) | ((a4 & 0xFFFFu) << 8), (a4 >> 16) | ((a5 & 0xFFu) << 16), a5 >> 8};
        uint dh[8] = {c0 & 0xFFFFFFu, (c0 >> 24) | ((c1 & 0xFFFFu) << 8), (c1 >> 16) | ((c2 & 0xFFu) << 16), c2 >> 8,
                      c3 & 0xFFFFFFu, (c3 >> 24) | ((c4 & 0xFFFFu) << 8), (c4 >> 16) | ((c5 & 0xFFu) << 16), c5 >> 8};
        uint nb[8], nu[8];
        if (tier == 4) {
            const uint l0 = lr0[64 * g], l1 = lr0[64 * g + 1], m0 = lr1[64 * g], m1 = lr1[64 * g + 1];
            for (int q = 0; q < 8; ++q) {
                nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
                nu[q] = spread8(dh[q], ((q < 4 ? m0 : m1) >> (8 * (q & 3))) & 0xFFu);
            }
        } else if (tier == 2) {
            for (int q = 0; q < 8; ++q) { nb[q] = spread3(ch[q]) & 0x66666666u; nu[q] = spread3(dh[q]) & 0x66666666u; }
        } else {
            for (int q = 0; q < 8; ++q) { nb[q] = spread3(ch[q]); nu[q] = spread3(dh[q]); }
        }
        threadgroup uint4* d0 = (threadgroup uint4*)(tb0w + lane * 8);
        d0[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
        d0[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
        threadgroup uint4* d1 = (threadgroup uint4*)(tb1w + lane * 8);
        d1[0] = uint4(nu[0], nu[1], nu[2], nu[3]);
        d1[1] = uint4(nu[4], nu[5], nu[6], nu[7]);
        threadgroup uint4* xd = (threadgroup uint4*)(taw + sr * 64 + 32 * shf);
        xd[0] = xa; xd[1] = xb; xd[2] = xc; xd[3] = xd4;
        sb0[lane] = (float)sr0[32 * g] * smul;
        sb0[32 + lane] = (float)br0[32 * g];
        sb1[lane] = (float)sr1[32 * g] * smul;
        sb1[32 + lane] = (float)br1[32 * g];
        if (lane < 16) rxw[lane] = rxv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        op.run(ta0, g0, cg);
        op.run(ta1, g1, cg);
        for (int i = 0; i < 16; ++i) accg[i] += sb0[ecol[i]] * cg[i] + sb0[32 + ecol[i]] * rxw[erow[i]];
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        op.run(ta0, u0, cg);
        op.run(ta1, u1, cg);
        for (int i = 0; i < 16; ++i) accu[i] += sb1[ecol[i]] * cg[i] + sb1[32 + ecol[i]] * rxw[erow[i]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);               // every simdgroup is done with its slice
    threadgroup float* part = (threadgroup float*)tg_all;           // [projection][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) {
        part[kc * 512 + erow[i] * 32 + ecol[i]] = accg[i];
        part[(KS_C + kc) * 512 + erow[i] * 32 + ecol[i]] = accu[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = thread_index_in_threadgroup; t < 512u; t += KS_C * 32) {
        const int row = (int)(t >> 5), col = (int)(t & 31);
        float gs = part[row * 32 + col], us = part[KS_C * 512 + row * 32 + col];
        for (int c = 1; c < KS_C; ++c) { gs += part[c * 512 + row * 32 + col]; us += part[(KS_C + c) * 512 + row * 32 + col]; }
        if (row < nq) {
            const T gv = static_cast<T>(gs);
            const T ub = static_cast<T>(us);
            h[(size_t)pidx[q0 + row] * N + n0 + col] = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);
        }
    }
"""


def source_ag():
    import ooo1c as OC
    import ooo1cq as OQ
    import coh1cq as CQ
    return CQ._ren(CQ._body(OQ.a_one_call_q(_PRE_AG1 + OC._OLD_HEAD + "\n" + _CORE_G)))


def _kag():
    if "ag" not in _K:
        _K["ag"] = mx.fast.metal_kernel(
            name="tgl16g_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=source_ag(), header=W._hdr())
    return _K["ag"]


def _common(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, grp, ks, ks_b, nsg_b):
    import ooo1cq as OQ
    import p11 as P11
    from tierglu import plan_masks, PROJ
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    assert msb_n["gate_proj"]["s3"].dtype == mx.uint32, "p11 store only"
    N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
    pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]
    assert pwm >= P11.piece_words(N * 40) and pwm % 4 == 0 and pwl >= P11.piece_words(N * 40) and pwl % 4 == 0
    assert K % (64 * ks) == 0 and N % (64 * ks_b) == 0 and D % (32 * nsg_b) == 0
    if grp is None:
        grp = OQ.gpu_grp_1cq(pre_m, pre_l, ids, plan)
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ms = pre_m.astype(mx.int32).reshape(-1)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("PWM_C", pwm),
            ("PWL_C", pwl)]
    return M, K, R, r4, N, D, grp, bb, ls, ms, tmpl, PROJ


def _b16(h, msb_n, lsb_n, rs_m, rs_l, grp, tmpl, bb, M, R, N, D, nsg_b, ks_b, dt):
    from tierglu_nax import rowsum64
    pidx, seg, nsg_, nseg = grp
    d, L_ = msb_n["down_proj"], lsb_n
    hs = rowsum64(h)
    (y,) = G._kb()(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L_["down_proj"]["l1"], L_["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_],
        template=tmpl + [("SGW_C", G.SGW), ("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[dt])
    return y.reshape(M, R, D)


def tier_glu_nax6n_1cq_tgl16g(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None,
                              nsg_b=2, ks=4, ks_b=2):
    from tierglu_nax import rowsum64
    M, K, R, r4, N, D, grp, bb, ls, ms, tmpl, PROJ = _common(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, grp, ks, ks_b, nsg_b)
    pidx, seg, nsg_, nseg = grp
    g, u = msb_n["gate_proj"], msb_n["up_proj"]
    L_ = lsb_n
    xs = rowsum64(x) if xs is None else xs
    h, rs_m, rs_l = _kag()(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L_["gate_proj"]["l1"], L_["up_proj"]["l1"], L_["gate_proj"]["b"], L_["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tmpl + [("SGWG_C", SGWG), ("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 32 * ks, nseg, 1),
        threadgroup=(32 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    return _b16(h, msb_n, lsb_n, rs_m, rs_l, grp, tmpl, bb, M, R, N, D, nsg_b, ks_b, x.dtype), rs_m, rs_l


def tier_glu_nax6n_1cq_a8b16(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None,
                             nsg_b=2, ks=4, ks_b=2):
    M, K, R, r4, N, D, grp, bb, ls, ms, tmpl, PROJ = _common(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, grp, ks, ks_b, nsg_b)
    pidx, seg, nsg_, nseg = grp
    g, u = msb_n["gate_proj"], msb_n["up_proj"]
    L_ = lsb_n
    xq, sx, xs_ = W.quant64(x)
    h, rs_m, rs_l = W._ka()(
        inputs=[xq, sx, xs_, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L_["gate_proj"]["l1"], L_["up_proj"]["l1"], L_["gate_proj"]["b"], L_["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tmpl + [("SGW_C", W.SGW), ("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
        threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    return _b16(h, msb_n, lsb_n, rs_m, rs_l, grp, tmpl, bb, M, R, N, D, nsg_b, ks_b, x.dtype), rs_m, rs_l


if __name__ == "__main__":
    s = source_ag()
    print(f"tgl16g A: {len(s)} chars; coherent pointers {s.count('coherent(system) const device')}; head polls "
          f"{s.count('oc_poll(')}; plain pool pointer decls: "
          f"{sum(s.count(f'const device uint32_t* {n} =') for n in ('wr0', 'wr1', 'lr0', 'lr1'))} (all coherent)")


# ---- ablations of tgl16 (timing only; outputs are garbage): ABL_C bits 1 no code loads (fake words keep the decode),
# 2 no decode (raw words stored), 4 no MMA, 8 no epilogue (acc += cg), 16 no activation loads (fake x)
def _core_abl():
    c = G._CORE
    reps = [
        ("const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];",
         "const uint fk = ((uint)lane * 2654435761u) ^ (uint)g;\n"
         "        const uint w0 = (ABL_C & 1) ? fk : c[0], w1 = (ABL_C & 1) ? fk * 3u : c[1], w2 = (ABL_C & 1) ? fk * 5u : c[2],\n"
         "                   w3 = (ABL_C & 1) ? fk * 7u : c[3], w4 = (ABL_C & 1) ? fk * 9u : c[4], w5 = (ABL_C & 1) ? fk * 11u : c[5];"),
        ("const uint4 xa = xsrc[0], xb = xsrc[1], xc = xsrc[2], xd4 = xsrc[3];",
         "const uint4 xa = (ABL_C & 16) ? uint4((uint)lane, (uint)g, 1u, 2u) : xsrc[0], xb = (ABL_C & 16) ? xa + 1u : xsrc[1],\n"
         "                    xc = (ABL_C & 16) ? xa + 2u : xsrc[2], xd4 = (ABL_C & 16) ? xa + 3u : xsrc[3];"),
        ("const uint l0 = lrow[64 * g], l1 = lrow[64 * g + 1];",
         "const uint l0 = (ABL_C & 1) ? fk * 13u : lrow[64 * g], l1 = (ABL_C & 1) ? fk * 17u : lrow[64 * g + 1];"),
        ("        if (tier == 4) {\n            const uint l0",
         "        if (ABL_C & 2) {\n            for (int q = 0; q < 8; ++q) nb[q] = ch[q] ^ ((uint)q * 0x11111111u);\n"
         "        } else if (tier == 4) {\n            const uint l0"),
        ("        op.run(ta0, tb0, cg);\n        op.run(ta1, tb1, cg);",
         "        if (!(ABL_C & 4)) { op.run(ta0, tb0, cg); op.run(ta1, tb1, cg); }"),
        ("            acc[i] += sbw[ecol[i]] * cg[i] + sbw[32 + ecol[i]] * rxw[erow[i]];",
         "            acc[i] += (ABL_C & 8) ? cg[i] : sbw[ecol[i]] * cg[i] + sbw[32 + ecol[i]] * rxw[erow[i]];"),
    ]
    for a, b in reps:
        assert c.count(a) == 1, a
        c = c.replace(a, b)
    return c


def _kabl(which, abl):
    key = (which, abl)
    if key not in _K:
        import ooo1c as OC
        import ooo1cq as OQ
        import coh1cq as CQ
        core = _core_abl()
        if which == "a":
            src = OQ.a_one_call_q(W._PRE_A1 + OC._OLD_HEAD + "\n" + G._PRE_A2 + core + W._POST_A)
            names = ["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"]
            outs = ["h", "rs_m", "rs_l"]
        else:
            src = OQ.b_drop(W._PRE_B1.replace(G._PRE_B_STAGE_OLD, G._PRE_B_STAGE_NEW) + core + W._POST_B)
            names = ["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"]
            outs = ["y"]
        _K[key] = mx.fast.metal_kernel(name=f"tgl16abl{abl}_{which}", input_names=names, output_names=outs,
                                       source=CQ._ren(CQ._body(src)), header=W._hdr())
    return _K[key]


def tier_glu_nax6n_1cq_tgl16abl(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None,
                                nsg_b=2, ks=4, ks_b=2, abl=0):
    from tierglu_nax import rowsum64
    M, K, R, r4, N, D, grp, bb, ls, ms, tmpl, PROJ = _common(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, grp, ks, ks_b, nsg_b)
    pidx, seg, nsg_, nseg = grp
    g, u, d = (msb_n[p] for p in PROJ)
    L_ = lsb_n
    xs = rowsum64(x) if xs is None else xs
    tm = tmpl + [("SGW_C", G.SGW), ("ABL_C", abl)]
    h, rs_m, rs_l = _kabl("a", abl)(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L_["gate_proj"]["l1"], L_["up_proj"]["l1"], L_["gate_proj"]["b"], L_["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tm + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
        threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    hs = rowsum64(h)
    (y,) = _kabl("b", abl)(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L_["down_proj"]["l1"], L_["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_],
        template=tm + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D), rs_m, rs_l
