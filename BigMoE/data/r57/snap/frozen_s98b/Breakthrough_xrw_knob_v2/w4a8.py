"""w4a8.py (Breakthrough, Tue 17 Nov 2026 10:0x; EXPERT REWRITE candidate 1, docs/EXPERT_REWRITE.md): the FAST-NAX expert
one-call (Leap's ooo1cq on the p11 store, COH reads) with the accelerator's MMA in int8 x int4b -> int32 instead of
bf16 x uint4b -> float. FAST (not bit-exact): EvalSuite's same-scores gate decides adoption; an OPT-IN knob for c25.

What changes, per kernel (A = gate | up + swiglu, B = down):
  - the activations: quant64 (one simdgroup per (row, 64-value group)) gives int8 codes q = rint(v * 127 / absmax), the
    group's scale sx = absmax / 127 and its float sum xs; it replaces rowsum64(x) / rowsum64(h) (same dispatch count);
  - the codes: the same decode (spread3 / spread8) XOR 0x88888888, so the int4b MMA reads q - 8 (x2_mma_layout.py PASS);
    the bias absorbs the offset: w = s' q + b = s' (q - 8) + (b + 8 s'), s' = smul * s;
  - the MMA: left = the tile's 16 activation rows staged in threadgroup memory (int8), right = the codes (int4b), int32
    dest; per quant group acc += s' sx (int32 sum) + (b + 8 s') xs (xs from the bf16 values, as today);
  - one slice of threadgroup memory per simdgroup (codes 1 KB, x 1 KB, scales 256 B, row stats 128 B; single-buffered,
    a simdgroup barrier at the top of each group) and the chunk partials summed by all threads.
Unchanged: the grid (segments x column tiles; A: 2 projections x 4 K chunks, B: 2 tiles x 2 K chunks), the grouping
(ooo1cq's _GRP1CQ), the head (ooo1cq.a_one_call_q: wait on late slots, QOM drops, rs_m / rs_l posts), kernel B's
dropped-group zeros (ooo1cq.b_drop), every pool read coherent(system) (coh1cq's _body / _p11_hdr / _ren), the outputs.

    w4a8.tier_glu_nax6n_1cq_w4a8(...)   # ooo1cq.tier_glu_nax6n_1cq's signature and outputs; the knob is xknob.py"""
import os
import sys

import mlx.core as mx

for _p in ("/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_scripts", "/Users/kamen/BigMoE/data/r57/snap/frozen_s98b/Leap_third_party/ms_gdn"):
    _p = os.path.expanduser(_p)
    if _p not in sys.path:
        sys.path.append(_p)                               # read-only imports of Leap's modules (append, not insert)

_K = {}
SGW = 608                                                  # words per simdgroup slice: codes 256 + x 256 + sb 64 + rx 32

_QSRC = r"""
    // one simdgroup per (row, 64-value group): int8 codes rint(v * 127 / absmax), scale absmax / 127, float sum
    const uint lane = thread_index_in_simdgroup;
    const uint gi = threadgroup_position_in_grid.x * 8 + simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.y;
    if (gi >= (uint)G_C) return;
    const device T* src = a + (size_t)row * L_C + 64 * gi + 2 * lane;
    const float v0 = (float)src[0], v1 = (float)src[1];
    const float m = simd_max(max(fabs(v0), fabs(v1)));
    const float sm = simd_sum(v0 + v1);
    const float inv = m > 0.0f ? 127.0f / m : 0.0f;
    const int c0 = clamp((int)rint(v0 * inv), -127, 127), c1 = clamp((int)rint(v1 * inv), -127, 127);
    device char2* dst = (device char2*)(q + (size_t)row * L_C + 64 * gi + 2 * lane);
    *dst = char2((char)c0, (char)c1);
    if (lane == 0) { sc[(size_t)row * G_C + gi] = m / 127.0f; su[(size_t)row * G_C + gi] = sm; }
"""

_PRE_A1 = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // projection pj = sg & 1 (0 gate, 1 up), K chunk kc = sg >> 1
    const uint pj = sg & 1u, kc = sg >> 1;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier0 = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
"""
_PRE_A2 = r"""
    const int sr = lane >> 1, shf = lane & 1, rr = lane & 15;   // x staging: row sr, half shf; row stats: row rr
    const uint xr_s = (uint)pidx[q0 + (sr < nq ? sr : 0)] / NR;  // the tile's x rows (rows past nq repeat the first)
    const uint xr_r = (uint)pidx[q0 + (rr < nq ? rr : 0)] / NR;
    const device int8_t* xq_s = xq + (size_t)xr_s * K + 32 * shf;
    const device float* rx_src = (lane < 16 ? sx : xs) + (size_t)xr_r * (K / 64);
    const device uint32_t* w3p = pj == 0 ? w3g : w3u;
    const device uint* s3p = (const device uint*)(pj == 0 ? s3g : s3u);
    const device uint* b3p = (const device uint*)(pj == 0 ? b3g : b3u);
    const device uint* b2p = (const device uint*)(pj == 0 ? b2g : b2u);
    const device uint32_t* l1p = pj == 0 ? l1g : l1u;
    const device uint* blp = (const device uint*)(pj == 0 ? bg : bu);
    threadgroup uint32_t tg_all[2 * KS_C * SGW_C];               // per-simdgroup slices; later the chunk partials
    #define S_K K
    #define S_O N
"""
_PRE_B1 = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // column tile ti = sg % NSG_C, K chunk kc = sg / NSG_C
    const uint ti = sg % NSG_C, kc = sg / NSG_C;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = (threadgroup_position_in_grid.x * NSG_C + ti) * 32;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier0 = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pf];
    int tier = tier0;
    uint el = 0;
    if (tier == 4) {
        const int el_ = lsb_slot[(pf / NR) * R4_C + popcount((uint)MASK4_C & ((1u << r0) - 1u))];
        if (el_ < 0) tier = 3; else el = (uint)el_;                // LP1: no LSB read -> tier 3
    }
    const int sr = lane >> 1, shf = lane & 1, rr = lane & 15;   // h staging: row sr, half shf; row stats: row rr
    const uint xr_s = (uint)pidx[q0 + (sr < nq ? sr : 0)];       // B's rows are the pairs
    const uint xr_r = (uint)pidx[q0 + (rr < nq ? rr : 0)];
    const device int8_t* xq_s = hq + (size_t)xr_s * N + 32 * shf;
    const device float* rx_src = (lane < 16 ? hsc : hs) + (size_t)xr_r * (N / 64);
    const device uint32_t* w3p = w3d;
    const device uint* s3p = (const device uint*)s3d;
    const device uint* b3p = (const device uint*)b3d;
    const device uint* b2p = (const device uint*)b2d;
    const device uint32_t* l1p = l1d;
    const device uint* blp = (const device uint*)bd;
    threadgroup uint32_t tg_all[NSG_C * KS_C * SGW_C];
    #define S_K N
    #define S_O D
"""

_CORE = r"""
    constexpr int KG = S_K / 64; constexpr uint NB11 = S_O * KG / 32, PW11 = PWM_C, PWL11 = PWL_C;
    const device uint32_t* wrow = (const device uint32_t*)w3p + ((size_t)e * (S_O / 32) + n0 / 32) * (KG * 32 * 6) + lane * 6;
    const P11R srow = p11r(s3p, e, PW11, NB11, (n0 / 32) * (KG * 32) + lane);
    P11R brow = {};
    const device uint32_t* lrow = (const device uint32_t*)l1p;
    if (tier == 4) {
        lrow = (const device uint32_t*)l1p + ((size_t)el * (S_O / 32) + n0 / 32) * (KG * 32 * 2) + lane * 2;
        brow = p11r(blp, el, PWL11, NB11, (n0 / 32) * (KG * 32) + lane);
    } else {
        brow = p11r(tier == 2 ? b2p : b3p, e, PW11, NB11, (n0 / 32) * (KG * 32) + lane);
    }
    const float smul = tier == 4 ? 0.5f : 1.0f;
    threadgroup uint32_t* tbw = tg_all + sg * SGW_C;               // codes: 32 rows x 64 int4b (256 words)
    threadgroup int8_t* taw = (threadgroup int8_t*)(tbw + 256);    // x: 16 rows x 64 int8 (256 words)
    threadgroup float* sbw = (threadgroup float*)(tbw + 512);      // 32 scales s', 32 biases b + 8 s'
    threadgroup float* rxw = (threadgroup float*)(tbw + 576);      // 16 row scales, 16 row sums
    constexpr int GPC = KG / KS_C;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TA = tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<threadgroup int4b_format, dextents<int32_t, 2>, tensor_inline>;
    TA ta0(taw, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TA ta1(taw + 32, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB tb0((threadgroup uchar*)tbw, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB tb1((threadgroup uchar*)tbw + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, int32_t>();
    float acc[16];
    short ecol[16], erow[16];
    for (int i = 0; i < 16; ++i) {                              // x2_mma_layout: capacity 16, index (column, row)
        acc[i] = 0.0f;
        const auto ix = cg.get_multidimensional_index(i);
        ecol[i] = (short)ix[0]; erow[i] = (short)ix[1];
    }
    for (int g = kc * GPC; g < (kc + 1) * GPC; ++g) {
        simdgroup_barrier(mem_flags::mem_threadgroup);              // the previous group's operands are consumed
        const device uint32_t* c = wrow + 192 * g;
        const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
        const uint4 xa = *(const device uint4*)(xq_s + 64 * g);
        const uint4 xb = *(const device uint4*)(xq_s + 64 * g + 16);
        const float rxv = rx_src[g];
        uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                      w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
        uint nb[8];
        if (tier == 4) {
            const uint l0 = lrow[64 * g], l1 = lrow[64 * g + 1];
            for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu) ^ 0x88888888u;
        } else if (tier == 2) {
            for (int q = 0; q < 8; ++q) nb[q] = (spread3(ch[q]) & 0x66666666u) ^ 0x88888888u;
        } else {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) ^ 0x88888888u;
        }
        threadgroup uint4* dst = (threadgroup uint4*)(tbw + lane * 8);
        dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
        dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
        threadgroup uint4* xd = (threadgroup uint4*)(taw + sr * 64 + 32 * shf);
        xd[0] = xa;
        xd[1] = xb;
        const float sv = (float)srow[32 * g] * smul;
        sbw[lane] = sv;
        sbw[32 + lane] = (float)brow[32 * g] + 8.0f * sv;
        rxw[lane] = rxv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 16; ++i) cg[i] = 0;
        op.run(ta0, tb0, cg);
        op.run(ta1, tb1, cg);
        for (int i = 0; i < 16; ++i)
            acc[i] += sbw[ecol[i]] * rxw[erow[i]] * (float)cg[i] + sbw[32 + ecol[i]] * rxw[16 + erow[i]];
    }
"""

_POST_A = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);               // every simdgroup is done with its slice
    threadgroup float* part = (threadgroup float*)tg_all;           // [projection][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) part[(pj * KS_C + kc) * 512 + erow[i] * 32 + ecol[i]] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = thread_index_in_threadgroup; t < 512u; t += 2 * KS_C * 32) {   // gate / up chunk sums, swiglu
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
_POST_B = r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* part = (threadgroup float*)tg_all;           // [tile][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) part[(ti * KS_C + kc) * 512 + erow[i] * 32 + ecol[i]] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = thread_index_in_threadgroup; t < (uint)(NSG_C * 512); t += NSG_C * KS_C * 32) {
        const int tt = (int)(t >> 9), row = (int)((t >> 5) & 15), col = (int)(t & 31);
        float ys = part[(tt * KS_C) * 512 + row * 32 + col];
        for (int c = 1; c < KS_C; ++c) ys += part[(tt * KS_C + c) * 512 + row * 32 + col];
        if (row < nq) y[(size_t)pidx[q0 + row] * D + (threadgroup_position_in_grid.x * NSG_C + tt) * 32 + col] = static_cast<T>(ys);
    }
"""


def sources():
    """kernel A / B sources after Leap's transforms (head, drop, coherent pool reads); no kernel is built"""
    import ooo1c as OC
    import ooo1cq as OQ
    import coh1cq as CQ
    a = OQ.a_one_call_q(_PRE_A1 + OC._OLD_HEAD + "\n" + _PRE_A2 + _CORE + _POST_A)
    b = OQ.b_drop(_PRE_B1 + _CORE + _POST_B)
    return CQ._ren(CQ._body(a)), CQ._ren(CQ._body(b))


def _hdr():
    import tierglu_nax6 as N6
    import ooo1c as OC
    import coh1cq as CQ
    import p11 as P11
    return N6._HDR4 + P11.P11_METAL + OC.SYS + CQ._p11_hdr()


def _kq():
    if "q" not in _K:
        _K["q"] = mx.fast.metal_kernel(name="w4a8_quant64", input_names=["a"], output_names=["q", "sc", "su"], source=_QSRC)
    return _K["q"]


def _ka():
    if "a" not in _K:
        a, _ = sources()
        _K["a"] = mx.fast.metal_kernel(
            name="w4a8_a",
            input_names=["xq", "sx", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=a, header=_hdr())
    return _K["a"]


def _kb():
    if "b" not in _K:
        _, b = sources()
        _K["b"] = mx.fast.metal_kernel(
            name="w4a8_b",
            input_names=["hq", "hsc", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg",
                         "nseg"],
            output_names=["y"], source=b, header=_hdr())
    return _K["b"]


def quant64(a):
    """[rows, L] (bf16) -> (q [rows, L] int8, scale [rows, L / 64] f32, sum [rows, L / 64] f32)"""
    R_, L = a.shape
    G = L // 64
    return _kq()(inputs=[a], template=[("T", a.dtype), ("L_C", L), ("G_C", G)], grid=(((G + 7) // 8) * 256, R_, 1),
                 threadgroup=(256, 1, 1), output_shapes=[(R_, L), (R_, G), (R_, G)],
                 output_dtypes=[mx.int8, mx.float32, mx.float32])


def tier_glu_nax6n_1cq_w4a8(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2,
                            ks=4, ks_b=2):
    """ooo1cq.tier_glu_nax6n_1cq's signature and outputs (y [M, R, D], rs_m, rs_l) with the W4A8 kernels. xs is ignored
    (quant64 makes its own sums); the p11 store only"""
    import ooo1cq as OQ
    import p11 as P11
    from tierglu import plan_masks, PROJ
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    assert msb_n["gate_proj"]["s3"].dtype == mx.uint32, "W4A8 covers the p11 store only"
    N, D = msb_n["gate_proj"]["w3"].shape[1], msb_n["down_proj"]["w3"].shape[1]
    pwm, pwl = msb_n["gate_proj"]["s3"].shape[-1], lsb_n["gate_proj"]["b"].shape[-1]
    assert pwm >= P11.piece_words(N * 40) and pwm % 4 == 0 and pwl >= P11.piece_words(N * 40) and pwl % 4 == 0
    assert K % (64 * ks) == 0 and N % (64 * ks_b) == 0 and D % (32 * nsg_b) == 0
    if grp is None:
        grp = OQ.gpu_grp_1cq(pre_m, pre_l, ids, plan)
    pidx, seg, nsg_, nseg = grp
    bb = {p: msb_n[p]["b3"] for p in PROJ}
    ls = pre_l.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ms = pre_m.astype(mx.int32).reshape(-1)
    L_ = lsb_n
    g, u, d = (msb_n[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("PWM_C", pwm),
            ("PWL_C", pwl), ("SGW_C", SGW)]
    xq, sx, xs_ = quant64(x)
    h, rs_m, rs_l = _ka()(
        inputs=[xq, sx, xs_, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L_["gate_proj"]["l1"], L_["up_proj"]["l1"], L_["gate_proj"]["b"], L_["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
        threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    hq, hsc, hs = quant64(h)
    (y,) = _kb()(
        inputs=[hq, hsc, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L_["down_proj"]["l1"], L_["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_],
        template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D), rs_m, rs_l


if __name__ == "__main__":
    a, b = sources()
    print(f"w4a8 sources: A {len(a)} chars, B {len(b)} chars; coherent pool pointers: "
          f"{a.count('coherent(system) const device')} / {b.count('coherent(system) const device')}; "
          f"head polls: {a.count('oc_poll(')}; B drop: {b.count('FAST-NAX: a dropped group')}")
