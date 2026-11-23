"""tgl16.py (Breakthrough, Tue 17 Nov 2026 10:3x; EXPERT REWRITE candidate 2, docs/EXPERT_REWRITE.md): the FAST-NAX
expert one-call with the MMA's LEFT operand (the tile's 16 bf16 activation rows) staged in threadgroup memory once per
quant group, instead of each lane loading its fragment from device memory into a cooperative tensor (Leap's form).
The operands, their types and the per-group epilogue are ooo1cq's (bf16 x uint4b -> float; acc += s' cg + b xs with
xs = rowsum64), so the arithmetic is the same; whether the accelerator's sums come out BITWISE equal is checked, not
assumed (xrw_check.py). X1's smoke (shared GPU): the threadgroup left cut the MMA path ~13 %.
Shares w4a8.py's grid, head (ooo1cq.a_one_call_q), drop (ooo1cq.b_drop), coherent pool reads (coh1cq), per-simdgroup
threadgroup slices (codes 1 KB, x 2 KB, scales 256 B, row sums 64 B) and layout-agnostic partials.
    tier_glu_nax6n_1cq_tgl16(...)   # ooo1cq.tier_glu_nax6n_1cq's signature and outputs"""
import os
import sys

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import w4a8 as W                                            # noqa: E402  (its preludes / posts; Leap's paths appended)

SGW = 848                                                  # words per simdgroup slice: codes 256 + x 512 + sb 64 + rx 16
_K = {}

_PRE_A2 = r"""
    const int sr = lane >> 1, shf = lane & 1, rr = lane & 15;   // x staging: row sr, half shf; row sums: row rr
    const uint xr_s = (uint)pidx[q0 + (sr < nq ? sr : 0)] / NR;  // the tile's x rows (rows past nq repeat the first)
    const uint xr_r = (uint)pidx[q0 + (rr < nq ? rr : 0)] / NR;
    const device T* xb_s = x + (size_t)xr_s * K + 32 * shf;
    const device float* rx_src = xs + (size_t)xr_r * (K / 64);
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
_PRE_B_STAGE_OLD = r"""    const device int8_t* xq_s = hq + (size_t)xr_s * N + 32 * shf;
    const device float* rx_src = (lane < 16 ? hsc : hs) + (size_t)xr_r * (N / 64);"""
_PRE_B_STAGE_NEW = r"""    const device T* xb_s = h + (size_t)xr_s * N + 32 * shf;
    const device float* rx_src = hs + (size_t)xr_r * (N / 64);"""

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
    threadgroup uint32_t* tbw = tg_all + sg * SGW_C;               // codes: 32 rows x 64 uint4b (256 words)
    threadgroup T* taw = (threadgroup T*)(tbw + 256);               // x: 16 rows x 64 bf16 (512 words)
    threadgroup float* sbw = (threadgroup float*)(tbw + 768);      // 32 scales s', 32 biases b
    threadgroup float* rxw = (threadgroup float*)(tbw + 832);      // 16 row sums (rowsum64)
    constexpr int GPC = KG / KS_C;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    using TA = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    TA ta0(taw, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TA ta1(taw + 32, dextents<int32_t, 2>(32, 16), array<int32_t, 2>{1, 64});
    TB tb0((threadgroup uchar*)tbw, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    TB tb1((threadgroup uchar*)tbw + 16, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
    auto cg = op.template get_destination_cooperative_tensor<TA, TB, float>();
    float acc[16];
    short ecol[16], erow[16];
    for (int i = 0; i < 16; ++i) {
        acc[i] = 0.0f;
        const auto ix = cg.get_multidimensional_index(i);
        ecol[i] = (short)ix[0]; erow[i] = (short)ix[1];
    }
    for (int g = kc * GPC; g < (kc + 1) * GPC; ++g) {
        simdgroup_barrier(mem_flags::mem_threadgroup);              // the previous group's operands are consumed
        const device uint32_t* c = wrow + 192 * g;
        const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
        const device uint4* xsrc = (const device uint4*)(xb_s + 64 * g);
        const uint4 xa = xsrc[0], xb = xsrc[1], xc = xsrc[2], xd4 = xsrc[3];
        const float rxv = rx_src[g];
        uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                      w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
        uint nb[8];
        if (tier == 4) {
            const uint l0 = lrow[64 * g], l1 = lrow[64 * g + 1];
            for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
        } else if (tier == 2) {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
        } else {
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
        }
        threadgroup uint4* dst = (threadgroup uint4*)(tbw + lane * 8);
        dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
        dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
        threadgroup uint4* xd = (threadgroup uint4*)(taw + sr * 64 + 32 * shf);
        xd[0] = xa; xd[1] = xb; xd[2] = xc; xd[3] = xd4;
        sbw[lane] = (float)srow[32 * g] * smul;
        sbw[32 + lane] = (float)brow[32 * g];
        if (lane < 16) rxw[lane] = rxv;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        op.run(ta0, tb0, cg);
        op.run(ta1, tb1, cg);
        for (int i = 0; i < 16; ++i)
            acc[i] += sbw[ecol[i]] * cg[i] + sbw[32 + ecol[i]] * rxw[erow[i]];
    }
"""


def sources():
    import ooo1c as OC
    import ooo1cq as OQ
    import coh1cq as CQ
    a = OQ.a_one_call_q(W._PRE_A1 + OC._OLD_HEAD + "\n" + _PRE_A2 + _CORE + W._POST_A)
    pb = W._PRE_B1
    assert pb.count(_PRE_B_STAGE_OLD) == 1
    b = OQ.b_drop(pb.replace(_PRE_B_STAGE_OLD, _PRE_B_STAGE_NEW) + _CORE + W._POST_B)
    return CQ._ren(CQ._body(a)), CQ._ren(CQ._body(b))


def _ka():
    if "a" not in _K:
        a, _ = sources()
        _K["a"] = mx.fast.metal_kernel(
            name="tgl16_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=a, header=W._hdr())
    return _K["a"]


def _kb():
    if "b" not in _K:
        _, b = sources()
        _K["b"] = mx.fast.metal_kernel(
            name="tgl16_b",
            input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
            output_names=["y"], source=b, header=W._hdr())
    return _K["b"]


def tier_glu_nax6n_1cq_tgl16(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2,
                             ks=4, ks_b=2):
    """ooo1cq.tier_glu_nax6n_1cq's signature and outputs (y [M, R, D], rs_m, rs_l); the p11 store only"""
    import ooo1cq as OQ
    import p11 as P11
    from tierglu import plan_masks, PROJ
    from tierglu_nax import rowsum64
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert m2 == 0
    assert msb_n["gate_proj"]["s3"].dtype == mx.uint32, "tgl16 covers the p11 store only"
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
    xs = rowsum64(x) if xs is None else xs
    h, rs_m, rs_l = _ka()(
        inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                L_["gate_proj"]["l1"], L_["up_proj"]["l1"], L_["gate_proj"]["b"], L_["up_proj"]["b"], ms, ls, pidx, seg, nsg_,
                region, offs, status],
        template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)], grid=(N // 32 * 64 * ks, nseg, 1),
        threadgroup=(64 * ks, 1, 1), output_shapes=[(M * R, N), (M * R,), (max(M * r4, 1),)],
        output_dtypes=[x.dtype, mx.int32, mx.int32])
    hs = rowsum64(h)
    (y,) = _kb()(
        inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L_["down_proj"]["l1"], L_["down_proj"]["b"],
                rs_m, rs_l, pidx, seg, nsg_],
        template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
        grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
        output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D), rs_m, rs_l


if __name__ == "__main__":
    a, b = sources()
    print(f"tgl16 sources: A {len(a)} chars, B {len(b)} chars; coherent pool pointers: "
          f"{a.count('coherent(system) const device')} / {b.count('coherent(system) const device')}; "
          f"head polls: {a.count('oc_poll(')}; B drop: {b.count('FAST-NAX: a dropped group')}")
