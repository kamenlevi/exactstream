"""o1p.py (Breakthrough, Tue 17 Nov 2026 11:3x; EXPERT REWRITE / MOE BLOCK): ooo1cq's NAX6-N kernels (cooperative left
from device, double-buffered code tiles, COH reads) with a PIPELINED group loop. Each iteration issues group g's two MMAs,
then decodes group g + 1 into the other buffer, then runs group g's epilogue (which uses the MMA result).
- X7 (x7_async.py): a simdgroup's own ALU work hides ~50 % behind its MMA (~180 ns of latency per 16x32x32 op).
- ooo1cq already double-buffers its code tiles (the gp & 1 buffers), so threadgroup memory is unchanged (~20 KB).
- The ops and their accumulation order are ooo1cq's, so the output should be BITWISE == ooo1cq (checked).
- GS_C = 1 only (the live value).
    tier_glu_nax6n_1cq_o1p(...)   # ooo1cq.tier_glu_nax6n_1cq's signature and outputs (grp may be the upgrade's)"""
import os
import sys

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import w4a8 as W                                            # noqa: E402  (Leap's paths; _hdr)

_K = {}

# the pipelined core: ooo1cq's _CORE_NP (tierglu_nax6n) with GS_C = 1 and the decode of g + 1 between g's MMA and epilogue
_CORE_P = r"""
    constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8; constexpr uint NB11 = S_O * KG / 32, PW11 = PWM_C, PWL11 = PWL_C;
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
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
    threadgroup uint32_t* tbw = tb_all + sg * (2 * 32 * 8);             // 2 buffers x 32 rows x 8 words
    threadgroup float* sbw = sb_all + sg * (2 * 64);                     // 2 buffers x (32 scales, 32 biases)
    constexpr int GPC = KG / KS_C;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    const int g0 = kc * GPC, g1 = (kc + 1) * GPC;
#define XRW_DECODE(G_, BUF_)                                                                                     \
    {                                                                                                            \
        threadgroup uint32_t* tbd = tbw + (BUF_) * 256;                                                          \
        threadgroup float* sbd = sbw + (BUF_) * 64;                                                              \
        const device uint32_t* c = wrow + 192 * (G_);                                                            \
        const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];                              \
        uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8, \
                      w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};  \
        uint nb[8];                                                                                              \
        if (tier == 4) {                                                                                         \
            const uint l0 = lrow[64 * (G_)], l1 = lrow[64 * (G_) + 1];                                           \
            for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);     \
        } else if (tier == 2) {                                                                                  \
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;                                    \
        } else {                                                                                                 \
            for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);                                                  \
        }                                                                                                        \
        threadgroup uint4* dst = (threadgroup uint4*)(tbd + lane * 8);                                           \
        dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);                                                              \
        dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);                                                              \
        sbd[lane] = (float)srow[32 * (G_)] * smul;                                                               \
        sbd[32 + lane] = (float)brow[32 * (G_)];                                                                 \
    }
    XRW_DECODE(g0, 0)
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = g0; g < g1; ++g) {
        const int buf = (g - g0) & 1;
        threadgroup uint32_t* tb = tbw + buf * 256;
        threadgroup float* sb = sbw + buf * 64;
        for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
        for (int hh = 0; hh < 2; ++hh) {
            const int k0 = 64 * g + 32 * hh;
            const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
            const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
            const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
            const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
            for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
            TB tbt((threadgroup uchar*)tb + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64});
            op.run(ca, tbt, cg);
        }
        if (g + 1 < g1) XRW_DECODE(g + 1, buf ^ 1)                      // overlaps g's MMAs (X7)
        const float x0 = as0[g], x1 = as1[g];
        for (int i = 0; i < 16; ++i) {
            const int col = 16 * (i >> 3) + fn0 + (i & 3);
            acc[i] += sb[col] * cg[i] + sb[32 + col] * (((i >> 2) & 1) ? x1 : x0);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);                  // g + 1's tile visible; g's buffer free
    }
#undef XRW_DECODE
"""


def sources():
    import tierglu_nax6n as NN
    import ooo1cq as OQ
    import coh1cq as CQ
    a_src = NN._A_NP
    b_src = NN._B_NP
    assert a_src.count(NN._CORE_NP) == 1 and b_src.count(NN._CORE_NP) == 1
    a = OQ.a_one_call_q(a_src.replace(NN._CORE_NP, _CORE_P))
    b = OQ.b_drop(b_src.replace(NN._CORE_NP, _CORE_P))
    return CQ._ren(CQ._body(a)), CQ._ren(CQ._body(b))


def _ka():
    if "a" not in _K:
        a, _ = sources()
        _K["a"] = mx.fast.metal_kernel(
            name="o1p_a",
            input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                         "msb_slot", "lsb_slot", "pidx", "seg", "nseg", "region", "offs", "status"],
            output_names=["h", "rs_m", "rs_l"], source=a, header=W._hdr())
    return _K["a"]


def _kb():
    if "b" not in _K:
        _, b = sources()
        _K["b"] = mx.fast.metal_kernel(
            name="o1p_b",
            input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
            output_names=["y"], source=b, header=W._hdr())
    return _K["b"]


def tier_glu_nax6n_1cq_o1p(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2,
                           ks=4, ks_b=2):
    """ooo1cq.tier_glu_nax6n_1cq with the pipelined kernels: ooo1cq's own function (not a knob's wrapper) with its _ka / _kb
    swapped for this call, as coh1cq does"""
    import ooo1cq as OQ
    import xknob
    f = xknob._ORIG[0] if xknob._ORIG[0] is not None else OQ.tier_glu_nax6n_1cq
    k0, k1 = OQ._ka, OQ._kb
    try:
        OQ._ka, OQ._kb = (lambda p11: _ka()), (lambda p11: _kb())
        return f(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=xs, grp=grp, nsg_b=nsg_b, ks=ks,
                 ks_b=ks_b)
    finally:
        OQ._ka, OQ._kb = k0, k1


def tier_glu_nax6n_1cq_o1pu(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=None, grp=None, nsg_b=2,
                            ks=4, ks_b=2):
    """o1p + the tier-4 upgrade grouping (xgrp v2)"""
    import xgrp
    if grp is None:
        grp = xgrp.gpu_grp_up2(pre_m, pre_l, ids, plan)
    return tier_glu_nax6n_1cq_o1p(x, msb_n, lsb_n, pre_m, pre_l, ids, plan, region, offs, status, xs=xs, grp=grp,
                                  nsg_b=nsg_b, ks=ks, ks_b=ks_b)


if __name__ == "__main__":
    a, b = sources()
    print(f"o1p sources: A {len(a)} chars, B {len(b)} chars; coherent pointers {a.count('coherent(system) const device')} / "
          f"{b.count('coherent(system) const device')}; head polls {a.count('oc_poll(')}; drop {b.count('FAST-NAX: a dropped group')}")
