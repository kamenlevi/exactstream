"""K-62i NAX5 (Leap, Sun 2026-11-01): NAX4 (tierglu_nax4.py) with its two rowsum64 dispatches folded in, for small calls
(K-62h: S 1 0.57x, S 3 0.8x vs 3c, partly dispatch cost):
- kernel A: the 64-value x sums from the A-operand loads (16 values per lane per group, fixed order), reduced over the 4
  lanes that share an A row (shuffle_xor 1, then 8: each step one commutative add, so all 4 lanes hold equal bits);
- kernel A's epilogue: per row the 32-column sum of the bf16 h it stores (same reduction) -> hs2 [P, N / 32]; kernel B's
  64-value h sum = hs2[2g] + hs2[2g + 1].
Dispatches per call: grouping, A, B (NAX4: + 2 rowsums). Per-pair bits differ from NAX4 (the sums' order), still
batch-invariant by construction (every sum reads only its own row). tier_glu_nax5(...) == tier_glu_nax4's signature."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
from tierglu import _HDR, PROJ, plan_masks
from tierglu_grouped import make_grp
from tierglu_nax import rowsum64

_HDR4 = _HDR + r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline uint spread3(uint x) {                              // 8 3-bit codes (24 bits) -> 8 nibbles (code in bits 0-2)
    uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
    a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
    return (a & 0x07070707u) | ((a & 0x38383838u) << 1);
}
"""

# S_K reduction length, S_O output rows of the projection; e / el / tier / n0 / lane / sg set by the caller; ar0 / ar1 the
# A rows (bf16, row starts), as0 / as1 their 64-value sums; result in acc[16] (C element i: n = 16 (i >> 3) + fn0 + (i & 3),
# m = fm0 + 8 ((i >> 2) & 1))
_CORE4 = r"""
    constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8;
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const device uint32_t* wrow = (const device uint32_t*)((const device uint8_t*)w3p + ((size_t)e * S_O + n0 + (DIAG_C == 3 ? 0 : lane)) * KB3);
    const device T* srow = s3p + ((size_t)e * S_O + n0 + lane) * KG;
    const device T* brow;
    const device uint32_t* lrow = (const device uint32_t*)l1p;
    if (tier == 4) {
        lrow = (const device uint32_t*)((const device uint8_t*)l1p + ((size_t)el * S_O + n0 + lane) * KL);
        brow = blp + ((size_t)el * S_O + n0 + lane) * KG;
    } else {
        brow = (tier == 2 ? b2p : b3p) + ((size_t)e * S_O + n0 + lane) * KG;
    }
    const float smul = tier == 4 ? 0.5f : 1.0f;
    threadgroup uint32_t* tbw = tb_all + sg * (2 * 32 * 8 * GS_C);  // 2 buffers x 32 rows x 8 GS_C words (64 GS_C nibbles)
    threadgroup float* sbw = sb_all + sg * (2 * 64 * GS_C);          // 2 buffers x GS_C x (32 scales, 32 biases)
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int gp = 0; gp < KG / GS_C; ++gp) {                   // GS_C quant groups per decode stage
        threadgroup uint32_t* tb = tbw + (gp & 1) * (32 * 8 * GS_C);
        threadgroup float* sb = sbw + (gp & 1) * (64 * GS_C);
        for (int j = 0; j < GS_C; ++j) {                          // decode this lane's row: 64 codes -> 64 nibbles per group
            const int g = gp * GS_C + j;
            const device uint32_t* c = wrow + 6 * g;
            const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
            uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                          w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
            uint nb[8];
            if (tier == 4) {
                const uint l0 = lrow[2 * g], l1 = lrow[2 * g + 1];
                for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
            } else if (tier == 2) {
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & 0x66666666u;
            } else {
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
            }
            threadgroup uint4* dst = (threadgroup uint4*)(tb + lane * 8 * GS_C + 8 * j);
            dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
            dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
            sb[64 * j + lane] = (float)srow[g] * smul;
            sb[64 * j + 32 + lane] = (float)brow[g];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < GS_C; ++j) {
            const int g = gp * GS_C + j;
            for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
#if XS_LOADS
            float xsum0 = 0.0f, xsum1 = 0.0f;
#endif
            for (int hh = 0; hh < 2; ++hh) {
                const int k0 = 64 * g + 32 * hh;
                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
                for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
#if XS_LOADS
                for (int q = 0; q < 4; ++q) { xsum0 += (float)a00[q]; xsum1 += (float)a10[q]; }
                for (int q = 0; q < 4; ++q) { xsum0 += (float)a01[q]; xsum1 += (float)a11[q]; }
#endif
                TB tbt((threadgroup uchar*)tb + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                op.run(ca, tbt, cg);
            }
#if XS_LOADS
            xsum0 += simd_shuffle_xor(xsum0, 1); xsum1 += simd_shuffle_xor(xsum1, 1);
            xsum0 += simd_shuffle_xor(xsum0, 8); xsum1 += simd_shuffle_xor(xsum1, 8);
#endif
    #if XS_LOADS
        const float x0 = xsum0, x1 = xsum1;
#else
        const float x0 = as0[2 * g] + as0[2 * g + 1], x1 = as1[2 * g] + as1[2 * g + 1];
#endif
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + fn0 + (i & 3);
                acc[i] += sb[64 * j + col] * cg[i] + sb[64 * j + 32 + col] * (((i >> 2) & 1) ? x1 : x0);
            }
        }
    }
"""

_A4 = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0: gate, 1: up
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
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
    const short fmA = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint ma0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)] / NR, ma1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)] / NR;
    const device T* ar0 = x + (size_t)ma0 * K;
    const device T* ar1 = x + (size_t)ma1 * K;
    const device float* as0 = nullptr;
    const device float* as1 = nullptr;
    const device uint32_t* w3p = sg == 0 ? w3g : w3u;
    const device T* s3p = sg == 0 ? s3g : s3u;
    const device T* b3p = sg == 0 ? b3g : b3u;
    const device T* b2p = sg == 0 ? b2g : b2u;
    const device uint32_t* l1p = sg == 0 ? l1g : l1u;
    const device T* blp = sg == 0 ? bg : bu;
    threadgroup uint32_t tb_all[2 * 2 * 32 * 8 * GS_C];
    threadgroup float sb_all[2 * 2 * 64 * GS_C];
    #define S_K K
    #define S_O N
    #define XS_LOADS 1
""" + _CORE4 + r"""
    threadgroup T gate_t[16][32];
    if (sg == 0) {
        for (int i = 0; i < 16; ++i) gate_t[fm0 + ((i >> 2) & 1) * 8][16 * (i >> 3) + fn0 + (i & 3)] = static_cast<T>(acc[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1) {
        float hs_0 = 0.0f, hs_1 = 0.0f;                          // this lane's 8 columns of rows fm0 and fm0 + 8
        for (int i = 0; i < 16; ++i) {
            const int row = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
            const T gv = gate_t[row][col];
            const T ub = static_cast<T>(acc[i]);
            const T hv = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);
            if ((i >> 2) & 1) hs_1 += (float)hv; else hs_0 += (float)hv;
            if (row < nq) h[(size_t)pidx[q0 + row] * N + n0 + col] = hv;
        }
        hs_0 += simd_shuffle_xor(hs_0, 1); hs_1 += simd_shuffle_xor(hs_1, 1);
        hs_0 += simd_shuffle_xor(hs_0, 8); hs_1 += simd_shuffle_xor(hs_1, 8);
        if (fn0 == 0) {                                           // the 32-column sums, one lane per row pair
            if (fm0 < nq) hs2[(size_t)pidx[q0 + fm0] * (N / 32) + n0 / 32] = hs_0;
            if (fm0 + 8 < nq) hs2[(size_t)pidx[q0 + fm0 + 8] * (N / 32) + n0 / 32] = hs_1;
        }
    }
"""

_B4 = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = (threadgroup_position_in_grid.x * nsg + sg) * 32;
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
    const short fmA = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint pa0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)], pa1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)];
    const device T* ar0 = h + (size_t)pa0 * N;
    const device T* ar1 = h + (size_t)pa1 * N;
    const device float* as0 = hs2 + (size_t)pa0 * (N / 32);
    const device float* as1 = hs2 + (size_t)pa1 * (N / 32);
    const device uint32_t* w3p = w3d;
    const device T* s3p = s3d;
    const device T* b3p = b3d;
    const device T* b2p = b2d;
    const device uint32_t* l1p = l1d;
    const device T* blp = bd;
    threadgroup uint32_t tb_all[NSG_C * 2 * 32 * 8 * GS_C];
    threadgroup float sb_all[NSG_C * 2 * 64 * GS_C];
    #define S_K N
    #define S_O D
    #define XS_LOADS 0
""" + _CORE4 + r"""
    for (int i = 0; i < 16; ++i) {
        const int row = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
        if (row < nq) y[(size_t)pidx[q0 + row] * D + n0 + col] = static_cast<T>(acc[i]);
    }
"""


_GRP = r"""
    // one threadgroup, thread i = pair i (P <= 1024): key = slot << 11 | tier code << 9 | (sidecar slot + 1 for tier 4)
    const int P = P_C;
    const uint i = thread_position_in_grid.x;
    threadgroup int key_s[P_C];
    threadgroup int skey[P_C];
    threadgroup int start_s[P_C];
    if ((int)i < P) {
        const int r = (int)i % NR_C, m = (int)i / NR_C;
        int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0;
        if (tier == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el < 0) { tier = 3; el = 0; } else el += 1;
        }
        key_s[i] = (msb_slot[i] << 11) | ((tier - 2) << 9) | el;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        const int k = key_s[i];
        int rank = 0;
        for (int j = 0; j < P; ++j) { const int kj = key_s[j]; rank += (kj < k) || (kj == k && j < (int)i); }
        pidx[rank] = (int)i;
        skey[rank] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {                                          // a group starts at a new key, and every 16 pairs inside one
        int r0 = (int)i;
        while (r0 > 0 && skey[r0 - 1] == skey[i]) --r0;
        start_s[i] = (((int)i - r0) % 16) == 0 ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && start_s[i]) {
        int g = 0;
        for (int j = 0; j < (int)i; ++j) g += start_s[j];
        seg[g] = (int)i;
    }
    if ((int)i == 0) {
        int n = 0;
        for (int j = 0; j < P; ++j) n += start_s[j];
        nseg[0] = n;
        seg[n] = P;
    }
"""


@lru_cache(None)
def _kgrp():
    return mx.fast.metal_kernel(name="tierglu_nax5_grp", input_names=["msb_slot", "lsb_slot"],
                                output_names=["pidx", "seg", "nseg"], source=_GRP)


def gpu_grp(msb_slot, lsb_slot, plan):
    """the pair grouping on the GPU (no host sync): (pidx [P], seg [P + 1], nseg [1], P) for tier_glu_nax4(grp=...);
    groups past nseg are empty and their threadgroups return at once"""
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = msb_slot.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024, P
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    pidx, seg, nseg = _kgrp()(inputs=[ms, ls], template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2),
                                                          ("R4_C", max(r4, 1))],
                              grid=(max(32, (P + 31) // 32 * 32), 1, 1), threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                              output_shapes=[(P,), (P + 1,), (1,)], output_dtypes=[mx.int32] * 3)
    return pidx, seg, nseg, P


@lru_cache(None)
def _ka4():
    return mx.fast.metal_kernel(
        name="tierglu_nax5_a",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h", "hs2"], source=_A4, header=_HDR4)


@lru_cache(None)
def _kb4():
    return mx.fast.metal_kernel(
        name="tierglu_nax5_b",
        input_names=["h", "hs2", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_B4, header=_HDR4)


def tier_glu_nax5(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, grp=None, nsg_b=2, ms_np=None, ls_np=None,
                  diag=0, gs=1):
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 64 and K % 64 == 0 and N % 64 == 0 and D % (32 * nsg_b) == 0     # groups are capped at 16 pairs
    if grp is None:
        grp = make_grp(msb_slot, lsb_slot, plan, M, ms_np, ls_np, 16)
    pidx, seg, nsg_, nseg = grp
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    if r4:
        ls = lsb_slot.astype(mx.int32).reshape(-1)
        L = lsb
    else:
        ls = mx.zeros((1,), mx.int32)
        L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("DIAG_C", diag), ("GS_C", gs)]
    h, hs2 = _ka4()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N)],
                  grid=(N // 32 * 64, nseg, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(M * R, N), (M * R, N // 32)], output_dtypes=[x.dtype, mx.float32])
    (y,) = _kb4()(inputs=[h, hs2, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b, nseg, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
