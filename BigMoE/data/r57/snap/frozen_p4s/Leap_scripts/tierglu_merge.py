"""TIER-MERGE (Leap, Wed 2026-11-04; TABLE_350 'TIER-MERGE'; FableGod EXPERT_KERNEL.md design B): NAX6 (tierglu_nax6.py)
with one pair group per expert whatever its pairs' tiers. NAX6 groups by (expert, tier), so an expert picked at a 4-bit
rank by some rows and a 3-bit rank by others gets two groups: its codes loaded and decoded twice, two epilogues, two
threadgroup rows per output tile (S 16: 3,680 groups per check vs 3,112 distinct experts).
Merged grouping key: slot << 11 | tier-2 flag << 10 | the slot's sidecar slot + 1 (a 3-bit pair adopts the smallest
one among its slot's 4-bit pairs, 0 if none), so a group holds 3- and 4-bit rows of one expert (<= 16 pairs).
  mode 1, MERGE-X (EXACT): one code load and decode per group feeds both nibble sets; the op runs on the 3-bit nibbles
    for the 3-bit rows and on the 4-bit nibbles (q = 2m + l) for the 4-bit rows, both over all 16 op rows; every row
    takes NAX6's epilogue with its own operands. A pair's bits == NAX6's: the op's rows are independent (INV), the
    split-K chunks and the epilogue expression are NAX6's. Groups with one tier keep NAX6's double-buffered path; a
    group with both tiers uses the two buffers for the two nibble sets.
  mode 2, MERGE-B (a reference change): one op on the 3-bit nibbles for all rows; a 4-bit row adds L = x . l (its LSB
    plane: a masked fp32 sum over each 64-group in index order, on the SIMD ALUs) as P4' = 2 P3 + L, then NAX6's 4-bit
    epilogue (scale s3 / 2, the LSB part's bias). P4' == NAX6's P4 wherever the fp32 sums are exact.
tier_glu_merge(x, msb, lsb, msb_slot, lsb_slot, plan, b2=None, grp=None, nsg_b=2, ks=4, ks_b=2, xs=None, mode=1):
tier_glu_nax6's signature; grp must come from gpu_grp_merge / gpu_grp_phase_merge (NAX6's groups hold one tier each)."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
from tierglu import PROJ, plan_masks
from tierglu_nax import rowsum64
from tierglu_nax6 import _HDR4

# the group's rows: has3 / has4 (a 4-bit row = a 4-bit rank with a sidecar slot), t2g (tier-2 group), el (the 4-bit
# rows' sidecar slot: one per group by the key), m4 (bit j = row j is 4-bit)
_ROWS = r"""
    bool has3 = false, has4 = false, t2g = false;
    uint el = 0, m4 = 0;
    for (int j = 0; j < nq; ++j) {
        const uint pq = (uint)pidx[q0 + j], rq = pq % NR;
        bool t4 = ((MASK4_C >> rq) & 1) != 0;
        if (t4) {
            const int el_ = lsb_slot[(pq / NR) * R4_C + popcount((uint)MASK4_C & ((1u << rq) - 1u))];
            if (el_ < 0) t4 = false; else el = (uint)el_;            // LP1: no LSB read -> 3-bit
        }
        if (t4) { has4 = true; m4 |= 1u << j; }
        else { has3 = true; if ((MASK2_C >> rq) & 1) t2g = true; }
    }
"""

# NAX6's _CORE4 for merged groups. S_K reduction length, S_O output rows; e / el / has3 / has4 / t2g / m4 / n0 / lane /
# sg / kc / nq set by the caller; ar0 / ar1 the A rows, as0 / as1 their 64-value sums; XROW(m) row m's activations
# (MERGE-B's LSB term). Result in acc[16] (C element i: n = 16 (i >> 3) + fn0 + (i & 3), m = fm0 + 8 ((i >> 2) & 1)).
_CORE4M = r"""
    constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8;
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const bool r4a = ((m4 >> fm0) & 1u) != 0, r4b = ((m4 >> (fm0 + 8)) & 1u) != 0;
    const device uint32_t* wrow = (const device uint32_t*)((const device uint8_t*)w3p + ((size_t)e * S_O + n0 + lane) * KB3);
    const device T* srow = s3p + ((size_t)e * S_O + n0 + lane) * KG;
    const device T* brow3 = (t2g ? b2p : b3p) + ((size_t)e * S_O + n0 + lane) * KG;
    const device uint32_t* lrow = (const device uint32_t*)l1p;
    const device T* brow4 = blp;
    if (has4) {
        lrow = (const device uint32_t*)((const device uint8_t*)l1p + ((size_t)el * S_O + n0 + lane) * KL);
        brow4 = blp + ((size_t)el * S_O + n0 + lane) * KG;
    }
    const bool use3 = MODE_C == 2 || has3;                          // ops on the 3-bit nibbles
    const bool use4 = MODE_C == 1 && has4;                          // ops on the 4-bit nibbles (MERGE-X)
    const bool both = use3 && use4;                                 // both sets: the two buffers, one stage at a time
    threadgroup uint32_t* tbw = tb_all + sg * (2 * 32 * 8 * GS_C);  // 2 buffers x 32 rows x 8 GS_C words (64 GS_C nibbles)
    threadgroup float* sbw = sb_all + sg * (2 * 128 * GS_C);         // 2 buffers x GS_C x (s3, b3, s3 / 2, b4) x 32
    constexpr int GPC = KG / GS_C / KS_C;                            // decode stages per K chunk
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
    using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
    auto cg3 = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    auto cg4 = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {        // this simdgroup's K chunk; GS_C quant groups per stage
        threadgroup uint32_t* tb3 = tbw + (both ? 0 : (gp & 1) * (32 * 8 * GS_C));
        threadgroup uint32_t* tb4 = both ? tbw + 32 * 8 * GS_C : tb3;
        threadgroup float* sb = sbw + (gp & 1) * (128 * GS_C);
        if (both) simdgroup_barrier(mem_flags::mem_threadgroup);   // the last stage's ops are done with both sets
        for (int j = 0; j < GS_C; ++j) {                          // decode this lane's row: 64 codes -> 64 nibbles per group
            const int g = gp * GS_C + j;
            const device uint32_t* c = wrow + 6 * g;
            const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
            uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                          w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
            if (use3) {
                uint nb[8];
                const uint msk = t2g ? 0x66666666u : 0xFFFFFFFFu;
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]) & msk;
                threadgroup uint4* dst = (threadgroup uint4*)(tb3 + lane * 8 * GS_C + 8 * j);
                dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
                dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
            }
            if (use4) {
                uint nb[8];
                const uint l0 = lrow[2 * g], l1 = lrow[2 * g + 1];
                for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
                threadgroup uint4* dst = (threadgroup uint4*)(tb4 + lane * 8 * GS_C + 8 * j);
                dst[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
                dst[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
            }
            const float s = (float)srow[g];
            sb[128 * j + lane] = s * 1.0f;
            sb[128 * j + 32 + lane] = (float)brow3[g];
            sb[128 * j + 64 + lane] = s * 0.5f;
            sb[128 * j + 96 + lane] = has4 ? (float)brow4[g] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < GS_C; ++j) {
            const int g = gp * GS_C + j;
            for (int i = 0; i < 16; ++i) { cg3[i] = 0.0f; cg4[i] = 0.0f; }
            for (int hh = 0; hh < 2; ++hh) {
                const int k0 = 64 * g + 32 * hh;
                const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
                const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
                const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
                const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
                for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
                if (use3) {
                    TB t3((threadgroup uchar*)tb3 + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                    op.run(ca, t3, cg3);
                }
                if (use4) {
                    TB t4((threadgroup uchar*)tb4 + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                    op.run(ca, t4, cg4);
                }
            }
            float lv[16];
            for (int i = 0; i < 16; ++i) lv[i] = 0.0f;
            if (MODE_C == 2 && has4) {                            // MERGE-B: L = x . l per 4-bit row, lane = output column
                const uint l0 = lrow[2 * g], l1 = lrow[2 * g + 1];
                for (int m = 0; m < nq; ++m) {
                    if (((m4 >> m) & 1u) == 0) continue;
                    const device vec<T, 4>* xr = (const device vec<T, 4>*)(XROW(m) + 64 * g);
                    float L = 0.0f;
                    for (int kk = 0; kk < 8; ++kk) {
                        const vec<T, 4> v = xr[kk];
                        for (int t = 0; t < 4; ++t) L += ((l0 >> (4 * kk + t)) & 1u) ? (float)v[t] : 0.0f;
                    }
                    for (int kk = 0; kk < 8; ++kk) {
                        const vec<T, 4> v = xr[8 + kk];
                        for (int t = 0; t < 4; ++t) L += ((l1 >> (4 * kk + t)) & 1u) ? (float)v[t] : 0.0f;
                    }
                    for (int i = 0; i < 16; ++i) {
                        const float v = simd_shuffle(L, (ushort)(16 * (i >> 3) + fn0 + (i & 3)));
                        if (fm0 + 8 * ((i >> 2) & 1) == m) lv[i] = v;
                    }
                }
            }
            const float x0 = as0[g], x1 = as1[g];
            for (int i = 0; i < 16; ++i) {
                const int col = 16 * (i >> 3) + fn0 + (i & 3);
                const bool r4 = ((i >> 2) & 1) ? r4b : r4a;
                const float cv = MODE_C == 1 ? (r4 ? (float)cg4[i] : (float)cg3[i]) : (r4 ? 2.0f * (float)cg3[i] + lv[i] : (float)cg3[i]);
                const float sv = sb[128 * j + (r4 ? 64 : 0) + col];
                const float bv = sb[128 * j + (r4 ? 96 : 32) + col];
                acc[i] += sv * cv + bv * (((i >> 2) & 1) ? x1 : x0);
            }
        }
    }
"""

# mode 3, MERGE-X2 (EXACT, lean): single-tier groups run NAX6's core verbatim (tierglu_nax6._CORE4: the same
# registers and hot loop); a group with both tiers decodes the codes once into the two buffers (no double buffering) and
# runs two passes with one op tensor: the op on the 3-bit nibbles + the epilogue on its 3-bit rows, then the 4-bit pass.
import tierglu_nax6 as _N6
_C4 = _N6._CORE4
for _d in ("    const short qid = lane >> 2;\n", "    const short fm0 = (qid & 4) | ((lane >> 1) & 3);\n",
           "    const short fn0 = ((qid & 2) | (lane & 1)) * 4;\n", "    float acc[16];\n    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;\n"):
    assert _C4.count(_d) == 1, _d
    _C4 = _C4.replace(_d, "")
_CORE4X2 = r"""
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    if (!(has3 && has4)) {                                         // one tier: NAX6's core, verbatim
    const int tier = has4 ? 4 : (t2g ? 2 : 3);
""" + _C4 + r"""
    } else {                                                       // both tiers: one decode, two passes
        constexpr int KB3 = S_K * 3 / 8, KG = S_K / 64, KL = S_K / 8;
        const device uint32_t* wrow = (const device uint32_t*)((const device uint8_t*)w3p + ((size_t)e * S_O + n0 + lane) * KB3);
        const device T* srow = s3p + ((size_t)e * S_O + n0 + lane) * KG;
        const device T* brow3 = b3p + ((size_t)e * S_O + n0 + lane) * KG;
        const device uint32_t* lrow = (const device uint32_t*)((const device uint8_t*)l1p + ((size_t)el * S_O + n0 + lane) * KL);
        const device T* brow4 = blp + ((size_t)el * S_O + n0 + lane) * KG;
        const bool r4a = ((m4 >> fm0) & 1u) != 0, r4b = ((m4 >> (fm0 + 8)) & 1u) != 0;
        threadgroup uint32_t* tb3 = tb_all + sg * (2 * 32 * 8 * GS_C);
        threadgroup uint32_t* tb4 = tb3 + 32 * 8 * GS_C;
        threadgroup float* sb = sb_all + sg * (2 * 64 * GS_C);        // one buffer: GS_C x (s3, b3, s3 / 2, b4) x 32
        constexpr int GPC = KG / GS_C / KS_C;
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
        auto ca = op.template get_left_input_cooperative_tensor<T, uint4b_format, float>();
        using TB = tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline>;
        auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TB, float>();
        for (int gp = kc * GPC; gp < (kc + 1) * GPC; ++gp) {
            simdgroup_barrier(mem_flags::mem_threadgroup);          // the last stage's ops are done with both sets
            for (int j = 0; j < GS_C; ++j) {
                const int g = gp * GS_C + j;
                const device uint32_t* c = wrow + 6 * g;
                const uint w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3], w4 = c[4], w5 = c[5];
                uint ch[8] = {w0 & 0xFFFFFFu, (w0 >> 24) | ((w1 & 0xFFFFu) << 8), (w1 >> 16) | ((w2 & 0xFFu) << 16), w2 >> 8,
                              w3 & 0xFFFFFFu, (w3 >> 24) | ((w4 & 0xFFFFu) << 8), (w4 >> 16) | ((w5 & 0xFFu) << 16), w5 >> 8};
                const uint l0 = lrow[2 * g], l1 = lrow[2 * g + 1];
                uint nb[8];
                for (int q = 0; q < 8; ++q) nb[q] = spread3(ch[q]);
                threadgroup uint4* d3 = (threadgroup uint4*)(tb3 + lane * 8 * GS_C + 8 * j);
                d3[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
                d3[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
                for (int q = 0; q < 8; ++q) nb[q] = spread8(ch[q], ((q < 4 ? l0 : l1) >> (8 * (q & 3))) & 0xFFu);
                threadgroup uint4* d4 = (threadgroup uint4*)(tb4 + lane * 8 * GS_C + 8 * j);
                d4[0] = uint4(nb[0], nb[1], nb[2], nb[3]);
                d4[1] = uint4(nb[4], nb[5], nb[6], nb[7]);
                const float s = (float)srow[g];
                sb[128 * j + lane] = s * 1.0f;
                sb[128 * j + 32 + lane] = (float)brow3[g];
                sb[128 * j + 64 + lane] = s * 0.5f;
                sb[128 * j + 96 + lane] = (float)brow4[g];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (int j = 0; j < GS_C; ++j) {
                const int g = gp * GS_C + j;
                const float x0 = as0[g], x1 = as1[g];
                for (int ps = 0; ps < 2; ++ps) {
                    threadgroup uint32_t* tb = ps ? tb4 : tb3;
                    for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
                    for (int hh = 0; hh < 2; ++hh) {
                        const int k0 = 64 * g + 32 * hh;
                        const vec<T, 4> a00 = *(const device vec<T, 4>*)(ar0 + k0 + fn0);
                        const vec<T, 4> a10 = *(const device vec<T, 4>*)(ar1 + k0 + fn0);
                        const vec<T, 4> a01 = *(const device vec<T, 4>*)(ar0 + k0 + 16 + fn0);
                        const vec<T, 4> a11 = *(const device vec<T, 4>*)(ar1 + k0 + 16 + fn0);
                        for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
                        TB tbt((threadgroup uchar*)tb + 32 * j + 16 * hh, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 64 * GS_C});
                        op.run(ca, tbt, cg);
                    }
                    for (int i = 0; i < 16; ++i) {
                        const int col = 16 * (i >> 3) + fn0 + (i & 3);
                        const bool r4 = ((i >> 2) & 1) ? r4b : r4a;
                        if (r4 == (ps == 1))
                            acc[i] += sb[128 * j + 64 * ps + col] * cg[i] + sb[128 * j + 64 * ps + 32 + col] * (((i >> 2) & 1) ? x1 : x0);
                    }
                }
            }
        }
    }
"""

_AM = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // projection pj = sg & 1 (0 gate, 1 up), K chunk kc = sg >> 1
    const uint pj = sg & 1u, kc = sg >> 1;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint e = (uint)msb_slot[(uint)pidx[q0]];
""" + _ROWS + r"""
    const short fmA = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint ma0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)] / NR, ma1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)] / NR;
    const device T* ar0 = x + (size_t)ma0 * K;
    const device T* ar1 = x + (size_t)ma1 * K;
    const device float* as0 = xs + (size_t)ma0 * (K / 64);
    const device float* as1 = xs + (size_t)ma1 * (K / 64);
    const device uint32_t* w3p = pj == 0 ? w3g : w3u;
    const device T* s3p = pj == 0 ? s3g : s3u;
    const device T* b3p = pj == 0 ? b3g : b3u;
    const device T* b2p = pj == 0 ? b2g : b2u;
    const device uint32_t* l1p = pj == 0 ? l1g : l1u;
    const device T* blp = pj == 0 ? bg : bu;
    threadgroup uint32_t tb_all[2 * KS_C * 2 * 32 * 8 * GS_C];     // later reused for the chunk accumulators
    threadgroup float sb_all[2 * KS_C * 2 * 128 * GS_C];
    #define S_K K
    #define S_O N
    #define XROW(m) (x + (size_t)((uint)pidx[q0 + (m)] / NR) * K)
""" + _CORE4M + r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);               // every simdgroup is done with its code buffers
    threadgroup float* part = (threadgroup float*)tb_all;           // [projection][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) part[(pj * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1) {                                                  // gate = chunk sums in order, up likewise, then swiglu
        for (int i = 0; i < 16; ++i) {
            const int row = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
            float gs = part[(0 * KS_C + 0) * 512 + row * 32 + col], us = part[(1 * KS_C + 0) * 512 + row * 32 + col];
            for (int c = 1; c < KS_C; ++c) { gs += part[(0 * KS_C + c) * 512 + row * 32 + col]; us += part[(1 * KS_C + c) * 512 + row * 32 + col]; }
            if (row < nq) {
                const T gv = static_cast<T>(gs);
                const T ub = static_cast<T>(us);
                h[(size_t)pidx[q0 + row] * N + n0 + col] = TgMultiply()(TgMultiply()(gv, TgSigmoid()(gv)), ub);
            }
        }
    }
"""

_BM = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // column tile ti = sg % NSG_C, K chunk kc = sg / NSG_C
    const uint ti = sg % NSG_C, kc = sg / NSG_C;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = (threadgroup_position_in_grid.x * NSG_C + ti) * 32;
    const uint e = (uint)msb_slot[(uint)pidx[q0]];
""" + _ROWS + r"""
    const short fmA = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint pa0 = (uint)pidx[q0 + (fmA < nq ? fmA : 0)], pa1 = (uint)pidx[q0 + (fmA + 8 < nq ? fmA + 8 : 0)];
    const device T* ar0 = h + (size_t)pa0 * N;
    const device T* ar1 = h + (size_t)pa1 * N;
    const device float* as0 = hs + (size_t)pa0 * (N / 64);
    const device float* as1 = hs + (size_t)pa1 * (N / 64);
    const device uint32_t* w3p = w3d;
    const device T* s3p = s3d;
    const device T* b3p = b3d;
    const device T* b2p = b2d;
    const device uint32_t* l1p = l1d;
    const device T* blp = bd;
    threadgroup uint32_t tb_all[NSG_C * KS_C * 2 * 32 * 8 * GS_C];  // later reused for the chunk accumulators
    threadgroup float sb_all[NSG_C * KS_C * 2 * 128 * GS_C];
    #define S_K N
    #define S_O D
    #define XROW(m) (h + (size_t)pidx[q0 + (m)] * N)
""" + _CORE4M + r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* part = (threadgroup float*)tb_all;           // [tile][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) part[(ti * KS_C + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kc == 0) {
        for (int i = 0; i < 16; ++i) {
            const int row = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
            float ys = part[(ti * KS_C + 0) * 512 + row * 32 + col];
            for (int c = 1; c < KS_C; ++c) ys += part[(ti * KS_C + c) * 512 + row * 32 + col];
            if (row < nq) y[(size_t)pidx[q0 + row] * D + n0 + col] = static_cast<T>(ys);
        }
    }
"""

# merged grouping: thread i = pair i (P <= 1024). PH_OK: this phase's pair (the phase form), else every pair.
_GRPM = r"""
    const int P = P_C;
    const uint i = thread_position_in_grid.x;
    threadgroup int key_s[P_C];
    threadgroup int skey[P_C];
    threadgroup int start_s[P_C];
    threadgroup int t_s[P_C];                                  // 2 / 3 / 4 (4 = a 4-bit rank with a sidecar slot); 0 = other phase
    threadgroup int e_s[P_C];                                  // a 4-bit pair's sidecar slot + 1
    if ((int)i < P) {
        const int r = (int)i % NR_C, m = (int)i / NR_C;
        int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
        int el = 0;
        if (tier == 4) {
            el = lsb_slot[m * R4_C + popcount((uint)MASK4_C & ((1u << r) - 1u))];
            if (el < 0) { tier = 3; el = 0; } else el += 1;
        }
        t_s[i] = (PH_OK) ? tier : 0;
        e_s[i] = el;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P) {
        const int t = t_s[i];
        int key = 0x7FFFFFFF;
        if (t != 0) {
            int el = e_s[i];
            if (t == 3) {                                      // adopt the smallest sidecar slot + 1 of this slot's 4-bit pairs
                const int s = msb_slot[i];
                el = 0;
                for (int j = 0; j < P; ++j)
                    if (t_s[j] == 4 && msb_slot[j] == s && (el == 0 || e_s[j] < el)) el = e_s[j];
            }
            key = (msb_slot[i] << 11) | ((t == 2 ? 1 : 0) << 10) | el;
        }
        key_s[i] = key;
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
        start_s[i] = (skey[i] != 0x7FFFFFFF && (((int)i - r0) % 16) == 0) ? 1 : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)i < P && start_s[i]) {
        int g = 0;
        for (int j = 0; j < (int)i; ++j) g += start_s[j];
        seg[g] = (int)i;
    }
    if ((int)i == 0) {
        int n = 0, nv = 0;
        for (int j = 0; j < P; ++j) { n += start_s[j]; nv += skey[j] != 0x7FFFFFFF; }
        nseg[0] = n;
        seg[n] = nv;
    }
"""


@lru_cache(None)
def _kgrpm(phase):
    if phase:
        return mx.fast.metal_kernel(name="tierglu_merge_grpp", input_names=["msb_slot", "lsb_slot", "ph"],
                                    output_names=["pidx", "seg", "nseg"], source=_GRPM.replace("PH_OK", "ph[i] == WHICH_C"))
    return mx.fast.metal_kernel(name="tierglu_merge_grp", input_names=["msb_slot", "lsb_slot"],
                                output_names=["pidx", "seg", "nseg"], source=_GRPM.replace("PH_OK", "true"))


def _grp(msb_slot, lsb_slot, plan, phase=None, which=0):
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    ms = msb_slot.astype(mx.int32).reshape(-1)
    P = ms.size
    assert P <= 1024, P
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ins = [ms, ls] + ([phase.astype(mx.int32).reshape(-1)] if phase is not None else [])
    tmpl = [("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    if phase is not None:
        tmpl.append(("WHICH_C", which))
    pidx, seg, nseg = _kgrpm(phase is not None)(inputs=ins, template=tmpl,
                                                 grid=(max(32, (P + 31) // 32 * 32), 1, 1),
                                                 threadgroup=(max(32, (P + 31) // 32 * 32), 1, 1),
                                                 output_shapes=[(P,), (P + 1,), (1,)], output_dtypes=[mx.int32] * 3)
    return pidx, seg, nseg, P


def gpu_grp_merge(msb_slot, lsb_slot, plan):
    """the merged pair grouping on the GPU (no host sync): (pidx, seg, nseg, P) for tier_glu_merge(grp=...)"""
    return _grp(msb_slot, lsb_slot, plan)


def gpu_grp_phase_merge(msb_slot, lsb_slot, plan, phase, which):
    """gpu_grp_merge over the pairs with phase == which only (OOO's phases; the others join no group)"""
    return _grp(msb_slot, lsb_slot, plan, phase, which)


_AX2 = _AM.replace(_CORE4M, _CORE4X2).replace("sb_all[2 * KS_C * 2 * 128 * GS_C]", "sb_all[2 * KS_C * 2 * 64 * GS_C]")
_BX2 = _BM.replace(_CORE4M, _CORE4X2).replace("sb_all[NSG_C * KS_C * 2 * 128 * GS_C]", "sb_all[NSG_C * KS_C * 2 * 64 * GS_C]")
assert _AX2 != _AM and _BX2 != _BM and "128 * GS_C]" not in _AX2.split("tb_all")[1][:200]


@lru_cache(None)
def _kam3():
    return mx.fast.metal_kernel(
        name="tierglu_merge_a3",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_AX2, header=_HDR4)


@lru_cache(None)
def _kbm3():
    return mx.fast.metal_kernel(
        name="tierglu_merge_b3",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_BX2, header=_HDR4)


@lru_cache(None)
def _kam():
    return mx.fast.metal_kernel(
        name="tierglu_merge_a",
        input_names=["x", "xs", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_AM, header=_HDR4)


@lru_cache(None)
def _kbm():
    return mx.fast.metal_kernel(
        name="tierglu_merge_b",
        input_names=["h", "hs", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_BM, header=_HDR4)


def tier_glu_merge(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, grp=None, nsg_b=2, gs=1, ks=4, ks_b=2,
                   xs=None, mode=1, **_):
    """NAX6 on merged groups: mode 1 MERGE-X / mode 3 MERGE-X2 (== NAX6 bitwise), mode 2 MERGE-B; y [M, R, D]"""
    assert mode in (1, 2, 3), mode
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 64 and K % 64 == 0 and N % 64 == 0 and D % (32 * nsg_b) == 0
    if grp is None:
        grp = gpu_grp_merge(msb_slot, lsb_slot, plan)
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
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("GS_C", gs)]
    tmpl += [("DIAG_C", 0)] if mode == 3 else [("MODE_C", mode)]
    ka, kb = (_kam3(), _kbm3()) if mode == 3 else (_kam(), _kbm())
    assert (K // 64) % (gs * ks) == 0 and (N // 64) % (gs * ks_b) == 0, (K, N, gs, ks, ks_b)
    xs = rowsum64(x) if xs is None else xs
    (h,) = ka(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("KS_C", ks)],
                  grid=(N // 32 * 64 * ks, nseg, 1), threadgroup=(64 * ks, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    hs = rowsum64(h)
    (y,) = kb(inputs=[h, hs, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"],
                          ms, ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("KS_C", ks_b)],
                  grid=(D // (32 * nsg_b) * 32 * nsg_b * ks_b, nseg, 1), threadgroup=(32 * nsg_b * ks_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
