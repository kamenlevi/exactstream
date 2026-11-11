"""K-61 TIERGRP (Leap, Sun 2026-11-01; NewIdeas 06:1x routing, REF2c bundle item A1): MoonShots' fused 3c tier GLU
(tierglu.tier_glu, frozen copy in /Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn) with the (row, rank) pairs grouped by (expert slot, tier,
sidecar slot). tier_glu runs one threadgroup per pair x output tile, so a check's pairs that share an expert each read its
weights again: K-60-T measured ~150 GB/s over the distinct experts' bytes, ~285 GB/s over the per-pair bytes it reads.
Here one threadgroup per group x output tile decodes each weight chunk once and applies it to every pair of the group.
Each pair's arithmetic is _A / _B's, in the same order: per 512-value block (A) or 256-value block plus the clamped tail
(B), the pair's x / h load (tg_load_vector[_safe]), the same code decode, qdot, block-ordered accumulation, simd_sum,
then A's swiglu (TgSigmoid, TgMultiply) or B's bf16 store. So the outputs are meant to be bit-identical to tier_glu
(k61_tiergrp_test.py). A group holds at most M pairs (an expert appears once per row); groups are split into chunks of <= cap pairs
(MP_C = cap: registers grow with it).
tier_glu_grouped(x, msb, lsb, msb_slot, lsb_slot, plan, ms_np=None, ls_np=None, cap=16, grp=None) -> y [M, R, D]
(no fused combine; grp = make_grp(...) precomputed).
The grouping is built on the host from the slot maps (ms_np / ls_np, else read back): REF2-T's slot path has them there."""
import os, sys
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn"))
from tierglu import _HDR, PROJ, plan_masks

_AG = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C, RA = RA_C, G = 64, MP = MP_C;
    constexpr int KB3 = K * 3 / 8, KG = K / G, KL = K / 8;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0: gate rows, 1: up rows
    const uint gq = threadgroup_position_in_grid.y;             // group of pairs: one (expert slot, tier, sidecar slot)
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;              // nq <= MP
    const uint j0 = threadgroup_position_in_grid.x * RA;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pf];
    const device uint8_t* w3 = (const device uint8_t*)(sg == 0 ? w3g : w3u) + ((size_t)e * N + j0) * KB3;
    const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * N + j0) * KG;
    float result[MP][RA];
    for (int q = 0; q < MP; ++q) for (int row = 0; row < RA; ++row) result[q][row] = 0.0f;
    if (tier == 4) {
        const uint mf = pf / NR;
        const uint i4 = popcount((uint)MASK4_C & ((1u << r0) - 1u));
        const uint el = (uint)lsb_slot[mf * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)(sg == 0 ? l1g : l1u) + ((size_t)el * N + j0) * KL;
        const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            uint16_t wsr[RA][4];
            float sr[RA], br[RA];
            for (int row = 0; row < RA; ++row) {
                uint q0w, q1w;
                if (LD_C == 1) {
                    const device uint16_t* c = (const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6);
                    const uint u0 = c[0], u1 = c[1], u2 = c[2];
                    const uint lw = *(const device uint16_t*)(l1 + row * KL + b * 64 + lane * 2);
                    q0w = spread8(u0 | ((u1 & 0xFFu) << 16), lw & 0xFFu);
                    q1w = spread8((u1 >> 8) | (u2 << 8), lw >> 8);
                } else {
                    const device uint8_t* c = w3 + row * KB3 + b * 192 + lane * 6;
                    const device uint8_t* lb = l1 + row * KL + b * 64 + lane * 2;
                    q0w = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), lb[0]);
                    q1w = spread8((uint)c[3] | ((uint)c[4] << 8) | ((uint)c[5] << 16), lb[1]);
                }
                wsr[row][0] = (uint16_t)(q0w & 0xFFFFu); wsr[row][1] = (uint16_t)(q0w >> 16);
                wsr[row][2] = (uint16_t)(q1w & 0xFFFFu); wsr[row][3] = (uint16_t)(q1w >> 16);
                const int gi = row * KG + b * 8 + lane / 4;
                sr[row] = (float)s3[gi] * 0.5f;
                br[row] = bl[gi];
            }
            for (int q = 0; q < nq; ++q) {
                const uint m = (uint)pidx[q0 + q] / NR;
                const device T* xr = x + (size_t)m * K + lane * 16;
                float xt[16];
                const float sum = tg_load_vector<T, float, 16, 4>(xr + b * 512, xt);
                for (int row = 0; row < RA; ++row) result[q][row] += tg_qdot4<float, 16>(wsr[row], xt, sr[row], br[row], sum);
            }
        }
    } else {
        const device T* b3 = tier == 2 ? (sg == 0 ? b2g : b2u) + ((size_t)e * N + j0) * KG
                                       : (sg == 0 ? b3g : b3u) + ((size_t)e * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            uint8_t wbr[RA][6];
            float sr[RA], br[RA];
            for (int row = 0; row < RA; ++row) {
                if (LD_C == 1) {
                    const device uint16_t* c = (const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6);
                    const uint16_t u0 = c[0] & (tier == 2 ? 0x6DB6 : 0xFFFF), u1 = c[1] & (tier == 2 ? 0xB6DB : 0xFFFF),
                                   u2 = c[2] & (tier == 2 ? 0xDB6D : 0xFFFF);
                    wbr[row][0] = (uint8_t)u0; wbr[row][1] = (uint8_t)(u0 >> 8); wbr[row][2] = (uint8_t)u1;
                    wbr[row][3] = (uint8_t)(u1 >> 8); wbr[row][4] = (uint8_t)u2; wbr[row][5] = (uint8_t)(u2 >> 8);
                } else {
                    tg_codes3(w3 + row * KB3, b * 512 + lane * 16, 6, wbr[row], tier == 2);
                }
                const int gi = row * KG + b * 8 + lane / 4;
                sr[row] = s3[gi];
                br[row] = b3[gi];
            }
            for (int q = 0; q < nq; ++q) {
                const uint m = (uint)pidx[q0 + q] / NR;
                const device T* xr = x + (size_t)m * K + lane * 16;
                float xt[16];
                const float sum = tg_load_vector<T, float, 16, 3>(xr + b * 512, xt);
                for (int row = 0; row < RA; ++row) result[q][row] += tg_qdot3<float, 16>(wbr[row], xt, sr[row], br[row], sum);
            }
        }
    }
    threadgroup T gate_v[MP][RA];
    for (int q = 0; q < nq; ++q) {
        for (int row = 0; row < RA; ++row) result[q][row] = simd_sum(result[q][row]);
        if (sg == 0 && lane == 0) {
            for (int row = 0; row < RA; ++row) gate_v[q][row] = static_cast<T>(result[q][row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1 && lane < (uint)RA) {
        for (int q = 0; q < nq; ++q) {
            float u = 0.0f;
            for (int row = 0; row < RA; ++row) u = (row == (int)lane) ? result[q][row] : u;
            const T g = gate_v[q][lane];
            const T ub = static_cast<T>(u);
            h[(size_t)pidx[q0 + q] * N + j0 + lane] = TgMultiply()(TgMultiply()(g, TgSigmoid()(g)), ub);
        }
    }
"""

_BG = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C, RB = RB_C, G = 64, MP = MP_C;
    constexpr int NB3 = N * 3 / 8, NG = N / G, NL = N / 8;
    constexpr int BS = 256;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
    const uint pf = (uint)pidx[q0];
    const uint r0 = pf % NR;
    const int tier = ((MASK4_C >> r0) & 1) ? 4 : (((MASK2_C >> r0) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[pf];
    const device uint8_t* w3 = (const device uint8_t*)w3d + ((size_t)e * D + d0) * NB3;
    const device T* s3 = s3d + ((size_t)e * D + d0) * NG;
    float result[MP][RB];
    for (int q = 0; q < MP; ++q) for (int row = 0; row < RB; ++row) result[q][row] = 0.0f;
    if (tier == 4) {
        const uint mf = pf / NR;
        const uint i4 = popcount((uint)MASK4_C & ((1u << r0) - 1u));
        const uint el = (uint)lsb_slot[mf * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)l1d + ((size_t)el * D + d0) * NL;
        const device T* bl = bd + ((size_t)el * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            uint16_t wsr[RB][2];
            float sr[RB], br[RB];
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint qw = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                wsr[row][0] = (uint16_t)(qw & 0xFFFFu); wsr[row][1] = (uint16_t)(qw >> 16);
                const int gi = row * NG + (k + lane * 8) / G;
                sr[row] = (float)s3[gi] * 0.5f; br[row] = (float)bl[gi];
            }
            for (int q = 0; q < nq; ++q) {
                const device T* hr = h + (size_t)pidx[q0 + q] * N + lane * 8;
                float xt[8];
                const float sum = tg_load_vector<T, float, 8, 4>(hr + k, xt);
                for (int row = 0; row < RB; ++row) result[q][row] += tg_qdot4<float, 8>(wsr[row], xt, sr[row], br[row], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            uint16_t wsr[RB][2];
            float sr[RB], br[RB];
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint qw = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                wsr[row][0] = (uint16_t)(qw & 0xFFFFu); wsr[row][1] = (uint16_t)(qw >> 16);
                const int gi = row * NG + (k + lane * 8) / G;
                sr[row] = (float)s3[gi] * 0.5f; br[row] = (float)bl[gi];
            }
            for (int q = 0; q < nq; ++q) {
                const device T* hr = h + (size_t)pidx[q0 + q] * N + lane * 8;
                float xt[8];
                const float sum = tg_load_vector_safe<T, float, 8, 4>(hr + k, xt, remaining);
                for (int row = 0; row < RB; ++row) result[q][row] += tg_qdot4<float, 8>(wsr[row], xt, sr[row], br[row], sum);
            }
        }
    } else {
        const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            uint8_t wbr[RB][3];
            float sr[RB], br[RB];
            for (int row = 0; row < RB; ++row) {
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wbr[row], tier == 2);
                const int gi = row * NG + (k + lane * 8) / G;
                sr[row] = (float)s3[gi]; br[row] = (float)b3[gi];
            }
            for (int q = 0; q < nq; ++q) {
                const device T* hr = h + (size_t)pidx[q0 + q] * N + lane * 8;
                float xt[8];
                const float sum = tg_load_vector<T, float, 8, 3>(hr + k, xt);
                for (int row = 0; row < RB; ++row) result[q][row] += tg_qdot3<float, 8>(wbr[row], xt, sr[row], br[row], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            uint8_t wbr[RB][3];
            float sr[RB], br[RB];
            for (int row = 0; row < RB; ++row) {
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wbr[row], tier == 2);
                const int gi = row * NG + (k + lane * 8) / G;
                sr[row] = (float)s3[gi]; br[row] = (float)b3[gi];
            }
            for (int q = 0; q < nq; ++q) {
                const device T* hr = h + (size_t)pidx[q0 + q] * N + lane * 8;
                float xt[8];
                const float sum = tg_load_vector_safe<T, float, 8, 3>(hr + k, xt, remaining);
                for (int row = 0; row < RB; ++row) result[q][row] += tg_qdot3<float, 8>(wbr[row], xt, sr[row], br[row], sum);
            }
        }
    }
    for (int q = 0; q < nq; ++q) {
        for (int row = 0; row < RB; ++row) {
            const float v = simd_sum(result[q][row]);
            if (lane == 0) y[(size_t)pidx[q0 + q] * D + d0 + row] = static_cast<T>(v);
        }
    }
"""


@lru_cache(None)
def _kag():
    return mx.fast.metal_kernel(
        name="tierglu_ag",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["h"], source=_AG, header=_HDR)


@lru_cache(None)
def _kbg():
    return mx.fast.metal_kernel(
        name="tierglu_bg",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "seg", "nseg"],
        output_names=["y"], source=_BG, header=_HDR)


def groups(ms_np, ls_np, plan, cap=16):
    """pairs p = m * R + r sorted by (base slot, tier, sidecar slot); group starts, groups split into chunks of <= cap
    pairs. ms_np [M, R], ls_np [M, R4] or None"""
    M, R = ms_np.shape
    tiers = np.array([4 if c == "4" else 2 if c == "2" else 3 for c in plan])
    i4 = np.cumsum([c == "4" for c in plan]) - 1
    p = np.arange(M * R)
    m, r = p // R, p % R
    e = ms_np.reshape(-1).astype(np.int64)
    t = tiers[r]
    el = np.where(t == 4, ls_np[m, np.maximum(i4[r], 0)] if ls_np is not None else 0, -1).astype(np.int64)
    order = np.lexsort((p, el, t, e))
    key = np.stack([e[order], t[order], el[order]], axis=1)
    starts = np.flatnonzero(np.r_[True, np.any(key[1:] != key[:-1], axis=1)])
    bounds = np.r_[starts, M * R]
    seg = np.concatenate([np.arange(a, b, cap) for a, b in zip(bounds[:-1], bounds[1:])] + [[M * R]]).astype(np.int32)
    return order.astype(np.int32), seg, len(seg) - 1


def make_grp(msb_slot, lsb_slot, plan, M, ms_np=None, ls_np=None, cap=16):
    """(pidx, seg, nseg array, nseg) for tier_glu_grouped(grp=...)"""
    R, r4 = len(plan), plan.count("4")
    ms_np = np.array(msb_slot).reshape(M, R) if ms_np is None else ms_np.reshape(M, R)
    if r4:
        ls_np = np.array(lsb_slot).reshape(M, r4) if ls_np is None else ls_np.reshape(M, r4)
    pidx, seg, nseg = groups(ms_np, ls_np if r4 else None, plan, cap)
    return mx.array(pidx), mx.array(seg), mx.array([nseg], dtype=mx.int32), nseg


def tier_glu_grouped(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, ld=0,
                     ms_np=None, ls_np=None, cap=16, grp=None):
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert M <= 16 and K % 512 == 0 and N % ra == 0 and D % (rb * nsg_b) == 0 and N % 64 == 0
    if grp is None:
        grp = make_grp(msb_slot, lsb_slot, plan, M, ms_np, ls_np, cap)
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
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("MP_C", cap)]
    (h,) = _kag()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls,
                          pidx, seg, nsg_],
                  template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                  grid=(N // ra * 64, nseg, 1), threadgroup=(64, 1, 1),
                  output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    (y,) = _kbg()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms,
                          ls, pidx, seg, nsg_],
                  template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                  grid=(D // (rb * nsg_b) * 32 * nsg_b, nseg, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)
