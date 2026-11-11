"""Dequant-once multi-row form of MoonShots' 3c fused GLU (the kernels BigMoE v3 runs; defer/v3d/tierglu.py _A / _B), for wide
verify passes AND for today's 4-row passes. Measured 13:50 (wide/MICROBENCH.md): 3c is bound by per-pair work (each
(row, rank) pair re-loads, masks and converts its expert's 3-bit codes); reading each expert once alone buys nothing.
Here one threadgroup takes one expert (and tier) and up to RM of its pairs. Per lane and 512-value block it loads the codes,
masks and converts them to floats ONCE per weight row (the terms of MLX's qdot: float(w & mask)), then for each pair runs
EXACTLY MLX's qdot arithmetic: the same pre-scaled x (load_vector), the same terms in the same order, the same
scale * accum + sum * bias, the same block order, the same simd_sum. So each pair's h / y rows, and the LY=8 combine, are
meant to be BIT-IDENTICAL to tier_glu(..., scores=); the test is microbench.py's check. Tiers 4 and 3 (no 2-bit tier).
Pairs are grouped on the GPU by (expert, tier) in chunks of <= RM (csr kernel; empty chunks return at once).
Dispatches: csr + A + B + combine. Templates: RA (gate/up rows per threadgroup) and RB (down rows per simdgroup) do not
change any row's arithmetic (3c uses 4 / 4; this uses 2 / 2 to keep RM x RA accumulators in registers)."""
import os, sys
from functools import lru_cache
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/MoonShots/defer/v3d"))
import tierglu as TG

_HDR = TG._HDR + r"""
// MLX's qdot3 / qdot4 with the weight terms precomputed (wf[k] = float(code bits & mask), the value MLX promotes)
template <int values_per_thread>
inline float mr_qdot3(const thread float* wf, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 8); i++) {
    const thread float* xt = x_thread + 8 * i;
    const thread float* w = wf + 10 * i;
    accum += w[0] * xt[0];
    accum += w[1] * xt[1];
    accum += w[2] * xt[2];
    accum += w[3] * (xt[2] * 256.0f);
    accum += w[4] * xt[3];
    accum += w[5] * xt[4];
    accum += w[6] * xt[5];
    accum += w[7] * (xt[5] * 256.0f);
    accum += w[8] * xt[6];
    accum += w[9] * xt[7];
  }
  return scale * accum + sum * bias;
}

template <int values_per_thread>
inline float mr_qdot4(const thread float* wq, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * wq[4 * i] +
         x_thread[4 * i + 1] * wq[4 * i + 1] +
         x_thread[4 * i + 2] * wq[4 * i + 2] +
         x_thread[4 * i + 3] * wq[4 * i + 3]);
  }
  return scale * accum + sum * bias;
}

// the 10 qdot3 terms of 8 values from 3 code bytes
inline void mr_terms3(const thread uint8_t* w, thread float* wf) {
  wf[0] = (float)(w[0] & 0x07); wf[1] = (float)(w[0] & 0x38); wf[2] = (float)(w[0] & 0xc0); wf[3] = (float)(w[1] & 0x01);
  wf[4] = (float)(w[1] & 0x0e); wf[5] = (float)(w[1] & 0x70); wf[6] = (float)(w[1] & 0x80); wf[7] = (float)(w[2] & 0x03);
  wf[8] = (float)(w[2] & 0x1c); wf[9] = (float)(w[2] & 0xe0);
}

// the 4 qdot4 terms of one 16-bit nibble word
inline void mr_terms4(uint16_t ws, thread float* wq) {
  wq[0] = (float)(ws & 0x000f); wq[1] = (float)(ws & 0x00f0); wq[2] = (float)(ws & 0x0f00); wq[3] = (float)(ws & 0xf000);
}
"""

_A = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C, RA = RA_C, RM = RM_C, G = 64;
    constexpr int KB3 = K * 3 / 8, KG = K / G, KL = K / 8;
    threadgroup T gate_v[RM][RA];
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0: gate rows, 1: up rows
    const uint g_ = threadgroup_position_in_grid.y;             // chunk of <= RM pairs of one (expert, tier)
    const uint j0 = threadgroup_position_in_grid.x * RA;
    const int off = grp_off[g_];
    const int cnt = min(grp_off[g_ + 1] - off, RM);
    if (cnt <= 0) return;                                       // uniform per threadgroup, before any barrier
    const uint p0 = (uint)gpairs[off];
    const uint r0 = p0 % NR;
    const bool t4 = ((MASK4_C >> r0) & 1) != 0;
    const uint e = (uint)msb_slot[p0];
    const device uint8_t* w3 = (const device uint8_t*)(sg == 0 ? w3g : w3u) + ((size_t)e * N + j0) * KB3;
    const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * N + j0) * KG;
    float result[RM][RA];
    _Pragma("unroll") for (int i = 0; i < RM; ++i) for (int row = 0; row < RA; ++row) result[i][row] = 0.0f;
    if (t4) {
        const uint i4 = popcount((uint)MASK4_C & ((1u << r0) - 1u));
        const uint el = (uint)lsb_slot[(p0 / NR) * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)(sg == 0 ? l1g : l1u) + ((size_t)el * N + j0) * KL;
        const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            float wq[RA][16], sc[RA], bi[RA];
            for (int row = 0; row < RA; ++row) {
                const device uint8_t* c = w3 + row * KB3 + b * 192 + lane * 6;
                const device uint8_t* lb = l1 + row * KL + b * 64 + lane * 2;
                const uint q0 = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), lb[0]);
                const uint q1 = spread8((uint)c[3] | ((uint)c[4] << 8) | ((uint)c[5] << 16), lb[1]);
                mr_terms4((uint16_t)(q0 & 0xFFFFu), wq[row]);
                mr_terms4((uint16_t)(q0 >> 16), wq[row] + 4);
                mr_terms4((uint16_t)(q1 & 0xFFFFu), wq[row] + 8);
                mr_terms4((uint16_t)(q1 >> 16), wq[row] + 12);
                const int gi = row * KG + b * 8 + lane / 4;
                sc[row] = (float)s3[gi] * 0.5f;
                bi[row] = bl[gi];
            }
            _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
                const uint m = (uint)gpairs[off + i] / NR;
                float xt[16];
                const float sum = tg_load_vector<T, float, 16, 4>(x + (size_t)m * K + lane * 16 + b * 512, xt);
                for (int row = 0; row < RA; ++row) result[i][row] += mr_qdot4<16>(wq[row], xt, sc[row], bi[row], sum);
            }
        }
    } else {
        const device T* b3 = (sg == 0 ? b3g : b3u) + ((size_t)e * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            float wf[RA][20], sc[RA], bi[RA];
            for (int row = 0; row < RA; ++row) {
                uint8_t wb[6];
                tg_codes3(w3 + row * KB3, b * 512 + lane * 16, 6, wb, false);
                mr_terms3(wb, wf[row]);
                mr_terms3(wb + 3, wf[row] + 10);
                const int gi = row * KG + b * 8 + lane / 4;
                sc[row] = s3[gi];
                bi[row] = b3[gi];
            }
            _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
                const uint m = (uint)gpairs[off + i] / NR;
                float xt[16];
                const float sum = tg_load_vector<T, float, 16, 3>(x + (size_t)m * K + lane * 16 + b * 512, xt);
                for (int row = 0; row < RA; ++row) result[i][row] += mr_qdot3<16>(wf[row], xt, sc[row], bi[row], sum);
            }
        }
    }
    _Pragma("unroll") for (int i = 0; i < RM; ++i)
        for (int row = 0; row < RA; ++row) result[i][row] = simd_sum(result[i][row]);
    if (sg == 0 && lane == 0) {
        _Pragma("unroll") for (int i = 0; i < RM; ++i) {
            if (i >= cnt) break;
            for (int row = 0; row < RA; ++row) gate_v[i][row] = static_cast<T>(result[i][row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1 && lane < (uint)RA) {
        _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
            float u = 0.0f;
            for (int row = 0; row < RA; ++row) u = (row == (int)lane) ? result[i][row] : u;
            const T g = gate_v[i][lane];
            const T ub = static_cast<T>(u);
            h[(size_t)gpairs[off + i] * N + j0 + lane] = TgMultiply()(TgMultiply()(g, TgSigmoid()(g)), ub);
        }
    }
"""

_B = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C, RB = RB_C, RM = RM_C, G = 64;
    constexpr int NB3 = N * 3 / 8, NG = N / G, NL = N / 8;
    constexpr int BS = 256;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint g_ = threadgroup_position_in_grid.y;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
    const int off = grp_off[g_];
    const int cnt = min(grp_off[g_ + 1] - off, RM);
    if (cnt <= 0) return;
    const uint p0 = (uint)gpairs[off];
    const uint r0 = p0 % NR;
    const bool t4 = ((MASK4_C >> r0) & 1) != 0;
    const uint e = (uint)msb_slot[p0];
    const device uint8_t* w3 = (const device uint8_t*)w3d + ((size_t)e * D + d0) * NB3;
    const device T* s3 = s3d + ((size_t)e * D + d0) * NG;
    float result[RM][RB];
    _Pragma("unroll") for (int i = 0; i < RM; ++i) for (int row = 0; row < RB; ++row) result[i][row] = 0.0f;
    if (t4) {
        const uint i4 = popcount((uint)MASK4_C & ((1u << r0) - 1u));
        const uint el = (uint)lsb_slot[(p0 / NR) * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)l1d + ((size_t)el * D + d0) * NL;
        const device T* bl = bd + ((size_t)el * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float wq[RB][8], sc[RB], bi[RB];
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint q = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                mr_terms4((uint16_t)(q & 0xFFFFu), wq[row]);
                mr_terms4((uint16_t)(q >> 16), wq[row] + 4);
                const int gi = row * NG + (k + lane * 8) / G;
                sc[row] = (float)s3[gi] * 0.5f;
                bi[row] = (float)bl[gi];
            }
            _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
                float xt[8];
                const float sum = tg_load_vector<T, float, 8, 4>(h + (size_t)gpairs[off + i] * N + lane * 8 + k, xt);
                for (int row = 0; row < RB; ++row) result[i][row] += mr_qdot4<8>(wq[row], xt, sc[row], bi[row], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            float wq[RB][8], sc[RB], bi[RB];
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint q = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                mr_terms4((uint16_t)(q & 0xFFFFu), wq[row]);
                mr_terms4((uint16_t)(q >> 16), wq[row] + 4);
                const int gi = row * NG + (k + lane * 8) / G;
                sc[row] = (float)s3[gi] * 0.5f;
                bi[row] = (float)bl[gi];
            }
            _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
                float xt[8];
                const float sum = tg_load_vector_safe<T, float, 8, 4>(h + (size_t)gpairs[off + i] * N + lane * 8 + k, xt, remaining);
                for (int row = 0; row < RB; ++row) result[i][row] += mr_qdot4<8>(wq[row], xt, sc[row], bi[row], sum);
            }
        }
    } else {
        const device T* b3 = b3d + ((size_t)e * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float wf[RB][10], sc[RB], bi[RB];
            for (int row = 0; row < RB; ++row) {
                uint8_t wb[3];
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, false);
                mr_terms3(wb, wf[row]);
                const int gi = row * NG + (k + lane * 8) / G;
                sc[row] = (float)s3[gi];
                bi[row] = (float)b3[gi];
            }
            _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
                float xt[8];
                const float sum = tg_load_vector<T, float, 8, 3>(h + (size_t)gpairs[off + i] * N + lane * 8 + k, xt);
                for (int row = 0; row < RB; ++row) result[i][row] += mr_qdot3<8>(wf[row], xt, sc[row], bi[row], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            float wf[RB][10], sc[RB], bi[RB];
            for (int row = 0; row < RB; ++row) {
                uint8_t wb[3];
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, false);
                mr_terms3(wb, wf[row]);
                const int gi = row * NG + (k + lane * 8) / G;
                sc[row] = (float)s3[gi];
                bi[row] = (float)b3[gi];
            }
            _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
                float xt[8];
                const float sum = tg_load_vector_safe<T, float, 8, 3>(h + (size_t)gpairs[off + i] * N + lane * 8 + k, xt, remaining);
                for (int row = 0; row < RB; ++row) result[i][row] += mr_qdot3<8>(wf[row], xt, sc[row], bi[row], sum);
            }
        }
    }
    _Pragma("unroll") for (int i = 0; i < RM; ++i) { if (i >= cnt) break;
        for (int row = 0; row < RB; ++row) {
            const float v = simd_sum(result[i][row]);
            if (lane == 0) y[(size_t)gpairs[off + i] * D + d0 + row] = static_cast<T>(v);
        }
    }
"""

_CSR = r"""
    // one threadgroup, thread i = pair i. key = expert x 2 + (rank i % NR is 4-bit); pairs sorted by (key, i); each run
    // of a key split into chunks of <= RM pairs, numbered in sorted order; grp_off[c] = first sorted position of chunk c
    const uint i = thread_position_in_grid.x;
    const int P = (int)P_C;
    threadgroup int key_s[P_C];
    threadgroup int first_s[P_C];
    threadgroup int cnt_s[P_C];
    int key = 0;
    if ((int)i < P) {
        key = ids[i] * 2 + (int)(((uint)MASK4_C >> ((int)i % NR_C)) & 1u);
        key_s[i] = key;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int pos = 0;
    if ((int)i < P) {
        int cnt = 0;
        for (int j = 0; j < P; ++j) {
            const int f = key_s[j];
            cnt += (f == key);
            pos += (f == key && j < (int)i);
        }
        first_s[i] = (pos == 0) ? 1 : 0;
        cnt_s[i] = cnt;
    }
    for (int t = (int)i; t <= P; t += (int)threads_per_threadgroup.x) grp_off[t] = P;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if ((int)i < P) {
        int rank = 0, before = 0;
        for (int j = 0; j < P; ++j) {
            const int f = key_s[j];
            rank += (f < key) || (f == key && j < (int)i);
            before += (f < key && first_s[j] != 0) ? (cnt_s[j] + RM_C - 1) / RM_C : 0;
        }
        gpairs[rank] = (int)i;
        if (pos % RM_C == 0) grp_off[before + pos / RM_C] = rank;
    }
"""


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(
        name="tierglu_mr_a",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "l1g", "l1u", "bg", "bu", "msb_slot", "lsb_slot",
                     "grp_off", "gpairs"],
        output_names=["h"], source=_A, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(
        name="tierglu_mr_b",
        input_names=["h", "w3d", "s3d", "b3d", "l1d", "bd", "msb_slot", "lsb_slot", "grp_off", "gpairs"],
        output_names=["y"], source=_B, header=_HDR)


@lru_cache(None)
def _kcsr():
    return mx.fast.metal_kernel(name="tierglu_mr_csr", input_names=["ids"], output_names=["gpairs", "grp_off"], source=_CSR)


def csr(ids, R, m4, rm):
    """ids [P] (pair p = m * R + r) -> gpairs [P] grouped by (expert, tier), grp_off [P + 1] (chunks of <= rm pairs)"""
    P = ids.size
    assert P + 1 <= 1024, P
    return _kcsr()(inputs=[ids.astype(mx.int32).reshape(-1)], template=[("P_C", P), ("NR_C", R), ("MASK4_C", m4), ("RM_C", rm)],
                   grid=(max(P + 1, 32), 1, 1), threadgroup=(max(P + 1, 32), 1, 1), output_shapes=[(P,), (P + 1,)],
                   output_dtypes=[mx.int32, mx.int32])


def tier_glu_mr(x, msb, lsb, msb_slot, lsb_slot, scores, plan="4433333333", rm=8, ra=2, rb=2, nsg_b=2):
    """== tierglu.tier_glu(x, ..., plan, scores=scores) bit for bit (the intent; checked in microbench.py)"""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = TG.plan_masks(plan)
    assert m2 == 0, "no 2-bit tier in the multi-row kernels"
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert K % 512 == 0 and N % ra == 0 and D % (rb * nsg_b) == 0
    if r4:
        ls = lsb_slot.astype(mx.int32).reshape(-1); L = lsb
    else:
        ls = mx.zeros((1,), mx.int32); L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in TG.PROJ}
    ms = msb_slot.astype(mx.int32).reshape(-1)
    gp, go = csr(ms, R, m4, rm)
    P = M * R
    g, u, d = (msb[p] for p in TG.PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("R4_C", max(r4, 1)), ("RM_C", rm)]
    (h,) = _ka()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], L["gate_proj"]["l1"], L["up_proj"]["l1"],
                         L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, go, gp],
                 template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra)],
                 grid=(N // ra * 64, P, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    (y,) = _kb()(inputs=[h, d["w3"], d["s3"], d["b3"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls, go, gp],
                 template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                 grid=(D // (rb * nsg_b) * 32 * nsg_b, P, 1), threadgroup=(32 * nsg_b, 1, 1),
                 output_shapes=[(P, D)], output_dtypes=[x.dtype])
    return TG.combine_dh(y, y, mx.ones((M, R), mx.int32), scores.reshape(M, R), 0)


def compile_graph():
    """for patches/sol_layer/compile_check.py (offline xcrun metal of every generated kernel; no GPU work)"""
    E, M, R = 4, 3, 10
    SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
    msb = {p: {"w3": mx.zeros((E, o, i * 3 // 32), mx.uint32), "s3": mx.zeros((E, o, i // 64), mx.bfloat16),
               "b3": mx.zeros((E, o, i // 64), mx.bfloat16)} for p, (o, i) in SH.items()}
    lsb = {p: {"l1": mx.zeros((E, o, i // 32), mx.uint32), "b": mx.zeros((E, o, i // 64), mx.bfloat16)} for p, (o, i) in SH.items()}
    x = mx.zeros((M, 2560), mx.bfloat16)
    ids = mx.zeros((M, R), mx.int32)
    sc = mx.zeros((M, R), mx.bfloat16)
    tier_glu_mr(x, msb, lsb, ids, ids[:, :2], sc, plan="4433333333")
    tier_glu_mr(x, msb, None, ids, None, sc, plan="3333333333")
