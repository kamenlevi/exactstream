"""TRL-G (Leap, Wed 2026-11-04 20:4x; TRELLIS FULL-STORE prep, the orchestrator 20:0x item 2): the routed expert GLU served
straight from trellis codes (trl_encode.py's format), grouped per distinct expert like NAX (gpu_grp; no tiers).
  Format per expert (trl_encode.py): {p}_w uint32 [O/16, K/16, TW] (16 x 16 tiles of W' = W R, row-major in the tile;
  bitstream = s_0's 12 bits then k bits per value, MSB first; TW = 17 words at k 2, 25 at k 3), {p}_s bf16 [O] (row RMS
  sc). Weight (o, i) = inst3(state) / RMS3 x sc[o]; state of tile value t = bits [k t, k t + 12) of the tile's stream.
  So tile row r (16 values) = bits [16 k r, 16 k r + 15 k + 12): k 2 -> words r, r + 1; k 3 -> words 3r/2 .. +2 (bit
  offset 16 when r is odd).
  y = W x = W' (R^T x): x' = H512(s_L * x) once per token (per-layer signs, seed L); h' = H128(s_e * h) per pair (seed
  1000 L + e); H = Sylvester / sqrt(n) blocks (fit_q.hadamard; mx.hadamard_transform's order).
Kernels: A = gate + up + SwiGLU, B = down. One threadgroup = (group, block of output rows); lane = one output row: it walks
its row's tiles, decodes 16 states from a 64-bit window, computes 3INST (no LUT), and serves up to 4 rows of the group per
pass (a group of m rows takes ceil(m / 4) passes; S 16 calls have ~2.6 rows per expert).
API: tier_glu_trl(x, P, slot, grp=None) -> y per pair [M R, D] bf16; trl_moe(...) adds the combine (combine_phase, as
NAX6-N's harness). P = pools: wg / wu [E, 40, 160, TWg], wd [E, 160, 40, TWd] uint32; sg / su [E, 640], sd [E, 2560] bf16;
sx [2560] f32; sdn [E, 640] f32; kg / kd bits.
usage: [BigMoE venv] tierglu_trl.py test -> ~/Leap/data/tierglu_trl_test.txt (GPU; MEM < 2 GB; share lock)"""
import math, os, sys, time
from functools import lru_cache
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs2/Leap_scripts"))
import trl_metal as TM

K, N, D = 2560, 640, 2560
TWK = {2: 17, 3: 25}
INV_RMS3 = 1.0 / TM.RMS3

_HDR = f"constant constexpr float INV_C = {INV_RMS3!r}f;\n" + r"""
inline float inst3(uint s) {                                // QTIP 3INST: an LCG, a mask, a sum of two fp16 halves
    const uint xq = 89226354u * s + 64248484u;
    const uint mq = (xq & 0x8FFF8FFFu) ^ 0x3B603B60u;
    const half2 hv = as_type<half2>(mq);
    return static_cast<float>(hv.x) + static_cast<float>(hv.y);
}
// the 64-bit window holding tile row rr's states (bits [16 KBT rr, 16 KBT rr + 15 KBT + 12) of the tile's stream)
template <int KBT> inline ulong trl_win(const device uint* wt, uint rr) {
    if (KBT == 2) return ((ulong)wt[rr] << 32) | (ulong)wt[rr + 1];
    const uint wi = (3u * rr) >> 1;
    const uint w0 = wt[wi], w1 = wt[wi + 1];
    if ((rr & 1u) == 0u) return ((ulong)w0 << 32) | (ulong)w1;
    const uint w2 = wt[wi + 2];
    return ((ulong)((w0 << 16) | (w1 >> 16)) << 32) | (ulong)((w1 << 16) | (w2 >> 16));
}
// 16 states of the window -> dot products with up to RP_C rows (16 values each, float4 x 4)
#define TRL_DOT4(KBT, WIN, XR, NROW, ACC)                                                           \
    {                                                                                               \
        float wv[16];                                                                               \
        for (int c = 0; c < 16; ++c) wv[c] = inst3((uint)((WIN) >> (52 - KBT * c)) & 0xFFFu);       \
        for (int i = 0; i < RP_C; ++i) {                                                            \
            if (i < (NROW)) {                                                                       \
                const device float4* xv = (const device float4*)(XR[i]);                            \
                float a = 0.0f;                                                                     \
                for (int q = 0; q < 4; ++q) {                                                       \
                    const float4 t = xv[q];                                                         \
                    a = fma(wv[4 * q], t.x, a); a = fma(wv[4 * q + 1], t.y, a);                     \
                    a = fma(wv[4 * q + 2], t.z, a); a = fma(wv[4 * q + 3], t.w, a);                 \
                }                                                                                   \
                ACC[i] += a;                                                                        \
            }                                                                                       \
        }                                                                                           \
    }
"""

_A = r"""
    constexpr int KK = K_C, NN = N_C, NSG = NSG_C, HS = NSG / 2, KT = KK / 16, NT = NN / 16, TW = KG == 2 ? 17 : 25;
    threadgroup float gv[RP_C][HS * 32];
    const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.y;
    if (g >= (uint)nseg[0]) return;
    const int s0 = seg[g], m = seg[g + 1] - s0;
    const bool up = sgi >= (uint)HS;
    const uint lr = (sgi % HS) * 32 + lane;
    const uint row = threadgroup_position_in_grid.x * (HS * 32) + lr;
    const uint e = (uint)slot[pidx[s0]];
    const device uint* W = (up ? wu : wg) + (((size_t)e * NT + (row >> 4)) * KT) * TW;
    const uint rr = row & 15u;
    const float sc = static_cast<float>((up ? su : sg)[(size_t)e * NN + row]) * INV_C;
    for (int c0 = 0; c0 < m; c0 += RP_C) {
        const int nrow = min(RP_C, m - c0);
        float acc[RP_C];
        const device float* xb[RP_C];
        for (int i = 0; i < RP_C; ++i) { acc[i] = 0.0f; xb[i] = xq + (size_t)(pidx[s0 + c0 + min(i, nrow - 1)] / NR_C) * KK; }
        for (int tc = 0; tc < KT; ++tc) {
            const ulong win = trl_win<KG>(W + tc * TW, rr);
            const device float* xr[RP_C];
            for (int i = 0; i < RP_C; ++i) xr[i] = xb[i] + tc * 16;
            TRL_DOT4(KG, win, xr, nrow, acc)
        }
        if (!up) for (int i = 0; i < RP_C; ++i) gv[i][lr] = acc[i] * sc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (up) {
            for (int i = 0; i < nrow; ++i) {
                const float gg = gv[i][lr];
                h[(size_t)pidx[s0 + c0 + i] * NN + row] = gg / (1.0f + metal::exp(-gg)) * (acc[i] * sc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_B = r"""
    constexpr int NN = N_C, DD = D_C, NSG = NSG_C, NT = NN / 16, TW = KD == 2 ? 17 : 25;
    const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.y;
    if (g >= (uint)nseg[0]) return;
    const int s0 = seg[g], m = seg[g + 1] - s0;
    const uint row = (threadgroup_position_in_grid.x * NSG + sgi) * 32 + lane;
    const uint e = (uint)slot[pidx[s0]];
    const device uint* W = wd + (((size_t)e * (DD / 16) + (row >> 4)) * NT) * TW;
    const uint rr = row & 15u;
    const float sc = static_cast<float>(sd[(size_t)e * DD + row]) * INV_C;
    for (int c0 = 0; c0 < m; c0 += RP_C) {
        const int nrow = min(RP_C, m - c0);
        float acc[RP_C];
        const device float* hb[RP_C];
        for (int i = 0; i < RP_C; ++i) { acc[i] = 0.0f; hb[i] = hq + (size_t)pidx[s0 + c0 + min(i, nrow - 1)] * NN; }
        for (int tc = 0; tc < NT; ++tc) {
            const ulong win = trl_win<KD>(W + tc * TW, rr);
            const device float* hr[RP_C];
            for (int i = 0; i < RP_C; ++i) hr[i] = hb[i] + tc * 16;
            TRL_DOT4(KD, win, hr, nrow, acc)
        }
        for (int i = 0; i < nrow; ++i) y[(size_t)pidx[s0 + c0 + i] * DD + row] = static_cast<TO>(acc[i] * sc);
    }
"""

# A2: lane = gate row o AND up row o; x' rows loaded once for both; SwiGLU in registers
_A2 = r"""
    constexpr int KK = K_C, NN = N_C, NSG = NSG_C, KT = KK / 16, NT = NN / 16, TW = KG == 2 ? 17 : 25;
    const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.y;
    if (g >= (uint)nseg[0]) return;
    const int s0 = seg[g], m = seg[g + 1] - s0;
    const uint row = (threadgroup_position_in_grid.x * NSG + sgi) * 32 + lane;
    const uint e = (uint)slot[pidx[s0]];
    const size_t tb = (((size_t)e * NT + (row >> 4)) * KT) * TW;
    const device uint* Wg = wg + tb;
    const device uint* Wu = wu + tb;
    const uint rr = row & 15u;
    const float scg = static_cast<float>(sg[(size_t)e * NN + row]) * INV_C;
    const float scu = static_cast<float>(su[(size_t)e * NN + row]) * INV_C;
    for (int c0 = 0; c0 < m; c0 += RP_C) {
        const int nrow = min(RP_C, m - c0);
        float ag[RP_C], au[RP_C];
        const device float* xb[RP_C];
        for (int i = 0; i < RP_C; ++i) { ag[i] = 0.0f; au[i] = 0.0f; xb[i] = xq + (size_t)(pidx[s0 + c0 + min(i, nrow - 1)] / NR_C) * KK; }
        for (int tc = 0; tc < KT; ++tc) {
            const ulong w1 = trl_win<KG>(Wg + tc * TW, rr), w2 = trl_win<KG>(Wu + tc * TW, rr);
            float vg[16], vu[16];
            for (int c = 0; c < 16; ++c) {
                vg[c] = inst3((uint)(w1 >> (52 - KG * c)) & 0xFFFu);
                vu[c] = inst3((uint)(w2 >> (52 - KG * c)) & 0xFFFu);
            }
            for (int i = 0; i < RP_C; ++i) {
                if (i < nrow) {
                    const device float4* xv = (const device float4*)(xb[i] + tc * 16);
                    float a = 0.0f, b = 0.0f;
                    for (int q = 0; q < 4; ++q) {
                        const float4 t = xv[q];
                        a = fma(vg[4 * q], t.x, a); a = fma(vg[4 * q + 1], t.y, a);
                        a = fma(vg[4 * q + 2], t.z, a); a = fma(vg[4 * q + 3], t.w, a);
                        b = fma(vu[4 * q], t.x, b); b = fma(vu[4 * q + 1], t.y, b);
                        b = fma(vu[4 * q + 2], t.z, b); b = fma(vu[4 * q + 3], t.w, b);
                    }
                    ag[i] += a; au[i] += b;
                }
            }
        }
        for (int i = 0; i < nrow; ++i) {
            const float gg = ag[i] * scg;
            h[(size_t)pidx[s0 + c0 + i] * NN + row] = gg / (1.0f + metal::exp(-gg)) * (au[i] * scu);
        }
    }
"""

# ---- TRL-TA (20:5x, after TRL-G's KILL): B10(b)'s TA / TB mapping, grouped. Layout v2 per projection: words
# [E][row block rb][word w][tile column tc] (tile column fastest), so the lanes' word loads are contiguous.
_HDR_TA = r"""
template <int KBT> inline ulong trl_win_i(const device uint* wt, uint rr, uint st) {   // words at stride st
    if (KBT == 2) return ((ulong)wt[rr * st] << 32) | (ulong)wt[(rr + 1) * st];
    const uint wi = (3u * rr) >> 1;
    const uint w0 = wt[wi * st], w1 = wt[(wi + 1) * st];
    if ((rr & 1u) == 0u) return ((ulong)w0 << 32) | (ulong)w1;
    const uint w2 = wt[(wi + 2) * st];
    return ((ulong)((w0 << 16) | (w1 >> 16)) << 32) | (ulong)((w1 << 16) | (w2 >> 16));
}
"""

# A-TA: simdgroups [0, HS) gate rows, [HS, NSG) the same up rows, RA rows each; lane l = tile columns l, l + 32, ...;
# each group row's 16 x' values per tile column sit in registers and serve all RA output rows
_A_TA = r"""
    constexpr int KK = K_C, NN = N_C, RA = RA_C, NSG = NSG_C, HS = NSG / 2, ROWS = HS * RA, KT = KK / 16, NT = NN / 16;
    constexpr int TW = KG == 2 ? 17 : 25, RP = RP_C;
    threadgroup float gv[RP][ROWS];
    const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.y;
    if (g >= (uint)nseg[0]) return;
    const int s0 = seg[g], m = seg[g + 1] - s0;
    const bool up = sgi >= (uint)HS;
    const uint jl = (sgi % HS) * RA;
    const uint j0 = threadgroup_position_in_grid.x * ROWS + jl;
    const uint e = (uint)slot[pidx[s0]];
    const device uint* W = (up ? wu : wg) + (size_t)e * NT * TW * KT;
    const device bfloat16_t* scp = (up ? su : sg) + (size_t)e * NN;
    for (int c0 = 0; c0 < m; c0 += RP) {
        const int nrow = min(RP, m - c0);
        float acc[RA][RP];
        for (int i = 0; i < RA; ++i) for (int r = 0; r < RP; ++r) acc[i][r] = 0.0f;
        for (int b = 0; b < KT / 32; ++b) {
            const uint tc = b * 32 + lane;
            float xv[RP][16];
            for (int r = 0; r < RP; ++r) {
                if (r < nrow) {
                    const device float4* xp = (const device float4*)(xq + (size_t)(pidx[s0 + c0 + r] / NR_C) * KK + tc * 16);
                    for (int q = 0; q < 4; ++q) { const float4 t = xp[q]; xv[r][4 * q] = t.x; xv[r][4 * q + 1] = t.y; xv[r][4 * q + 2] = t.z; xv[r][4 * q + 3] = t.w; }
                }
            }
            for (int i = 0; i < RA; ++i) {
                const uint o = j0 + i;
                const ulong win = trl_win_i<KG>(W + (size_t)(o >> 4) * TW * KT + tc, o & 15u, KT);
                float wv[16];
                for (int c = 0; c < 16; ++c) wv[c] = inst3((uint)(win >> (52 - KG * c)) & 0xFFFu);
                for (int r = 0; r < RP; ++r) {
                    if (r < nrow) {
                        float a = 0.0f;
                        for (int c = 0; c < 16; ++c) a = fma(wv[c], xv[r][c], a);
                        acc[i][r] += a;
                    }
                }
            }
        }
        for (int i = 0; i < RA; ++i) {
            const float sc = static_cast<float>(scp[j0 + i]) * INV_C;
            for (int r = 0; r < RP; ++r) {
                const float v = simd_sum(acc[i][r]) * sc;
                if (!up && lane == 0) gv[r][jl + i] = v;
                acc[i][r] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (up && lane == 0) {
            for (int r = 0; r < nrow; ++r) {
                for (int i = 0; i < RA; ++i) {
                    const float gg = gv[r][jl + i];
                    h[(size_t)pidx[s0 + c0 + r] * NN + j0 + i] = gg / (1.0f + metal::exp(-gg)) * acc[i][r];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# B-TA: each simdgroup = RB down rows; its RB x NT (row, tile column) items spread over the lanes (TB's mapping)
_B_TA = r"""
    constexpr int NN = N_C, DD = D_C, RB = RB_C, NSG = NSG_C, NT = NN / 16, Q = RB * NT, TW = KD == 2 ? 17 : 25, RP = RP_C;
    static_assert(Q % 32 == 0, "RB * N / 16 must be a multiple of 32");
    const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.y;
    if (g >= (uint)nseg[0]) return;
    const int s0 = seg[g], m = seg[g + 1] - s0;
    const uint d0 = (threadgroup_position_in_grid.x * NSG + sgi) * RB;
    const uint e = (uint)slot[pidx[s0]];
    const device uint* W = wd + (size_t)e * (DD / 16) * TW * NT;
    for (int c0 = 0; c0 < m; c0 += RP) {
        const int nrow = min(RP, m - c0);
        float acc[RB][RP];
        for (int k = 0; k < RB; ++k) for (int r = 0; r < RP; ++r) acc[k][r] = 0.0f;
        for (int it = 0; it < Q / 32; ++it) {
            const uint q = lane + 32 * it;
            const uint row = q / NT, tc = q % NT;
            const uint o = d0 + row;
            const ulong win = trl_win_i<KD>(W + (size_t)(o >> 4) * TW * NT + tc, o & 15u, NT);
            float wv[16];
            for (int c = 0; c < 16; ++c) wv[c] = inst3((uint)(win >> (52 - KD * c)) & 0xFFFu);
            for (int r = 0; r < RP; ++r) {
                if (r < nrow) {
                    const device float4* hp = (const device float4*)(hq + (size_t)pidx[s0 + c0 + r] * NN + tc * 16);
                    float a = 0.0f;
                    for (int qq = 0; qq < 4; ++qq) {
                        const float4 t = hp[qq];
                        a = fma(wv[4 * qq], t.x, a); a = fma(wv[4 * qq + 1], t.y, a);
                        a = fma(wv[4 * qq + 2], t.z, a); a = fma(wv[4 * qq + 3], t.w, a);
                    }
                    for (int k = 0; k < RB; ++k) acc[k][r] += row == (uint)k ? a : 0.0f;
                }
            }
        }
        for (int k = 0; k < RB; ++k) {
            const float sc = static_cast<float>(sd[(size_t)e * DD + d0 + k]) * INV_C;
            for (int r = 0; r < RP; ++r) {
                const float v = simd_sum(acc[k][r]) * sc;
                if (lane == 0 && r < nrow) y[(size_t)pidx[s0 + c0 + r] * DD + d0 + k] = static_cast<TO>(v);
            }
        }
    }
"""


@lru_cache(None)
def _ka_ta():
    return mx.fast.metal_kernel(name="trl_ta_a", input_names=["xq", "wg", "wu", "sg", "su", "slot", "pidx", "seg", "nseg"],
                                output_names=["h"], source=_A_TA, header=_HDR + _HDR_TA)


@lru_cache(None)
def _kb_ta():
    return mx.fast.metal_kernel(name="trl_ta_b", input_names=["hq", "wd", "sd", "slot", "pidx", "seg", "nseg"],
                                output_names=["y"], source=_B_TA, header=_HDR + _HDR_TA)


def to_v2(P):
    """layout v1 (tiles [rb][tc][w]) -> v2 ([rb][w][tc]) for TRL-TA"""
    Q = dict(P)
    for nm in ("wg", "wu", "wd"):
        Q[nm + "2"] = mx.contiguous(P[nm].transpose(0, 1, 3, 2))
    mx.eval(Q["wg2"], Q["wu2"], Q["wd2"])
    return Q


def tier_glu_trl_ta(x, P, slot, R=10, grp=None, ra=4, nsg_a=8, rb=4, nsg_b=8, rp=4, out=mx.bfloat16):
    """TRL-TA: x [M, K] bf16, slot [M, R] -> y per pair [M R, D] (P needs wg2 / wu2 / wd2: to_v2)"""
    M = x.shape[0]
    NP = M * R
    slot = slot.astype(mx.int32).reshape(-1)
    if grp is None:
        grp = grp_of(slot, R)
    pidx, seg, nseg = grp[:3]
    xq = mx.hadamard_transform((x.astype(mx.float32) * P["sx"]).reshape(M, K // 512, 512),
                               scale=1 / math.sqrt(512)).reshape(M, K)
    rows = (nsg_a // 2) * ra
    (h,) = _ka_ta()(inputs=[xq, P["wg2"], P["wu2"], P["sg"], P["su"], slot, pidx, seg, nseg],
                    template=[("KG", P["kg"]), ("K_C", K), ("N_C", N), ("RA_C", ra), ("NSG_C", nsg_a), ("NR_C", R), ("RP_C", rp)],
                    grid=(N // rows * 32 * nsg_a, NP, 1), threadgroup=(32 * nsg_a, 1, 1),
                    output_shapes=[(NP, N)], output_dtypes=[mx.float32])
    hq = mx.hadamard_transform((h * P["sdn"][slot]).reshape(NP, N // 128, 128), scale=1 / math.sqrt(128)).reshape(NP, N)
    (y,) = _kb_ta()(inputs=[hq, P["wd2"], P["sd"], slot, pidx, seg, nseg],
                    template=[("KD", P["kd"]), ("N_C", N), ("D_C", D), ("RB_C", rb), ("NSG_C", nsg_b), ("TO", out), ("RP_C", rp)],
                    grid=(D // (rb * nsg_b) * 32 * nsg_b, NP, 1), threadgroup=(32 * nsg_b, 1, 1),
                    output_shapes=[(NP, D)], output_dtypes=[out])
    return y


def trl_moe_ta(x, P, slot, scores, R=10, grp=None, **kw):
    import tierglu as TG
    y = tier_glu_trl_ta(x, P, slot, R, grp, **kw)
    M = x.shape[0]
    return TG.combine_phase(y, y, mx.ones((M, R), mx.int32), scores)

# ---- TRL-NAX (21:1x, after TRL-G / TRL-TA's KILLs): NAX6's MPP structure with the trellis decode. Each lane decodes its
# output row's next 32 weights (two tiles) into a 32 x 32 half tile in threadgroup memory; matmul2d (16 x 32 x 32, half x
# half -> float) multiplies all <= 16 rows of the group at once (gpu_grp's groups), so every weight is decoded once per
# group and the FMAs run on the matrix units. The row scale (constant over K) applies once at the end.
_HDR_NAX = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
"""

_A_NAX = r"""
    constexpr int KK = K_C, NN = N_C, NR = NR_C, KT = KK / 16, NT = NN / 16, TW = KG == 2 ? 17 : 25, KS = KS_C;
    constexpr int KCH = KK / KS;                                  // K values per chunk (simdgroup)
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;               // pj = sg & 1 (0 gate, 1 up), kc = sg >> 1
    const uint pj = sg & 1u, kc = sg >> 1;
    threadgroup half tb_all[2 * KS * 32 * 32];                    // reused for the chunk partials
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint n0 = threadgroup_position_in_grid.x * 32;
    const uint e = (uint)slot[pidx[q0]];
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const uint ma0 = (uint)pidx[q0 + (fm0 < nq ? fm0 : 0)] / NR, ma1 = (uint)pidx[q0 + (fm0 + 8 < nq ? fm0 + 8 : 0)] / NR;
    const device half* ar0 = xh + (size_t)ma0 * KK;
    const device half* ar1 = xh + (size_t)ma1 * KK;
    const uint row = n0 + lane;
    const device uint* W = (pj == 0 ? wg : wu) + (((size_t)e * NT + (row >> 4)) * KT) * TW;
    const uint rr = row & 15u;
    threadgroup half* tb = tb_all + sg * 32 * 32;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    using TBT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TBT, float>();
    for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
    for (int st = 0; st < KCH / 32; ++st) {
        const int k0 = (int)kc * KCH + st * 32;
        for (int u = 0; u < 2; ++u) {
            const ulong win = trl_win<KG>(W + (k0 / 16 + u) * TW, rr);
            threadgroup half4* dst = (threadgroup half4*)(tb + lane * 32 + u * 16);
            for (int c4 = 0; c4 < 4; ++c4) {
                half4 v;
                for (int c = 0; c < 4; ++c) v[c] = (half)inst3((uint)(win >> (52 - KG * (4 * c4 + c))) & 0xFFFu);
                dst[c4] = v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const half4 a00 = *(const device half4*)(ar0 + k0 + fn0);
        const half4 a10 = *(const device half4*)(ar1 + k0 + fn0);
        const half4 a01 = *(const device half4*)(ar0 + k0 + 16 + fn0);
        const half4 a11 = *(const device half4*)(ar1 + k0 + 16 + fn0);
        for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
        TBT tbt(tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 32});
        op.run(ca, tbt, cg);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* part = (threadgroup float*)tb_all;         // [projection][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) part[(pj * KS + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = cg[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const device bfloat16_t* scg = sg_ + (size_t)e * NN + n0;
        const device bfloat16_t* scu = su + (size_t)e * NN + n0;
        for (int i = 0; i < 16; ++i) {
            const int r = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
            float gs = 0.0f, us = 0.0f;
            for (int c = 0; c < KS; ++c) { gs += part[(0 * KS + c) * 512 + r * 32 + col]; us += part[(1 * KS + c) * 512 + r * 32 + col]; }
            if (r < nq) {
                const float gg = gs * static_cast<float>(scg[col]) * INV_C, uu = us * static_cast<float>(scu[col]) * INV_C;
                h[(size_t)pidx[q0 + r] * NN + n0 + col] = gg / (1.0f + metal::exp(-gg)) * uu;
            }
        }
    }
"""

# B: simdgroup = (column tile ti, K chunk kc); output tile of 32 down rows; K = N = 640 over KSB chunks
_B_NAX = r"""
    constexpr int NN = N_C, DD = D_C, NT = NN / 16, TW = KD == 2 ? 17 : 25, KSB = KSB_C, NSB = NSB_C;
    constexpr int KCH = NN / KSB;
    const ushort lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;               // ti = sg % NSB, kc = sg / NSB
    const uint ti = sg % NSB, kc = sg / NSB;
    threadgroup half tb_all[NSB * KSB * 32 * 32];
    const uint gq = threadgroup_position_in_grid.y;
    if ((int)gq >= nseg[0]) return;
    const int q0 = seg[gq], nq = seg[gq + 1] - q0;
    const uint d0 = (threadgroup_position_in_grid.x * NSB + ti) * 32;
    const uint e = (uint)slot[pidx[q0]];
    const short qid = lane >> 2;
    const short fm0 = (qid & 4) | ((lane >> 1) & 3);
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const device half* ar0 = hh + (size_t)pidx[q0 + (fm0 < nq ? fm0 : 0)] * NN;
    const device half* ar1 = hh + (size_t)pidx[q0 + (fm0 + 8 < nq ? fm0 + 8 : 0)] * NN;
    const uint row = d0 + lane;
    const device uint* W = wd + (((size_t)e * (DD / 16) + (row >> 4)) * NT) * TW;
    const uint rr = row & 15u;
    threadgroup half* tb = tb_all + sg * 32 * 32;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<half, half, float>();
    using TBT = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;
    auto cg = op.template get_destination_cooperative_tensor<decltype(ca), TBT, float>();
    for (int i = 0; i < 16; ++i) cg[i] = 0.0f;
    for (int st = 0; st < KCH / 32; ++st) {
        const int k0 = (int)kc * KCH + st * 32;
        for (int u = 0; u < 2; ++u) {
            const ulong win = trl_win<KD>(W + (k0 / 16 + u) * TW, rr);
            threadgroup half4* dst = (threadgroup half4*)(tb + lane * 32 + u * 16);
            for (int c4 = 0; c4 < 4; ++c4) {
                half4 v;
                for (int c = 0; c < 4; ++c) v[c] = (half)inst3((uint)(win >> (52 - KD * (4 * c4 + c))) & 0xFFFu);
                dst[c4] = v;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const half4 a00 = *(const device half4*)(ar0 + k0 + fn0);
        const half4 a10 = *(const device half4*)(ar1 + k0 + fn0);
        const half4 a01 = *(const device half4*)(ar0 + k0 + 16 + fn0);
        const half4 a11 = *(const device half4*)(ar1 + k0 + 16 + fn0);
        for (int q = 0; q < 4; ++q) { ca[q] = a00[q]; ca[4 + q] = a10[q]; ca[8 + q] = a01[q]; ca[12 + q] = a11[q]; }
        TBT tbt(tb, dextents<int32_t, 2>(32, 32), array<int32_t, 2>{1, 32});
        op.run(ca, tbt, cg);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* part = (threadgroup float*)tb_all;         // [tile][chunk][16 x 32]
    for (int i = 0; i < 16; ++i) part[(ti * KSB + kc) * 512 + (fm0 + ((i >> 2) & 1) * 8) * 32 + 16 * (i >> 3) + fn0 + (i & 3)] = cg[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kc == 0) {
        const device bfloat16_t* scd = sd + (size_t)e * DD + d0;
        for (int i = 0; i < 16; ++i) {
            const int r = fm0 + ((i >> 2) & 1) * 8, col = 16 * (i >> 3) + fn0 + (i & 3);
            float ys = 0.0f;
            for (int c = 0; c < KSB; ++c) ys += part[(ti * KSB + c) * 512 + r * 32 + col];
            if (r < nq) y[(size_t)pidx[q0 + r] * DD + d0 + col] = static_cast<TO>(ys * static_cast<float>(scd[col]) * INV_C);
        }
    }
"""


@lru_cache(None)
def _ka_nax():
    return mx.fast.metal_kernel(name="trl_nax_a", input_names=["xh", "wg", "wu", "sg_", "su", "slot", "pidx", "seg", "nseg"],
                                output_names=["h"], source=_A_NAX, header=_HDR + _HDR_NAX)


@lru_cache(None)
def _kb_nax():
    return mx.fast.metal_kernel(name="trl_nax_b", input_names=["hh", "wd", "sd", "slot", "pidx", "seg", "nseg"],
                                output_names=["y"], source=_B_NAX, header=_HDR + _HDR_NAX)


def tier_glu_trl_nax(x, P, slot, R=10, grp=None, ks=4, ksb=2, nsb=2, out=mx.bfloat16):
    """TRL-NAX: x [M, K] bf16, slot [M, R] -> y per pair [M R, D]; v1 layout pools (as TRL-G); groups <= 16 rows"""
    M = x.shape[0]
    NP = M * R
    slot = slot.astype(mx.int32).reshape(-1)
    if grp is None:
        grp = grp_of(slot, R)
    pidx, seg, nseg = grp[:3]
    xh = mx.hadamard_transform((x.astype(mx.float32) * P["sx"]).reshape(M, K // 512, 512),
                               scale=1 / math.sqrt(512)).reshape(M, K).astype(mx.float16)
    (h,) = _ka_nax()(inputs=[xh, P["wg"], P["wu"], P["sg"], P["su"], slot, pidx, seg, nseg],
                     template=[("KG", P["kg"]), ("K_C", K), ("N_C", N), ("NR_C", R), ("KS_C", ks)],
                     grid=(N // 32 * 32 * 2 * ks, NP, 1), threadgroup=(32 * 2 * ks, 1, 1),
                     output_shapes=[(NP, N)], output_dtypes=[mx.float32])
    hh = mx.hadamard_transform((h * P["sdn"][slot]).reshape(NP, N // 128, 128),
                               scale=1 / math.sqrt(128)).reshape(NP, N).astype(mx.float16)
    (y,) = _kb_nax()(inputs=[hh, P["wd"], P["sd"], slot, pidx, seg, nseg],
                     template=[("KD", P["kd"]), ("N_C", N), ("D_C", D), ("KSB_C", ksb), ("NSB_C", nsb), ("TO", out)],
                     grid=(D // (32 * nsb) * 32 * nsb * ksb, NP, 1), threadgroup=(32 * nsb * ksb, 1, 1),
                     output_shapes=[(NP, D)], output_dtypes=[out])
    return y


def trl_moe_nax(x, P, slot, scores, R=10, grp=None, **kw):
    import tierglu as TG
    y = tier_glu_trl_nax(x, P, slot, R, grp, **kw)
    M = x.shape[0]
    return TG.combine_phase(y, y, mx.ones((M, R), mx.int32), scores)


@lru_cache(None)
def _ka2():
    return mx.fast.metal_kernel(name="trl_g_a2", input_names=["xq", "wg", "wu", "sg", "su", "slot", "pidx", "seg", "nseg"],
                                output_names=["h"], source=_A2, header=_HDR)


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(name="trl_g_a", input_names=["xq", "wg", "wu", "sg", "su", "slot", "pidx", "seg", "nseg"],
                                output_names=["h"], source=_A, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(name="trl_g_b", input_names=["hq", "wd", "sd", "slot", "pidx", "seg", "nseg"],
                                output_names=["y"], source=_B, header=_HDR)


def signs(L, E=512):
    """fit_b3.rot's sign vectors: per layer (gate / up, seed L) and per expert (down, seed 1000 L + e)"""
    sx = np.random.default_rng(L).choice(np.array([-1.0, 1.0]), K).astype(np.float32)
    sdn = np.stack([np.random.default_rng(1000 * L + e).choice(np.array([-1.0, 1.0]), N) for e in range(E)]).astype(np.float32)
    return sx, sdn


def grp_of(slot, R):
    """gpu_grp with a tier-free plan: groups = distinct experts (split every 16 pairs)"""
    import tierglu_nax6 as N6
    ms = slot.reshape(-1, R)
    return N6.gpu_grp(ms, ms[:, :1], "3" * R)


def tier_glu_trl(x, P, slot, R=10, grp=None, nsg_a=4, nsg_b=4, out=mx.bfloat16, a2=False, rp=4):
    """x [M, K] bf16; slot [M, R] int (experts, rank order) -> y per pair [M R, D] bf16"""
    M = x.shape[0]
    NP = M * R
    slot = slot.astype(mx.int32).reshape(-1)
    if grp is None:
        grp = grp_of(slot, R)
    pidx, seg, nseg = grp[:3]
    xq = mx.hadamard_transform((x.astype(mx.float32) * P["sx"]).reshape(M, K // 512, 512),
                               scale=1 / math.sqrt(512)).reshape(M, K)
    hs = (nsg_a // 2) * 32 if not a2 else nsg_a * 32
    (h,) = (_ka2() if a2 else _ka())(inputs=[xq, P["wg"], P["wu"], P["sg"], P["su"], slot, pidx, seg, nseg],
                                     template=[("KG", P["kg"]), ("K_C", K), ("N_C", N), ("NSG_C", nsg_a), ("NR_C", R), ("RP_C", rp)],
                                     grid=(N // hs * 32 * nsg_a, NP, 1), threadgroup=(32 * nsg_a, 1, 1),
                                     output_shapes=[(NP, N)], output_dtypes=[mx.float32])
    hq = mx.hadamard_transform((h * P["sdn"][slot]).reshape(NP, N // 128, 128), scale=1 / math.sqrt(128)).reshape(NP, N)
    (y,) = _kb()(inputs=[hq, P["wd"], P["sd"], slot, pidx, seg, nseg],
                 template=[("KD", P["kd"]), ("N_C", N), ("D_C", D), ("NSG_C", nsg_b), ("TO", out), ("RP_C", rp)],
                 grid=(D // (32 * nsg_b) * 32 * nsg_b, NP, 1), threadgroup=(32 * nsg_b, 1, 1),
                 output_shapes=[(NP, D)], output_dtypes=[out])
    return y


def trl_moe(x, P, slot, scores, R=10, grp=None, **kw):
    import tierglu as TG
    y = tier_glu_trl(x, P, slot, R, grp, **kw)
    M = x.shape[0]
    return TG.combine_phase(y, y, mx.ones((M, R), mx.int32), scores)


def random_pools(E=512, kg=2, kd=3, seed=0, L=18):
    rng = np.random.default_rng(seed)
    P = {"kg": kg, "kd": kd}
    for nm, (O, I, k) in (("g", (N, K, kg)), ("u", (N, K, kg)), ("d", (D, N, kd))):
        P["w" + nm] = mx.array(rng.integers(0, 1 << 32, (E, O // 16, I // 16, TWK[k]), dtype=np.uint32))
        P["s" + nm] = mx.array((rng.uniform(0.6, 1.4, (E, O)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    sx, sdn = signs(L, E)
    P["sx"], P["sdn"] = mx.array(sx), mx.array(sdn)
    mx.eval([v for v in P.values() if isinstance(v, mx.array)])
    return P


def random_pools_v2(E=512, kg=2, kd=3, seed=0, L=18, share_d=None):
    """random-stream pools directly in layout v2 (TRL-TA); share_d = another v2 pool dict whose wd2 / sd to reuse"""
    rng = np.random.default_rng(seed)
    P = {"kg": kg, "kd": kd}
    for nm, (O, I, k) in (("g", (N, K, kg)), ("u", (N, K, kg)), ("d", (D, N, kd))):
        if nm == "d" and share_d is not None:
            P["wd2"], P["sd"] = share_d["wd2"], share_d["sd"]
            continue
        P["w" + nm + "2"] = mx.array(rng.integers(0, 1 << 32, (E, O // 16, TWK[k], I // 16), dtype=np.uint32))
        P["s" + nm] = mx.array((rng.uniform(0.6, 1.4, (E, O)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    sx, sdn = signs(L, E)
    P["sx"], P["sdn"] = mx.array(sx), mx.array(sdn)
    mx.eval([v for v in P.values() if isinstance(v, mx.array)])
    return P


def ref_forward(x, P, ids, scores, pairs=False):
    """float64 numpy: dense Q = LUT3[states] x sc per expert from the streams (TM.unpack_states), the same rotations"""
    import fit_q as F                                             # Fit12's hadamard (Sylvester / sqrt n), read only
    xs = np.array(x.astype(mx.float32)).astype(np.float64)
    ids = np.array(ids).astype(np.int64)
    sc = {k: np.array(P[k].astype(mx.float32)).astype(np.float64) for k in ("sg", "su", "sd")}
    sx, sdn = np.array(P["sx"]).astype(np.float64), np.array(P["sdn"]).astype(np.float64)
    H512, H128 = F.hadamard(512).astype(np.float64), F.hadamard(128).astype(np.float64)
    cache = {}

    def Q(nm, e, O, I, k):
        if (nm, e) not in cache:
            w = np.array(P["w" + nm][e]).reshape(-1, TWK[k])
            st = TM.unpack_states(w, k).reshape(O // 16, I // 16, 16, 16)
            cache[(nm, e)] = TM.LUT3[st.astype(np.int64)].astype(np.float64).transpose(0, 2, 1, 3).reshape(O, I) \
                * sc["s" + nm][e][:, None]
        return cache[(nm, e)]
    M, R = ids.shape
    out = np.zeros((M, D))
    yp = np.zeros((M * R, D))
    sco = np.array(scores.astype(mx.float32)).astype(np.float64)
    for m in range(M):
        xp = ((xs[m] * sx).reshape(-1, 512) @ H512).reshape(-1)
        for r in range(R):
            e = int(ids[m, r])
            g, u = Q("g", e, N, K, P["kg"]) @ xp, Q("u", e, N, K, P["kg"]) @ xp
            hh = g / (1 + np.exp(-g)) * u
            hp = ((hh * sdn[e]).reshape(-1, 128) @ H128).reshape(-1)
            yp[m * R + r] = Q("d", e, D, N, P["kd"]) @ hp
            out[m] += sco[m, r] * yp[m * R + r]
    return yp if pairs else out


def _say(path):
    def say(s=""):
        print(s, flush=True)
        open(path, "a").write(s + "\n")
    return say


def test():
    sys.path.insert(0, os.path.expanduser("~/Fit12/scripts"))
    say = _say(os.path.expanduser("~/Leap/data/tierglu_trl_test.txt"))
    say(f"\nTRL-G test {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
    ok = True
    NAXR = []
    for kg, kd in ((2, 3), (3, 3), (2, 2)):
        P = random_pools(E=64, kg=kg, kd=kd, seed=kg * 10 + kd)
        P2 = to_v2(P)
        rng = np.random.default_rng(5)
        for M in (1, 3, 16):
            ids = np.stack([rng.choice(64 if M < 16 else 12, 10, replace=False) for _ in range(M)]).astype(np.int32)
            x = mx.array(rng.standard_normal((M, K)).astype(np.float32) * 0.5).astype(mx.bfloat16)
            sco = mx.softmax(mx.array(rng.standard_normal((M, 10)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
            yk = trl_moe(x, P, mx.array(ids), sco)
            y32 = tier_glu_trl(x, P, mx.array(ids), out=mx.float32)
            y32b = tier_glu_trl(x, P, mx.array(ids), out=mx.float32, a2=True, nsg_a=2)
            y32c = tier_glu_trl(x, P, mx.array(ids), out=mx.float32, a2=True, nsg_a=2, rp=8)
            mx.eval(yk, y32, y32b, y32c)
            a2same = bool(mx.array_equal(y32, y32b).item()) and bool(mx.array_equal(y32, y32c).item())
            yr = ref_forward(x, P, ids, sco)
            yrp = ref_forward(x, P, ids, sco, pairs=True)
            err = float(np.abs(np.array(yk.astype(mx.float32)) - yr).max() / np.abs(yr).max())
            ynx = tier_glu_trl_nax(x, P, mx.array(ids), out=mx.float32)
            mx.eval(ynx)
            dnx = np.array(ynx) - yrp
            enx = float(np.abs(dnx).max() / np.abs(yrp).max())
            rnx = float(np.sqrt((dnx ** 2).mean() / (yrp ** 2).mean()))
            NAXR.append((kg, kd, M, enx, rnx))
            yta = [tier_glu_trl_ta(x, P2, mx.array(ids), out=mx.float32, rp=rp_) for rp_ in (2, 4)]
            mx.eval(yta)
            eta = max(float(np.abs(np.array(v) - yrp).max() / np.abs(yrp).max()) for v in yta)
            ok &= eta <= 1e-4
            d32 = np.array(y32) - yrp
            e32 = float(np.abs(d32).max() / np.abs(yrp).max())
            r32 = float(np.sqrt((d32 ** 2).mean() / (yrp ** 2).mean()))
            ok &= e32 <= 1e-4
            say(f"k {kg} / {kd}, M {M:2d} ({'shared experts, groups up to 16 rows' if M == 16 else 'random experts'}): "
                f"fp32 per pair: max |dy| / max |y| {e32:.2e}, rms {r32:.2e} {'ok' if e32 <= 1e-4 else 'FAIL'}; "
                f"bf16 combined (engine path): {err:.2e}; A2 and A2 rp 8 == A bitwise {a2same}; TRL-TA (rp 2 / 4) fp32 {eta:.2e}")
    for kg, kd, M, enx, rnx in NAXR:
        say(f"TRL-NAX k {kg} / {kd}, M {M:2d}: fp32 out, half operands: max |dy| / max |y| {enx:.2e}, rms {rnx:.2e} "
            f"({'ok' if enx <= 5e-3 else 'FAIL'}: half A / B rounding ~5e-4 per operand)")
        ok &= enx <= 5e-3
    say(f"TRL-G test: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if test() else 1)
