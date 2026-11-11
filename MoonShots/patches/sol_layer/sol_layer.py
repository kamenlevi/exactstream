"""Speed-of-light GDN layer for the 125B (NewIdeas task, 2026-10-21): how fast can one GDN layer be when each step is a
fused, bandwidth-shaped kernel? Non-exact (fp32 accumulation, simplified elementwise math), resident weights, the true
dependency chain and the real byte counts (shapes from config.json; hc 4 x 2560, low-rank 320; GDN 16 k-heads + 48
v-heads of 128, conv 4; experts 640 wide, 3-bit tier plan 4433333333 via patches/planes_engine/tierglu.py).
Dispatches per layer (each a global dependency in a non-persistent kernel model):
  1 hc_down   x_stream -> grouped RMSNorm (4 x 2560) -> W_down [320+4, 10240] bf16 -> silu(./4) | inject 2 sigmoid(./4)
  2 hc_up     mix320 -> W_up [10240, 320] bf16 -> mixed[d] = mean_k sigmoid(up[k,d]) * xn[k,d]
  3 gdn_in    RMSNorm(mixed) -> W_in 4-bit [16480, 2560] -> q|k: conv4 + silu + per-head l2norm, v: conv4 + silu, z|b|a
  4 gdn_rec   per v-head: S = g S + beta (v - S k) k^T, o = S q (fp32 state 128 x 128, rows in order), gated RMSNorm(o, z)
  5 gdn_out   W_out 4-bit [2560, 6144] -> stream += inject (x) out          (HC write)
  6 hc_down   (MLP side)          7 hc_up
  8 router+sh RMSNorm(mixed) -> W [512 router | 1 shared gate | 2 x 640 shared gate/up interleaved] 4-bit -> logits,
              silu(g) * u
  9 3c A      top-10 per row from the 512 logits (in-kernel), gate/up/act of the routed experts (3-bit tiers)
 10 3c B      down + the gate-weighted combine (bitwise kernel), + shared down + HC write in a final small dispatch (11)
Reported: ms per layer (48 chained layers over NLP distinct weight sets), and the byte floor at 300 GB/s.
usage: sol_layer.py   (GPU ~1.5 GB; quiet lock)"""
import os, sys, time
from functools import lru_cache
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planes_engine"))
import mlx.core as mx
from tierglu import tier_glu

D, HC, LR, G = 2560, 4, 320, 64
QK, VD, CONV = 2048, 6144, 10240
NIN = CONV + VD + 48 + 48                                   # 16480 in_proj rows
E, TOPK, FF = 512, 10, 640

_HDR = r"""
inline float silu_f(float x) { return x / (1.0f + metal::exp(-x)); }
inline float sigm_f(float x) { return 1.0f / (1.0f + metal::exp(-x)); }
"""

# ---- bf16 GEMV: y[m, n] = W[n, :] . xin[m, :]; lane l takes 8-value chunks l, l+32, ...; RB rows per simdgroup
_G16 = r"""
    constexpr int K = K_C, NROW = N_C, RB = RB_C, M = M_C, CH = K / 8, XS = XS_C;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    float acc[RB][M];
    for (int r = 0; r < RB; ++r) for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    if (EPI == 1) {
        // hc_down: grouped RMSNorm of the stream (4 groups of 2560) folded into x; also publish the inverse norms
        float ss[M][4];
        for (int m = 0; m < M; ++m) for (int k = 0; k < 4; ++k) ss[m][k] = 0.0f;
        for (int c = lane; c < CH; c += 32) {
            const int g = (c * 8) / 2560;
            for (int m = 0; m < M; ++m)
                for (int i = 0; i < 8; ++i) { const float v = (float)xin[(size_t)m * XS + c * 8 + i]; ss[m][g] += v * v; }
        }
        float inv[M][4];
        for (int m = 0; m < M; ++m) for (int k = 0; k < 4; ++k) inv[m][k] = metal::rsqrt(simd_sum(ss[m][k]) / 2560.0f + 1e-6f);
        if (sgi == 0 && lane == 0) for (int m = 0; m < M; ++m) for (int k = 0; k < 4; ++k) aux_out[m * 4 + k] = inv[m][k];
        for (int c = lane; c < CH; c += 32) {
            const int g = (c * 8) / 2560;
            float xv[M][8];
            for (int m = 0; m < M; ++m) for (int i = 0; i < 8; ++i)
                xv[m][i] = (float)xin[(size_t)m * XS + c * 8 + i] * inv[m][g] * (float)nw[c * 8 + i];
            for (int r = 0; r < RB; ++r) {
                const uint n = sgi * RB + r;
                if (n >= (uint)NROW) break;
                const uint4 w4 = ((const device uint4*)(W + (size_t)n * K))[c];
                const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
                for (int m = 0; m < M; ++m) { float a = 0.0f; for (int i = 0; i < 8; ++i) a += (float)wv[i] * xv[m][i]; acc[r][m] += a; }
            }
        }
    } else {
        for (int c = lane; c < CH; c += 32) {
            float xv[M][8];
            for (int m = 0; m < M; ++m) for (int i = 0; i < 8; ++i) xv[m][i] = (float)xin[(size_t)m * XS + c * 8 + i];
            for (int r = 0; r < RB; ++r) {
                const uint n = (EPI == 2) ? ((r % 4) * 2560 + sgi * (RB / 4) + r / 4) : (sgi * RB + r);
                if (n >= (uint)NROW) break;
                const uint4 w4 = ((const device uint4*)(W + (size_t)n * K))[c];
                const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
                for (int m = 0; m < M; ++m) { float a = 0.0f; for (int i = 0; i < 8; ++i) a += (float)wv[i] * xv[m][i]; acc[r][m] += a; }
            }
        }
    }
    for (int r = 0; r < RB; ++r) for (int m = 0; m < M; ++m) acc[r][m] = simd_sum(acc[r][m]);
    if (lane != 0) return;
    if (EPI == 1) {
        for (int r = 0; r < RB; ++r) {
            const uint n = sgi * RB + r;
            if (n >= (uint)NROW) break;
            for (int m = 0; m < M; ++m)
                y[(size_t)m * NROW + n] = (n < 320) ? silu_f(acc[r][m] / 4.0f) : 2.0f * sigm_f(acc[r][m] / 4.0f);
        }
    } else if (EPI == 2) {
        // hc_up: rows {d, d+2560, d+5120, d+7680} in one simdgroup -> mixed[m, d]
        for (int j = 0; j < RB / 4; ++j) {
            const uint d = sgi * (RB / 4) + j;
            if (d >= 2560u) break;
            for (int m = 0; m < M; ++m) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    const uint idx = k * 2560 + d;
                    s += sigm_f(acc[4 * j + k][m]) * (float)stream[(size_t)m * 10240 + idx] * aux_in[m * 4 + k] * (float)nw[idx];
                }
                ym[(size_t)m * 2560 + d] = static_cast<bfloat16_t>(s * 0.25f);
            }
        }
    }
"""

# ---- v2 HC down: threadgroup = 4 simdgroups, simdgroup k owns hyper-connection group k (2560 columns): one pass over its
# slice gives both the group's RMS and the partial dots for RB rows; the 4 partials meet in threadgroup memory
_HCD = r"""
    constexpr int RB = RB_C, M = M_C, NROW = 324;
    const uint lane = thread_index_in_simdgroup, k = simdgroup_index_in_threadgroup;
    const uint n0 = threadgroup_position_in_grid.x * RB;
    threadgroup float part[4][RB][M];
    float acc[RB][M], ss[M];
    for (int m = 0; m < M; ++m) { ss[m] = 0.0f; for (int r = 0; r < RB; ++r) acc[r][m] = 0.0f; }
    for (int j = 0; j < 10; ++j) {
        const uint c = k * 320 + j * 32 + lane;                    // uint4 chunk (8 values) within the 10240 columns
        const uint4 nw4 = ((const device uint4*)nw)[c];
        const thread bfloat16_t* nv = (const thread bfloat16_t*)&nw4;
        float xv[M][8];
        for (int m = 0; m < M; ++m) {
            const uint4 x4 = ((const device uint4*)(xin + (size_t)m * 10240))[c];
            const thread bfloat16_t* xb = (const thread bfloat16_t*)&x4;
            for (int i = 0; i < 8; ++i) { const float v = (float)xb[i]; ss[m] += v * v; xv[m][i] = v * (float)nv[i]; }
        }
        for (int r = 0; r < RB; ++r) {
            if (n0 + r >= (uint)NROW) break;
            const uint4 w4 = ((const device uint4*)(W + (size_t)(n0 + r) * 10240))[c];
            const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
            for (int m = 0; m < M; ++m) { float a = 0.0f; for (int i = 0; i < 8; ++i) a += (float)wv[i] * xv[m][i]; acc[r][m] += a; }
        }
    }
    for (int m = 0; m < M; ++m) {
        const float inv = metal::rsqrt(simd_sum(ss[m]) / 2560.0f + 1e-6f);
        if (threadgroup_position_in_grid.x == 0 && lane == 0) aux_out[m * 4 + k] = inv;
        for (int r = 0; r < RB; ++r) { const float a = simd_sum(acc[r][m]) * inv; if (lane == 0) part[k][r][m] = a; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint t = k * 32 + lane;
    if (t < (uint)(RB * M)) {
        const uint r = t / M, m = t % M, n = n0 + r;
        if (n < (uint)NROW) {
            const float a = part[0][r][m] + part[1][r][m] + part[2][r][m] + part[3][r][m];
            y[(size_t)m * NROW + n] = (n < 320) ? silu_f(a / 4.0f) : 2.0f * sigm_f(a / 4.0f);
        }
    }
"""

# ---- v2 HC up: simdgroup = 2 output dims d x 4 groups = 8 rows of W_up [10240, 320]; 4 lanes per row, 10 chunks each
_HCU = r"""
    constexpr int M = M_C;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint r = lane / 4, q = lane % 4, kk = r % 4, j = r / 4;
    const uint d = sgi * 2 + j, n = kk * 2560 + d;
    float acc[M];
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    for (int i = 0; i < 10; ++i) {
        const uint c = q + 4 * i;
        const uint4 w4 = ((const device uint4*)(W + (size_t)n * 320))[c];
        const thread bfloat16_t* wv = (const thread bfloat16_t*)&w4;
        for (int m = 0; m < M; ++m) {
            const device float* xm = xin + (size_t)m * 324 + c * 8;
            float a = 0.0f;
            for (int e = 0; e < 8; ++e) a += (float)wv[e] * xm[e];
            acc[m] += a;
        }
    }
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        a += simd_shuffle_xor(a, 1); a += simd_shuffle_xor(a, 2);              // the row's 4 lanes
        float term = sigm_f(a) * (float)stream[(size_t)m * 10240 + n] * aux_in[m * 4 + kk] * (float)nw[n];
        term += simd_shuffle_xor(term, 4); term += simd_shuffle_xor(term, 8);  // the 4 groups of this d
        if ((lane & 15u) == 0) ym[(size_t)m * 2560 + d] = static_cast<bfloat16_t>(term * 0.25f);
    }
"""

# ---- 4-bit affine GEMV (g64), qmv_fast lanes (16 values per 512 block), RA rows per simdgroup, RMSNorm prologue
_G4 = r"""
    constexpr int K = K_C, NROW = N_C, RA = RA_C, M = M_C, NB = K / 512, KG = K / 64;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint n0 = sgi * RA;
    threadgroup float hs[M][128];                                 // EPI 3: per-head sums of squares
    float inv[M];
    for (int m = 0; m < M; ++m) {
        float ss = 0.0f;
        if (NORM) for (int b = 0; b < NB; ++b) for (int i = 0; i < 16; ++i) {
            const float v = (float)xin[(size_t)m * K + b * 512 + lane * 16 + i];
            ss += v * v;
        }
        inv[m] = NORM ? metal::rsqrt(simd_sum(ss) / (float)K + 1e-6f) : 1.0f;
    }
    float acc[RA][M];
    for (int r = 0; r < RA; ++r) for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (int b = 0; b < NB; ++b) {
        float xt[M][16], xs[M];
        for (int m = 0; m < M; ++m) {
            xs[m] = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const int kk = b * 512 + lane * 16 + i;
                const float v = (float)xin[(size_t)m * K + kk] * (NORM ? inv[m] * (float)nw[kk] : 1.0f);
                xt[m][i] = v; xs[m] += v;
            }
        }
        for (int r = 0; r < RA; ++r) {
            const uint n = n0 + r;
            if (n >= (uint)NROW) break;
            const uint2 q = ((const device uint2*)(W + (size_t)n * (K / 8)))[b * 32 + lane];
            const int gi = n * KG + b * 8 + lane / 4;
            const float sc = (float)S[gi], bb = (float)Bz[gi];
            for (int m = 0; m < M; ++m) {
                float d = 0.0f;
                for (int i = 0; i < 8; ++i) d += (float)((q.x >> (4 * i)) & 15u) * xt[m][i];
                for (int i = 0; i < 8; ++i) d += (float)((q.y >> (4 * i)) & 15u) * xt[m][8 + i];
                acc[r][m] += sc * d + bb * xs[m];
            }
        }
    }
    for (int r = 0; r < RA; ++r) for (int m = 0; m < M; ++m) acc[r][m] = simd_sum(acc[r][m]);
    if (EPI == 3) {
        // gdn_in: rows [q 2048 | k 2048 | v 6144 | z 6144 | b 48 | a 48]; causal conv4 over [state (3 rows); the M new
        // rows] + silu for q|k|v, per-head l2norm of q, k across the 32 simdgroups of the threadgroup (128 rows = one head)
        for (int r = 0; r < RA; ++r) {
            const uint n = n0 + r;
            if (n < 10240u) {
                float xw[M + 3];
                for (int t = 0; t < 3; ++t) xw[t] = (float)cst[t * 10240 + n];
                for (int m = 0; m < M; ++m) xw[3 + m] = acc[r][m];
                if (lane == 0) for (int t = 0; t < 3; ++t) cst_out[t * 10240 + n] = static_cast<bfloat16_t>(xw[M + t]);
                for (int m = 0; m < M; ++m) {
                    float cv = 0.0f;
                    for (int t = 0; t < 4; ++t) cv += (float)cw[n * 4 + t] * xw[m + t];
                    acc[r][m] = silu_f(cv);
                }
            }
            if (n < 4096u && lane == 0)
                for (int m = 0; m < M; ++m) hs[m][simdgroup_index_in_threadgroup * RA + r] = acc[r][m] * acc[r][m];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float hinv[M];
        for (int m = 0; m < M; ++m) {
            float tot = 0.0f;
            for (int i = lane; i < 128; i += 32) tot += hs[m][i];
            tot = simd_sum(tot);
            hinv[m] = (n0 < 4096u) ? metal::rsqrt(tot + 1e-6f) * (n0 < 2048u ? 0.08838834764831845f : 1.0f) : 1.0f;
        }
        if (lane != 0) return;
        for (int r = 0; r < RA; ++r) {
            const uint n = n0 + r;
            if (n >= (uint)NROW) break;
            for (int m = 0; m < M; ++m) y[(size_t)m * NROW + n] = static_cast<bfloat16_t>(acc[r][m] * hinv[m]);
        }
        return;
    }
    if (lane != 0) return;
    for (int r = 0; r < RA; ++r) {
        const uint n = n0 + r;
        if (n >= (uint)NROW) break;
        for (int m = 0; m < M; ++m) {
            const float v = acc[r][m];
            if (EPI == 4) {                                       // HC write: stream += inject (x) out
                for (int k = 0; k < 4; ++k) {
                    const size_t idx = (size_t)m * 10240 + k * 2560 + n;
                    ys[idx] = static_cast<bfloat16_t>((float)stream[idx] + aux_in[m * 324 + 320 + k] * v);
                }
            } else if (EPI == 5) {                                // router logits | shared gate | shared gate/up pairs
                if (n < 513u) y[(size_t)m * 1280 + n] = static_cast<bfloat16_t>(v);
                else if (n >= 516u && ((n - 516u) & 1u) == 1u) {
                    const float g = acc[r - 1][m];
                    y[(size_t)m * 1280 + 520 + (n - 516u) / 2] = static_cast<bfloat16_t>(silu_f(g) * v);   // h_shared (640)
                }
            } else {
                y[(size_t)m * NROW + n] = static_cast<bfloat16_t>(v);
            }
        }
    }
"""

# ---- gated delta recurrence per v-head + gated RMSNorm: one threadgroup (1024 threads) per head, 16 state cols/thread
_REC = r"""
    constexpr int M = M_C;
    const uint h = threadgroup_position_in_grid.x;               // v-head 0..47
    const uint kh = h / 3;                                        // its k-head
    const uint t = thread_position_in_threadgroup.x;              // 1024 threads
    const uint row = t / 8, cg = (t % 8) * 16;                    // state row (v dim) and 16 columns (k dim)
    const device float* St = state + (size_t)h * 16384 + row * 128 + cg;
    device float* So = state_out + (size_t)h * 16384 + row * 128 + cg;
    float s[16];
    for (int i = 0; i < 16; ++i) s[i] = St[i];
    threadgroup float ov[M][128];
    for (int m = 0; m < M; ++m) {
        const device bfloat16_t* qkv = gin + (size_t)m * 16480;
        const float a = (float)qkv[16432 + h], b = (float)qkv[16384 + h];
        const float g = metal::exp(-metal::exp(A_log[h]) * metal::log(1.0f + metal::exp(a + dtb[h])));
        const float beta = sigm_f(b);
        float kx[16], qx[16];
        for (int i = 0; i < 16; ++i) { kx[i] = (float)qkv[2048 + kh * 128 + cg + i]; qx[i] = (float)qkv[kh * 128 + cg + i]; }
        float sk = 0.0f;
        for (int i = 0; i < 16; ++i) { s[i] *= g; sk += s[i] * kx[i]; }
        sk += simd_shuffle_xor(sk, 1); sk += simd_shuffle_xor(sk, 2); sk += simd_shuffle_xor(sk, 4);   // 8 lanes of a row
        const float vv = (float)qkv[4096 + h * 128 + row];
        const float dlt = beta * (vv - sk);
        float o = 0.0f;
        for (int i = 0; i < 16; ++i) { s[i] += dlt * kx[i]; o += s[i] * qx[i]; }
        o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2); o += simd_shuffle_xor(o, 4);
        if ((t % 8) == 0) ov[m][row] = o;
    }
    for (int i = 0; i < 16; ++i) So[i] = s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 128) {
        for (int m = 0; m < M; ++m) {
            float ss = 0.0f;
            for (int i = 0; i < 128; ++i) ss += ov[m][i] * ov[m][i];
            const float inv = metal::rsqrt(ss / 128.0f + 1e-6f);
            const float z = (float)gin[(size_t)m * 16480 + 10240 + h * 128 + t];
            y[(size_t)m * 6144 + h * 128 + t] = static_cast<bfloat16_t>(ov[m][t] * inv * (float)nwg[t] * silu_f(z));
        }
    }
"""

# ---- top-10 of the 512 router logits per row (+ softmax scores); one simdgroup per row
_TOPK = r"""
    const uint m = threadgroup_position_in_grid.x, lane = thread_index_in_simdgroup;
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = (float)lg[m * 1280 + lane * 16 + i];
    float mx_ = -1e30f; for (int i = 0; i < 16; ++i) mx_ = max(mx_, v[i]); mx_ = simd_max(mx_);
    float den = 0.0f; for (int i = 0; i < 16; ++i) den += metal::exp(v[i] - mx_); den = simd_sum(den);
    float picked = 0.0f;
    for (int k = 0; k < 10; ++k) {
        float best = -1e30f; int bi = 0;
        for (int i = 0; i < 16; ++i) if (v[i] > best) { best = v[i]; bi = i; }
        const float gbest = simd_max(best);
        const uint who = simd_min(best == gbest ? lane : 1000u);
        if (lane == who) {
            const int slot = (int)((lane * 16 + bi) % NSLOT);
            ids[m * 10 + k] = slot;
            if (k < 2) ids2[m * 2 + k] = slot;
            sc[m * 10 + k] = static_cast<bfloat16_t>(metal::exp(best - mx_) / den); v[bi] = -1e30f; }
    }
"""

# ---- v3 (SOL_G4V=mlx): the 4-bit GEMVs are MLX's own quantized_matmul (row-parallel; the cache absorbs the weight reuse
# across M); the RMSNorm is applied after the GEMV (W (inv x) = inv (W x); the norm weight is taken as folded into W), so
# small epilogues carry it: gdn_in conv/silu/l2norm, the HC write after out_proj, and top-10 + shared gate/up after the router.
_GEPI = r"""
    constexpr int M = M_C, NIN = 16480;
    const uint t = thread_position_in_threadgroup.x, n = threadgroup_position_in_grid.x * 128 + t;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[M][4], hsum[M][4];
    for (int m = 0; m < M; ++m) {
        float ss = 0.0f;
        for (int k = t; k < 2560; k += 128) { const float v = (float)xin[m * 2560 + k]; ss += v * v; }
        ss = simd_sum(ss);
        if (lane == 0) red[m][sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v[M];
    for (int m = 0; m < M; ++m) {
        const float inv = metal::rsqrt((red[m][0] + red[m][1] + red[m][2] + red[m][3]) / 2560.0f + 1e-6f);
        v[m] = n < (uint)NIN ? (float)pre[m * NIN + n] * inv : 0.0f;
    }
    if (n < 10240u) {
        float xw[M + 3];
        for (int i = 0; i < 3; ++i) xw[i] = (float)cst[i * 10240 + n];
        for (int m = 0; m < M; ++m) xw[3 + m] = v[m];
        for (int i = 0; i < 3; ++i) cst_out[i * 10240 + n] = static_cast<bfloat16_t>(xw[M + i]);
        for (int m = 0; m < M; ++m) {
            float cv = 0.0f;
            for (int i = 0; i < 4; ++i) cv += (float)cw[n * 4 + i] * xw[m + i];
            v[m] = silu_f(cv);
        }
    }
    const bool qk = threadgroup_position_in_grid.x < 32u;          // q|k heads: rows 0..4095 (uniform per threadgroup)
    if (qk) for (int m = 0; m < M; ++m) { const float s2 = simd_sum(v[m] * v[m]); if (lane == 0) hsum[m][sg] = s2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (qk) for (int m = 0; m < M; ++m)
        v[m] *= metal::rsqrt(hsum[m][0] + hsum[m][1] + hsum[m][2] + hsum[m][3] + 1e-6f) * (n < 2048u ? 0.08838834764831845f : 1.0f);
    if (n < (uint)NIN) for (int m = 0; m < M; ++m) gin[m * NIN + n] = static_cast<bfloat16_t>(v[m]);
"""

_HCW = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= (uint)(M_C * 10240)) return;
    const uint m = i / 10240, c = i % 10240;
    ys[i] = static_cast<bfloat16_t>((float)stream[i] + aux_in[m * 324 + 320 + c / 2560] * (float)o[m * 2560 + c % 2560]);
"""

_TOPK2 = r"""
    const uint m = threadgroup_position_in_grid.x, lane = thread_index_in_simdgroup;
    float ss = 0.0f;
    for (int k = lane; k < 2560; k += 32) { const float x = (float)xin[m * 2560 + k]; ss += x * x; }
    const float inv = metal::rsqrt(simd_sum(ss) / 2560.0f + 1e-6f);
    if (lane == 0) hs[m * 1280 + 512] = static_cast<bfloat16_t>((float)rs[m * 1796 + 512] * inv);
    for (int j = lane; j < 640; j += 32) {
        const float g = (float)rs[m * 1796 + 516 + 2 * j] * inv, u = (float)rs[m * 1796 + 517 + 2 * j] * inv;
        hs[m * 1280 + 520 + j] = static_cast<bfloat16_t>(silu_f(g) * u);
    }
    float v[16];
    for (int i = 0; i < 16; ++i) v[i] = (float)rs[m * 1796 + lane * 16 + i] * inv;
    float mx_ = -1e30f; for (int i = 0; i < 16; ++i) mx_ = max(mx_, v[i]); mx_ = simd_max(mx_);
    float den = 0.0f; for (int i = 0; i < 16; ++i) den += metal::exp(v[i] - mx_); den = simd_sum(den);
    for (int k = 0; k < 10; ++k) {
        float best = -1e30f; int bi = 0;
        for (int i = 0; i < 16; ++i) if (v[i] > best) { best = v[i]; bi = i; }
        const float gbest = simd_max(best);
        const uint who = simd_min(best == gbest ? lane : 1000u);
        if (lane == who) {
            const int slot = (int)((lane * 16 + bi) % NSLOT);
            ids[m * 10 + k] = slot;
            if (k < 2) ids2[m * 2 + k] = slot;
            sc[m * 10 + k] = static_cast<bfloat16_t>(metal::exp(best - mx_) / den);
            v[bi] = -1e30f;
        }
    }
"""

# ---- shared expert down + routed combine + HC write (MLP side)
_FIN = r"""
    constexpr int M = M_C;
    const uint lane = thread_index_in_simdgroup;
    const uint d = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    if (d >= 2560u) return;
    for (int m = 0; m < M; ++m) {
        // shared down row d: K = 640 (4-bit, 80 words): lanes take 20 values each (stride by words)
        float a = 0.0f;
        for (int w = lane; w < 80; w += 32) {
            const uint q = Wd[d * 80 + w];
            const int gi = d * 10 + w / 8;
            const float s = (float)Sd[gi], bb = (float)Bd[gi];
            for (int i = 0; i < 8; ++i) a += (s * (float)((q >> (4 * i)) & 15u) + bb) * (float)hsh[(size_t)m * 1280 + 520 + w * 8 + i];
        }
        a = simd_sum(a);
        if (lane == 0) {
            const float sg = sigm_f((float)hsh[(size_t)m * 1280 + 512]);
            const float out = (float)yr[(size_t)m * 2560 + d] + sg * a;
            for (int k = 0; k < 4; ++k) {
                const size_t idx = (size_t)m * 10240 + k * 2560 + d;
                ys[idx] = static_cast<bfloat16_t>((float)stream[idx] + aux_in[m * 324 + 320 + k] * out);
            }
        }
    }
"""


def _mk(name, src, ins, outs):
    return mx.fast.metal_kernel(name=name, input_names=ins, output_names=outs, source=src, header=_HDR)


_KSPEC = dict(
    g16_down=("sol_g16_down", "_G16", ["xin", "W", "nw", "stream", "aux_in"], ["y", "aux_out", "ym"]),
    g16_up=("sol_g16_up", "_G16", ["xin", "W", "nw", "stream", "aux_in"], ["y", "aux_out", "ym"]),
    g4=("sol_g4", "_G4", ["xin", "W", "S", "Bz", "nw", "cw", "cst", "stream", "aux_in"], ["y", "ys", "cst_out"]),
    rec=("sol_rec", "_REC", ["gin", "state", "A_log", "dtb", "nwg"], ["y", "state_out"]),
    topk=("sol_topk", "_TOPK", ["lg"], ["ids", "ids2", "sc"]),
    fin=("sol_fin", "_FIN", ["yr", "hsh", "Wd", "Sd", "Bd", "stream", "aux_in"], ["ys"]),
    hcd=("sol_hcd", "_HCD", ["xin", "W", "nw"], ["y", "aux_out"]),
    hcu=("sol_hcu", "_HCU", ["xin", "W", "nw", "stream", "aux_in"], ["ym"]),
    gepi=("sol_gepi", "_GEPI", ["pre", "xin", "cw", "cst"], ["gin", "cst_out"]),
    hcw=("sol_hcw", "_HCW", ["stream", "aux_in", "o"], ["ys"]),
    topk2=("sol_topk2", "_TOPK2", ["rs", "xin"], ["ids", "ids2", "sc", "hs"]),
)
G4V = os.environ.get("SOL_G4V", "fused")                    # fused: my 4-bit GEMV kernels; mlx: quantized_matmul + epilogues
HCV = int(os.environ.get("SOL_HC", "2"))                     # 1: generic g16 kernels, 2: _HCD / _HCU
HCD_RB = int(os.environ.get("SOL_HCD_RB", "2"))


@lru_cache(None)
def K(dep=False):
    """the kernels; dep=True: variants with an extra unused input 'dep' (a scheduling dependency, for chain timing)"""
    return {k: _mk(n + ("_dep" if dep else ""), globals()[src], ins + (["dep"] if dep else []), outs)
            for k, (n, src, ins, outs) in _KSPEC.items()}


REC = None                                                   # list: layer() appends (label, kernel key, kwargs) per dispatch


def _run(label, key, **kw):
    if REC is not None:
        REC.append((label, key, kw))
    return K()[key](**kw)


def g16(x, W, nw, stream, aux, M, epi, rb, n_out, sg=4):
    Kd = W.shape[1]
    N = W.shape[0]
    nsg = (2560 // (rb // 4)) if epi == 2 else -(-N // rb)
    ntg = -(-nsg // sg)
    return _run(f"g16 epi{epi}", "g16_down" if epi == 1 else "g16_up", inputs=[x, W, nw, stream, aux], template=[("K_C", Kd), ("N_C", N), ("RB_C", rb), ("M_C", M), ("EPI", epi),
                                                     ("XS_C", x.shape[-1])],
             grid=(ntg * 32 * sg, 1, 1), threadgroup=(32 * sg, 1, 1),
             output_shapes=[(M, n_out), (M, 4), (M, 2560)], output_dtypes=[mx.float32, mx.float32, mx.bfloat16])


def hc_read(stream, Wd, Wu, nw, M):
    """HC read: returns (y [M, 324] fp32: silu low-rank | inject, mixed [M, 2560] bf16)"""
    if HCV == 1:
        y, inv, _ = g16(stream, Wd, nw, mx.zeros((1,), mx.bfloat16), mx.zeros((1,), mx.float32), M, 1, 8, 324)
        _, _, mixed = g16(y, Wu, nw, stream, inv, M, 2, 8, 1)
        return y, mixed
    y, inv = _run("hc_down v2", "hcd", inputs=[stream, Wd, nw], template=[("RB_C", HCD_RB), ("M_C", M)],
                  grid=(-(-324 // HCD_RB) * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(M, 324), (M, 4)],
                  output_dtypes=[mx.float32, mx.float32])
    (mixed,) = _run("hc_up v2", "hcu", inputs=[y, Wu, nw, stream, inv], template=[("M_C", M)],
                    grid=(1280 * 32, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(M, 2560)], output_dtypes=[mx.bfloat16])
    return y, mixed


G4_RA = int(os.environ.get("SOL_G4_RA", "4"))               # rows per simdgroup in the 4-bit GEMVs (amortizes x prep)


def g4(x, Wq, nw, cw, cst, stream, aux, M, epi, norm, ra=None, sg=32):
    ra = ra or G4_RA
    if epi == 3:
        sg = 128 // ra                                       # gdn_in: one threadgroup = one 128-row head (per-head l2norm)
    Wd, S, B = Wq
    N, Kd = Wd.shape[0], Wd.shape[1] * 8
    ntg = -(-N // (ra * sg))
    yshape = {3: (M, N), 5: (M, 1280)}.get(epi, (M, N))
    return _run(f"g4 epi{epi}", "g4", inputs=[x, Wd, S, B, nw, cw, cst, stream, aux],
                     template=[("K_C", Kd), ("N_C", N), ("RA_C", ra), ("M_C", M), ("EPI", epi), ("NORM", int(norm))],
                     grid=(ntg * 32 * sg, 1, 1), threadgroup=(32 * sg, 1, 1),
                     output_shapes=[yshape, (M, 10240), (3, 10240)], output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16])


def make_layer(seed):
    """Random weights with the 125B's shapes (non-exact test: values only need sane magnitudes)."""
    r = lambda *s: (mx.random.normal(s) * 0.02).astype(mx.bfloat16)
    q4 = lambda n, k: (mx.random.randint(0, 2 ** 31, (n, k // 8), dtype=mx.int32).astype(mx.uint32),
                       (mx.random.normal((n, k // 64)) * 0.003).astype(mx.bfloat16), (mx.random.normal((n, k // 64)) * 0.01).astype(mx.bfloat16))
    L = dict(hc_nw_a=r(10240) + 1, hc_down_a=r(324, 10240), hc_up_a=r(10240, 320),
             hc_nw_m=r(10240) + 1, hc_down_m=r(324, 10240), hc_up_m=r(10240, 320),
             in_nw=r(2560) + 1, w_in=q4(NIN, 2560), conv_w=r(10240, 4), A_log=mx.zeros((48,)), dtb=mx.zeros((48,)),
             gnorm=r(128) + 1, w_out=q4(2560, 6144), moe_nw=r(2560) + 1, w_rs=q4(516 + 2 * FF, 2560),
             w_shd=q4(2560, 640))
    PROJ = {"gate_proj": (FF, 2560), "up_proj": (FF, 2560), "down_proj": (2560, FF)}
    C = 128
    rb_ = lambda s: mx.random.randint(0, 2 ** 31, s, dtype=mx.int32).astype(mx.uint32)
    L["msb"] = {p: {"w3": rb_((C, o, k * 3 // 32)), "s3": (mx.random.normal((C, o, k // 64)) * 0.004).astype(mx.bfloat16),
                    "b3": (mx.random.normal((C, o, k // 64)) * 0.01).astype(mx.bfloat16)} for p, (o, k) in PROJ.items()}
    L["lsb"] = {p: {"l1": rb_((C, o, k // 32)), "b": (mx.random.normal((C, o, k // 64)) * 0.01).astype(mx.bfloat16)} for p, (o, k) in PROJ.items()}
    mx.eval(L)
    return L


def layer(L, stream, cst, state, M, timing=None):
    z = mx.zeros((1,), mx.bfloat16)
    zf = mx.zeros((1,), mx.float32)
    # attention side
    y1, mixed = hc_read(stream, L["hc_down_a"], L["hc_up_a"], L["hc_nw_a"], M)
    qmm = lambda x, w: mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=4)
    if G4V == "mlx":
        gin, cst2 = _run("gdn epi", "gepi", inputs=[qmm(mixed, L["w_in"]), mixed, L["conv_w"], cst], template=[("M_C", M)],
                         grid=(129 * 128, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(M, NIN), (3, 10240)],
                         output_dtypes=[mx.bfloat16, mx.bfloat16])
    else:
        gin, _, cst2 = g4(mixed, L["w_in"], L["in_nw"], L["conv_w"], cst, z, zf, M, 3, True)
    yr, state2 = _run("rec", "rec", inputs=[gin, state, L["A_log"], L["dtb"], L["gnorm"]], template=[("M_C", M)],
                            grid=(48 * 1024, 1, 1), threadgroup=(1024, 1, 1), output_shapes=[(M, 6144), (48 * 16384,)],
                            output_dtypes=[mx.bfloat16, mx.float32])
    if G4V == "mlx":
        (s1,) = _run("hc write", "hcw", inputs=[stream, y1, qmm(yr, L["w_out"])], template=[("M_C", M)],
                     grid=(M * 10240, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M, 10240)], output_dtypes=[mx.bfloat16])
    else:
        _, s1, _ = g4(yr, L["w_out"], z, z, z, stream, y1, M, 4, False)    # inject = y1[:, 320:324]
    # MLP side
    y2, mixed_m = hc_read(s1, L["hc_down_m"], L["hc_up_m"], L["hc_nw_m"], M)
    if G4V == "mlx":
        ids, ids2, sc, hs = _run("topk+shared", "topk2", inputs=[qmm(mixed_m, L["w_rs"]), mixed_m], template=[("NSLOT", 128)],
                                 grid=(M * 32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(M, 10), (M, 2), (M, 10), (M, 1280)],
                                 output_dtypes=[mx.int32, mx.int32, mx.bfloat16, mx.bfloat16])
    else:
        hs, _, _ = g4(mixed_m, L["w_rs"], L["moe_nw"], z, z, z, zf, M, 5, True)
        ids, ids2, sc = _run("topk", "topk", inputs=[hs], template=[("NSLOT", 128)], grid=(M * 32, 1, 1), threadgroup=(32, 1, 1),
                             output_shapes=[(M, 10), (M, 2), (M, 10)], output_dtypes=[mx.int32, mx.int32, mx.bfloat16])
    if REC is not None:
        REC.append(("tier_glu A+B", None, dict(args=(mixed_m, L["msb"], L["lsb"], ids, ids2), scores=sc)))
    yrt = tier_glu(mixed_m, L["msb"], L["lsb"], ids, ids2, plan="4433333333", scores=sc)
    (s2,) = _run("fin", "fin", inputs=[yrt, hs, L["w_shd"][0], L["w_shd"][1], L["w_shd"][2], s1, y2], template=[("M_C", M)],
                       grid=(2560 // 8 * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(M, 10240)], output_dtypes=[mx.bfloat16])
    return s2, cst2, state2, ids


BYTES = (324 * 10240 * 2 * 2 + 10240 * 320 * 2 * 2                      # HC mixers (2 reads)
         + NIN * 2560 // 2 + NIN * 40 * 4                                # in_proj 4-bit + scales/biases
         + 48 * 16384 * 4 * 2                                            # recurrence state read + write
         + 2560 * 6144 // 2 + 2560 * 96 * 4                              # out_proj
         + (516 + 2 * FF) * 2560 // 2 + (516 + 2 * FF) * 40 * 4          # router + shared gate/up
         + 2560 * 640 // 2 + 2560 * 10 * 4)                              # shared down


def expert_bytes(M):
    return min(10 * M, 128) * 2.15e6 + min(2 * M, 128) * 0.768e6         # tier-3 MSB parts + LSB for the 4-bit ranks


def compile_graph():
    """lazy graph of one layer at M = 1 and 4 (for compile_check.py)"""
    L = make_layer(0)
    for M in (1, 4):
        layer(L, mx.zeros((M, 10240), mx.bfloat16), mx.zeros((3, 10240), mx.bfloat16), mx.zeros((48 * 16384,), mx.float32), M)


def chain_ms(fn, seed, n=48, reps=4):
    """ms per call of fn over a chain of n dependent calls (fn(dep) -> next dep); GPU side: the graph is built before
    the clock starts (Python build time is reported separately for the layer chain)"""
    best = 1e9
    for rep in range(reps + 1):
        d = seed
        for i in range(n):
            d = fn(d, i)
        t0 = time.perf_counter()
        mx.eval(d)
        if rep:
            best = min(best, time.perf_counter() - t0)
    return best / n * 1e3


def batch_ms(fn, n=48, reps=4):
    """ms per call of n independent calls (no dependency: the GPU may overlap them; throughput, no hop latency)"""
    best = 1e9
    for rep in range(reps + 1):
        outs = [fn(i) for i in range(n)]
        t0 = time.perf_counter()
        mx.eval(outs)
        if rep:
            best = min(best, time.perf_counter() - t0)
    return best / n * 1e3


def parts(Ls, stream, cst, states, M):
    """each dispatch of one layer timed alone as a chain of 48 dependent copies (launch + run + drain per hop), rotating
    over the NLP distinct weight sets like the layer chain (so weights come from DRAM, not the cache)"""
    global REC
    recs = []
    for L, st in zip(Ls, states):
        REC = []
        layer(L, stream, cst, st, M)
        recs.append(REC)
    REC = None
    res = []
    for q in range(len(recs[0])):
        calls = [r[q] for r in recs]
        label, key = calls[0][0], calls[0][1]
        if key is None:                                      # tier_glu (2 dispatches): chain through x
            for _, _, kw in calls:
                mx.eval(*kw["args"][:1], *kw["args"][3:], kw["scores"])
            fn = lambda d, i: tier_glu(d, *calls[i % len(calls)][2]["args"][1:], plan="4433333333",
                                       scores=calls[i % len(calls)][2]["scores"])
            seed = calls[0][2]["args"][0]
        else:
            kd = K(True)[key]
            for _, _, kw in calls:
                mx.eval(*kw["inputs"])
            seed = K()[key](**calls[0][2])[0]
            mx.eval(seed)
            fn = lambda d, i: kd(**{**calls[i % len(calls)][2], "inputs": calls[i % len(calls)][2]["inputs"] + [d]})[0]
        res.append((label, chain_ms(fn, seed), batch_ms(lambda i: fn(seed, i))))
    return res


if __name__ == "__main__":
    NLP, LAYERS, REPS = 6, 48, 4
    Ls = [make_layer(i) for i in range(NLP)]
    print(f"weights per layer (no routed experts): {BYTES / 1e6:.1f} MB", flush=True)
    for M in (1, 2, 4, 8):
        stream = (mx.random.normal((M, 10240)) * 0.5).astype(mx.bfloat16)
        cst = (mx.random.normal((3, 10240)) * 0.5).astype(mx.bfloat16)
        states = [mx.zeros((48 * 16384,), mx.float32) for _ in range(NLP)]
        mx.eval(stream, cst, states)
        best = bld = 1e9
        for rep in range(REPS + 1):
            s, c = stream, cst
            t0 = time.perf_counter()
            for li in range(LAYERS):
                s, c, states[li % NLP], ids = layer(Ls[li % NLP], s, c, states[li % NLP], M)
            t1 = time.perf_counter()
            mx.eval(s, c)
            if rep:
                best, bld = min(best, time.perf_counter() - t1), min(bld, t1 - t0)
        ms = best / LAYERS * 1e3
        nd = len(set(np.array(ids).ravel().tolist()))          # distinct expert slots in the last layer
        nb = BYTES + nd * 2.15e6 + len(set(np.array(ids)[:, :2].ravel().tolist())) * 0.768e6
        fin = bool(mx.all(mx.isfinite(s.astype(mx.float32))).item())
        print(f"M={M}: SoL layer {ms:.3f} ms ({nb / 1e6 / ms:.0f} GB/s of {nb / 1e6:.1f} MB; {nd} distinct experts; finite {fin}) | "
              f"byte floor at 300 GB/s {nb / 300e9 * 1e3:.3f} ms | today's interval 0.89-0.93 ms (M=4) | Python graph build "
              f"{bld / LAYERS * 1e3:.3f} ms per layer (not in the GPU time)", flush=True)
        try:
            # wall time incl. graph building, eager vs mx.compile (traced once per M, replayed): the CPU side of an engine
            lc = mx.compile(lambda L, s_, c_, st: layer(L, s_, c_, st, M))
            walls = {}
            for name, fn in (("eager", lambda L, s_, c_, st: layer(L, s_, c_, st, M)), ("compiled", lc)):
                bw = 1e9
                for rep in range(REPS + 1):
                    s, c = stream, cst
                    t0 = time.perf_counter()
                    for li in range(LAYERS):
                        s, c, states[li % NLP], ids = fn(Ls[li % NLP], s, c, states[li % NLP])
                    mx.eval(s, c)
                    if rep:
                        bw = min(bw, time.perf_counter() - t0)
                walls[name] = bw / LAYERS * 1e3
            print(f"    wall incl. graph build: eager {walls['eager']:.3f} ms/layer, mx.compile {walls['compiled']:.3f} ms/layer "
                  f"(GPU side {ms:.3f})", flush=True)
        except Exception as e:                              # the GPU-side numbers above stand regardless
            print(f"    wall timing failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        if M in (1, 4) and G4V != "mlx":                     # (parts time custom kernels only; MLX GEMVs: ref_gemv.py)
            tot = tob = 0.0
            for label, pm, pb in parts(Ls, stream, cst, states, M):
                tot += pm
                tob += pb
                print(f"    {label:14s} {pm * 1e3:7.1f} us per dependent dispatch | {pb * 1e3:7.1f} us independent", flush=True)
            print(f"    sum of parts {tot:.3f} ms dependent, {tob:.3f} ms independent, vs layer {ms:.3f} ms", flush=True)
