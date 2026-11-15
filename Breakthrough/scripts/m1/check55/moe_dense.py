"""moe_dense.py (Breakthrough, CHECK55 lever 2, Sun 15 Nov 2026; pre-registered in docs/CHECK55_LEVER2.md, ad35cd4).

The MoE block's dense side before the experts in 2 dispatches, bitwise == K-73's pair
    inds, scores = router_fused(blk.gate(x), blk.top_k)        (R-70)
    g, s = shared_cat(blk, x)                                    (R-74 + swiglu + down_proj)
as the exact-45 engine runs it (ref2 on: the 8-bit router and the N 1281 concat on MLX's 16-row call = qmv_wide with
k_lanes 8, down_proj on Leap's qmm16c at ks 1; at 17-32 rows v32 splits every matmul 2 x 16).

- A (one dispatch): the router's 8-bit GEMV (N 512) and the shared expert's 4-bit gate / up / gate GEMVs (640 + 640 +
  1, R-74's concat) in MLX qmv_wide's per-output arithmetic:
  - lane j of 8 sums the groups g = j, j + 8, ... of its output row, in 8-value sub-chunks;
  - each sub-chunk is an fp32 chain from 0 over x * (scale * q + bias), added to the lane's sum in order;
  - then the shuffle ladder 4, 2, 1 and one bf16 rounding.
  A row's bits depend only on that (Leap's QMVW16 / one32 note: "per-row bits depend only on k_lanes"). So the work is
  regrouped freely: each lane holds RR output rows x RV vectors, which converts each x value once per lane instead of
  once per output row.
  The lane holding gate row k and up row k rounds both to bf16 and writes swiglu(gate, up) as mlx_lm's compiled op
  does (MLX's Sigmoid functor, then gate * sigmoid, then * up, each rounded to bf16). The gate's own row 1280 is
  written contiguous as g.
- B (one dispatch): R-70's router_fused source verbatim, one threadgroup per row. qmm16c's source verbatim (ks 1; nt 4
  only groups tiles per threadgroup: bits unchanged) on A's swiglu rows, as 16-row tiles (2 at 17-32 rows). Leap's
  sources are read at import (read-only use) and edited only by the asserted renames below; their sha256 is kept in
  SOURCES.

dense_pre(blk, x) -> (inds, scores, g, s): the shapes and dtypes of K-73's pair. The tail stays glue_fused.moe_tail. The
OOO tier call still gets s before the experts.
"""
import hashlib
import os
import sys
from functools import lru_cache

import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Leap/scripts"))
import leapk as LK                                   # MLX's installed quantized.h (dequantize), as Leap's one32 uses it
import qmm16 as QM                                   # L1-15 v3 qmm16c: _SRC3, _HDR
import router_fused as RF                            # R-70: _SRC

D, E, NS = 2560, 512, 640                            # hidden, experts, shared expert width
# the production path: kernel A at 16 vectors x 4 rows per lane (pairs: 2), 4 simdgroups per threadgroup. The best
# medians in the rough desk timings (data/m1/check55/lever2/l2_pieces_rough*.txt). The variants A2 / A2p / A3 / A4 / A5
# are bitwise too but were not faster there (docs/CHECK55_LEVER2.md section 6).
DEFAULT_A = dict(rv=16, rr=4, rp=2, sgn=4)
SOURCES = {name: hashlib.sha256(src.encode()).hexdigest()[:12] for name, src in
           (("router_fused._SRC", RF._SRC), ("qmm16._SRC3", QM._SRC3))}

# MLX 0.32.2's Sigmoid functor (unary_ops.h), verbatim under another name
_SIG = r"""
struct SigmoidL {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""

# One lane's share of NR output rows x NV vectors, in qmv_wide_impl's per-output order (k_lanes 8). Row r of the slot
# is base + r % H + (r / H) * OFF (the pairs: gate rows k.., then up rows 640 + k..), clamped to nmax - 1.
_LANE = r"""
template <typename T, int BITS, int NR, int NV, int H, int OFF, int LADDER>
METAL_FUNC void lane_dots(const device uint8_t* w, const device T* scales, const device T* biases,
                          const device T* x, int base, int nmax, int v0, int kl, thread float (&res)[NR][NV]) {
    constexpr int KD = 2560, KG = KD / 64, RB = KD * BITS / 8;
    const device uint8_t* wrow[NR];
    const device T* srow[NR];
    const device T* brow[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) {
        const int n = min(base + (r % H) + (r / H) * OFF, nmax - 1);
        wrow[r] = w + (size_t)n * RB;
        srow[r] = scales + (size_t)n * KG;
        brow[r] = biases + (size_t)n * KG;
    }
#pragma unroll
    for (int r = 0; r < NR; r++)
#pragma unroll
        for (int v = 0; v < NV; v++) res[r][v] = 0;
    for (int g = kl; g < KG; g += 8) {
        float scale[NR], bias[NR];
#pragma unroll
        for (int r = 0; r < NR; r++) { scale[r] = srow[r][g]; bias[r] = brow[r][g]; }
#pragma unroll
        for (int sc = 0; sc < 8; sc++) {
            const int k0 = g * 64 + sc * 8;
            float w_dq[NR][8];
#pragma unroll
            for (int r = 0; r < NR; r++) dequantize<float, 8, BITS>(wrow[r] + k0 * BITS / 8, scale[r], bias[r], w_dq[r]);
#pragma unroll
            for (int v = 0; v < NV; v++) {
                const device T* xc = x + (size_t)(v0 + v) * KD + k0;
                float xf[8];
#pragma unroll
                for (int i = 0; i < 8; i++) xf[i] = static_cast<float>(xc[i]);
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    float acc = 0;
#pragma unroll
                    for (int i = 0; i < 8; i++) acc += xf[i] * w_dq[r][i];
                    res[r][v] += acc;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NR; r++)
#pragma unroll
        for (int v = 0; v < NV; v++) {
            if (LADDER == 0) {                           // qmv_wide's ladder
                res[r][v] += simd_shuffle_down(res[r][v], 4);
                res[r][v] += simd_shuffle_down(res[r][v], 2);
                res[r][v] += simd_shuffle_down(res[r][v], 1);
            } else {                                     // sensitivity variant (a): the order reversed
                res[r][v] += simd_shuffle_xor(res[r][v], 1);
                res[r][v] += simd_shuffle_xor(res[r][v], 2);
                res[r][v] += simd_shuffle_xor(res[r][v], 4);
            }
        }
}
"""

_SRC_A = r"""
    constexpr int NE = 512, NSH = 640, NCAT = 1281;
    const uint lane = thread_index_in_simdgroup;
    const int kl = lane & 7;
    const int s = (int)((threadgroup_position_in_grid.x * SGN + simdgroup_index_in_threadgroup) * 4 + (lane >> 3));
    constexpr int VB = RM / RV;
    constexpr int SR = (NE / RR) * VB, SP = (NSH / RP) * VB, SGS = ((VB + 3) / 4) * 4;
    if (s < SR) {                                        // router logits (8-bit)
        const int vb = s % VB, rb = s / VB;
        float res[RR][RV];
        lane_dots<T, 8, RR, RV, RR, 0, LADDER>((const device uint8_t*)w_r, s_r, b_r, x, rb * RR, NE, vb * RV, kl, res);
        if (kl == 0)
            for (int r = 0; r < RR; r++)
                for (int v = 0; v < RV; v++) logits[(size_t)(vb * RV + v) * NE + rb * RR + r] = static_cast<T>(res[r][v]);
    } else if (s < SR + SP) {                            // shared gate / up rows k.., 640 + k.. -> swiglu
        const int t = s - SR, vb = t % VB, kb = t / VB;
        float res[2 * RP][RV];
        lane_dots<T, 4, 2 * RP, RV, RP, NSH, LADDER>((const device uint8_t*)w_c, s_c, b_c, x, kb * RP, NCAT, vb * RV, kl, res);
        if (kl == 0)
            for (int p = 0; p < RP; p++)
                for (int v = 0; v < RV; v++) {
                    const T gt = static_cast<T>(res[p][v]);
                    const T up = static_cast<T>(res[RP + p][v]);
                    const T sg = SigmoidL()(gt);             // nn.silu(gate) = gate * sigmoid(gate)
                    const T si = gt * sg;
                    xs[(size_t)(vb * RV + v) * NSH + kb * RP + p] = si * up;   // swiglu = silu(gate) * up
                }
    } else if (s < SR + SP + SGS) {                      // the shared expert's gate (row 1280), contiguous
        const int vb = s - SR - SP;
        float res[1][RV];
        lane_dots<T, 4, 1, RV, 1, 0, LADDER>((const device uint8_t*)w_c, s_c, b_c, x, 2 * NSH, NCAT, min(vb, VB - 1) * RV, kl, res);
        if (kl == 0 && vb < VB)
            for (int v = 0; v < RV; v++) g[vb * RV + v] = static_cast<T>(res[0][v]);
    }
"""


# A2: the same per-output arithmetic with x staged in threadgroup memory as fp32, one 512-column tile (one group per
# k-lane) per round: round t holds groups 8 t .. 8 t + 7, so lane j's groups still come in the order j, j + 8, ...
# A threadgroup serves one block of RV vectors; its slots are row blocks of the three sections.
_TILE = r"""
template <typename T, int BITS, int NR, int NRM, int NV, int H, int OFF>
METAL_FUNC void tile_dots(const device uint8_t* w, const device T* scales, const device T* biases, int base, int nmax,
                          int g, int kl, const threadgroup float* xt, thread float (&res)[NRM][NV]) {
    constexpr int KD = 2560, KG = KD / 64, RB = KD * BITS / 8;
    int nrow[NR];
    float scale[NR], bias[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) {
        nrow[r] = min(base + (r % H) + (r / H) * OFF, nmax - 1);
        scale[r] = scales[(size_t)nrow[r] * KG + g];
        bias[r] = biases[(size_t)nrow[r] * KG + g];
    }
#pragma unroll
    for (int sc = 0; sc < 8; sc++) {
        const int k0 = g * 64 + sc * 8;
        float w_dq[NR][8];
#pragma unroll
        for (int r = 0; r < NR; r++) dequantize<float, 8, BITS>(w + (size_t)nrow[r] * RB + k0 * BITS / 8, scale[r], bias[r], w_dq[r]);
#pragma unroll
        for (int v = 0; v < NV; v++) {
            const threadgroup float* xc = xt + v * 512 + kl * 64 + sc * 8;
            float xf[8];
#pragma unroll
            for (int i = 0; i < 8; i++) xf[i] = xc[i];
#pragma unroll
            for (int r = 0; r < NR; r++) {
                float acc = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) acc += xf[i] * w_dq[r][i];
                res[r][v] += acc;
            }
        }
    }
}
"""

_SRC_A2 = r"""
    constexpr int NE = 512, NSH = 640, NCAT = 1281, KD = 2560, TW = 512;
    constexpr int NRM = (RR > 2 * RP ? RR : 2 * RP);
    threadgroup float xt[RV * TW];
    const uint lane = thread_index_in_simdgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int kl = lane & 7;
    constexpr int VB = RM / RV;
    const int vb = threadgroup_position_in_grid.x % VB;
    const int sb = threadgroup_position_in_grid.x / VB;
    const int s = (sb * SGN + (int)simdgroup_index_in_threadgroup) * 4 + (int)(lane >> 3);
    constexpr int SR = NE / RR, SP = NSH / RP;
    const int sec = s < SR ? 0 : (s < SR + SP ? 1 : (s < SR + SP + 4 ? 2 : 3));
    float res[NRM][RV];
#pragma unroll
    for (int r = 0; r < NRM; r++)
#pragma unroll
        for (int v = 0; v < RV; v++) res[r][v] = 0;
    for (int tr = 0; tr < KD / TW; tr++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < RV * TW; i += SGN * 32) {
            const int v = i / TW, j = i % TW;
            xt[i] = static_cast<float>(x[(size_t)(vb * RV + v) * KD + tr * TW + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int g = tr * 8 + kl;
        if (sec == 0)
            tile_dots<T, 8, RR, NRM, RV, RR, 0>((const device uint8_t*)w_r, s_r, b_r, s * RR, NE, g, kl, xt, res);
        else if (sec == 1)
            tile_dots<T, 4, 2 * RP, NRM, RV, RP, NSH>((const device uint8_t*)w_c, s_c, b_c, (s - SR) * RP, NCAT, g, kl, xt, res);
        else if (sec == 2)
            tile_dots<T, 4, 1, NRM, RV, 1, 0>((const device uint8_t*)w_c, s_c, b_c, 2 * NSH, NCAT, g, kl, xt, res);
    }
    if (sec == 3) return;
#pragma unroll
    for (int r = 0; r < NRM; r++)
#pragma unroll
        for (int v = 0; v < RV; v++) {
            res[r][v] += simd_shuffle_down(res[r][v], 4);
            res[r][v] += simd_shuffle_down(res[r][v], 2);
            res[r][v] += simd_shuffle_down(res[r][v], 1);
        }
    if (kl != 0) return;
    if (sec == 0) {
        for (int r = 0; r < RR; r++)
            for (int v = 0; v < RV; v++) logits[(size_t)(vb * RV + v) * NE + s * RR + r] = static_cast<T>(res[r][v]);
    } else if (sec == 1) {
        const int kb = s - SR;
        for (int p = 0; p < RP; p++)
            for (int v = 0; v < RV; v++) {
                const T gt = static_cast<T>(res[p][v]);
                const T up = static_cast<T>(res[RP + p][v]);
                const T sg = SigmoidL()(gt);
                const T si = gt * sg;
                xs[(size_t)(vb * RV + v) * NSH + kb * RP + p] = si * up;
            }
    } else if (s == SR + SP) {
        for (int v = 0; v < RV; v++) g[vb * RV + v] = static_cast<T>(res[0][v]);
    }
"""


# A3: kernel A with wide loads. The weight bytes of a sub-chunk come in one load (8-bit: uint2; 4-bit: uint) and x's 8
# values in one uint4. Each weight is decoded with MLX dequantize's own expressions (bits 8: s * q + b; bits 4: s * (w &
# 0x0f) + b and (s / 16) * (w & 0xf0) + b, q from the same byte), so every value and every fp32 chain is unchanged.
_WIDE = r"""
template <int BITS>
METAL_FUNC void dq8(const device uint8_t* p, float scale, float bias, thread float* w_local) {
    const float s = float(scale);
    const float b = float(bias);
    if (BITS == 8) {
        const uint2 u = *(const device uint2*)p;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t q = uint8_t(((i < 4 ? u.x : u.y) >> (8 * (i & 3))) & 0xffu);
            w_local[i] = static_cast<float>(s * q + b);
        }
    } else {
        float sc[2] = {s, s / 16.0f};
        const uint u = *(const device uint*)p;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint8_t q = uint8_t((u >> (8 * i)) & 0xffu);
            w_local[2 * i] = static_cast<float>(sc[0] * (q & 0x0f) + b);
            w_local[2 * i + 1] = static_cast<float>(sc[1] * (q & 0xf0) + b);
        }
    }
}

template <typename T, int BITS, int NR, int NV, int H, int OFF>
METAL_FUNC void lane_dots_w(const device uint8_t* w, const device T* scales, const device T* biases,
                            const device T* x, int base, int nmax, int v0, int kl, thread float (&res)[NR][NV]) {
    constexpr int KD = 2560, KG = KD / 64, RB = KD * BITS / 8;
    int nrow[NR];
#pragma unroll
    for (int r = 0; r < NR; r++) nrow[r] = min(base + (r % H) + (r / H) * OFF, nmax - 1);
#pragma unroll
    for (int r = 0; r < NR; r++)
#pragma unroll
        for (int v = 0; v < NV; v++) res[r][v] = 0;
    for (int g = kl; g < KG; g += 8) {
        float scale[NR], bias[NR];
#pragma unroll
        for (int r = 0; r < NR; r++) { scale[r] = scales[(size_t)nrow[r] * KG + g]; bias[r] = biases[(size_t)nrow[r] * KG + g]; }
#pragma unroll
        for (int sc = 0; sc < 8; sc++) {
            const int k0 = g * 64 + sc * 8;
            float w_dq[NR][8];
#pragma unroll
            for (int r = 0; r < NR; r++) dq8<BITS>(w + (size_t)nrow[r] * RB + k0 * BITS / 8, scale[r], bias[r], w_dq[r]);
#pragma unroll
            for (int v = 0; v < NV; v++) {
                const uint4 u = *(const device uint4*)(x + (size_t)(v0 + v) * KD + k0);
                float xf[8];
                xf[0] = as_type<float>(u.x << 16); xf[1] = as_type<float>(u.x & 0xffff0000u);
                xf[2] = as_type<float>(u.y << 16); xf[3] = as_type<float>(u.y & 0xffff0000u);
                xf[4] = as_type<float>(u.z << 16); xf[5] = as_type<float>(u.z & 0xffff0000u);
                xf[6] = as_type<float>(u.w << 16); xf[7] = as_type<float>(u.w & 0xffff0000u);
#pragma unroll
                for (int r = 0; r < NR; r++) {
                    float acc = 0;
#pragma unroll
                    for (int i = 0; i < 8; i++) acc += xf[i] * w_dq[r][i];
                    res[r][v] += acc;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NR; r++)
#pragma unroll
        for (int v = 0; v < NV; v++) {
            res[r][v] += simd_shuffle_down(res[r][v], 4);
            res[r][v] += simd_shuffle_down(res[r][v], 2);
            res[r][v] += simd_shuffle_down(res[r][v], 1);
        }
}
"""


# A4: the same per-output arithmetic, with the lanes laid out for the memory system (diag_moe_dense.py: loads, not FMAs,
# bound A). A threadgroup is 8 simdgroups; simdgroup j does k-lane j's share (groups j, j + 8, ...) for 32 output rows,
# one per lane, x 16 vectors. Every x load is one address for the whole simdgroup (a broadcast). Each row's 64-value
# block of a group is read once in uint4 words and decoded with MLX dequantize's expressions. The 8 k-lane partials go
# through threadgroup memory and are summed in qmv_wide's ladder tree: ((p0 + p4) + (p2 + p6)) + ((p1 + p5) + (p3 + p7)).
# Threadgroups per 16-vector block: 16 router (32 rows each), 40 shared (16 gate rows k.. + 16 up rows 640 + k..),
# 1 for the shared expert's gate (row 1280).
_A4 = r"""
template <typename T, int BITS>
METAL_FUNC void a4_lane(const device uint8_t* w, const device T* scales, const device T* biases, const device T* x,
                        int n, int kl, int v0, thread float (&res)[16]) {
    constexpr int KD = 2560, KG = 40, RB = KD * BITS / 8, GB = 64 * BITS / 8, NW = GB / 16;
    for (int g = kl; g < KG; g += 8) {
        const float s = scales[(size_t)n * KG + g];
        const float b = biases[(size_t)n * KG + g];
        uint4 blk[NW];
        const device uint4* wp = (const device uint4*)(w + (size_t)n * RB + g * GB);
#pragma unroll
        for (int j = 0; j < NW; j++) blk[j] = wp[j];
#pragma unroll
        for (int sc = 0; sc < 8; sc++) {
            float w_dq[8];
            if (BITS == 8) {
                const uint4 u = blk[sc / 2];
                const uint lo = (sc % 2 == 0) ? u.x : u.z;
                const uint hi = (sc % 2 == 0) ? u.y : u.w;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const uint8_t q = uint8_t(((i < 4 ? lo : hi) >> (8 * (i & 3))) & 0xffu);
                    w_dq[i] = static_cast<float>(s * q + b);
                }
            } else {
                const uint4 u = blk[sc / 4];
                const uint wd = (sc % 4 == 0) ? u.x : ((sc % 4 == 1) ? u.y : ((sc % 4 == 2) ? u.z : u.w));
                float scl[2] = {s, s / 16.0f};
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const uint8_t q = uint8_t((wd >> (8 * i)) & 0xffu);
                    w_dq[2 * i] = static_cast<float>(scl[0] * (q & 0x0f) + b);
                    w_dq[2 * i + 1] = static_cast<float>(scl[1] * (q & 0xf0) + b);
                }
            }
            const int k0 = g * 64 + sc * 8;
#pragma unroll
            for (int v = 0; v < 16; v++) {
                const uint4 u = *(const device uint4*)(x + (size_t)(v0 + v) * KD + k0);
                float xf[8];
                xf[0] = as_type<float>(u.x << 16); xf[1] = as_type<float>(u.x & 0xffff0000u);
                xf[2] = as_type<float>(u.y << 16); xf[3] = as_type<float>(u.y & 0xffff0000u);
                xf[4] = as_type<float>(u.z << 16); xf[5] = as_type<float>(u.z & 0xffff0000u);
                xf[6] = as_type<float>(u.w << 16); xf[7] = as_type<float>(u.w & 0xffff0000u);
                float acc = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) acc += xf[i] * w_dq[i];
                res[v] += acc;
            }
        }
    }
}
"""

_SRC_A4 = r"""
    constexpr int NE = 512, NSH = 640;
    threadgroup float part[8 * 16 * 32];
    const int kl = (int)simdgroup_index_in_threadgroup;
    const int l = (int)thread_index_in_simdgroup;
    const int tid = (int)thread_position_in_threadgroup.x;
    const int tg = (int)threadgroup_position_in_grid.x;
    const int t = tg % 57, v0 = (tg / 57) * 16;
    float res[16];
#pragma unroll
    for (int v = 0; v < 16; v++) res[v] = 0;
    if (t < 16) {
        a4_lane<T, 8>((const device uint8_t*)w_r, s_r, b_r, x, 32 * t + l, kl, v0, res);
    } else if (t < 56) {
        const int bb = t - 16;
        a4_lane<T, 4>((const device uint8_t*)w_c, s_c, b_c, x, l < 16 ? 16 * bb + l : NSH + 16 * bb + (l - 16), kl, v0, res);
    } else {
        a4_lane<T, 4>((const device uint8_t*)w_c, s_c, b_c, x, 2 * NSH, kl, v0, res);
    }
#pragma unroll
    for (int v = 0; v < 16; v++) part[(kl * 16 + v) * 32 + l] = res[v];
    threadgroup_barrier(mem_flags::mem_threadgroup);
#define P_(k_, v_, l_) part[((k_) * 16 + (v_)) * 32 + (l_)]
#define TREE(l_, v_) (((P_(0, v_, l_) + P_(4, v_, l_)) + (P_(2, v_, l_) + P_(6, v_, l_))) + ((P_(1, v_, l_) + P_(5, v_, l_)) + (P_(3, v_, l_) + P_(7, v_, l_))))
    if (t < 16) {
        for (int o = tid; o < 512; o += 256) {
            const int ll = o % 32, v = o / 32;
            logits[(size_t)(v0 + v) * NE + 32 * t + ll] = static_cast<T>(TREE(ll, v));
        }
    } else if (t < 56) {
        const int bb = t - 16, kk = tid % 16, v = tid / 16;
        const T gt = static_cast<T>(TREE(kk, v));
        const T up = static_cast<T>(TREE(kk + 16, v));
        const T sg = SigmoidL()(gt);
        const T si = gt * sg;
        xs[(size_t)(v0 + v) * NSH + 16 * bb + kk] = si * up;
    } else if (tid < 16) {
        g[v0 + tid] = static_cast<T>(TREE(0, tid));
    }
"""


def _edit(src, old, new, count=1):
    n = src.count(old)
    assert n == count, (old, n, count)
    return src.replace(old, new)


def _src_b():
    """R-70's router body and qmm16c's body in one kernel: threadgroups [0, NTD) are down_proj tiles, [NTD, NTD + rows)
    router rows. Threadgroup arrays are hoisted to kernel scope; each body sits in its own block."""
    r = RF._SRC
    r = _edit(r, "K = K_C,", "K = TOPK_C,")
    r = _edit(r, "const uint row = threadgroup_position_in_grid.x;", "const uint row = threadgroup_position_in_grid.x - NTD;")
    r = _edit(r, "    threadgroup float local_max[32];\n    threadgroup float local_norm[32];\n    threadgroup uint keys[E];\n", "")
    q = QM._SRC3
    q = _edit(q, "x + (size_t)fm * K_C", "xt + (size_t)fm * K_C")
    q = _edit(q, "x + (size_t)(fm + 8) * K_C", "xt + (size_t)(fm + 8) * K_C")
    q = _edit(q, "y[(size_t)m * N_C + n]", "yt[(size_t)m * N_C + n]")
    q = _edit(q, "    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];\n", "")
    return ("#define ACC_F32 0\n"
            "    threadgroup float local_max[32];\n    threadgroup float local_norm[32];\n    threadgroup uint keys[E_C];\n"
            "    threadgroup float part[(KS > 1 ? KS - 1 : 1) * NT * 16 * 32];\n"
            "    if (threadgroup_position_in_grid.x >= NTD) {\n" + r + "\n    } else {\n"
            "    for (int rt = 0; rt < RT; rt++) {\n"
            "        const device T* xt = xs + (size_t)rt * 16 * K_C;\n"
            "        device T* yt = s_out + (size_t)rt * 16 * N_C;\n"
            "        {\n" + q + "\n        }\n    }\n    }\n")


@lru_cache(None)
def _kA(ladder=0):
    return mx.fast.metal_kernel(name=f"b2_moe_dense_a{'_sens' if ladder else ''}",
                                input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"],
                                header=LK._q_header() + _SIG + _LANE, source=_SRC_A)


_LANE5 = _LANE.replace("lane_dots(", "lane_dots5(").replace(
    "    for (int g = kl; g < KG; g += 8) {\n", "#pragma unroll\n    for (int j_ = 0; j_ < 5; j_++) {\n        const int g = kl + 8 * j_;\n")
assert _LANE5.count("lane_dots5(") == 1 and "const int g = kl + 8 * j_;" in _LANE5
_SRC_A5 = _SRC_A.replace("lane_dots<", "lane_dots5<")
assert _SRC_A5.count("lane_dots5<") == 3


@lru_cache(None)
def _kA5():
    return mx.fast.metal_kernel(name="b2_moe_dense_a5", input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"], header=LK._q_header() + _SIG + _LANE5, source=_SRC_A5)


def dense_a5(pk, xp, rv=4, rr=4, rp=2, sgn=4):
    """A5 alone (kernel A, group loop unrolled): the same outputs as dense_a, bitwise"""
    RM = xp.shape[0]
    assert RM in (16, 32) and RM % rv == 0 and E % rr == 0 and NS % rp == 0
    (wr, sr, br), (wc, sc, bc), _ = pk
    vb = RM // rv
    slots = (E // rr) * vb + (NS // rp) * vb + ((vb + 3) // 4) * 4
    tgs = (slots // 4 + sgn - 1) // sgn
    return _kA5()(inputs=[xp, wr, sr, br, wc, sc, bc],
                  template=[("T", xp.dtype), ("RM", RM), ("RV", rv), ("RR", rr), ("RP", rp), ("SGN", sgn), ("LADDER", 0)],
                  grid=(tgs * sgn * 32, 1, 1), threadgroup=(sgn * 32, 1, 1),
                  output_shapes=[(RM, E), (RM, NS), (RM, 1)], output_dtypes=[xp.dtype] * 3)


# A2p: A2 with each 64-value group segment of the tile padded to 65 floats, so the 8 k-lanes read 8 different banks
_TILEP = _TILE.replace("tile_dots(", "tile_dots_p(").replace(
    "const threadgroup float* xc = xt + v * 512 + kl * 64 + sc * 8;", "const threadgroup float* xc = xt + (v * 8 + kl) * 65 + sc * 8;")
assert _TILEP.count("(v * 8 + kl) * 65") == 1
_SRC_A2P = _SRC_A2.replace("tile_dots<", "tile_dots_p<").replace("threadgroup float xt[RV * TW];", "threadgroup float xt[RV * 520];").replace(
    "            xt[i] = static_cast<float>(x[(size_t)(vb * RV + v) * KD + tr * TW + j]);",
    "            xt[(v * 8 + j / 64) * 65 + j % 64] = static_cast<float>(x[(size_t)(vb * RV + v) * KD + tr * TW + j]);")
assert _SRC_A2P.count("tile_dots_p<") == 3 and "xt[RV * 520]" in _SRC_A2P and "* 65 + j % 64]" in _SRC_A2P


@lru_cache(None)
def _kA2p():
    return mx.fast.metal_kernel(name="b2_moe_dense_a2p", input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"], header=LK._q_header() + _SIG + _TILEP, source=_SRC_A2P)


def dense_a2p(pk, xp, rv=8, rr=2, rp=1, sgn=4):
    """A2p alone (A2 with a padded tile): the same outputs as dense_a, bitwise"""
    RM = xp.shape[0]
    assert RM in (16, 32) and RM % rv == 0 and rv * 520 * 4 <= 32768 and E % (4 * rr) == 0 and NS % (4 * rp) == 0
    (wr, sr, br), (wc, sc, bc), _ = pk
    slots = E // rr + NS // rp + 4
    tgs = (RM // rv) * ((slots // 4 + sgn - 1) // sgn)
    return _kA2p()(inputs=[xp, wr, sr, br, wc, sc, bc],
                   template=[("T", xp.dtype), ("RM", RM), ("RV", rv), ("RR", rr), ("RP", rp), ("SGN", sgn)],
                   grid=(tgs * sgn * 32, 1, 1), threadgroup=(sgn * 32, 1, 1),
                   output_shapes=[(RM, E), (RM, NS), (RM, 1)], output_dtypes=[xp.dtype] * 3)


_SRC_A3 = _SRC_A.replace("lane_dots<", "lane_dots_w<").replace(", LADDER>(", ">(")
assert _SRC_A3.count("lane_dots_w<") == 3 and "LADDER" not in _SRC_A3


@lru_cache(None)
def _kA3():
    return mx.fast.metal_kernel(name="b2_moe_dense_a3", input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"], header=LK._q_header() + _SIG + _WIDE, source=_SRC_A3)


def dense_a3(pk, xp, rv=4, rr=4, rp=2, sgn=4):
    """A3 alone (kernel A with wide loads): the same outputs as dense_a, bitwise"""
    RM = xp.shape[0]
    assert RM in (16, 32) and RM % rv == 0 and E % rr == 0 and NS % rp == 0
    (wr, sr, br), (wc, sc, bc), _ = pk
    vb = RM // rv
    slots = (E // rr) * vb + (NS // rp) * vb + ((vb + 3) // 4) * 4
    tgs = (slots // 4 + sgn - 1) // sgn
    return _kA3()(inputs=[xp, wr, sr, br, wc, sc, bc],
                  template=[("T", xp.dtype), ("RM", RM), ("RV", rv), ("RR", rr), ("RP", rp), ("SGN", sgn)],
                  grid=(tgs * sgn * 32, 1, 1), threadgroup=(sgn * 32, 1, 1),
                  output_shapes=[(RM, E), (RM, NS), (RM, 1)], output_dtypes=[xp.dtype] * 3)


@lru_cache(None)
def _kA4():
    return mx.fast.metal_kernel(name="b2_moe_dense_a4", input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"], header=_SIG + _A4, source=_SRC_A4)


def dense_a4(pk, xp, **_):
    """A4 alone (k-lane per simdgroup, rows across lanes): the same outputs as dense_a, bitwise"""
    RM = xp.shape[0]
    assert RM in (16, 32)
    (wr, sr, br), (wc, sc, bc), _ = pk
    return _kA4()(inputs=[xp, wr, sr, br, wc, sc, bc], template=[("T", xp.dtype)],
                  grid=(57 * (RM // 16) * 256, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(RM, E), (RM, NS), (RM, 1)], output_dtypes=[xp.dtype] * 3)


@lru_cache(None)
def _kA2():
    return mx.fast.metal_kernel(name="b2_moe_dense_a2", input_names=["x", "w_r", "s_r", "b_r", "w_c", "s_c", "b_c"],
                                output_names=["logits", "xs", "g"], header=LK._q_header() + _SIG + _TILE, source=_SRC_A2)


def dense_a2(pk, xp, rv=8, rr=2, rp=1, sgn=4):
    """A2 alone (x tiles in threadgroup memory): the same outputs as dense_a, bitwise"""
    RM = xp.shape[0]
    assert RM in (16, 32) and RM % rv == 0 and rv * 512 * 4 <= 32768 and E % (4 * rr) == 0 and NS % (4 * rp) == 0
    (wr, sr, br), (wc, sc, bc), _ = pk
    slots = E // rr + NS // rp + 4
    tgs = (RM // rv) * ((slots // 4 + sgn - 1) // sgn)
    return _kA2()(inputs=[xp, wr, sr, br, wc, sc, bc],
                  template=[("T", xp.dtype), ("RM", RM), ("RV", rv), ("RR", rr), ("RP", rp), ("SGN", sgn)],
                  grid=(tgs * sgn * 32, 1, 1), threadgroup=(sgn * 32, 1, 1),
                  output_shapes=[(RM, E), (RM, NS), (RM, 1)], output_dtypes=[xp.dtype] * 3)


@lru_cache(None)
def _kB():
    return mx.fast.metal_kernel(name="b2_moe_dense_b", input_names=["logits", "xs", "w", "scales", "biases"],
                                output_names=["out_inds", "out_scores", "s_out"], header=QM._HDR, source=_src_b())


def packs(blk):
    """per block, once: (router w, s, b), R-74's concat (gate | up | gate: w, s, b), down_proj (w, s, b)"""
    pk = blk.__dict__.get("_b2l2")
    if pk is None:
        se = blk.shared_expert
        ps = [se.gate_proj, se.up_proj, blk.shared_expert_gate]
        for p in ps + [se.down_proj]:
            assert p.bits == 4 and p.group_size == 64 and "bias" not in p, (p.bits, p.group_size)
        assert blk.gate.bits == 8 and blk.gate.group_size == 64 and "bias" not in blk.gate
        assert [int(p.weight.shape[0]) for p in ps] == [NS, NS, 1] and int(se.down_proj.weight.shape[0]) == D
        k74 = blk.__dict__.get("_k74")                   # Leap's shared_cat concat (the same rows in the same order), if built
        cat = k74[0] if k74 is not None else tuple(mx.concatenate([p[key] for p in ps]) for key in ("weight", "scales", "biases"))
        rt = tuple(blk.gate[key] for key in ("weight", "scales", "biases"))
        dn = tuple(se.down_proj[key] for key in ("weight", "scales", "biases"))
        mx.eval(cat)
        pk = (rt, cat, dn)
        object.__setattr__(blk, "_b2l2", pk)
    return pk


def dense_a(pk, xp, rv=4, rr=4, rp=2, sgn=4, ladder=0):
    """A alone: xp [RM, 2560] (RM 16 or 32) -> logits [RM, 512], swiglu rows [RM, 640], g [RM, 1]"""
    RM = xp.shape[0]
    assert RM in (16, 32) and RM % rv == 0 and E % rr == 0 and NS % rp == 0
    (wr, sr, br), (wc, sc, bc), _ = pk
    vb = RM // rv
    slots = (E // rr) * vb + (NS // rp) * vb + ((vb + 3) // 4) * 4
    tgs = (slots // 4 + sgn - 1) // sgn
    return _kA(ladder)(inputs=[xp, wr, sr, br, wc, sc, bc],
                       template=[("T", xp.dtype), ("RM", RM), ("RV", rv), ("RR", rr), ("RP", rp), ("SGN", sgn),
                                 ("LADDER", ladder)],
                       grid=(tgs * sgn * 32, 1, 1), threadgroup=(sgn * 32, 1, 1),
                       output_shapes=[(RM, E), (RM, NS), (RM, 1)], output_dtypes=[xp.dtype] * 3)


def dense_b(pk, logits, xs, rows, k):
    """B alone: -> inds [rows, k] uint32, scores [rows, k], s [RM, 2560]"""
    RM = xs.shape[0]
    wd, sd, bd = pk[2]
    ntd = D // (32 * 4)
    return _kB()(inputs=[logits, xs, wd, sd, bd],
                 template=[("T", xs.dtype), ("E_C", E), ("TOPK_C", k), ("K_C", NS), ("N_C", D), ("KS", 1), ("NT", 4),
                           ("NTD", ntd), ("RT", RM // 16)],
                 grid=((ntd + rows) * 128, 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[(rows, k), (rows, k), (RM, D)], output_dtypes=[mx.uint32, xs.dtype, xs.dtype])


def dense_pre(blk, x, a="a", **kw):
    """x [..., 2560] bf16 with <= 32 rows -> (inds [..., k] uint32, scores [..., k], g [..., 1], s [..., 2560]);
    a = "a" (register-blocked A) or "a2" (x tiles in threadgroup memory): the same bits"""
    lead = x.shape[:-1]
    rows = 1
    for d_ in lead:
        rows *= d_
    assert x.dtype == mx.bfloat16 and x.shape[-1] == D and 1 <= rows <= 32, (x.shape, x.dtype)
    k = int(blk.top_k)
    assert k <= 32 and blk.norm_topk_prob is not False
    RM = 16 if rows <= 16 else 32
    x2 = x.reshape(rows, D)
    xp = x2 if rows == RM else mx.concatenate([x2, mx.zeros((RM - rows, D), x.dtype)], axis=0)
    pk = packs(blk)
    if a == "a" and not kw:
        kw = DEFAULT_A
    logits, xs, g = {"a": dense_a, "a2": dense_a2, "a3": dense_a3, "a4": dense_a4, "a5": dense_a5, "a2p": dense_a2p}[a](pk, xp, **kw)
    inds, scores, s = dense_b(pk, logits, xs, rows, k)
    if rows != RM:
        g, s = g[:rows], s[:rows]
    return inds.reshape(*lead, k), scores.reshape(*lead, k), g.reshape(*lead, 1), s.reshape(*lead, D)
