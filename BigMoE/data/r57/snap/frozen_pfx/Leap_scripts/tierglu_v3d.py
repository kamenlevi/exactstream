# Snapshot (Leap, Sun 2026-10-25) of ~/MoonShots/defer/v3d/tierglu.py (MoonShots' 3c kernels, as BigMoE v3 runs them); read-only source, copied verbatim below for B10(b) timing.
"""3c: fused routed-expert GLU over rank tiers (BigMoE DESIGN_3bit.md), reading the MSB-3 pool and the LSB-1 sidecar
directly: two dispatches per MoE layer for all rows x ranks, no 4-bit scratch.
  A  h[p, j] = swiglu(gate_p . x_m, up_p . x_m)       p = m * R + r (row m, gate rank r), j < N
  B  y[p, d] = down_p . h[p, :]                        per-rank outputs [M, R, D]; the block combines (y * scores).sum
Equivalence target = BigMoE's 3b path (bigmoe/tier.py: rebuild4 + gather_qmm bits=4 for 4-bit ranks, gather_qmm bits=3
for 3-bit ranks, SwiGLU). Every output is MLX's own arithmetic on the same values in the same order, so it is meant to
be bit-identical: gate/up rows follow MLX's affine qmv_fast (16 values per lane, 512-value blocks, load_vector's x
scaling, qdot's expressions, block-ordered accumulation, simd_sum), down rows follow qmv (8 values per lane, 256-value
blocks, the clamped tail of qmv_impl), h is MLX's compiled swiglu (Sigmoid, Multiply in bfloat16). Only where the code
bits come from differs: 4-bit ranks rebuild q = 2m + l in registers (spread8) and use scale s3 * 0.5 (exact) and the
LSB part's bias; 3-bit ranks read the MSB part as is.
Tiers per rank are compile-time masks, so any plan of 4 / 3 / 2 works: tier 2 = the MSB-3 codes with their low bit
cleared (m & 6, i.e. 4s * (q >> 2)) under MLX's 3-bit arithmetic, with its own bias pool b2 (bf16(b + s * mean(q & 3)),
a record part BigMoE does not store yet: pass it when a plan has 2-bit ranks).
tier_glu(x, msb, lsb, msb_slot, lsb_slot, plan) -> y [M, R, D]"""
from functools import lru_cache
import mlx.core as mx

PROJ = ("gate_proj", "up_proj", "down_proj")

_HDR = r"""
inline uint spread8(uint x, uint l) {                       // 24 code bits (8 x m) + 8 low bits -> 8 nibbles q = 2m + l
    uint a = (x & 0xFFFu) | ((x & 0xFFF000u) << 4);
    a = (a & 0x003F003Fu) | ((a & 0x0FC00FC0u) << 2);
    a = (a & 0x07070707u) | ((a & 0x38383838u) << 1);
    uint y = l & 0xFFu;
    y = (y | (y << 12)) & 0x000F000Fu;
    y = (y | (y << 6)) & 0x03030303u;
    y = (y | (y << 3)) & 0x11111111u;
    return (a << 1) | y;
}

// ---- MLX quantized.h (0.32.2) load_vector / load_vector_safe / qdot, bits 3 and 4, verbatim; qdot reads its code
// bytes from registers instead of device memory (same expressions, same types after promotion, same order)
template <typename T, typename U, int values_per_thread, int bits>
inline U tg_load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }
  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }
  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U tg_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  U sum = 0;
  if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }
  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
  return sum;
}

template <typename U, int values_per_thread>
inline U tg_qdot3(const thread uint8_t* w, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 8); i++) {
    x_thread += 8 * i;
    w += 3 * i;

    accum += (w[0] & 0x07) * x_thread[0];
    accum += (w[0] & 0x38) * x_thread[1];
    accum += (w[0] & 0xc0) * x_thread[2];
    accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

    accum += (w[1] & 0x0e) * x_thread[3];
    accum += (w[1] & 0x70) * x_thread[4];
    accum += (w[1] & 0x80) * x_thread[5];
    accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

    accum += (w[2] & 0x1c) * x_thread[6];
    accum += (w[2] & 0xe0) * x_thread[7];
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread>
inline U tg_qdot4(const thread uint16_t* ws, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

// MLX unary_ops.h Sigmoid / binary_ops.h Multiply (0.32.2), verbatim: mlx_lm's compiled swiglu = silu(gate) * x
struct TgSigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct TgMultiply {
  template <typename T>
  T operator()(T x, T y) thread {
    return x * y;
  }
};

// One lane's codes for nv consecutive values starting at value v0 of a row: 3-bit stream bytes (tier 3), the same
// with each code's low bit cleared (tier 2), or rebuilt 4-bit nibble words (tier 4).
inline void tg_codes3(const device uint8_t* row, int v0, int nb, thread uint8_t* wb, bool clear_low) {
  const device uint8_t* src = row + v0 * 3 / 8;
  for (int i = 0; i < nb; i++) wb[i] = src[i];
  if (clear_low) {                                          // 0b110 per 3-bit field: 0xB6, 0x6D, 0xDB repeating
    for (int i = 0; i < nb; i++) wb[i] &= (i % 3 == 0) ? 0xB6 : ((i % 3 == 1) ? 0x6D : 0xDB);
  }
}
"""

_A = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C, RA = RA_C, G = 64;
    constexpr int KB3 = K * 3 / 8, KG = K / G, KL = K / 8;      // bytes of 3-bit codes, groups, bytes of low bits per row
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;             // 0: gate rows, 1: up rows
    const uint p = threadgroup_position_in_grid.y;              // pair = m * NR + r
    const uint j0 = threadgroup_position_in_grid.x * RA;
    const uint m = p / NR, r = p % NR;
    const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[p];
    const device uint8_t* w3 = (const device uint8_t*)(sg == 0 ? w3g : w3u) + ((size_t)e * N + j0) * KB3;
    const device T* s3 = (sg == 0 ? s3g : s3u) + ((size_t)e * N + j0) * KG;
    const device T* xr = x + (size_t)m * K + lane * 16;
    float result[RA];
    for (int row = 0; row < RA; ++row) result[row] = 0.0f;
    if (tier == 4) {
        const uint i4 = popcount((uint)MASK4_C & ((1u << r) - 1u));
        const uint el = (uint)lsb_slot[m * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)(sg == 0 ? l1g : l1u) + ((size_t)el * N + j0) * KL;
        const device T* bl = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            float xt[16];
            const float sum = tg_load_vector<T, float, 16, 4>(xr + b * 512, xt);
            for (int row = 0; row < RA; ++row) {
                uint q0, q1;
                if (LD_C == 1) {                                 // 3 ushort loads + 1 ushort (6l and 2l are even)
                    const device uint16_t* c = (const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6);
                    const uint u0 = c[0], u1 = c[1], u2 = c[2];
                    const uint lw = *(const device uint16_t*)(l1 + row * KL + b * 64 + lane * 2);
                    q0 = spread8(u0 | ((u1 & 0xFFu) << 16), lw & 0xFFu);
                    q1 = spread8((u1 >> 8) | (u2 << 8), lw >> 8);
                } else {
                    const device uint8_t* c = w3 + row * KB3 + b * 192 + lane * 6;
                    const device uint8_t* lb = l1 + row * KL + b * 64 + lane * 2;
                    q0 = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), lb[0]);
                    q1 = spread8((uint)c[3] | ((uint)c[4] << 8) | ((uint)c[5] << 16), lb[1]);
                }
                const uint16_t ws[4] = {(uint16_t)(q0 & 0xFFFFu), (uint16_t)(q0 >> 16), (uint16_t)(q1 & 0xFFFFu), (uint16_t)(q1 >> 16)};
                const int gi = row * KG + b * 8 + lane / 4;
                const float s = (float)s3[gi] * 0.5f;
                const float bb = bl[gi];
                result[row] += tg_qdot4<float, 16>(ws, xt, s, bb, sum);
            }
        }
    } else {
        const device T* b3 = tier == 2 ? (sg == 0 ? b2g : b2u) + ((size_t)e * N + j0) * KG
                                       : (sg == 0 ? b3g : b3u) + ((size_t)e * N + j0) * KG;
        for (int b = 0; b < K / 512; ++b) {
            float xt[16];
            const float sum = tg_load_vector<T, float, 16, 3>(xr + b * 512, xt);
            for (int row = 0; row < RA; ++row) {
                uint8_t wb[6];
                if (LD_C == 1) {
                    const device uint16_t* c = (const device uint16_t*)(w3 + row * KB3 + b * 192 + lane * 6);
                    const uint16_t u0 = c[0] & (tier == 2 ? 0x6DB6 : 0xFFFF), u1 = c[1] & (tier == 2 ? 0xB6DB : 0xFFFF),
                                   u2 = c[2] & (tier == 2 ? 0xDB6D : 0xFFFF);
                    wb[0] = (uint8_t)u0; wb[1] = (uint8_t)(u0 >> 8); wb[2] = (uint8_t)u1;
                    wb[3] = (uint8_t)(u1 >> 8); wb[4] = (uint8_t)u2; wb[5] = (uint8_t)(u2 >> 8);
                } else {
                    tg_codes3(w3 + row * KB3, b * 512 + lane * 16, 6, wb, tier == 2);
                }
                const int gi = row * KG + b * 8 + lane / 4;
                const float s = s3[gi];
                const float bb = b3[gi];
                result[row] += tg_qdot3<float, 16>(wb, xt, s, bb, sum);
            }
        }
    }
    threadgroup T gate_v[RA];
    for (int row = 0; row < RA; ++row) result[row] = simd_sum(result[row]);
    if (sg == 0 && lane == 0) {
        for (int row = 0; row < RA; ++row) gate_v[row] = static_cast<T>(result[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1 && lane < (uint)RA) {
        float u = 0.0f;
        for (int row = 0; row < RA; ++row) u = (row == (int)lane) ? result[row] : u;
        const T g = gate_v[lane];
        const T ub = static_cast<T>(u);
        h[(size_t)p * N + j0 + lane] = TgMultiply()(TgMultiply()(g, TgSigmoid()(g)), ub);
    }
"""

_B = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C, RB = RB_C, G = 64;
    constexpr int NB3 = N * 3 / 8, NG = N / G, NL = N / 8;      // per down row
    constexpr int BS = 256;                                     // qmv: 8 values per lane x 32 lanes
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint p = threadgroup_position_in_grid.y;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
    const uint m = p / NR, r = p % NR;
    const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK2_C >> r) & 1) ? 2 : 3);
    const uint e = (uint)msb_slot[p];
    const device uint8_t* w3 = (const device uint8_t*)w3d + ((size_t)e * D + d0) * NB3;
    const device T* s3 = s3d + ((size_t)e * D + d0) * NG;
    const device T* hr = h + (size_t)p * N + lane * 8;
    float result[RB];
    for (int row = 0; row < RB; ++row) result[row] = 0.0f;
    if (tier == 4) {
        const uint i4 = popcount((uint)MASK4_C & ((1u << r) - 1u));
        const uint el = (uint)lsb_slot[m * R4_C + i4];
        const device uint8_t* l1 = (const device uint8_t*)l1d + ((size_t)el * D + d0) * NL;
        const device T* bl = bd + ((size_t)el * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float xt[8];
            const float sum = tg_load_vector<T, float, 8, 4>(hr + k, xt);
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint q = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                const uint16_t ws[2] = {(uint16_t)(q & 0xFFFFu), (uint16_t)(q >> 16)};
                const int gi = row * NG + (k + lane * 8) / G;
                result[row] += tg_qdot4<float, 8>(ws, xt, (float)s3[gi] * 0.5f, (float)bl[gi], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            float xt[8];
            const float sum = tg_load_vector_safe<T, float, 8, 4>(hr + k, xt, remaining);
            for (int row = 0; row < RB; ++row) {
                const device uint8_t* c = w3 + row * NB3 + (k + lane * 8) * 3 / 8;
                const uint q = spread8((uint)c[0] | ((uint)c[1] << 8) | ((uint)c[2] << 16), l1[row * NL + (k + lane * 8) / 8]);
                const uint16_t ws[2] = {(uint16_t)(q & 0xFFFFu), (uint16_t)(q >> 16)};
                const int gi = row * NG + (k + lane * 8) / G;
                result[row] += tg_qdot4<float, 8>(ws, xt, (float)s3[gi] * 0.5f, (float)bl[gi], sum);
            }
        }
    } else {
        const device T* b3 = (tier == 2 ? b2d : b3d) + ((size_t)e * D + d0) * NG;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float xt[8];
            const float sum = tg_load_vector<T, float, 8, 3>(hr + k, xt);
            for (int row = 0; row < RB; ++row) {
                uint8_t wb[3];
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, tier == 2);
                const int gi = row * NG + (k + lane * 8) / G;
                result[row] += tg_qdot3<float, 8>(wb, xt, (float)s3[gi], (float)b3[gi], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 8), 0, 8);
        if (remaining > 0) {
            float xt[8];
            const float sum = tg_load_vector_safe<T, float, 8, 3>(hr + k, xt, remaining);
            for (int row = 0; row < RB; ++row) {
                uint8_t wb[3];
                tg_codes3(w3 + row * NB3, k + lane * 8, 3, wb, tier == 2);
                const int gi = row * NG + (k + lane * 8) / G;
                result[row] += tg_qdot3<float, 8>(wb, xt, (float)s3[gi], (float)b3[gi], sum);
            }
        }
    }
    for (int row = 0; row < RB; ++row) {
        result[row] = simd_sum(result[row]);
        if (lane == 0) y[(size_t)p * D + d0 + row] = static_cast<T>(result[row]);
    }
"""


# B with the block's combine fused: one threadgroup = one row m x RB down rows, simdgroup r = rank r (kernel B's
# arithmetic for pair m * NR + r); each rank's bf16 output goes to threadgroup memory, then lanes 0..RB-1 of simdgroup 0
# form sum_r bf16(y_r * score_r) in MLX's col_reduce order: LY partial bf16 totals (ranks j, j + LY, ...; total = val +
# total from 0), combined j = 1..LY-1 (LY = 1: one sequential bf16 sum; combine_probe.py says which LY MLX uses).
_BC = _B.replace("""    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint p = threadgroup_position_in_grid.y;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
    const uint m = p / NR, r = p % NR;""", """    const uint r = simdgroup_index_in_threadgroup;               // rank
    const uint m = threadgroup_position_in_grid.y;
    const uint d0 = threadgroup_position_in_grid.x * RB;
    const uint p = m * NR + r;""").replace("""    for (int row = 0; row < RB; ++row) {
        result[row] = simd_sum(result[row]);
        if (lane == 0) y[(size_t)p * D + d0 + row] = static_cast<T>(result[row]);
    }
""", """    threadgroup T ys[NR * RB];
    for (int row = 0; row < RB; ++row) {
        result[row] = simd_sum(result[row]);
        if (lane == 0) ys[r * RB + row] = static_cast<T>(result[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0 && lane < (uint)RB) {
        T part[LY_C];
        for (int j = 0; j < LY_C; ++j) part[j] = static_cast<T>(0);
        for (int rr = 0; rr < NR; ++rr) {
            const T prod = TgMultiply()(ys[rr * RB + lane], scores[m * NR + rr]);
            part[rr % LY_C] = prod + part[rr % LY_C];
        }
        T tot = part[0];
        for (int j = 1; j < LY_C; ++j) tot = part[j] + tot;
        y[(size_t)m * D + d0 + lane] = tot;
    }
""")
assert "ys[r * RB + row]" in _BC and "const uint p = m * NR + r;" in _BC


@lru_cache(None)
def _kbc():
    return mx.fast.metal_kernel(
        name="tierglu_bc",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "scores"],
        output_names=["y"], source=_BC, header=_HDR)


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(
        name="tierglu_a",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot"],
        output_names=["h"], source=_A, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(
        name="tierglu_b",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot"],
        output_names=["y"], source=_B, header=_HDR)


def plan_masks(plan):
    m4 = sum(1 << r for r, c in enumerate(plan) if c == "4")
    m2 = sum(1 << r for r, c in enumerate(plan) if c == "2")
    assert all(c in "432" for c in plan), plan
    return m4, m2, plan.count("4")


def tier_glu(x, msb, lsb, msb_slot, lsb_slot, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, scores=None, ly=8, rbc=4, ld=0):
    """x [M, K] bf16; msb[proj] = {"w3": [C_m, out, in*3/32] u32, "s3", "b3": [C_m, out, in/64] bf16}; lsb[proj] =
    {"l1": [C_l, out, in/32] u32, "b": [C_l, out, in/64] bf16}; msb_slot int [M, R] (rank order), lsb_slot int [M, R4]
    (the 4-bit ranks in rank order); b2[proj]: [C_m, out, in/64] bf16 bias pool of the 2-bit tier (only with '2' ranks).
    Returns y [M, R, D] bf16: the routed experts' outputs per rank, before the gate-weighted sum; with scores ([M, R]
    bf16, rank order) y [M, D] = the block's (y * scores[..., None]).sum(-2), combined in-kernel (ly: MLX's order)."""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    assert msb_slot.shape == (M, R), (msb_slot.shape, M, R)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    assert K % 512 == 0 and N % ra == 0 and D % (rb * nsg_b) == 0 and N % 64 == 0
    assert m2 == 0 or b2 is not None, "plan has 2-bit ranks: pass the b2 bias pools"
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}            # unread when the plan has no 2-bit rank
    if r4:
        assert lsb_slot is not None and lsb_slot.shape == (M, r4), (None if lsb_slot is None else lsb_slot.shape, r4)
        ls = lsb_slot.astype(mx.int32).reshape(-1)
        L = lsb
    else:
        ls = mx.zeros((1,), mx.int32)
        L = {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}        # unread
    ms = msb_slot.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    (h,) = _ka()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                         L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls],
                 template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                 grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1),
                 output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    if scores is not None:
        assert D % rbc == 0 and R * 32 <= 1024
        (y,) = _kbc()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls,
                              scores.astype(x.dtype).reshape(-1)],
                      template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                      grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                      output_shapes=[(M, D)], output_dtypes=[x.dtype])
        return y
    (y,) = _kb()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls],
                 template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                 grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                 output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)


# ---- out-of-order experts: run the resident (hit) pairs while the misses load, the missed pairs after, then the
# block's combine in MLX's order. Subset kernels read the pair index from pidx (the full pair p = m * NR + r keeps
# every slot and tier lookup unchanged) and write their rows compactly; per-pair arithmetic is untouched, so
# tier_glu_split == tier_glu(..., scores=) bitwise for any hit mask.
_AS = _A.replace("    const uint p = threadgroup_position_in_grid.y;              // pair = m * NR + r",
                 "    const uint po = threadgroup_position_in_grid.y;\n    const uint p = (uint)pidx[po];                               // pair = m * NR + r") \
        .replace("h[(size_t)p * N + j0 + lane]", "h[(size_t)po * N + j0 + lane]")
_BS = _B.replace("    const uint p = threadgroup_position_in_grid.y;\n",
                 "    const uint po = threadgroup_position_in_grid.y;\n    const uint p = (uint)pidx[po];\n") \
        .replace("const device T* hr = h + (size_t)p * N", "const device T* hr = h + (size_t)po * N") \
        .replace("y[(size_t)p * D + d0 + row]", "y[(size_t)po * D + d0 + row]")
assert "pidx[po]" in _AS and "h[(size_t)po" in _AS and "pidx[po]" in _BS and "y[(size_t)po" in _BS and "h + (size_t)po" in _BS

_KC = r"""
    const uint d = thread_position_in_grid.x, m = thread_position_in_grid.y;
    if (d >= (uint)D_C) return;
    T part[LY_C];
    for (int j = 0; j < LY_C; ++j) part[j] = static_cast<T>(0);
    for (int rr = 0; rr < NR_C; ++rr) {
        const int lc = loc[m * NR_C + rr];
        const T yv = lc >= 0 ? yh[(size_t)lc * D_C + d] : ym[(size_t)(-lc - 1) * D_C + d];
        const T prod = TgMultiply()(yv, scores[m * NR_C + rr]);
        part[rr % LY_C] = prod + part[rr % LY_C];
    }
    T tot = part[0];
    for (int j = 1; j < LY_C; ++j) tot = part[j] + tot;
    out[(size_t)m * D_C + d] = tot;
"""


@lru_cache(None)
def _kas():
    return mx.fast.metal_kernel(
        name="tierglu_as",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx"],
        output_names=["h"], source=_AS, header=_HDR)


@lru_cache(None)
def _kbs():
    return mx.fast.metal_kernel(
        name="tierglu_bs",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx"],
        output_names=["y"], source=_BS, header=_HDR)


@lru_cache(None)
def _kc():
    return mx.fast.metal_kernel(name="tierglu_combine", input_names=["yh", "ym", "loc", "scores"], output_names=["out"],
                                source=_KC, header=_HDR)


def tier_glu_pairs(x, msb, lsb, msb_slot, lsb_slot, pidx, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, ld=0):
    """Per-rank outputs y [len(pidx), D] for the pairs pidx (int32, full pair index m * R + r) only."""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ms = msb_slot.astype(mx.int32).reshape(-1)
    P = pidx.size
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1))]
    (h,) = _kas()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx],
                  template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                  grid=(N // ra * 64, P, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    (y,) = _kbs()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls, pidx],
                  template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                  grid=(D // (rb * nsg_b) * 32 * nsg_b, P, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(P, D)], output_dtypes=[x.dtype])
    return y


def tier_glu_split(x, msb, lsb, msb_slot, lsb_slot, hit, scores, plan="4433333333", b2=None, ly=8):
    """Out-of-order experts: hit (bool [M, R], host) pairs first, missed pairs second, then the combine.
    == tier_glu(..., scores=scores) bitwise. In the engine the first call is issued before the spin on the misses."""
    import numpy as np
    M, R = len(hit), len(plan)
    flat = np.asarray(hit, bool).reshape(-1)
    hi, mi = np.nonzero(flat)[0], np.nonzero(~flat)[0]
    loc = np.empty(M * R, np.int32)
    loc[hi] = np.arange(hi.size)
    loc[mi] = -(np.arange(mi.size) + 1)
    D = msb["down_proj"]["s3"].shape[1]
    run = lambda ix: tier_glu_pairs(x, msb, lsb, msb_slot, lsb_slot, mx.array(ix.astype(np.int32)), plan, b2) if ix.size \
        else mx.zeros((1, D), x.dtype)
    yh, ym = run(hi), run(mi)
    (out,) = _kc()(inputs=[yh, ym, mx.array(loc), scores.astype(x.dtype).reshape(-1)],
                   template=[("T", x.dtype), ("D_C", D), ("NR_C", R), ("LY_C", ly)],
                   grid=(D, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, D)], output_dtypes=[x.dtype])
    return out


# ---- engine form of out-of-order experts (no host sync): a device array phase [M * R] (1 = the pair's experts were
# resident at routing time, 2 = it waits for the loader). The phase-P call runs the full pair grid; a threadgroup whose
# pair is not in phase P returns at once (uniform per threadgroup, before any barrier). Phase 1 takes the slot ids known
# before the spin (non-resident pairs may hold -1: never read), phase 2 the spin's ids. Each call writes only its own
# pairs' rows of y [M * R, D]; the combine picks each pair's row from y1 or y2 by phase, in MLX's LY=8 order.
_AP = _A.replace("    const uint p = threadgroup_position_in_grid.y;              // pair = m * NR + r",
                 "    const uint p = threadgroup_position_in_grid.y;              // pair = m * NR + r\n"
                 "    if (phase[p] != PHASE_C) return;")
_BP = _B.replace("    const uint p = threadgroup_position_in_grid.y;\n",
                 "    const uint p = threadgroup_position_in_grid.y;\n    if (phase[p] != PHASE_C) return;\n")
assert "phase[p] != PHASE_C" in _AP and "phase[p] != PHASE_C" in _BP
_KCP = _KC.replace("const int lc = loc[m * NR_C + rr];\n        const T yv = lc >= 0 ? yh[(size_t)lc * D_C + d] : ym[(size_t)(-lc - 1) * D_C + d];",
                   "const T yv = phase[m * NR_C + rr] == 1 ? y1[(size_t)(m * NR_C + rr) * D_C + d] : y2[(size_t)(m * NR_C + rr) * D_C + d];")
assert "phase[m * NR_C + rr] == 1" in _KCP


@lru_cache(None)
def _kap():
    return mx.fast.metal_kernel(
        name="tierglu_ap",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "phase"],
        output_names=["h"], source=_AP, header=_HDR)


@lru_cache(None)
def _kbp():
    return mx.fast.metal_kernel(
        name="tierglu_bp",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "phase"],
        output_names=["y"], source=_BP, header=_HDR)


@lru_cache(None)
def _kcp():
    return mx.fast.metal_kernel(name="tierglu_combine_phase", input_names=["y1", "y2", "phase", "scores"],
                                output_names=["out"], source=_KCP, header=_HDR)


def tier_glu_phase(x, msb, lsb, msb_slot, lsb_slot, phase, which, plan="4433333333", b2=None, ra=4, rb=4, nsg_b=2, ld=0):
    """Per-rank y [M * R, D] for the pairs with phase == which (rows of the other phase are left unwritten)."""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ms = msb_slot.astype(mx.int32).reshape(-1)
    ph = phase.astype(mx.int32).reshape(-1)
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("PHASE_C", which)]
    (h,) = _kap()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                          L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, ph],
                  template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                  grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    (y,) = _kbp()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls, ph],
                  template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                  grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y


def combine_phase(y1, y2, phase, scores, ly=8):
    """The block's (y * scores[..., None]).sum(-2) in MLX's order, each pair's y taken from y1 (phase 1) or y2."""
    M, R = scores.shape
    D = y1.shape[-1]
    (out,) = _kcp()(inputs=[y1, y2, phase.astype(mx.int32).reshape(-1), scores.astype(y1.dtype).reshape(-1)],
                    template=[("T", y1.dtype), ("D_C", D), ("NR_C", R), ("LY_C", ly)],
                    grid=(D, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, D)], output_dtypes=[y1.dtype])
    return out


# ---- [MoonShots v3d] half-layer deferral (tier.py BIGMOE_DH): phase 3 = late pairs, read after phase A and added half a
# layer late. combine_dh(..., which=0) sums phases 1-2 (y1 / y2 by phase, late pairs skipped), which=1 the late pairs only
# (y1 = the phase-3 call's rows); a skipped pair adds an exact bf16 zero, so with no late pair which=0 == combine_phase
# bitwise, and either part == the block's (y * masked scores).sum(-2) on the per-rank outputs.
_KCD = _KC.replace("const int lc = loc[m * NR_C + rr];\n        const T yv = lc >= 0 ? yh[(size_t)lc * D_C + d] : ym[(size_t)(-lc - 1) * D_C + d];",
                   "const int ph = phase[m * NR_C + rr];\n"
                   "        const size_t q = (size_t)(m * NR_C + rr) * D_C + d;\n"
                   "        const T yv = WHICH_C == 0 ? (ph == 1 ? y1[q] : ph == 2 ? y2[q] : static_cast<T>(0))\n"
                   "                   : WHICH_C == 1 ? (ph == 3 ? y1[q] : static_cast<T>(0))\n"
                   "                   : ((rr >= W0_C && ph == 3) ? y1[(size_t)(m * (NR_C - W0_C) + (rr - W0_C)) * D_C + d] : static_cast<T>(0));")
assert "WHICH_C == 0" in _KCD


@lru_cache(None)
def _kcd():
    return mx.fast.metal_kernel(name="tierglu_combine_dh", input_names=["y1", "y2", "phase", "scores"],
                                output_names=["out"], source=_KCD, header=_HDR)


def combine_dh(y1, y2, phase, scores, which, ly=8, w0=0):
    """which=0: phases 1-2 (each pair's row from y1 or y2); which=1: phase 3 (rows of y1); which=2: phase 3 from compact
    rows over ranks >= w0 (tier_glu_pairs_phase on the static subset). MLX's LY=8 order."""
    M, R = scores.shape
    D = y1.shape[-1]
    (out,) = _kcd()(inputs=[y1, y2, phase.astype(mx.int32).reshape(-1), scores.astype(y1.dtype).reshape(-1)],
                    template=[("T", y1.dtype), ("D_C", D), ("NR_C", R), ("LY_C", ly), ("WHICH_C", which), ("W0_C", w0)],
                    grid=(D, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, D)], output_dtypes=[y1.dtype])
    return out


# ---- [MoonShots v3d] phase 3 on a static pair subset (tier.py BIGMOE_DH_SUB=1): only ranks >= W can be late, so the late
# call runs the subset kernels over the pairs (m, r >= W) (pidx, a fixed list) with the phase check of the phase kernels:
# 40 % fewer empty threadgroups at W=4. Rows are compact (row po = pair pidx[po]); combine_dh(which=2, w0=W) reads them.
_ASP = _AS.replace("    const uint p = (uint)pidx[po];                               // pair = m * NR + r",
                   "    const uint p = (uint)pidx[po];                               // pair = m * NR + r\n"
                   "    if (phase[p] != PHASE_C) return;")
_BSP = _BS.replace("    const uint p = (uint)pidx[po];\n", "    const uint p = (uint)pidx[po];\n    if (phase[p] != PHASE_C) return;\n")
assert "phase[p] != PHASE_C" in _ASP and "phase[p] != PHASE_C" in _BSP


@lru_cache(None)
def _kasp():
    return mx.fast.metal_kernel(
        name="tierglu_asp",
        input_names=["x", "w3g", "w3u", "s3g", "s3u", "b3g", "b3u", "b2g", "b2u", "l1g", "l1u", "bg", "bu",
                     "msb_slot", "lsb_slot", "pidx", "phase"],
        output_names=["h"], source=_ASP, header=_HDR)


@lru_cache(None)
def _kbsp():
    return mx.fast.metal_kernel(
        name="tierglu_bsp",
        input_names=["h", "w3d", "s3d", "b3d", "b2d", "l1d", "bd", "msb_slot", "lsb_slot", "pidx", "phase"],
        output_names=["y"], source=_BSP, header=_HDR)


def tier_glu_pairs_phase(x, msb, lsb, msb_slot, lsb_slot, pidx, phase, which, plan="4433333333", b2=None, ra=4, rb=4,
                         nsg_b=2, ld=0):
    """Per-rank y [len(pidx), D] for the pairs pidx (full pair index m * R + r) whose phase == which (other rows unwritten)."""
    M, K = x.shape
    R = len(plan)
    m4, m2, r4 = plan_masks(plan)
    N = msb["gate_proj"]["s3"].shape[1]
    D = msb["down_proj"]["s3"].shape[1]
    bb = b2 if b2 is not None else {p: msb[p]["b3"] for p in PROJ}
    L = lsb if r4 else {p: {"l1": msb[p]["w3"], "b": msb[p]["b3"]} for p in PROJ}
    ls = lsb_slot.astype(mx.int32).reshape(-1) if r4 else mx.zeros((1,), mx.int32)
    ms = msb_slot.astype(mx.int32).reshape(-1)
    ph = phase.astype(mx.int32).reshape(-1)
    P = pidx.size
    g, u, d = (msb[p] for p in PROJ)
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", max(r4, 1)), ("PHASE_C", which)]
    (h,) = _kasp()(inputs=[x, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], bb["gate_proj"], bb["up_proj"],
                           L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, ph],
                   template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra), ("LD_C", ld)],
                   grid=(N // ra * 64, P, 1), threadgroup=(64, 1, 1), output_shapes=[(P, N)], output_dtypes=[x.dtype])
    (y,) = _kbsp()(inputs=[h, d["w3"], d["s3"], d["b3"], bb["down_proj"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls,
                           pidx, ph],
                   template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                   grid=(D // (rb * nsg_b) * 32 * nsg_b, P, 1), threadgroup=(32 * nsg_b, 1, 1),
                   output_shapes=[(P, D)], output_dtypes=[x.dtype])
    return y
