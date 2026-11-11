"""Fused routed GLU over rank tiers on records4 (2+1+1 planes, DESIGN_records4.md): per-rank tier 2 / 3 / 4 from the P2 pool
(2-bit codes + s2 + b2), the B3 sidecar (c bits + b3) and the B4 sidecar (l bits + b). Two dispatches (A: gate+up+swiglu,
B: down), per-rank outputs y [M*R, D] (the block's combine stays MLX's, or use tierglu.combine_phase).
Exact references per tier (MLX's own arithmetic, lane geometry and order, as tierglu.py):
  tier 2  gather_qmm(bits=2) on (w2, s2, b2)            qmv_fast 16 values/lane; down: qmv with 16 values/lane, 512 blocks
  tier 3  gather_qmm(bits=3) on m = 2 m2 + c, s2/2, b3   == records3's tier 3 (v3) bitwise
  tier 4  gather_qmm(bits=4) on q = 4 m2 + 2c + l, s2/4, b == the 4-bit expert bitwise
tier_glu4(x, p2, b3s, b4s, p2_slot, c_slot, l_slot, plan) with per-pair slots [M, R] (c_slot / l_slot -1 where unused)."""
from functools import lru_cache
import mlx.core as mx
from tierglu import _HDR as _HDR3

_HDR = _HDR3 + r"""
template <typename T, typename U, int values_per_thread>
inline U tg_load_vector2(const device T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < values_per_thread; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 4.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 64.0f;
  }
  return sum;
}
template <typename T, typename U, int values_per_thread>
inline U tg_load_vector2_safe(const device T* x, thread U* x_thread, int N) {
  U sum = 0;
  for (int i = 0; i < N; i += 4) {
    sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 4.0f;
    x_thread[i + 2] = x[i + 2] / 16.0f;
    x_thread[i + 3] = x[i + 3] / 64.0f;
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
  return sum;
}
template <typename U, int values_per_thread>
inline U tg_qdot2(const thread uint8_t* w, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (w[i] & 0x03) +
         x_thread[4 * i + 1] * (w[i] & 0x0c) +
         x_thread[4 * i + 2] * (w[i] & 0x30) +
         x_thread[4 * i + 3] * (w[i] & 0xc0));
  }
  return scale * accum + sum * bias;
}
// codes from planes: m2 (2 bits each, value i at bits 2i), c and l (1 bit each, value i at bit i); bit tricks:
inline uint tg_spread2x4(uint m2) {                          // 8 fields of 2 bits (bits 0..15) -> nibble slots (bits 4i..4i+1)
    uint x = m2 & 0xFFFFu;
    x = (x | (x << 8)) & 0x00FF00FFu;
    x = (x | (x << 4)) & 0x0F0F0F0Fu;
    x = (x | (x << 2)) & 0x33333333u;
    return x;
}
inline uint tg_spread1x4(uint b) {                           // 8 bits -> bit 0 of each nibble
    uint y = b & 0xFFu;
    y = (y | (y << 12)) & 0x000F000Fu;
    y = (y | (y << 6)) & 0x03030303u;
    y = (y | (y << 3)) & 0x11111111u;
    return y;
}
inline uint tg_m24(uint m2, uint c) {                        // 8 values: m = 2 m2 + c as MLX's 3-bit stream (24 bits)
    uint y = (tg_spread2x4(m2) << 1) | tg_spread1x4(c);
    y = (y & 0x07070707u) | ((y & 0x70707070u) >> 1);
    y = (y & 0x003F003Fu) | ((y & 0x3F003F00u) >> 2);
    return (y & 0x00000FFFu) | ((y & 0x0FFF0000u) >> 4);
}
inline uint tg_nib4(uint m2, uint c, uint l) {               // 8 values: q = 4 m2 + 2c + l as nibbles
    return (tg_spread2x4(m2) << 2) | (tg_spread1x4(c) << 1) | tg_spread1x4(l);
}
inline void tg_codes3_from(uint m2, uint c, int nv, thread uint8_t* wb) {   // MLX 3-bit stream bytes (nv = 8 or 16)
    for (int h = 0; h < nv / 8; ++h) {
        const uint acc = tg_m24(m2 >> (16 * h), c >> (8 * h));
        wb[3 * h] = acc & 0xFF; wb[3 * h + 1] = (acc >> 8) & 0xFF; wb[3 * h + 2] = (acc >> 16) & 0xFF;
    }
}
inline void tg_codes4_from(uint m2, uint c, uint l, int nv, thread uint16_t* ws) { // 4-bit nibble words (4 values each)
    for (int h = 0; h < nv / 8; ++h) {
        const uint w = tg_nib4(m2 >> (16 * h), c >> (8 * h), l >> (8 * h));
        ws[2 * h] = (uint16_t)(w & 0xFFFFu); ws[2 * h + 1] = (uint16_t)(w >> 16);
    }
}
"""

_A = r"""
    constexpr int K = K_C, N = N_C, NR = NR_C, RA = RA_C, G = 64;
    constexpr int KW2 = K / 16, KG = K / G, KB = K / 8;           // P2 u32 words, groups, bit-plane bytes per row
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint p = threadgroup_position_in_grid.y;
    const uint j0 = threadgroup_position_in_grid.x * RA;
    const uint m = p / NR, r = p % NR;
    const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK3_C >> r) & 1) ? 3 : 2);
    const uint e2 = (uint)p2_slot[p];
    const device uint* w2 = (sg == 0 ? w2g : w2u) + ((size_t)e2 * N + j0) * KW2;
    const device T* s2 = (sg == 0 ? s2g : s2u) + ((size_t)e2 * N + j0) * KG;
    const device T* xr = x + (size_t)m * K + lane * 16;
    const device uint8_t* cb = nullptr; const device uint8_t* lb = nullptr; const device T* bias;
    if (tier == 2) {
        bias = (sg == 0 ? b2g : b2u) + ((size_t)e2 * N + j0) * KG;
    } else {
        const uint ec = (uint)c_slot[p];
        cb = (const device uint8_t*)(sg == 0 ? cg : cu) + ((size_t)ec * N + j0) * KB;
        bias = (sg == 0 ? b3g : b3u) + ((size_t)ec * N + j0) * KG;
        if (tier == 4) {
            const uint el = (uint)l_slot[p];
            lb = (const device uint8_t*)(sg == 0 ? lg : lu) + ((size_t)el * N + j0) * KB;
            bias = (sg == 0 ? bg : bu) + ((size_t)el * N + j0) * KG;
        }
    }
    float result[RA];
    for (int row = 0; row < RA; ++row) result[row] = 0.0f;
    for (int b = 0; b < K / 512; ++b) {
        float xt[16];
        float sum;
        if (tier == 2) sum = tg_load_vector2<T, float, 16>(xr + b * 512, xt);
        else if (tier == 3) sum = tg_load_vector<T, float, 16, 3>(xr + b * 512, xt);
        else sum = tg_load_vector<T, float, 16, 4>(xr + b * 512, xt);
        for (int row = 0; row < RA; ++row) {
            const uint m2 = w2[row * KW2 + b * 32 + lane];
            const int gi = row * KG + b * 8 + lane / 4;
            const float s = (float)s2[gi];
            if (tier == 2) {
                const uint8_t wb[4] = {(uint8_t)(m2 & 0xFF), (uint8_t)((m2 >> 8) & 0xFF), (uint8_t)((m2 >> 16) & 0xFF), (uint8_t)(m2 >> 24)};
                result[row] += tg_qdot2<float, 16>(wb, xt, s, (float)bias[gi], sum);
            } else {
                const uint c = *(const device uint16_t*)(cb + row * KB + b * 64 + lane * 2);
                if (tier == 3) {
                    uint8_t wb[6];
                    tg_codes3_from(m2, c, 16, wb);
                    result[row] += tg_qdot3<float, 16>(wb, xt, s * 0.5f, (float)bias[gi], sum);
                } else {
                    const uint l = *(const device uint16_t*)(lb + row * KB + b * 64 + lane * 2);
                    uint16_t ws[4];
                    tg_codes4_from(m2, c, l, 16, ws);
                    result[row] += tg_qdot4<float, 16>(ws, xt, s * 0.25f, (float)bias[gi], sum);
                }
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
        h[(size_t)p * N + j0 + lane] = TgMultiply()(TgMultiply()(g, TgSigmoid()(g)), static_cast<T>(u));
    }
"""

_B = r"""
    constexpr int N = N_C, D = D_C, NR = NR_C, RB = RB_C, G = 64;
    constexpr int NW2 = N / 16, NG = N / G, NB = N / 8;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint p = threadgroup_position_in_grid.y;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
    const uint r = p % NR;
    const int tier = ((MASK4_C >> r) & 1) ? 4 : (((MASK3_C >> r) & 1) ? 3 : 2);
    const uint e2 = (uint)p2_slot[p];
    const device uint* w2 = w2d + ((size_t)e2 * D + d0) * NW2;
    const device uint16_t* w2h = (const device uint16_t*)w2;
    const device T* s2 = s2d + ((size_t)e2 * D + d0) * NG;
    const device T* hr = h + (size_t)p * N;
    const device uint8_t* cbp = nullptr; const device uint8_t* lbp = nullptr; const device T* bias;
    if (tier == 2) {
        bias = b2d + ((size_t)e2 * D + d0) * NG;
    } else {
        const uint ec = (uint)c_slot[p];
        cbp = (const device uint8_t*)cd + ((size_t)ec * D + d0) * NB;
        bias = b3d + ((size_t)ec * D + d0) * NG;
        if (tier == 4) {
            const uint el = (uint)l_slot[p];
            lbp = (const device uint8_t*)ld + ((size_t)el * D + d0) * NB;
            bias = bd + ((size_t)el * D + d0) * NG;
        }
    }
    float result[RB];
    for (int row = 0; row < RB; ++row) result[row] = 0.0f;
    if (tier == 2) {                                             // MLX qmv bits=2: 16 values per lane, 512-value blocks
        constexpr int BS = 512;
        int k = 0;
        for (; k < N - BS; k += BS) {
            float xt[16];
            const float sum = tg_load_vector2<T, float, 16>(hr + k + lane * 16, xt);
            for (int row = 0; row < RB; ++row) {
                const uint m2 = w2[row * NW2 + (k + lane * 16) / 16];
                const uint8_t wb[4] = {(uint8_t)(m2 & 0xFF), (uint8_t)((m2 >> 8) & 0xFF), (uint8_t)((m2 >> 16) & 0xFF), (uint8_t)(m2 >> 24)};
                const int gi = row * NG + (k + lane * 16) / G;
                result[row] += tg_qdot2<float, 16>(wb, xt, (float)s2[gi], (float)bias[gi], sum);
            }
        }
        const int remaining = clamp(static_cast<int>(N - k - lane * 16), 0, 16);
        if (remaining > 0) {
            float xt[16];
            const float sum = tg_load_vector2_safe<T, float, 16>(hr + k + lane * 16, xt, remaining);
            for (int row = 0; row < RB; ++row) {
                const uint m2 = w2[row * NW2 + (k + lane * 16) / 16];
                const uint8_t wb[4] = {(uint8_t)(m2 & 0xFF), (uint8_t)((m2 >> 8) & 0xFF), (uint8_t)((m2 >> 16) & 0xFF), (uint8_t)(m2 >> 24)};
                const int gi = row * NG + (k + lane * 16) / G;
                result[row] += tg_qdot2<float, 16>(wb, xt, (float)s2[gi], (float)bias[gi], sum);
            }
        }
    } else {                                                     // MLX qmv bits=3/4: 8 values per lane, 256-value blocks
        constexpr int BS = 256;
        int k = 0;
        for (int pass = 0; pass < 2; ++pass) {
            const bool tail = pass == 1;
            for (; tail ? (k < N) : (k < N - BS); k += BS) {
                const int remaining = tail ? clamp(static_cast<int>(N - k - lane * 8), 0, 8) : 8;
                if (remaining > 0) {
                    float xt[8];
                    float sum;
                    if (tier == 3) sum = tail ? tg_load_vector_safe<T, float, 8, 3>(hr + k + lane * 8, xt, remaining)
                                              : tg_load_vector<T, float, 8, 3>(hr + k + lane * 8, xt);
                    else sum = tail ? tg_load_vector_safe<T, float, 8, 4>(hr + k + lane * 8, xt, remaining)
                                    : tg_load_vector<T, float, 8, 4>(hr + k + lane * 8, xt);
                    for (int row = 0; row < RB; ++row) {
                        const uint v0 = k + lane * 8;
                        const uint m2 = w2h[row * (NW2 * 2) + v0 / 8];
                        const uint c = cbp[row * NB + v0 / 8];
                        const int gi = row * NG + v0 / G;
                        const float s = (float)s2[gi];
                        if (tier == 3) {
                            uint8_t wb[3];
                            tg_codes3_from(m2, c, 8, wb);
                            result[row] += tg_qdot3<float, 8>(wb, xt, s * 0.5f, (float)bias[gi], sum);
                        } else {
                            const uint l = lbp[row * NB + v0 / 8];
                            uint16_t ws[2];
                            tg_codes4_from(m2, c, l, 8, ws);
                            result[row] += tg_qdot4<float, 8>(ws, xt, s * 0.25f, (float)bias[gi], sum);
                        }
                    }
                }
                if (tail) break;
            }
        }
    }
    for (int row = 0; row < RB; ++row) {
        result[row] = simd_sum(result[row]);
        if (lane == 0) y[(size_t)p * D + d0 + row] = static_cast<T>(result[row]);
    }
"""


# ---- B with the gate-weighted combine in-kernel (one simdgroup per rank, MLX's col_reduce_small LY order), as tierglu's _BC
_BC = _B.replace("""    const uint sg = simdgroup_index_in_threadgroup;
    const uint nsg = threads_per_threadgroup.x / 32;
    const uint p = threadgroup_position_in_grid.y;
    const uint d0 = (threadgroup_position_in_grid.x * nsg + sg) * RB;
    if (d0 >= (uint)D) return;
    const uint r = p % NR;""", """    const uint r = simdgroup_index_in_threadgroup;               // rank
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
        name="tierglu4_bc",
        input_names=["h", "w2d", "s2d", "b2d", "cd", "b3d", "ld", "bd", "p2_slot", "c_slot", "l_slot", "scores"],
        output_names=["y"], source=_BC, header=_HDR)


@lru_cache(None)
def _ka():
    return mx.fast.metal_kernel(
        name="tierglu4_a",
        input_names=["x", "w2g", "w2u", "s2g", "s2u", "b2g", "b2u", "cg", "cu", "b3g", "b3u", "lg", "lu", "bg", "bu",
                     "p2_slot", "c_slot", "l_slot"],
        output_names=["h"], source=_A, header=_HDR)


@lru_cache(None)
def _kb():
    return mx.fast.metal_kernel(
        name="tierglu4_b",
        input_names=["h", "w2d", "s2d", "b2d", "cd", "b3d", "ld", "bd", "p2_slot", "c_slot", "l_slot"],
        output_names=["y"], source=_B, header=_HDR)


def tier_glu4(x, p2, b3s, b4s, p2_slot, c_slot, l_slot, plan="4422222222", ra=4, rb=4, nsg_b=2, scores=None, ly=8, rbc=4):
    """x [M, K] bf16; p2[proj] = {"w2": [C2, out, in/16] u32, "s2", "b2": [C2, out, in/64]}; b3s[proj] = {"c": [C3, out,
    in/32] u32, "b3"}; b4s[proj] = {"l": [C4, out, in/32] u32, "b"}; slots int [M, R] (c_slot, l_slot: -1 where unused).
    Returns per-rank y [M, R, D] bf16; with scores ([M, R] bf16, rank order) y [M, D] = (y * scores[..., None]).sum(-2)
    combined in-kernel in MLX's order (ly), as tierglu.tier_glu(scores=)."""
    M, K = x.shape
    R = len(plan)
    m4 = sum(1 << i for i, c in enumerate(plan) if c == "4")
    m3 = sum(1 << i for i, c in enumerate(plan) if c == "3")
    N = p2["gate_proj"]["s2"].shape[1]
    D = p2["down_proj"]["s2"].shape[1]
    sl = [a.astype(mx.int32).reshape(-1) for a in (p2_slot, c_slot, l_slot)]
    g2, u2, d2 = (p2[p] for p in ("gate_proj", "up_proj", "down_proj"))
    g3, u3, d3 = (b3s[p] for p in ("gate_proj", "up_proj", "down_proj"))
    g4, u4, d4 = (b4s[p] for p in ("gate_proj", "up_proj", "down_proj"))
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK3_C", m3)]
    (h,) = _ka()(inputs=[x, g2["w2"], u2["w2"], g2["s2"], u2["s2"], g2["b2"], u2["b2"], g3["c"], u3["c"], g3["b3"], u3["b3"],
                         g4["l"], u4["l"], g4["b"], u4["b"], *sl],
                 template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra)],
                 grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    if scores is not None:
        assert D % rbc == 0 and R * 32 <= 1024
        (y,) = _kbc()(inputs=[h, d2["w2"], d2["s2"], d2["b2"], d3["c"], d3["b3"], d4["l"], d4["b"], *sl,
                              scores.astype(x.dtype).reshape(-1)],
                      template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rbc), ("LY_C", ly)],
                      grid=(D // rbc * 32 * R, M, 1), threadgroup=(32 * R, 1, 1),
                      output_shapes=[(M, D)], output_dtypes=[x.dtype])
        return y
    (y,) = _kb()(inputs=[h, d2["w2"], d2["s2"], d2["b2"], d3["c"], d3["b3"], d4["l"], d4["b"], *sl],
                 template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                 grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                 output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y.reshape(M, R, D)



# ---- engine form of out-of-order experts on records4 (as tierglu.tier_glu_phase): a device array phase [M * R] (1 = the
# pair's planes were resident at routing time, 2 = it waits for the loader); a threadgroup whose pair is not in phase P
# returns at once (uniform per threadgroup, before any barrier). The combine is tierglu.combine_phase (format-free).
_AP = _A.replace("    const uint p = threadgroup_position_in_grid.y;\n",
                 "    const uint p = threadgroup_position_in_grid.y;\n    if (phase[p] != PHASE_C) return;\n", 1)
_BP = _B.replace("    const uint p = threadgroup_position_in_grid.y;\n",
                 "    const uint p = threadgroup_position_in_grid.y;\n    if (phase[p] != PHASE_C) return;\n", 1)
assert "phase[p] != PHASE_C" in _AP and "phase[p] != PHASE_C" in _BP


@lru_cache(None)
def _kap():
    return mx.fast.metal_kernel(
        name="tierglu4_ap",
        input_names=["x", "w2g", "w2u", "s2g", "s2u", "b2g", "b2u", "cg", "cu", "b3g", "b3u", "lg", "lu", "bg", "bu",
                     "p2_slot", "c_slot", "l_slot", "phase"],
        output_names=["h"], source=_AP, header=_HDR)


@lru_cache(None)
def _kbp():
    return mx.fast.metal_kernel(
        name="tierglu4_bp",
        input_names=["h", "w2d", "s2d", "b2d", "cd", "b3d", "ld", "bd", "p2_slot", "c_slot", "l_slot", "phase"],
        output_names=["y"], source=_BP, header=_HDR)


def tier_glu4_phase(x, p2, b3s, b4s, p2_slot, c_slot, l_slot, phase, which, plan="4422222222", ra=4, rb=4, nsg_b=2):
    """Per-rank y [M * R, D] for the pairs with phase == which (rows of the other phase are left unwritten); the block's
    output is tierglu.combine_phase(y1, y2, phase, scores), bitwise equal to tier_glu4(..., scores=)."""
    M, K = x.shape
    R = len(plan)
    m4 = sum(1 << i for i, c in enumerate(plan) if c == "4")
    m3 = sum(1 << i for i, c in enumerate(plan) if c == "3")
    N = p2["gate_proj"]["s2"].shape[1]
    D = p2["down_proj"]["s2"].shape[1]
    sl = [a.astype(mx.int32).reshape(-1) for a in (p2_slot, c_slot, l_slot)]
    ph = phase.astype(mx.int32).reshape(-1)
    g2, u2, d2 = (p2[p] for p in ("gate_proj", "up_proj", "down_proj"))
    g3, u3, d3 = (b3s[p] for p in ("gate_proj", "up_proj", "down_proj"))
    g4, u4, d4 = (b4s[p] for p in ("gate_proj", "up_proj", "down_proj"))
    tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK3_C", m3), ("PHASE_C", which)]
    (h,) = _kap()(inputs=[x, g2["w2"], u2["w2"], g2["s2"], u2["s2"], g2["b2"], u2["b2"], g3["c"], u3["c"], g3["b3"], u3["b3"],
                          g4["l"], u4["l"], g4["b"], u4["b"], *sl, ph],
                  template=tmpl + [("K_C", K), ("N_C", N), ("RA_C", ra)],
                  grid=(N // ra * 64, M * R, 1), threadgroup=(64, 1, 1), output_shapes=[(M * R, N)], output_dtypes=[x.dtype])
    (y,) = _kbp()(inputs=[h, d2["w2"], d2["s2"], d2["b2"], d3["c"], d3["b3"], d4["l"], d4["b"], *sl, ph],
                  template=tmpl + [("N_C", N), ("D_C", D), ("RB_C", rb)],
                  grid=(D // (rb * nsg_b) * 32 * nsg_b, M * R, 1), threadgroup=(32 * nsg_b, 1, 1),
                  output_shapes=[(M * R, D)], output_dtypes=[x.dtype])
    return y

def compile_graph():
    """lazy graphs of every kernel (for patches/sol_layer/compile_check.py)"""
    C, K, N = 8, 2560, 640
    z = lambda *sh, dt=mx.uint32: mx.zeros(sh, dt)
    SH = {"gate_proj": (N, K), "up_proj": (N, K), "down_proj": (K, N)}
    p2 = {p: {"w2": z(C, o, k // 16), "s2": z(C, o, k // 64, dt=mx.bfloat16), "b2": z(C, o, k // 64, dt=mx.bfloat16)}
          for p, (o, k) in SH.items()}
    b3 = {p: {"c": z(C, o, k // 32), "b3": z(C, o, k // 64, dt=mx.bfloat16)} for p, (o, k) in SH.items()}
    b4 = {p: {"l": z(C, o, k // 32), "b": z(C, o, k // 64, dt=mx.bfloat16)} for p, (o, k) in SH.items()}
    for M in (1, 4):
        x, sl, sc = z(M, K, dt=mx.bfloat16), z(M, 10, dt=mx.int32), z(M, 10, dt=mx.bfloat16)
        for plan in ("4422222222", "4322222222", "4433333333"):
            tier_glu4(x, p2, b3, b4, sl, sl, sl, plan=plan)
            tier_glu4(x, p2, b3, b4, sl, sl, sl, plan=plan, scores=sc)
            for which in (1, 2):
                tier_glu4_phase(x, p2, b3, b4, sl, sl, sl, sl, which, plan=plan)
