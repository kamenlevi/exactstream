"""Round 27: a bit-identical, higher-bandwidth replacement for MLX's affine qmv_wide (the kernel MLX runs for 2 <= M <= 5
rows, i.e. the 4-row MTP verify pass) on the 125B's 4-bit projections. Profile (round 26c): those matrices (9-17 MB)
run at 114-168 GB/s against a 310-326 GB/s read ceiling. MLX's qmv_wide_impl gives each output row 8 lanes; a lane walks
its K/(64*8) quant groups in a loop with a runtime bound, so it has ~32 B of weight loads in flight at a time.
qmv_hb keeps MLX's geometry and arithmetic exactly (k_lanes 8, 4 rows per simdgroup, 2 simdgroups, the lane -> group
assignment, the dequantize expressions, the 8-value sub-chunk sums in order, the shuffle ladder) and only changes when
the bytes are loaded: K is a compile-time constant and every group of the lane (weights with 16-byte loads, scale, bias)
is loaded before any arithmetic, so all of a lane's loads are in flight at once. Every output is MLX's arithmetic on the
same values in the same order -> bit-identical to mx.quantized_matmul at M = 2..5.
MLX's headers (utils, quantized.h) are inlined for bfloat16_t and friends, as PrefillKernels' fused_qmv.py does."""
import os, re
from functools import lru_cache
import mlx.core as mx

_INC = os.path.join(os.path.dirname(mx.__file__), "include")


def _inline(rel, seen):
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in open(os.path.join(_INC, rel)):
        m = re.match(r'\s*#include\s+"(mlx/backend/metal/kernels/[^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() == "#pragma once":
            continue
        else:
            out.append(line)
    return "".join(out)


_TEMPLATE = """
template <int GB>
inline uint hb_byte(const thread uint4 (&wv)[GB / 16], int k) {
  const uint4 v = wv[k >> 4];
  const int c = (k >> 2) & 3;
  const uint word = c == 0 ? v.x : (c == 1 ? v.y : (c == 2 ? v.z : v.w));
  return (word >> (8 * (k & 3))) & 0xffu;
}

template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, int K>
METAL_FUNC void qmv_wide_hb_impl(
    const device uint32_t* w, const device T* scales, const device T* biases, const device T* x, device T* y,
    const int out_vec_size, const int M, uint3 tid, uint simd_gid, uint simd_lid) {
  static_assert(bits == 4, "qmv_hb: 4-bit only");
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int sub = 8;
  constexpr int NGR = K / group_size;                         // groups per row
  constexpr int NGL = (NGR + k_lanes - 1) / k_lanes;          // groups per lane
  constexpr int GB = group_size * bits / 8;                   // bytes per group
  typedef float U;

  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;
  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) + results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;
  const int row = min(out_row, out_vec_size - 1);
  const device uint8_t* wrow = (const device uint8_t*)w + (size_t)row * (K * bits / 8);
  const device T* srow = scales + (size_t)row * NGR;
  const device T* brow = biases + (size_t)row * NGR;
  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) {
    xv[v] = x + min(vec0 + v, M - 1) * K;
  }

  // all of this lane's groups first: weights (16-byte loads), scales, biases
  uint4 wv[NGL][GB / 16];
  U scv[NGL], biv[NGL];
#pragma unroll
  for (int i = 0; i < NGL; i++) {
    const int g = k_lane + i * k_lanes;
    if (g < NGR) {
      const device uint4* src = (const device uint4*)(wrow + g * GB);
#pragma unroll
      for (int j = 0; j < GB / 16; j++) wv[i][j] = src[j];
      scv[i] = srow[g];
      biv[i] = brow[g];
    }
  }

  U result[vecs_per_tg] = {0};
#pragma unroll
  for (int i = 0; i < NGL; i++) {
    const int g = k_lane + i * k_lanes;
    if (g < NGR) {
      U scale = scv[i];
      U bias = biv[i];
      const float s = float(scale);
      const float b = float(bias);
      float sc[2] = {s, s / 16.0f};
#pragma unroll
      for (int scn = 0; scn < group_size / sub; scn++) {
        const int k0 = g * group_size + scn * sub;
        U w_dq[sub];
#pragma unroll
        for (int ii = 0; ii < (sub / 2); ii++) {            // MLX's dequantize<U, 8, 4> on the same bytes
          const uint wb = hb_byte<GB>(wv[i], scn * (sub * bits / 8) + ii);
          w_dq[2 * ii] = static_cast<U>(sc[0] * (wb & 0x0f) + b);
          w_dq[2 * ii + 1] = static_cast<U>(sc[1] * (wb & 0xf0) + b);
        }
#pragma unroll
        for (int v = 0; v < vecs_per_tg; v++) {
          const device T* xc = xv[v] + k0;
          U acc = 0;
#pragma unroll
          for (int ii = 0; ii < sub; ii++) {
            acc += static_cast<U>(xc[ii]) * w_dq[ii];
          }
          result[v] += acc;
        }
      }
    }
  }

  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) { result[v] += simd_shuffle_down(result[v], 16); }
    if constexpr (k_lanes >= 16) { result[v] += simd_shuffle_down(result[v], 8); }
    if constexpr (k_lanes >= 8) { result[v] += simd_shuffle_down(result[v], 4); }
    if constexpr (k_lanes >= 4) { result[v] += simd_shuffle_down(result[v], 2); }
    if constexpr (k_lanes >= 2) { result[v] += simd_shuffle_down(result[v], 1); }
  }
  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) {
        y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]);
      }
    }
  }
}

// MLX's qmv_wide_impl verbatim except: in_vec_size is the compile-time K, and NSG simdgroups per threadgroup
template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, int K, int NSG>
METAL_FUNC void qmv_wide_ck_impl(
    const device uint32_t* w, const device T* scales, const device T* biases, const device T* x, device T* y,
    const int out_vec_size, const int M, uint3 tid, uint simd_gid, uint simd_lid) {
  constexpr int num_simdgroups = NSG;
  constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
  constexpr int sub = 8;
  constexpr int in_vec_size = K;
  typedef float U;
  const short k_lane = simd_lid % k_lanes;
  const short sg_row = simd_lid / k_lanes;
  const int out_row = tid.y * (results_per_simdgroup * num_simdgroups) + results_per_simdgroup * simd_gid + sg_row;
  const int vec0 = tid.x * vecs_per_tg;
  const int row = min(out_row, out_vec_size - 1);
  constexpr int in_vec_size_w = in_vec_size * bits / 8;
  constexpr int in_vec_size_g = in_vec_size / group_size;
  const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
  const device T* srow = scales + row * in_vec_size_g;
  const device T* brow = biases + row * in_vec_size_g;
  const device T* xv[vecs_per_tg];
  for (int v = 0; v < vecs_per_tg; v++) { xv[v] = x + min(vec0 + v, M - 1) * in_vec_size; }
  U result[vecs_per_tg] = {0};
  for (int g = k_lane; g < in_vec_size_g; g += k_lanes) {
    U scale = srow[g];
    U bias = brow[g];
#pragma unroll
    for (int sc = 0; sc < group_size / sub; sc++) {
      const int k0 = g * group_size + sc * sub;
      const device uint8_t* wc = wrow + k0 * bits / 8;
      U w_dq[sub];
      dequantize<U, sub, bits>(wc, scale, bias, w_dq);
#pragma unroll
      for (int v = 0; v < vecs_per_tg; v++) {
        const device T* xc = xv[v] + k0;
        U acc = 0;
#pragma unroll
        for (int i = 0; i < sub; i++) { acc += static_cast<U>(xc[i]) * w_dq[i]; }
        result[v] += acc;
      }
    }
  }
  for (int v = 0; v < vecs_per_tg; v++) {
    if constexpr (k_lanes >= 32) { result[v] += simd_shuffle_down(result[v], 16); }
    if constexpr (k_lanes >= 16) { result[v] += simd_shuffle_down(result[v], 8); }
    if constexpr (k_lanes >= 8) { result[v] += simd_shuffle_down(result[v], 4); }
    if constexpr (k_lanes >= 4) { result[v] += simd_shuffle_down(result[v], 2); }
    if constexpr (k_lanes >= 2) { result[v] += simd_shuffle_down(result[v], 1); }
  }
  if (k_lane == 0 && out_row < out_vec_size) {
    for (int v = 0; v < vecs_per_tg; v++) {
      if (vec0 + v < M) { y[(vec0 + v) * out_vec_size + out_row] = static_cast<T>(result[v]); }
    }
  }
}
"""


@lru_cache(None)
def _header():
    seen = set()
    return "".join(_inline(f"mlx/backend/metal/kernels/{h}", seen) for h in ("steel/gemm/gemm.h", "quantized_utils.h", "quantized.h")) + _TEMPLATE


@lru_cache(None)
def _kernel(wide_impl):
    tail = "(w, s, b, x, y, Nc, Mc, uint3(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y, 0), " \
           "simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
    if wide_impl == "hb":
        src = "qmv_wide_hb_impl<bfloat16_t, 64, 4, VPT, 8, KC>" + tail
    elif wide_impl.startswith("ck"):                              # ck2 / ck4: NSG simdgroups per threadgroup
        src = "qmv_wide_ck_impl<bfloat16_t, 64, 4, VPT, 8, KC, NSG>" + tail
    else:                                                         # MLX's own impl (in_vec_size by reference)
        src = "qmv_wide_impl<bfloat16_t, 64, 4, VPT, 8>(w, s, b, x, y, Kc, Nc, Mc, uint3(threadgroup_position_in_grid.x, " \
              "threadgroup_position_in_grid.y, 0), simdgroup_index_in_threadgroup, thread_index_in_simdgroup);"
    return mx.fast.metal_kernel(name=f"qmv_{wide_impl}", input_names=["x", "w", "s", "b", "Mc", "Nc", "Kc"],
                                output_names=["y"], header=_header(), source="  " + src, ensure_row_contiguous=True)


def qmv(x, w, s, b, impl="hb"):
    """x [M, K] bf16 (2 <= M <= 5); w [N, K/8] uint32; s, b [N, K/64] bf16 -> [M, N] bf16 = x @ dequant(w).T."""
    M, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[0])
    assert 2 <= M <= 5 and K % 512 == 0
    nsg = 4 if impl == "ck4" else 2
    rows_tg = 4 * nsg                                             # 32 / k_lanes rows per simdgroup x NSG
    (y,) = _kernel(impl)(inputs=[x, w, s, b, M, N, K], template=[("VPT", M), ("KC", K), ("NSG", nsg)],
                         grid=(32, nsg * ((N + rows_tg - 1) // rows_tg), 1), threadgroup=(32, nsg, 1),
                         output_shapes=[(M, N)], output_dtypes=[x.dtype])
    return y
