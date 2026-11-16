"""WEEK98 BET 7: exact building blocks for the fused level (MoonShots, Mon 16 Nov 2026).

Metal device functions that reproduce MLX 0.32.2's stock kernels bit for bit, written so that ANY threadgroup can
recompute them in its prologue (the trick that removes a dependent dispatch): the stock lane layout is emulated as
"virtual lanes" inside whatever threadgroup runs it, so every simd_sum sees the same 32 values in the same lanes.

RMSNorm (stock rms_norm.metal is not shipped; its order, as mtplx's fused_norm.py documents and bit-tests it):
- axis <= 4096 (rms_single_row): ceil(axis / 4) lanes; lane i squares elements 4i..4i+3 in float, in order
  (acc += x * x); one simd_sum per simdgroup; the partials in threadgroup slots (unused slots 0) summed by one more
  simd_sum; inv = precise::rsqrt(acc / axis + eps); out = w * T(float(x) * inv).
- axis > 4096 (rms_looped): 1024 lanes; lane i walks r = 0, 4096, 8192, ... squaring elements r + 4i .. r + 4i + 3;
  then the same two simd_sum levels.
The bit test (fl_bittest.py) is what decides whether this is the stock order; nothing here is assumed exact."""
import mlx.core as mx

RMS_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Stock RMSNorm statistic for one row of AXIS values at x, computed by a threadgroup of NSG simdgroups (any NSG >= 1).
// part: threadgroup scratch of >= 33 floats. Returns inv = precise::rsqrt(sum / AXIS + eps) to every thread.
template <int AXIS, typename T>
METAL_FUNC float fl_rms_inv(const device T* x, float eps, threadgroup float* part, uint lane, uint sg, uint nsg) {
  constexpr int LOOPED = AXIS > 4096 ? 1 : 0;
  constexpr int NV = LOOPED ? 1024 : ((AXIS + 3) / 4 + 31) / 32 * 32;   // stock lanes (threadgroup size)
  constexpr int NVS = NV / 32;                                           // stock simdgroups
  for (uint g = sg; g < uint(NVS); g += nsg) {
    uint vl = g * 32 + lane;
    float acc = 0.0f;
    if (LOOPED) {
      for (uint r = 0; r < uint(AXIS); r += NV * 4) {
        for (int i = 0; i < 4; i++) {
          uint idx = r + vl * 4 + i;
          if (idx < uint(AXIS)) { float xi = x[idx]; acc += xi * xi; }
        }
      }
    } else {
      for (int i = 0; i < 4; i++) {
        uint idx = vl * 4 + i;
        if (idx < uint(AXIS)) { float xi = x[idx]; acc += xi * xi; }
      }
    }
    acc = simd_sum(acc);
    if (lane == 0) { part[g] = acc; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < uint(NVS)) ? part[lane] : 0.0f;
    float tot = simd_sum(v);
    if (lane == 0) { part[32] = metal::precise::rsqrt(tot / float(AXIS) + eps); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);   // part[] may be reused by the caller
  return inv;
}

// the same statistic for a row in any address space (P = a device / threadgroup / constant pointer)
template <int AXIS, typename P>
METAL_FUNC float fl_rms_inv_p(P x, float eps, threadgroup float* part, uint lane, uint sg, uint nsg) {
  constexpr int LOOPED = AXIS > 4096 ? 1 : 0;
  constexpr int NV = LOOPED ? 1024 : ((AXIS + 3) / 4 + 31) / 32 * 32;
  constexpr int NVS = NV / 32;
  for (uint g = sg; g < uint(NVS); g += nsg) {
    uint vl = g * 32 + lane;
    float acc = 0.0f;
    if (LOOPED) {
      for (uint r = 0; r < uint(AXIS); r += NV * 4) {
        for (int i = 0; i < 4; i++) {
          uint idx = r + vl * 4 + i;
          if (idx < uint(AXIS)) { float xi = x[idx]; acc += xi * xi; }
        }
      }
    } else {
      for (int i = 0; i < 4; i++) {
        uint idx = vl * 4 + i;
        if (idx < uint(AXIS)) { float xi = x[idx]; acc += xi * xi; }
      }
    }
    acc = simd_sum(acc);
    if (lane == 0) { part[g] = acc; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = (lane < uint(NVS)) ? part[lane] : 0.0f;
    float tot = simd_sum(v);
    if (lane == 0) { part[32] = metal::precise::rsqrt(tot / float(AXIS) + eps); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float inv = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return inv;
}
"""

# test kernel: one threadgroup per row (NSG simdgroups), writes the normed row (with weight, as nn.RMSNorm does)
_RMS_TEST_SRC = """
  uint row = threadgroup_position_in_grid.x;
  uint lid = thread_position_in_threadgroup.x;
  uint lane = thread_index_in_simdgroup;
  uint sg = simdgroup_index_in_threadgroup;
  uint nsg = threads_per_threadgroup.x / 32;
  threadgroup float part[33];
  const device T* xr = x + size_t(row) * AXIS;
  float inv = fl_rms_inv<AXIS, T>(xr, eps[0], part, lane, sg, nsg);
  for (uint i = lid; i < uint(AXIS); i += threads_per_threadgroup.x) {
    out[size_t(row) * AXIS + i] = w[i] * static_cast<T>(float(xr[i]) * inv);
  }
"""
_rms_test_kernel = mx.fast.metal_kernel(name="fl_rms_test", input_names=["x", "w", "eps"], output_names=["out"],
                                        header=RMS_HEADER, source=_RMS_TEST_SRC)


# ---- elementwise replicas (bf16). Two flavours of the same MLX functor:
# - "pre" = a STANDALONE stock op (precompiled metallib, -fno-fast-math): exp and divide are the precise ones;
# - "jit" = an op inside an mx.compile'd kernel (nn.silu, swiglu): the JIT library resolves unqualified metal::exp to
#   the fast one (mtplx/attention_math.py's finding); our kernels are JIT too, so the functor as written reproduces it.
# Every bf16 op boundary rounds (bfloat arithmetic computes in float, then rounds).
EW_HEADER = """
// stock Sigmoid on bf16, standalone (precise exp, correctly rounded divide)
METAL_FUNC bfloat16_t fl_sigmoid_pre(bfloat16_t x) {
  bfloat16_t ax = static_cast<bfloat16_t>(metal::precise::abs(float(x)));
  bfloat16_t e = static_cast<bfloat16_t>(metal::precise::exp(float(ax)));
  bfloat16_t d = static_cast<bfloat16_t>(1.0f + float(e));
  bfloat16_t y = static_cast<bfloat16_t>(metal::precise::divide(1.0f, float(d)));
  return (x < 0) ? y : static_cast<bfloat16_t>(1.0f - float(y));
}
// stock Sigmoid as the JIT compiles it (MLX's functor text, unqualified exp)
METAL_FUNC bfloat16_t fl_sigmoid_jit(bfloat16_t x) {
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}
// nn.silu (mx.compile'd): x * sigmoid(x), both bf16
METAL_FUNC bfloat16_t fl_silu_jit(bfloat16_t x) { return x * fl_sigmoid_jit(x); }
"""

_EW_SRC = """
  uint i = thread_position_in_grid.x;
  bfloat16_t x = inp[i];
  if (MODE == 0) out[i] = fl_sigmoid_pre(x);
  else if (MODE == 1) out[i] = fl_silu_jit(x);
  else if (MODE == 2) out[i] = fl_sigmoid_jit(x);
"""
_ew_kernel = mx.fast.metal_kernel(name="fl_ew_test", input_names=["inp"], output_names=["out"],
                                  header=EW_HEADER, source=_EW_SRC)


def ew_exact(x, mode):
    """mode 0: standalone mx.sigmoid; 1: nn.silu (compiled); 2: the JIT sigmoid (diagnostic)"""
    return _ew_kernel(inputs=[x], template=[("MODE", mode)], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


def rms_norm_exact(x, w, eps, tg=256):
    """x [rows, axis] (T), w [axis] (T): w * T(x * inv) per row, by the virtual-lane statistic"""
    rows, axis = x.shape
    return _rms_test_kernel(inputs=[x, w, mx.array([eps], mx.float32)], template=[("T", x.dtype), ("AXIS", axis)],
                            grid=(rows * tg, 1, 1), threadgroup=(tg, 1, 1), output_shapes=[x.shape],
                            output_dtypes=[x.dtype])[0]
