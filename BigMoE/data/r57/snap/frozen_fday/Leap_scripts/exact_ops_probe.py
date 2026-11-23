"""L1-15 (Leap, Sat 2026-10-24): exhaustive bf16 check of kernel-side copies of MLX 0.32.4's elementwise ops, over all
65,536 bf16 bit patterns (NaNs excluded): sigmoid (fast vs precise exp), silu, softplus (logaddexp(x, 0)) and mlx_lm's
compute_g (exp(-exp(A_log) * softplus(a + dt_bias)), 32 heads of random A_log / dt_bias). 0 mismatches = the copy is
bitwise MLX's op for every input.
Usage: [SpecDecode venv] exact_ops_probe.py  -> ~/Leap/data/exact_ops_probe.txt"""
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import gated_delta as gd
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

_HDR = r"""
struct SigFast {
  template <typename T> T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct SigPrecise {                      // MLX 0.32.4 unary_ops.h Sigmoid
  template <typename T> T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
struct LAE {                             // MLX 0.32.4 binary_ops.h LogAddExp
  template <typename T> T operator()(T x, T y) thread {
    if (metal::isnan(x) || metal::isnan(y)) return metal::numeric_limits<T>::quiet_NaN();
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf) ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
  }
};
"""
_SRC = r"""
    const uint i = thread_position_in_grid.x;
    if (i >= (uint)NX) return;
    const T x = xs[i];
    sig_f[i] = SigFast()(x);
    sig_p[i] = SigPrecise()(x);
    silu_p[i] = x * SigPrecise()(x);
    silu_f[i] = x * SigFast()(x);
    sp[i] = LAE()(x, T(0));
    for (int h = 0; h < 32; ++h) {                          // compute_g for a = x, head h
        const T apd = x + dtb[h];
        const T s = LAE()(apd, T(0));
        const float ea = metal::precise::exp(static_cast<float>(alog[h]));
        g[i * 32 + h] = metal::precise::exp(-ea * static_cast<float>(s));
    }
"""
k = mx.fast.metal_kernel(name="leap_exact_ops_probe", input_names=["xs", "alog", "dtb"],
                         output_names=["sig_f", "sig_p", "silu_p", "silu_f", "sp", "g"], source=_SRC, header=_HDR)
u = np.arange(65536, dtype=np.uint16)
f = (u.astype(np.uint32) << 16).view(np.float32)
keep = ~np.isnan(f)
xs = mx.array(u[keep]).view(mx.bfloat16)
NX = xs.size
mx.random.seed(3)
alog = (mx.random.uniform(-3, 3, (32,))).astype(mx.bfloat16)
dtb = (mx.random.normal((32,)) * 2).astype(mx.bfloat16)
outs = k(inputs=[xs, alog, dtb], template=[("T", mx.bfloat16), ("NX", NX)], grid=((NX + 255) // 256 * 256, 1, 1),
         threadgroup=(256, 1, 1), output_shapes=[(NX,)] * 5 + [(NX * 32,)],
         output_dtypes=[mx.bfloat16] * 5 + [mx.float32])
sig_f, sig_p, silu_p, silu_f, sp, g = outs
def mism(a, b):
    a = np.array(a.astype(mx.float32)).reshape(-1); b = np.array(b.astype(mx.float32)).reshape(-1)
    same = (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))
    return int((~same).sum())
say(f"exhaustive over {NX} bf16 values (mismatches vs MLX 0.32.4's op):")
say(f"  sigmoid, fast exp:     {mism(sig_f, mx.sigmoid(xs))}")
say(f"  sigmoid, precise exp:  {mism(sig_p, mx.sigmoid(xs))}")
say(f"  silu (nn.silu), fast:     {mism(silu_f, nn.silu(xs))}")
say(f"  silu (nn.silu), precise:  {mism(silu_p, nn.silu(xs))}")
say(f"  softplus (nn.softplus = logaddexp(x, 0)): {mism(sp, nn.softplus(xs))}")
g_ref = gd.compute_g(alog, mx.broadcast_to(xs[:, None], (NX, 32)), dtb)
say(f"  compute_g (32 heads x {NX}, dtype {g_ref.dtype}): {mism(g, g_ref)}")
open(os.path.expanduser("~/Leap/data/exact_ops_probe.txt"), "w").write("\n".join(lines) + "\n")
