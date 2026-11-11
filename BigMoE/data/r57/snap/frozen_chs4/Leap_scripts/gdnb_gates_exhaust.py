"""GDNB-g (Leap, Mon 2026-11-02): can g = compute_g(A_log, a, dt_bias) and beta = sigmoid(b) move into the one-dispatch GDN
body without changing a bit? MLX's functors (binary_ops.h LogAddExp, unary_ops.h Exp / Sigmoid, utils.h log1p) copied
verbatim into a custom kernel, evaluated on EVERY bf16 bit pattern of a (x the 48 heads' real dt_bias / A_log of layer 0)
and of b, against MLX's compute_g (mx.compile'd) and mx.sigmoid. Exhaustive, so a PASS is a proof for these inputs.
Usage: [BigMoE venv] gdnb_gates_exhaust.py -> ~/Leap/data/gdnb_gates_exhaust.txt"""
import json, os, sys
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g

HDR = r"""
struct GbLogAddExp {                                        // mlx binary_ops.h LogAddExp, verbatim (real part)
  template <typename T>
  T operator()(T x, T y) thread {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
};
struct GbExp { template <typename T> T operator()(T x) thread { return metal::precise::exp(x); }; };
struct GbSigmoid {                                          // mlx unary_ops.h Sigmoid; MLX's own build = precise exp
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};
"""
SRC = r"""
    const uint i = thread_position_in_grid.x;               // a / b bit pattern
    const uint h = thread_position_in_grid.y;               // head
    // compute_g: exp(-exp(A_log.astype(fp32)) * softplus(a + dt_bias)), softplus = logaddexp(x, 0)
    const T s = a[i] + dtb[h];                              // Add in bf16
    const T sp = GbLogAddExp()(s, static_cast<T>(0));       // logaddexp(x, 0) in bf16
    const float e = GbExp()(static_cast<float>(alog[h]));   // AsType -> Exp
    const float m = (-e) * static_cast<float>(sp);          // Negative, AsType, Multiply
    g[i * 48 + h] = GbExp()(m);
    if (h == 0) beta[i] = GbSigmoid()(a[i]);                // sigmoid on the same patterns (b's role)
"""
k = mx.fast.metal_kernel(name="gdnb_gates_x", input_names=["a", "dtb", "alog"], output_names=["g", "beta"], source=SRC, header=HDR)
D = os.path.expanduser("~/BigMoE/models/bare")
wmap = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{D}/config.json"))
gdn_layers = [i for i, t in enumerate(cfg["text_config"]["layer_types"]) if t == "linear_attention"]
pat = mx.array(np.arange(65536, dtype=np.uint16)).view(mx.bfloat16)
fin = np.isfinite(np.array(pat.astype(mx.float32)))
gbad = gfin = btot = 0
for L in gdn_layers:
    pre = f"language_model.model.layers.{L}.linear_attn."
    w = mx.load(f"{D}/{wmap[pre + 'A_log']}")
    A_log, dt_bias = w[pre + "A_log"], w[pre + "dt_bias"]
    g_k, b_k = k(inputs=[pat, dt_bias, A_log], template=[("T", mx.bfloat16)], grid=(65536, 48, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(65536 * 48,), (65536,)], output_dtypes=[mx.float32, mx.bfloat16])
    g_r = compute_g(A_log, mx.broadcast_to(pat[:, None], (65536, 48)), dt_bias)
    mx.eval(g_k, b_k, g_r)
    gk, gr = np.array(g_k).reshape(65536, 48).view(np.uint32), np.array(g_r).view(np.uint32)
    gbad += int((gk != gr).sum())
    gfin += int((gk[fin] != gr[fin]).sum())
    if L == gdn_layers[0]:
        br = np.array(mx.sigmoid(pat).view(mx.uint16))
        btot = int((np.array(b_k.view(mx.uint16)) != br).sum())
out = [f"GDNB-g exhaustive: all 65,536 bf16 patterns of a x 48 heads x {len(gdn_layers)} GDN layers' own A_log / dt_bias:",
       f"  g (compute_g replica): {gbad} / {65536 * 48 * len(gdn_layers)} bit patterns differ ({gfin} on finite inputs)",
       f"  beta (Sigmoid with metal::precise::exp): {btot} / 65536 differ"]
print("\n".join(out))
open(os.path.expanduser("~/Leap/data/gdnb_gates_exhaust.txt"), "w").write("\n".join(out) + "\n")
