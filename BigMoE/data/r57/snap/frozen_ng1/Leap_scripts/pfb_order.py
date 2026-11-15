"""PF-B ORDER (Leap, Sun 08:3x): does the unaligned big sorted gather zero rows only when it runs before any aligned call
(pipeline cache), or in a shared eval? up_proj alone, 79,320 rows (% 64 = 24), in separate processes per order (argv)."""
import sys, numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
E, TOPK, D, F, T = 512, 10, 2560, 640, 7932
rng = np.random.default_rng(11)
q = QuantizedSwitchLinear(D, F, E, bias=False, group_size=64, bits=4)
q.weight = mx.array(rng.integers(0, 2 ** 32, (E, F, D // 8), dtype=np.uint32))
q.scales = mx.random.uniform(0.002, 0.02, (E, F, D // 64)).astype(mx.bfloat16)
q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
ids = np.sort(rng.integers(0, E, T * TOPK)).astype(np.int32)
xs = (mx.random.normal((T * TOPK, 1, D)) * 0.5).astype(mx.bfloat16)
si = mx.array(ids)
mx.eval(q.weight, q.scales, q.biases, xs, si)
zr = lambda y: int((~mx.any(y.reshape(-1, F) != 0, axis=-1)).sum().item())
out = []
for step in sys.argv[1].split(","):
    if step == "U":                                          # unaligned big call
        y = q(xs, si, sorted_indices=True); mx.eval(y); out.append(f"U:{zr(y)}")
    elif step == "A":                                        # aligned call (first 79,296 rows = 1,239 x 64)
        y = q(xs[:79296], si[:79296], sorted_indices=True); mx.eval(y); out.append(f"A:{zr(y)}")
    elif step == "S":                                        # small unaligned call (10,000 rows)
        y = q(xs[:10000], si[:10000], sorted_indices=True); mx.eval(y); out.append(f"S:{zr(y)}")
    elif step == "UA":                                       # unaligned + aligned in one eval
        y1 = q(xs, si, sorted_indices=True); y2 = q(xs[:79296], si[:79296], sorted_indices=True); mx.eval(y1, y2)
        out.append(f"UA:{zr(y1)}/{zr(y2)}")
print(sys.argv[1], "->", " ".join(out), flush=True)
