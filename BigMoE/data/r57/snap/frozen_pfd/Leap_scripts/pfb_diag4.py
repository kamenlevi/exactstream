"""PF-B DIAG 4 (Leap, Sun 08:3x): race or kernel? The sorted NAX gather at 79K rows on (A) x rows gathered in the same graph
(as SwitchGLU), (B) x rows evaluated first (own command buffer), (C) a contiguous copy evaluated first; experts wrong vs fp32."""
import numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
E, TOPK, D, F, T = 512, 10, 2560, 640, 7932
rng = np.random.default_rng(11)
pop = 1.0 / np.arange(1, E + 1) ** 0.6; pop = pop[rng.permutation(E)]; pop /= pop.sum()
q = QuantizedSwitchLinear(D, F, E, bias=False, group_size=64, bits=4)
q.weight = mx.array(rng.integers(0, 2 ** 32, (E, F, D // 8), dtype=np.uint32))
q.scales = mx.random.uniform(0.002, 0.02, (E, F, D // 64)).astype(mx.bfloat16)
q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
ids = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32).reshape(-1)
x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
wd = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4).astype(mx.float32)
mx.eval(q.weight, q.scales, q.biases, x, wd)
o = np.argsort(ids, kind="stable")
si = mx.array(ids[o])
xf, W = np.array(x.astype(mx.float32)), np.array(wd)
segs = {e: np.nonzero(ids[o] == e)[0] for e in range(E)}
ref = {e: xf[o[r] // TOPK] @ W[e].T for e, r in segs.items() if len(r)}


def wrong(y):
    yn = np.array(y.reshape(-1, F).astype(mx.float32))
    return sum(1 for e, r in segs.items() if len(r) and np.abs(yn[r] - ref[e]).max() > 0.2)


for tag in ("A same graph", "B x evaluated first", "C contiguous x evaluated first", "A again"):
    xs = mx.expand_dims(x[mx.array(o // TOPK)], -2)
    if tag.startswith("B"):
        mx.eval(xs)
    if tag.startswith("C"):
        xs = mx.contiguous(xs); mx.eval(xs)
    y = q(xs, si, sorted_indices=True)
    mx.eval(y)
    print(f"  {tag:32s}: experts wrong {wrong(y)} / {E}", flush=True)
