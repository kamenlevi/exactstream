"""PF-B DIAG 2 (Leap, Sun 08:2x): size of the per-expert differences (1 bf16 ulp = rounding order, O(1) = wrong rows) and which
side matches an fp32 reference (dequantized expert weights, x @ W^T in fp32) on the up_proj gather alone."""
import numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, _gather_sort
E, TOPK, D, F, T = 512, 10, 2560, 640, 7932
rng = np.random.default_rng(11)
q = QuantizedSwitchLinear(D, F, E, bias=False, group_size=64, bits=4)
q.weight = mx.array(rng.integers(0, 2 ** 32, (E, F, D // 8), dtype=np.uint32))
q.scales = mx.random.uniform(0.002, 0.02, (E, F, D // 64)).astype(mx.bfloat16)
q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
pop = 1.0 / np.arange(1, E + 1) ** 0.6; pop = pop[rng.permutation(E)]; pop /= pop.sum()
ids = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32)
x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
mx.eval(q.weight, q.scales, q.biases, x)
xe = mx.expand_dims(x, (-2, -3))
outs = []
for a in range(0, T, 1024):
    xs, idx, inv = _gather_sort(xe[a:a + 1024], mx.array(ids[a:a + 1024]))
    outs.append(q(xs, idx, sorted_indices=True)[inv].reshape(-1, F))
yc = mx.concatenate(outs, axis=0)
xs, idx, inv = _gather_sort(xe, mx.array(ids))
yo = q(xs, idx, sorted_indices=True)[inv].reshape(-1, F)
yu = mx.concatenate([q(xe[a:a + 64], mx.array(ids[a:a + 64]), sorted_indices=False).reshape(-1, F) for a in range(0, T, 64)], 0)
mx.eval(yc, yo, yu)
e_flat = ids.reshape(-1)
bad = ~np.array(mx.all(yc == yo, axis=-1))
print(f"up_proj alone, T {T}: rows differing chunked vs one call {int(bad.sum())} / {len(bad)}")
a, b = np.array(yc.astype(mx.float32)), np.array(yo.astype(mx.float32))
d = np.abs(a - b)[bad]
rel = d / (np.abs(a[bad]) + 1e-6)
print(f"  differing rows: elements differing {int((d > 0).sum())} of {d.size}; max |diff| {d.max():.4g}, median |diff| of nonzero "
      f"{np.median(d[d > 0]):.4g}; median |value| {np.median(np.abs(a[bad])):.4g}; max rel {np.quantile(rel, 0.999):.3g} (99.9 %)")
# fp32 reference for 6 differing rows and 6 equal rows
wd = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4).astype(mx.float32)   # [E, F, D]
xf = np.array(x.astype(mx.float32))
for tag, rows in (("differing", np.nonzero(bad)[0][:6]), ("equal", np.nonzero(~bad)[0][:6])):
    for r in rows:
        t, e = r // TOPK, e_flat[r]
        ref = np.array(wd[int(e)]) @ xf[t]
        ec, eo, eu = (np.abs(v[r] - ref).max() for v in (a, b, np.array(yu.astype(mx.float32))))
        print(f"  {tag} row {r} (expert {e}): max |err| vs fp32  chunked {ec:.4g}  one-call {eo:.4g}  unsorted-gather {eu:.4g}")
