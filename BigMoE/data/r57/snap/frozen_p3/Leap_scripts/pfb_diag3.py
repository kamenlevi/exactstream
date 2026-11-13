"""PF-B DIAG 3 (Leap, Sun 08:2x): is the wrong-rows bug in mx.argsort (SwitchGLU's _gather_sort at ~80K ids) or in the sorted
NAX gather kernel? (1) argsort validity at 10K / 40K / 79K ids; (2) the up_proj gather on rows sorted by numpy (stable),
vs the fp32 reference, at 79K rows."""
import numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
E, TOPK, D, F = 512, 10, 2560, 640
rng = np.random.default_rng(11)
pop = 1.0 / np.arange(1, E + 1) ** 0.6; pop = pop[rng.permutation(E)]; pop /= pop.sum()
for T in (1024, 4096, 7932):
    ids = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32).reshape(-1)
    o = np.array(mx.argsort(mx.array(ids)))
    perm_ok = np.array_equal(np.sort(o), np.arange(len(ids)))
    sorted_ok = bool((np.diff(ids[o]) >= 0).all()) if perm_ok else False
    inv = np.array(mx.argsort(mx.array(o)))
    inv_ok = perm_ok and np.array_equal(o[inv], np.arange(len(ids)))
    print(f"argsort at {len(ids)} ids: permutation {'OK' if perm_ok else 'BROKEN'}, sorted {'OK' if sorted_ok else 'NO'}, "
          f"inverse {'OK' if inv_ok else 'BROKEN'}", flush=True)
T = 7932
q = QuantizedSwitchLinear(D, F, E, bias=False, group_size=64, bits=4)
q.weight = mx.array(rng.integers(0, 2 ** 32, (E, F, D // 8), dtype=np.uint32))
q.scales = mx.random.uniform(0.002, 0.02, (E, F, D // 64)).astype(mx.bfloat16)
q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
ids = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32).reshape(-1)
x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
mx.eval(q.weight, q.scales, q.biases, x)
o = np.argsort(ids, kind="stable")
xs = mx.expand_dims(x[mx.array(o // TOPK)], -2)              # [R, 1, D] rows in expert order (numpy's stable sort)
ys = q(xs, mx.array(ids[o]), sorted_indices=True).reshape(-1, F)
wd = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4).astype(mx.float32)
mx.eval(ys, wd)
ysn, xf = np.array(ys.astype(mx.float32)), np.array(x.astype(mx.float32))
bad_e = []
for e in range(E):
    r = np.nonzero(ids[o] == e)[0]
    if len(r) == 0:
        continue
    ref = xf[o[r] // TOPK] @ np.array(wd[e]).T
    if np.abs(ysn[r] - ref).max() > 0.2:
        bad_e.append((e, len(r), int(r[0])))
print(f"sorted NAX gather on numpy-sorted rows, {len(ids)} rows: experts wrong vs fp32 (max |err| > 0.2): {len(bad_e)} of {E}", flush=True)
if bad_e:
    st = np.array([b[2] for b in bad_e]); ln = np.array([b[1] for b in bad_e])
    print(f"  wrong experts: segment start % 64 histogram (first 16 bins of 64): {np.bincount(st % 64, minlength=64)[:16].tolist()}; "
          f"first starts {st[:8].tolist()}, lengths {ln[:8].tolist()}; min start {st.min()}", flush=True)
    good = sorted(set(range(E)) - {b[0] for b in bad_e})
    gs = [int(np.nonzero(ids[o] == e)[0][0]) for e in good if (ids == e).any()]
    print(f"  correct experts: min start {min(gs)}, max start {max(gs)}; wrong experts' max start {st.max()}", flush=True)
