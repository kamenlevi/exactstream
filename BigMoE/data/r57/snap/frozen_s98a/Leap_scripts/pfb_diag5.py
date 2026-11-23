"""PF-B DIAG 5 (Leap, Sun 08:2x): which weights did the sorted NAX gather use for a wrong expert's rows? For wrong rows at 79K
rows, match each against x_row @ W_e' for all 512 experts (fp32); and relabel the experts (a permutation) to see whether
the wrong set follows the expert ids or the row positions."""
import numpy as np, mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
E, TOPK, D, F, T = 512, 10, 2560, 640, 7932
rng = np.random.default_rng(11)
pop = 1.0 / np.arange(1, E + 1) ** 0.6; pop = pop[rng.permutation(E)]; pop /= pop.sum()
q = QuantizedSwitchLinear(D, F, E, bias=False, group_size=64, bits=4)
q.weight = mx.array(rng.integers(0, 2 ** 32, (E, F, D // 8), dtype=np.uint32))
q.scales = mx.random.uniform(0.002, 0.02, (E, F, D // 64)).astype(mx.bfloat16)
q.biases = (-7.5 * q.scales.astype(mx.float32)).astype(mx.bfloat16)
ids0 = np.stack([rng.choice(E, TOPK, replace=False, p=pop) for _ in range(T)]).astype(np.int32).reshape(-1)
x = (mx.random.normal((T, D)) * 0.5).astype(mx.bfloat16)
wd = mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=4).astype(mx.float32)
mx.eval(q.weight, q.scales, q.biases, x, wd)
W, xf = np.array(wd), np.array(x.astype(mx.float32))


def run(ids):
    o = np.argsort(ids, kind="stable")
    y = q(mx.expand_dims(x[mx.array(o // TOPK)], -2), mx.array(ids[o]), sorted_indices=True)
    mx.eval(y)
    yn = np.array(y.reshape(-1, F).astype(mx.float32))
    bad = {}
    for e in range(E):
        r = np.nonzero(ids[o] == e)[0]
        if len(r) and np.abs(yn[r] - xf[o[r] // TOPK] @ W[e].T).max() > 0.2:
            bad[e] = (int(r[0]), int(r[-1]))
    return o, yn, bad


o, yn, bad = run(ids0)
print(f"run 1: {len(bad)} experts wrong; their rows span {min(v[0] for v in bad.values())}..{max(v[1] for v in bad.values())}")
for e in list(bad)[:4]:
    r0 = bad[e][0] + 1                                      # one row inside the wrong segment
    xr = xf[o[r0] // TOPK]
    errs = np.abs(yn[r0][None, :] - np.einsum("efd,d->ef", W, xr)).max(axis=1)
    best = np.argsort(errs)[:3]
    print(f"  expert {e} row {r0}: best-matching weights = experts {best.tolist()} (max |err| {errs[best].round(4).tolist()}); "
          f"own expert err {errs[e]:.3f}")
perm = rng.permutation(E).astype(np.int32)                  # relabel: the same tokens -> experts relation under new ids
_, _, bad2 = run(perm[ids0])
inv = np.argsort(perm)
same_ids = len(set(inv[list(bad2)]) & set(bad)) if bad2 else 0
print(f"relabelled run: {len(bad2)} experts wrong; rows span {min(v[0] for v in bad2.values()) if bad2 else '-'}.."
      f"{max(v[1] for v in bad2.values()) if bad2 else '-'}; {same_ids} of them are run 1's wrong experts (same weights)")
