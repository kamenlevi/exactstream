"""PF-B DIAG 6 (Leap, Sun 08:3x): are the wrong rows exactly zero, does the band move with fresh buffers (mx.clear_cache), and
the device's working-set / wired figures while BigMoE / EvalSuite hold their 45 GB processes."""
import numpy as np, mlx.core as mx, subprocess
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
mx.eval(q.weight, q.scales, q.biases, x)
info = mx.metal.device_info()
print({k: info[k] for k in info if "size" in k or "memory" in k})
o = np.argsort(ids, kind="stable")
xs = mx.expand_dims(x[mx.array(o // TOPK)], -2)
si = mx.array(ids[o])
mx.eval(xs, si)
yref = None
for rep in range(3):
    if rep == 2:
        mx.clear_cache()
    y = q(xs, si, sorted_indices=True).reshape(-1, F)
    mx.eval(y)
    yn = np.array(y.astype(mx.float32))
    zero_rows = np.nonzero(~yn.any(axis=1))[0]
    if yref is None:
        yref = yn
    print(f"  rep {rep}{' (after clear_cache)' if rep == 2 else ''}: all-zero rows {len(zero_rows)}"
          f"{f' spanning {zero_rows.min()}..{zero_rows.max()}' if len(zero_rows) else ''}; == rep 0: {np.array_equal(yn, yref)}",
          flush=True)
yc = mx.concatenate([q(xs[a:a + 10240], si[a:a + 10240], sorted_indices=True).reshape(-1, F) for a in range(0, len(ids), 10240)], 0)
mx.eval(yc)
ycn = np.array(yc.astype(mx.float32))
print(f"  same sorted rows in 10,240-row calls: all-zero rows {int((~ycn.any(axis=1)).sum())}; rows == the big call "
      f"{int((ycn == yref).all(axis=1).sum())} / {len(ids)}", flush=True)
print("  peak", round(mx.get_peak_memory() / 1e9, 2), "GB;", subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
      capture_output=True, text=True).stdout.strip(), "MB iogpu.wired_limit_mb", flush=True)
