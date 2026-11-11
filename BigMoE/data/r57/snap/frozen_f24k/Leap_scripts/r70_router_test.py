"""R-70 FUSED ROUTER screen (Leap, Sun 2026-11-01; shared GPU, rough): router_fused vs the MLX chain it replaces.
1. Bitwise (inds uint32, scores bf16 bits) over input families, both accumulation modes (bf16 = MLX's row_reduce_small
   in the output type; fp32 = the control that should fail somewhere if the sum's rounding matters):
   normal sigma 0.5-8, tie-heavy (logits on a 1/4 or 1/16 grid), all-equal rows, peaked (one or two logits far above),
   and the live shapes [1, R, 512] for R = 1-16.
2. Speed: 48 dependent "layers" per eval (each layer's logits depend on the previous layer's scores, as in the check),
   chain vs fused, at R 16 / 10 / 1; best of 3 medians; per layer and x 48 layers per check.
Usage: [BigMoE venv] r70_router_test.py -> ~/Leap/data/r70_router.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_f24k/Leap_scripts"))
from router_fused import router_fused, router_chain

lines = []
def say(s):
    print(s, flush=True); lines.append(s)
E, K = 512, 10
mx.random.seed(0)

def fam(name, n):
    if name.startswith("normal"):
        return (mx.random.normal((n, E)) * float(name[6:])).astype(mx.bfloat16)
    if name == "grid4":
        return (mx.round(mx.random.normal((n, E)) * 8) / 4).astype(mx.bfloat16)
    if name == "grid16":
        return (mx.round(mx.random.normal((n, E)) * 4) / 16).astype(mx.bfloat16)
    if name == "equal":
        return mx.broadcast_to(mx.random.normal((n, 1)), (n, E)).astype(mx.bfloat16)
    if name == "fewlevels":
        return (mx.random.randint(0, 3, (n, E)).astype(mx.float32) - 1).astype(mx.bfloat16)
    if name == "peaked":
        x = mx.random.normal((n, E))
        hot = mx.random.randint(0, E, (n, 1))
        return mx.where(mx.arange(E)[None] == hot, x + 25.0, x).astype(mx.bfloat16)
    if name == "peaked2":
        x = mx.random.normal((n, E)) * 0.5
        h1, h2 = mx.random.randint(0, E, (n, 1)), mx.random.randint(0, E, (n, 1))
        x = mx.where(mx.arange(E)[None] == h1, x + 12.0, x)
        return mx.where(mx.arange(E)[None] == h2, x + 11.0, x).astype(mx.bfloat16)
    raise ValueError(name)

def cmp(x, acc):
    ia, sa = router_chain(x, K)
    ib, sb = router_fused(x, K, acc_f32=acc)
    mx.eval(ia, sa, ib, sb)
    di = np.array(ia) != np.array(ib)
    ds = np.array(sa.view(mx.uint16)) != np.array(sb.view(mx.uint16))
    rows = di.any(-1) | ds.any(-1)
    return int(rows.sum()), int(di.any(-1).sum()), int(ds.any(-1).sum())

fams = ["normal0.5", "normal1", "normal2", "normal4", "normal8", "grid4", "grid16", "equal", "fewlevels", "peaked",
        "peaked2"]
tot = {False: 0, True: 0}
for name in fams:
    res = {False: [0, 0, 0], True: [0, 0, 0]}
    nrows = 0
    for rep in range(6):
        x = fam(name, 8192)
        mx.eval(x)
        nrows += 8192
        for acc in (False, True):
            r = cmp(x, acc)
            res[acc] = [a + b for a, b in zip(res[acc], r)]
    for acc in (False, True):
        tot[acc] += res[acc][0]
    say(f"{name:10s} {nrows} rows: bf16-acc rows differing {res[False][0]} (inds {res[False][1]}, scores {res[False][2]}); "
        f"fp32-acc control {res[True][0]} (inds {res[True][1]}, scores {res[True][2]})")
live = 0
for R in range(1, 17):
    for name in ("normal2", "grid4", "peaked2"):
        for rep in range(20):
            x = fam(name, R).reshape(1, R, E)
            live += cmp(x, False)[0]
say(f"live shapes [1, R, 512], R 1-16 x 3 families x 20: rows differing {live}")
say(f"BITWISE: {'PASS' if tot[False] == 0 and live == 0 else 'FAIL'} (bf16 accumulation: {tot[False]} rows differ over "
    f"{len(fams) * 6 * 8192} rows; fp32 control: {tot[True]})")

def bench(fn, R, L=48, n=30):
    x0 = (mx.random.normal((1, R, E)) * 2).astype(mx.bfloat16)
    mx.eval(x0)
    def run():
        x = x0
        outs = []
        for _ in range(L):
            inds, scores = fn(x)
            outs.append(inds)
            x = x0 + scores[..., :1] * 0          # the next layer depends on this one (same extra cost in both arms)
        return outs + [x]
    for _ in range(5):
        mx.eval(run())
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(run()); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
for R in (16, 10, 1):
    tc, tf = [], []
    for _ in range(3):
        tc.append(bench(lambda x: router_chain(x, K), R)); tf.append(bench(lambda x: router_fused(x, K), R))
    c, f = 1e3 * min(tc), 1e3 * min(tf)
    say(f"R {R}: 48 dependent layers chain {c:.3f} ms, fused {f:.3f} ms per eval: {1e3 * (c - f) / 48:.1f} us per layer, "
        f"{c - f:+.2f} ms per 48-layer check saved (shared GPU, rough)")
open(os.path.expanduser("~/Leap/data/r70_router.txt"), "w").write("\n".join(lines) + "\n")
