"""L1-23 leftovers (Leap, Sat 2026-10-24): the bf16 shapes of a 16-row check. MLX's x @ W.T at 16 rows is padding-
independent but slow for some shapes (block_inject 4 x 10240: 4.06 ms per check vs 0.33 at 1 row; HC mix down 320 x
10240: 5.83 vs a 2.27 floor). Try other formulations that are still one fixed call per shape at 16 (padded) rows:
  a  x @ W.T            (L1-23's choice)
  b  (W @ x.T).T        (M = N_w, N = 16: a different tile config)
  c  mx.matmul with W.T materialized once ([K, N] contiguous)
  d  for N = 4: zero-padded W to 32 rows, then a
Each: ms per check (count per check, random weights, cold) and bitwise padding independence (real rows 1/2/5/8/12 padded
with zeros or random rows vs the all-real 16-row call).
Usage: [BigMoE venv] bf23.py -> ~/Leap/data/bf23.txt"""
import os, time
import numpy as np
import mlx.core as mx

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

SHAPES = [("HC mix up", 10240, 320, 97), ("HC mix down", 320, 10240, 97), ("ple key", 10240, 2560, 1),
          ("block_inject", 4, 10240, 96)]
mx.random.seed(23)
for name, N, K, cnt in SHAPES:
    Ws = [(mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16) for _ in range(cnt)]
    WT = [mx.contiguous(W.T) for W in Ws]
    WP = [mx.concatenate([W, mx.zeros((32 - N, K), W.dtype)]) for W in Ws] if N < 32 else None
    mx.eval(Ws, WT, *(WP or []))
    X = (mx.random.normal((32, K)) * 0.5).astype(mx.bfloat16)
    floor = cnt * Ws[0].nbytes / 280e9 * 1e3
    forms = {"a x@W.T": lambda x, i: x @ Ws[i].T, "b (W@x.T).T": lambda x, i: (Ws[i] @ x.T).T,
             "c x@WT": lambda x, i: x @ WT[i]}
    if WP is not None:
        forms["d padW x@W.T"] = lambda x, i: (x @ WP[i].T)[:, :N]
    out = []
    for fname, f in forms.items():
        t = timeit(lambda: [f(X[:16], i) for i in range(cnt)])
        ref = bits(f(X[:16], 0))
        ok = True
        for M in (1, 2, 5, 8, 12):
            for pad in (mx.zeros((16 - M, K), mx.bfloat16), X[16:32 - M]):
                if not np.array_equal(bits(f(mx.concatenate([X[:M], pad]), 0))[:M], ref[:M]):
                    ok = False
        same_a = np.array_equal(ref, bits(forms["a x@W.T"](X[:16], 0)))
        out.append(f"{fname}: {t:.2f} ms ({'invariant' if ok else 'NOT invariant'}{', = a' if same_a else ''})")
    say(f"{name:13s} {N:6d} x {K:5d} x {cnt:3d}: floor {floor:.2f} ms | " + " | ".join(out))
open(os.path.expanduser("~/Leap/data/bf23.txt"), "w").write("\n".join(lines) + "\n")
