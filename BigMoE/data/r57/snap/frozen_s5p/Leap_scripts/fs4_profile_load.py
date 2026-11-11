"""fs4 (B6's FATT16-S, KS 4) at N 10,000 keys, 16 rows, the indexer's mask shape (B6's synthetic setup), in a steady loop
for xctrace's Metal GPU Counters (attach after the marker). Leap, Tue 2026-11-03 15:2x. Usage: fs4_profile_load.py MARKER SECONDS"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from fatt16s import fatt16s
marker, secs = sys.argv[1], float(sys.argv[2])
N, R, SC = 10000, 16, 256 ** -0.5
rng = np.random.default_rng(3)
m = np.zeros((R, N), bool)
for r in range(R):
    last = N - R + r
    for b in rng.choice((last + 1) // 4, 512, replace=False):
        m[r, 4 * b:4 * b + 4] = True
    m[r, max(0, last - 63):last + 1] = True
    m[r, last + 1:] = False
mask = mx.array(m)
mx.random.seed(0)
q = (mx.random.normal((1, 24, R, 256)) * 0.8).astype(mx.bfloat16)
k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
mx.eval(q, k, v, mask)
t0 = time.time()
while time.time() - t0 < 1:                                      # warm-up
    mx.eval([fatt16s(q, k, v, SC, mask) for _ in range(8)])
open(marker, "w").write(f"{os.getpid()}\n")
n, t0 = 0, time.time()
while time.time() - t0 < secs:
    mx.eval([fatt16s(q, k, v, SC, mask) for _ in range(8)])
    n += 8
print(f"fs4 at 10K: {n} calls in {time.time() - t0:.1f} s = {1e3 * (time.time() - t0) / n:.3f} ms per call", flush=True)
