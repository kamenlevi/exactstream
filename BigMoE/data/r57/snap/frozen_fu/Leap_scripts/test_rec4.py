"""L1-15 (Leap, Sat 2026-10-24): recurrence_r4 / recurrence_tg vs mlx_lm's gated_delta kernel, bitwise, and ms for 30
layers (35B shapes: Hk 16, Hv 32, Dk = Dv = 128, fp32 state), random data at T = 1, 2, 8, 16.
Usage: [SpecDecode venv] test_rec4.py  -> appends to ~/Leap/data/test_qmm16.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from gdn35_exact import recurrence_r4, recurrence_tg
from mlx_lm.models import gated_delta as gd
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
def timeit(fn, reps=21):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3
def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)
mx.random.seed(2)
say("\nrecurrence kernels vs mlx_lm's (random data, 30 instances; mismatches y / state; ms for 30 calls):")
for T in (1, 2, 8, 16):
    I = []
    for _ in range(30):
        q = (mx.random.normal((1, T, 16, 128)) * 0.09).astype(mx.bfloat16)
        k = (mx.random.normal((1, T, 16, 128)) * 0.09).astype(mx.bfloat16)
        v = mx.random.normal((1, T, 32, 128)).astype(mx.bfloat16)
        g = mx.random.uniform(0.5, 1.0, (1, T, 32)).astype(mx.float32)
        beta = mx.random.uniform(0.0, 1.0, (1, T, 32)).astype(mx.bfloat16)
        st = mx.random.normal((1, 32, 128, 128)).astype(mx.float32)
        I.append((q, k, v, g, beta, st))
    mx.eval(I)
    res = []
    for nm, f in (("mlx_lm", gd.gated_delta_kernel), ("tg", recurrence_tg), ("r4", recurrence_r4)):
        my = ms = 0
        for a in I[:4]:
            y0, s0 = gd.gated_delta_kernel(*a); y1, s1 = f(*a)
            my += int((bits(y0) != bits(y1)).sum()); ms += int((bits(s0) != bits(s1)).sum())
        res.append(f"{nm} {timeit(lambda: [x for a in I for x in f(*a)]):.3f} ms ({my}/{ms})")
    say(f"  T = {T:2d}: " + "   ".join(res))
open(os.path.expanduser("~/Leap/data/test_qmm16.txt"), "a").write("\n".join(lines) + "\n")
