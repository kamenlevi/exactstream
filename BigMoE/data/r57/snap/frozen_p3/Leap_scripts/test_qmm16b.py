"""L1-15 (Leap, Sat 2026-10-24): qmm16b (v2: NB x 32 columns per simdgroup, prefetch) vs qmm16 (v1): bitwise agreement,
padding independence, ms for the 35B's GDN projection shapes (30 random instances).
Usage: [SpecDecode venv] test_qmm16b.py  -> appends to ~/Leap/data/test_qmm16.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p3/Leap_scripts"))
from qmm16 import qmm16, qmm16b
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
mx.random.seed(0)
say("\nqmm16b (v2) vs qmm16 (v1):")
for name, (N, K) in {"fused in_proj": (12352, 2048), "out_proj": (2048, 4096)}.items():
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(30)]
    mx.eval(Ws)
    fl = 30 * sum(a.nbytes for a in Ws[0]) / 280e9 * 1e3
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    ref = bits(qmm16(x, *Ws[0], ks=1))
    say(f"  {name} (floor {fl:.2f} ms): v1 KS1 {timeit(lambda: [qmm16(x, *W, ks=1) for W in Ws]):.3f} ms")
    for nb in (1, 2):
        if N % (32 * nb):
            continue
        for ks in (1, 2, 4, 8):
            if K % (64 * ks):
                continue
            y = qmm16b(x, *Ws[0], ks=ks, nb=nb)
            eq = int((bits(y) != ref).sum())
            pad = sum(int((bits(qmm16b(x[t:t + 1], *Ws[0], ks=ks, nb=nb)) != bits(y)[t:t + 1]).sum()) for t in (0, 7, 15))
            t2 = timeit(lambda: [qmm16b(x, *W, ks=ks, nb=nb) for W in Ws])
            say(f"    v2 NB {nb} KS {ks}: {t2:.3f} ms ({fl / t2:4.0%} of floor); vs v1 KS1 {eq} differ; padding {pad} differ")
    del Ws
open(os.path.expanduser("~/Leap/data/test_qmm16.txt"), "a").write("\n".join(lines) + "\n")
