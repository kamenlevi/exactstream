"""L1-15 (Leap, Sat 2026-10-24): qmm16 n-tiles per threadgroup (NT) x K split (KS) sweep for the 35B's GDN projection
shapes (30 random instances), bitwise check that NT never changes the bits (it only regroups simdgroups).
Usage: [SpecDecode venv] test_qmm16_nt.py  -> ~/Leap/data/test_qmm16_nt.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfa/Leap_scripts"))
from qmm16 import qmm16
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
for name, (N, K), cfgs in (("fused in_proj", (12352, 2048), [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4)]),
                           ("out_proj", (2048, 4096), [(2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (1, 4)])):
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(30)]
    mx.eval(Ws)
    fl = 30 * sum(a.nbytes for a in Ws[0]) / 280e9 * 1e3
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    say(f"{name}: floor {fl:.2f} ms (30 instances)")
    for ks, nt in cfgs:
        if N % (32 * nt) or K % (64 * ks) or (ks - 1) * nt > 16:
            continue
        same = int((bits(qmm16(x, *Ws[0], ks=ks, nt=nt)) != bits(qmm16(x, *Ws[0], ks=ks, nt=1))).sum())
        t = timeit(lambda: [qmm16(x, *W, ks=ks, nt=nt) for W in Ws])
        say(f"  KS {ks} NT {nt}: {t:.3f} ms ({fl / t:4.0%} of floor); NT vs NT 1 bits: {same} differ")
    del Ws
open(os.path.expanduser("~/Leap/data/test_qmm16_nt.txt"), "w").write("\n".join(lines) + "\n")
