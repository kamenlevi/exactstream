"""L1-15 (Leap, Sat 2026-10-24): qmm16c (code MMA) vs qmm16 (v1) and MLX: accuracy vs an fp32 reference, bitwise padding
independence, ms for the 35B's GDN projection shapes (30 random instances).
Usage: [SpecDecode venv] test_qmm16c.py  -> ~/Leap/data/test_qmm16c.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs1/Leap_scripts"))
from qmm16 import qmm16, qmm16c
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
for name, (N, K), cfgs in (("fused in_proj", (12352, 2048), [(1, 1), (1, 2), (2, 1), (2, 2)]),
                           ("out_proj", (2048, 4096), [(2, 1), (4, 2), (8, 1), (1, 1)])):
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(30)]
    mx.eval(Ws)
    fl = 30 * sum(a.nbytes for a in Ws[0]) / 280e9 * 1e3
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(mx.float32) @ mx.dequantize(*Ws[0], group_size=64, bits=4).astype(mx.float32).T
    err = lambda y: float(mx.abs(y.astype(mx.float32) - ref).max())
    say(f"{name}: floor {fl:.2f} ms; max |err| vs fp32: MLX {err(mx.quantized_matmul(x, *Ws[0], transpose=True, group_size=64, bits=4)):.4g}, "
        f"qmm16 {err(qmm16(x, *Ws[0], ks=1)):.4g}; qmm16 KS1 {timeit(lambda: [qmm16(x, *W, ks=1) for W in Ws]):.3f} ms")
    for ks, nt in cfgs:
        if N % (32 * nt) or K % (64 * ks):
            continue
        y = qmm16c(x, *Ws[0], ks=ks, nt=nt)
        pad = sum(int((bits(qmm16c(x[t:t + 1], *Ws[0], ks=ks, nt=nt)) != bits(y)[t:t + 1]).sum()) for t in range(16))
        t = timeit(lambda: [qmm16c(x, *W, ks=ks, nt=nt) for W in Ws])
        say(f"  qmm16c KS {ks} NT {nt}: max |err| {err(y):.4g}; padding {pad} differ; {t:.3f} ms ({fl / t:4.0%} of floor)")
    del Ws
open(os.path.expanduser("~/Leap/data/test_qmm16c.txt"), "w").write("\n".join(lines) + "\n")
