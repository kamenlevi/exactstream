"""L1-15 (Leap, Sat 2026-10-24): recurrence_gates (beta / g inside, R rows per thread) vs MLX sigmoid + compute_g +
mlx_lm's kernel, bitwise, and ms for 30 calls; real A_log / dt_bias would come from the model, here random (bf16).
Usage: [SpecDecode venv] test_recg.py  -> appends to ~/Leap/data/test_qmm16_nt.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from gdn35_exact import recurrence_gates, recurrence_r4
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
mx.random.seed(4)
say("recurrence_gates vs MLX gates + mlx_lm kernel (30 instances; mismatches y / state; ms for 30 calls incl. gates):")
for T in (1, 8, 16):
    I = []
    for _ in range(30):
        q = (mx.random.normal((1, T, 16, 128)) * 0.09).astype(mx.bfloat16)
        k = (mx.random.normal((1, T, 16, 128)) * 0.09).astype(mx.bfloat16)
        v = mx.random.normal((1, T, 32, 128)).astype(mx.bfloat16)
        p = (mx.random.normal((T, 12352)) * 3).astype(mx.bfloat16)
        A_log = mx.random.uniform(-2, 2, (32,)).astype(mx.bfloat16)
        dtb = (mx.random.normal((32,)) * 2).astype(mx.bfloat16)
        st = mx.random.normal((1, 32, 128, 128)).astype(mx.float32)
        I.append((q, k, v, p, A_log, dtb, st))
    mx.eval(I)
    def ref(q, k, v, p, A_log, dtb, st):
        beta = mx.sigmoid(p[:, 12288:12320])[None]; g = gd.compute_g(A_log, p[:, 12320:12352], dtb)[None]
        return gd.gated_delta_kernel(q, k, v, g, beta, st)
    def r4(q, k, v, p, A_log, dtb, st):
        beta = mx.sigmoid(p[:, 12288:12320])[None]; g = gd.compute_g(A_log, p[:, 12320:12352], dtb)[None]
        return recurrence_r4(q, k, v, g, beta, st)
    res = [f"MLX gates + mlx_lm {timeit(lambda: [x for a in I for x in ref(*a)]):.3f} ms",
           f"MLX gates + r4 {timeit(lambda: [x for a in I for x in r4(*a)]):.3f} ms"]
    for r in (2, 4, 8):
        my = ms = 0
        for a in I[:6]:
            y0, s0 = ref(*a); y1, s1 = recurrence_gates(*a, r=r)
            my += int((bits(y0) != bits(y1)).sum()); ms += int((bits(s0) != bits(s1)).sum())
        res.append(f"gates R{r} {timeit(lambda: [x for a in I for x in recurrence_gates(*a, r=r)]):.3f} ms ({my}/{ms})")
    say(f"  T = {T:2d}: " + ";  ".join(res))
open(os.path.expanduser("~/Leap/data/test_qmm16_nt.txt"), "a").write("\n".join(lines) + "\n")
