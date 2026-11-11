"""L1-29 test (Leap, Sun 2026-10-25): bmm16 (bf16 16-row tensor-unit matmul) on the HC mixer shapes of a 16-row check.
Per shape: (1) config sweep over (ks, nt), median of 7; (2) fair interleaved A/B (12 rounds) of bmm16's best config vs
MLX's best batch-invariant 16-row form (a: x @ W.T, b: (W @ x.T).T, c: x @ contiguous(W.T)); (3) row-independence
(rows 1 / 2 / 5 / 8 / 12 alone vs inside the 16-row call, bitwise) and max |err| vs fp32 for both.
Usage: [BigMoE venv] test_bmm30.py -> ~/Leap/data/bmm30.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5/Leap_scripts"))
from bmm30 import bmm16

lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def t_once(fn, inner=7):
    mx.eval(fn())
    ts = []
    for _ in range(inner):
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[inner // 2] * 1e3

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

mx.random.seed(30)
say("L1-29: bf16 16-row tensor-unit matmul for the HC mixers (per 16-row check)")
for name, N, K, cnt, form in (("HC mix down", 320, 10240, 97, "b"), ("HC mix up", 10240, 320, 97, "c"),
                              ("ple key", 10240, 2560, 1, "b")):
    Ws = [(mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16) for _ in range(cnt)]
    WT = [mx.contiguous(W.T) for W in Ws]
    mx.eval(Ws, WT)
    x = (mx.random.normal((32, K)) * 0.5).astype(mx.bfloat16)
    floor = cnt * Ws[0].nbytes / 280e9 * 1e3
    mlx = {"a": lambda: [x[:16] @ W.T for W in Ws], "b": lambda: [(W @ x[:16].T).T for W in Ws],
           "c": lambda: [x[:16] @ W for W in WT]}[form]
    cfgs = []
    for ks in (1, 2, 4, 5, 8, 10, 16, 20, 32):
        for nt in (1, 2):
            if N % (32 * nt) or K % (64 * ks) or 32 * ks * nt > 1024 or (ks - 1) * nt * 2048 > 32768:
                continue
            cfgs.append((t_once(lambda: [bmm16(x[:16], W, ks, nt) for W in Ws]), (ks, nt)))
    cfgs.sort()
    ks, nt = cfgs[0][1]
    rng = np.random.default_rng(30)
    res = {"mlx": [], "bmm16": []}
    for _ in range(12):
        for k in rng.permutation(["mlx", "bmm16"]):
            res[k].append(t_once(mlx if k == "mlx" else (lambda: [bmm16(x[:16], W, ks, nt) for W in Ws])))
    r = np.array(res["mlx"]) / np.array(res["bmm16"])
    ref = x[:16].astype(mx.float32) @ Ws[0].astype(mx.float32).T
    e_m = float(mx.abs((x[:16] @ Ws[0].T).astype(mx.float32) - ref).max())
    y16 = bmm16(x[:16], Ws[0], ks, nt)
    e_b = float(mx.abs(y16.astype(mx.float32) - ref).max())
    full = bits(y16)
    inv = all(np.array_equal(bits(bmm16(mx.concatenate([x[:M], x[16:32 - M]]), Ws[0], ks, nt))[:M], full[:M])
              for M in (1, 2, 5, 8, 12))
    say(f"  {name:11s} {N}x{K} x{cnt}: floor {floor:.2f} ms; MLX form {form} {np.median(res['mlx']):.2f} "
        f"({floor / np.median(res['mlx']):.0%}); bmm16 {cfgs[0][1]} {np.median(res['bmm16']):.2f} "
        f"({floor / np.median(res['bmm16']):.0%}); paired x{np.median(r):.2f} (IQR {np.percentile(r, 25):.2f}-"
        f"{np.percentile(r, 75):.2f}); max |err| vs fp32 MLX {e_m:.3g}, bmm16 {e_b:.3g}; invariant {inv}; next configs "
        + ", ".join(f"{t:.2f}{c}" for t, c in cfgs[1:4]))
    del Ws, WT
open(os.path.expanduser("~/Leap/data/bmm30.txt"), "w").write("\n".join(lines) + "\n")
