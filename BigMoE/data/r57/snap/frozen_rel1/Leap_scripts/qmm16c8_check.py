"""BP-5 check (Leap, Wed 2026-11-04 16:2x): qmm16c8.qmm16c8 (8-bit g64, 16-row code MMA).
 (a) accuracy: error vs an fp32 reference of the same dequantized weights, against MLX's own 8-bit quantized_matmul's
     error; row invariance (1 / 8 rows == the 16-row call's first rows, bitwise);
 (b) speed at the drafter's shapes (BP-2a's set), split-K ks 1 / 2 / 4 / 8 where it divides, vs bf16 x @ W.T, MLX's
     8-bit quantized_matmul and qmm16c (4-bit): us per call (20 independent calls per eval, median of 9 rounds).
usage: [BigMoE venv] qmm16c8_check.py -> ~/Leap/data/qmm16c8_check.txt (GPU, MEM ~1.5 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
from qmm16c8 import qmm16c8
from qmm16 import qmm16c

OUT = os.path.expanduser("~/Leap/data/qmm16c8_check.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nBP-5 qmm16c8 check {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
f32 = mx.float32
rel = lambda a, b: float(mx.sqrt(mx.mean((a.astype(f32) - b.astype(f32)) ** 2)).item() / mx.sqrt(mx.mean(b.astype(f32) ** 2)).item())
eqb = lambda a, b: bool(a.shape == b.shape and mx.all(a.view(mx.uint16) == b.view(mx.uint16)).item())
mx.random.seed(4)
ok = True
for N, K in ((4096, 2560), (1024, 2560), (2560, 7680)):
    W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    w8, s8, b8 = mx.quantize(W, group_size=64, bits=8)
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    ref = x.astype(f32) @ mx.dequantize(w8, s8, b8, group_size=64, bits=8).astype(f32).T
    y = qmm16c8(x, w8, s8, b8)
    ym = mx.quantized_matmul(x, w8, s8, b8, transpose=True, group_size=64, bits=8)
    y1, y8 = qmm16c8(x[:1], w8, s8, b8), qmm16c8(x[:8], w8, s8, b8)
    yk = [qmm16c8(x, w8, s8, b8, ks=k) for k in (2, 4) if K % (64 * k) == 0]
    mx.eval(ref, y, ym, y1, y8, *yk)
    inv = eqb(y1, y[:1]) and eqb(y8, y[:8])
    e, em = rel(y, ref), rel(ym, ref)
    ok_ = inv and e <= em * 1.05 and all(rel(v, ref) <= em * 1.05 for v in yk)
    ok &= ok_
    say(f"(a) N {N} K {K}: error vs fp32 {e:.2e} (MLX 8-bit {em:.2e}; ks 2/4 {[f'{rel(v, ref):.2e}' for v in yk]}); "
        f"1 / 8 rows == the 16-row call's rows {inv} -> {'OK' if ok_ else 'FAIL'}")
SHAPES = [("q", 4096, 2560), ("k / v", 1024, 2560), ("o", 2560, 4096), ("gate / up", 7680, 2560), ("down", 2560, 7680),
          ("fc", 2560, 12800)]
best = {}
for name, N, K in SHAPES:
    W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    w8, s8, b8 = mx.quantize(W, group_size=64, bits=8)
    w4, s4, b4 = mx.quantize(W, group_size=64, bits=4)
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(W, w8, s8, b8, w4, s4, b4, x)
    arms = {"bf16": (lambda: x @ W.T, N * K * 2), "mlx q8": (lambda: mx.quantized_matmul(x, w8, s8, b8, transpose=True, group_size=64, bits=8), N * K * 1.0625),
            "qmm16c 4b": (lambda: qmm16c(x, w4, s4, b4), N * K * 0.5625)}
    for ks in (1, 2, 4, 8):
        if K % (64 * ks) == 0:
            arms[f"c8 ks{ks}"] = (lambda ks=ks: qmm16c8(x, w8, s8, b8, ks=ks), N * K * 1.0625)
    for f, _ in arms.values():
        mx.eval([f() for _ in range(4)])
    T = {k: [] for k in arms}
    for r in range(9):
        for k in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval([arms[k][0]() for _ in range(20)]); mx.synchronize()
            T[k].append((time.perf_counter() - t0) / 20)
    med = {k: float(np.median(v)) for k, v in T.items()}
    c8 = min((k for k in med if k.startswith("c8")), key=lambda k: med[k])
    best[name] = (c8, med[c8], med["bf16"], med["qmm16c 4b"])
    say(f"(b) {name:9s} N {N:5d} K {K:5d}: " + "; ".join(f"{k} {1e6 * v:.1f} us ({arms[k][1] / v / 1e9:.0f} GB/s)" for k, v in med.items()))
say("(b) best c8 per shape: " + ", ".join(f"{k} {v[0]} {1e6 * v[1]:.1f} us (bf16 {1e6 * v[2]:.1f})" for k, v in best.items()))
say(f"qmm16c8 accuracy + invariance: {'PASS' if ok else 'FAIL'}")
