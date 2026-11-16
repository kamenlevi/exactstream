"""one32_test.py (Leap, Sun 15 Nov 2026 11:3x; CHECK55 step 1): bits of one32's one-dispatch pieces against the certified
16-row forms, per row, at 17 / 20 / 24 / 31 / 32 rows x 3 seeds, random weights quantized by MLX at the engine's shapes.
  qmm16c2 vs two qmm16c calls (rows 0-15, rows 16-n zero-padded to 16): the shapes REF2 sends to qmm16c;
  qmvw32 (MLX's qmv_wide at k_lanes 8, the kernel of MLX's 16-row call on these shapes), qmv32 and MLX's raw n-row call
  vs two padded 16-row MLX calls: the shapes REF2 leaves to MLX. (mlx_b16, a [2, 16, K] batch, was 0 / 372: MLX folds
  the batch into 32 rows.)
Line: a piece is OK for a shape iff every row is bitwise at every row count and seed.
usage: [BigMoE venv] one32_test.py   (MEM ~3 GB with the head; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_g1b/Leap_scripts/week98"))
from qmm16 import qmm16c
import one32

R = 16
ROWS = (17, 20, 24, 31, 32)
SEEDS = (1, 2, 3)
C16 = [("gdn qkv|z (R-83 qz)", 16384, 2560), ("gdn out_proj / att o_proj", 2560, 6144), ("att q_proj", 12288, 2560),
       ("shared down", 2560, 640), ("ple key", 10240, 2560), ("ple value", 2560, 2560), ("lm_head", 248320, 2560)]
MLXS = [("gdn b|a (R-83 ba)", 96, 2560, 4), ("att k / v_proj", 512, 2560, 4), ("indexer qk", 640, 2560, 4),
        ("router (8-bit)", 512, 2560, 8), ("shared gate|up|gate (K-73)", 1281, 2560, 4),
        ("8-bit 640 (attention layer)", 640, 2560, 8), ("shared_expert_gate N 1 (4-bit)", 1, 2560, 4),
        ("N 1 (8-bit)", 1, 2560, 8), ("N 48 (4-bit)", 48, 2560, 4)]


def say(s):
    print(s, flush=True)


def pad16(x):
    return x if x.shape[0] == R else mx.concatenate([x, mx.zeros((R - x.shape[0], x.shape[1]), x.dtype)], 0)


def score(forms, ref_fn, K):
    res = {f: [0, 0] for f in forms}
    for sd in SEEDS:
        x = mx.random.normal((32, K), dtype=mx.bfloat16, key=mx.random.key(sd)) * 0.7
        for M in ROWS:
            xm = x[:M]
            ref = mx.concatenate([ref_fn(xm[:R]), ref_fn(pad16(xm[R:]))[:M - R]], axis=0)
            for f, fn in forms.items():
                eq = mx.all(fn(xm) == ref, axis=-1)
                mx.eval(eq)
                res[f][0] += int(np.array(eq).sum()); res[f][1] += M
    return res


t0 = time.time()
say(f"ONE32-TEST {time.strftime('%a %d %b %H:%M:%S')}: MLX {mx.__version__}")
for name, N, K in C16:
    mx.random.seed(N + K)
    wq, s, b = mx.quantize(mx.random.normal((N, K), dtype=mx.bfloat16) * 0.02, group_size=64, bits=4)
    mx.eval(wq, s, b)
    res = score({"qmm16c2": lambda xm: one32.qmm16c2(xm, wq, s, b), "qmm16c2a": lambda xm: one32.qmm16c2a(xm, wq, s, b),
                 "qmm16c2i": lambda xm: one32.qmm16c2i(xm, wq, s, b)},
                lambda x16: qmm16c(x16, wq, s, b), K)
    say(f"  {name:28s} N {N:6d} K {K:5d}: " + ", ".join(
        f"{f} {g:4d} / {t:4d} -> {'OK' if g == t else 'NO'}" for f, (g, t) in res.items()))
    del wq, s, b
for name, N, K, bits in MLXS:
    mx.random.seed(N + K + bits)
    wq, s, b = mx.quantize(mx.random.normal((N, K), dtype=mx.bfloat16) * 0.02, group_size=64, bits=bits)
    mx.eval(wq, s, b)
    q = lambda xx: mx.quantized_matmul(xx, wq, s, b, transpose=True, group_size=64, bits=bits)
    res = score({"qmvw32 kl8": lambda xm: one32.qmvw32(xm, wq, s, b, bits, 8),
                 "qmvw32 kl8 vpt32": lambda xm: one32.qmvw32(xm, wq, s, b, bits, 8, 32),
                 "qmv32": lambda xm: one32.qmv32(xm, wq, s, b, bits), "mlx raw": q}, q, K)
    say(f"  {name:28s} N {N:6d} K {K:5d} {bits}-bit: " + ", ".join(
        f"{f} {g:4d} / {t:4d} -> {'OK' if g == t else 'NO'}" for f, (g, t) in res.items()))
    del wq, s, b
say(f"done in {time.time() - t0:.0f} s; peak {mx.get_peak_memory() / 1e9:.1f} GB")
