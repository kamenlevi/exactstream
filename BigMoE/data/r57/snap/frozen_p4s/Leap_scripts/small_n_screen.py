"""SMALL-N screen (Leap, Wed 2026-11-04 15:3x; TABLE_350 'SMALL-N'): MLX's 16-row 4-bit call vs qmm16n (rows padded to a
multiple of 32) on the default's small dense shapes, ms per check (random weights at the shapes x counts; interleaved).
Usage: [BigMoE venv] small_n_screen.py -> ~/Leap/data/small_n_screen.txt (GPU, MEM ~0.5 GB; share lock)"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_p4s/Leap_scripts"))
import mlx.core as mx
from qmm26 import repack, qmm16n
OUT = os.path.expanduser("~/Leap/data/small_n_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nSMALL-N screen {time.strftime('%a %H:%M:%S')}")
SHAPES = [("shared gate|up|gate (R-74)", 1281, 2560, 48), ("k_proj / v_proj", 512, 2560, 24), ("b|a (R-83)", 96, 2560, 36)]
def rbits(shape):
    hi = mx.random.randint(0, 1 << 16, shape).astype(mx.uint32); lo = mx.random.randint(0, 1 << 16, shape).astype(mx.uint32)
    return mx.bitwise_or(mx.left_shift(hi, 16), lo)
tot = {"mlx": 0.0, "n": 0.0}
for name, N, K, cnt in SHAPES:
    Np = (N + 31) // 32 * 32
    Ws, Wn = [], []
    for _ in range(cnt):
        w = rbits((N, K // 8)); s = (mx.random.uniform(0.5, 1.5, (N, K // 64)) * 0.004).astype(mx.bfloat16)
        b = (-8 * s.astype(mx.float32)).astype(mx.bfloat16)
        Ws.append((w, s, b))
        pad = lambda a: mx.concatenate([a, mx.zeros((Np - N,) + a.shape[1:], a.dtype)]) if Np > N else a
        Wn.append(repack(pad(w), pad(s), pad(b)))
    mx.eval(Ws, Wn)
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    arms = {"mlx": lambda: [mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4) for W in Ws],
            "n": lambda: [qmm16n(x, *W, Np)[:, :N] for W in Wn]}
    T = {a: [] for a in arms}
    for r in range(18):
        for a in (("mlx", "n") if r % 2 == 0 else ("n", "mlx")):
            mx.eval(arms[a]())
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(arms[a]()); mx.synchronize()
            T[a].append(1e3 * (time.perf_counter() - t0))
    med = {a: float(np.median(v[3:])) for a, v in T.items()}
    tot["mlx"] += med["mlx"]; tot["n"] += med["n"]
    say(f"  {name} N {N} (padded {Np}) x{cnt}: MLX 16-row {med['mlx']:.3f} ms, qmm16n {med['n']:.3f} ms per check")
v = tot["mlx"] - tot["n"]
say(f"SMALL-N: bundle {tot['mlx']:.3f} -> {tot['n']:.3f} ms per check ({v:+.3f} saved) -> "
    f"{'BUILD' if v >= 0.5 else 'KILL' if v < 0.2 else 'BETWEEN'}")
