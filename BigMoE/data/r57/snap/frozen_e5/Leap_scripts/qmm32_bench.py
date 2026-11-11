"""QMM32 bench (Leap, Fri 2026-11-06 09:1x): the 17-row drafter calls three ways on the D7b drafter's shapes (4-bit g64, bf16
scales): MLX quantized_matmul (today without D7BNAX_32), two qmm16c halves (D7BNAX_32 today) and qmm32.qmm32c (one weight read).
  bits: qmm32c(x17) vs the halves at the same ks: must be bitwise (same per-row arithmetic)
  time: dependent chains over cold copies, round-robin over the forms, median of 7 rounds; per propose = calls x saving
-> ~/Leap/data/qmm32_bench.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
import mlx.core as mx
import mlx.nn as nn
from qmm16 import qmm16c
from qmm32 import qmm32c

OUT = os.path.expanduser("~/Leap/data/qmm32_bench.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nQMM32 bench {time.strftime('%a %H:%M:%S')} ({os.environ.get('Q32_COND', 'share lock')}; MLX {mx.__version__})")
SHAPES = [("gate|up", 7680, 2560, 10), ("down", 2560, 7680, 5), ("q", 4096, 2560, 5), ("o", 2560, 4096, 5),
          ("k|v", 1024, 2560, 10), ("2560->640", 640, 2560, 10)]
rng = np.random.default_rng(21)
clock = time.perf_counter


def ks_of(N, K, cap=5):
    g, tiles, ks = K // 64, N // 32, 1
    for k in (1, 2, 4, 5):
        if g % k == 0 and k <= cap:
            ks = k
            if tiles * k >= 320:
                break
    return ks


def make(N, K, n):
    out = []
    for _ in range(n):
        w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=4)
        s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
        mx.eval(q, s, b)
        out.append((q, s, b))
    return out


def form(name, x, q, s, b, ks):
    if name == "mlx":
        return mx.quantized_matmul(x, q, s, b, transpose=True, group_size=64, bits=4)
    if name == "halves":
        return mx.concatenate([qmm16c(x[:16], q, s, b, ks), qmm16c(x[16:], q, s, b, ks)], axis=0)
    return qmm32c(x, q, s, b, ks)


def chain(W, x0, name, ks):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for q, s, b in W:
            x = x0 + (prev * 0).astype(x0.dtype)
            if calls:
                prev = form(name, x, q, s, b, ks).reshape(-1)[:1].astype(mx.float32)
            else:
                prev = x.reshape(-1)[:1].astype(mx.float32)
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(W)


tot_h = tot_m = 0.0
for name, N, K, per in SHAPES:
    mb = (N * K // 2 + 2 * N * (K // 64) * 2) / 1e6
    n = int(min(64, max(8, np.ceil(96.0 / mb))))
    W = make(N, K, n)
    x = mx.array(rng.standard_normal((17, K)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x)
    ks = ks_of(N, K)
    q, s, b = W[0]
    bit = {}
    for k in sorted({1, ks}):
        ya, yb = form("halves", x, q, s, b, k), form("q32", x, q, s, b, k)
        mx.eval(ya, yb)
        bit[k] = bool(mx.array_equal(ya, yb).item())
    for f in ("mlx", "halves", "q32"):
        chain(W, x, f, ks)
    acc = {f: [] for f in ("mlx", "halves", "q32")}
    for r in range(7):
        for f in (("mlx", "halves", "q32") if r % 2 == 0 else ("q32", "halves", "mlx")):
            acc[f].append(chain(W, x, f, ks))
    t = {f: float(np.median(v)) for f, v in acc.items()}
    tot_h += per * (t["halves"] - t["q32"]) / 1e3
    tot_m += per * (t["mlx"] - t["q32"]) / 1e3
    say(f"  {name:10s} N {N:5d} K {K:5d} ({mb:5.2f} MB, ks {ks}, {per} per propose): MLX {t['mlx']:6.1f} us, halves "
        f"{t['halves']:6.1f} us, qmm32c {t['q32']:6.1f} us ({mb * 1e-3 / max(1e-9, t['q32'] * 1e-6):4.0f} GB/s); "
        f"qmm32c == halves bitwise: {bit}")
    del W
    mx.clear_cache()
say(f"  per propose: qmm32c vs halves {tot_h:+.2f} ms saved, vs MLX {tot_m:+.2f} ms saved (all six shapes on qmm32c)")
