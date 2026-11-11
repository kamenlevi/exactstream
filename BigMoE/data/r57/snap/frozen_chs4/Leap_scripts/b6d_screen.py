"""B6-D screen (Leap, Tue 2026-11-03 15:2x; pre-registered in ~/Leap/TABLE_350.md 'B6-D'; FAST-K): fatt16d (fs4 with each
simdgroup holding half the output dims, against fs4's register spill) vs fatt16s (fs4), B6's synthetic setup.
1. Bits: fatt16d == fatt16s, all outputs, at N 2K / 10K / 20K, R 16 and R 8 (if not: B6's INV (a)-(e) on fatt16d).
2. ms per call (8 chained, median of 20): fs4, fatt16d KS 4 / 2 / 8 at 2K / 10K / 20K; R 8 at 10K.
Lines: PASS = bits (or INV) and fatt16d KS 4 >= 1.5x vs fs4 at 10K; KILL < 1.15x.
Usage: [BigMoE venv] b6d_screen.py -> ~/Leap/data/b6d_screen.txt (GPU, < 1 GB; share lock)"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs4/Leap_scripts"))
from fatt16s import fatt16s
from fatt16d import fatt16d

OUT = os.path.expanduser("~/Leap/data/b6d_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"B6-D screen {time.strftime('%a %H:%M')} ({os.environ.get('B6D_COND', 'shared GPU, rough')})")
SC = 256 ** -0.5
bits = lambda a: np.array(a.view(mx.uint16)).astype(np.int64)
rng = np.random.default_rng(3)
def selmask(N, R=16):
    m = np.zeros((R, N), bool)
    for r in range(R):
        last = N - R + r
        if last + 1 <= 2048:
            m[r, :last + 1] = True
            continue
        for b in rng.choice((last + 1) // 4, 512, replace=False):
            m[r, 4 * b:4 * b + 4] = True
        m[r, max(0, last - 63):last + 1] = True
        m[r, last + 1:] = False
    return mx.array(m)
def qkv(N, R=16, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((1, 24, R, 256)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 2, N, 256)) * 0.8).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, N, 256)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    return q, k, v
# ---------------------------------------------------------------- 1. bits vs fs4
bits_ok = True
RUN_BITS = os.environ.get("B6D_PART", "all") != "speed"
for N in ((2000, 10000, 20000) if RUN_BITS else ()):
    for R in (16, 8):
        q, k, v = qkv(N, R, seed=N + R)
        m = selmask(N, R)
        a, b = fatt16s(q, k, v, SC, m), fatt16d(q, k, v, SC, m)
        mx.eval(a, b)
        same = bool((bits(a) == bits(b)).all())
        bits_ok &= same
        say(f"1. N {N} R {R}: fatt16d == fs4 bitwise: {same}" + ("" if same else
            f" ({100 * float((bits(a) != bits(b)).mean()):.2f} % of values differ)"))
if not bits_ok:                                                   # then INV as B6 on fatt16d
    sys.argv = [sys.argv[0]]
    say("1. not bitwise: running B6's INV on fatt16d")
    for N in (2000, 10000, 20000):
        for R in (16, 8):
            q, k, v = qkv(N, R, seed=N + R + 1)
            m = selmask(N, R)
            P = 5 if R == 16 else 2
            base = bits(fatt16d(q, k, v, SC, m)[0, :, P])
            q2 = (mx.random.normal(q.shape) * 0.8).astype(mx.bfloat16)
            qa = mx.concatenate([q2[:, :, :P], q[:, :, P:P + 1], q2[:, :, P + 1:]], axis=2)
            ok_a = (bits(fatt16d(qa, k, v, SC, m)[0, :, P]) == base).all()
            k3 = mx.concatenate([k, (mx.random.normal((1, 2, 37, 256)) * 0.8).astype(mx.bfloat16)], axis=2)
            v3 = mx.concatenate([v, mx.random.normal((1, 2, 37, 256)).astype(mx.bfloat16)], axis=2)
            m3 = mx.concatenate([m, mx.zeros((R, 37), mx.bool_)], axis=1)
            ok_c = (bits(fatt16d(q, k3, v3, SC, m3)[0, :, P]) == base).all()
            bits_ok |= False
            say(f"1. INV fatt16d N {N} R {R}: (a) {'PASS' if ok_a else 'FAIL'}, (c) {'PASS' if ok_c else 'FAIL'}")
# ---------------------------------------------------------------- 2. speed
if os.environ.get("B6D_PART", "all") == "bits":
    say(f"bits part done: {'PASS' if bits_ok else 'not bitwise'}")
    raise SystemExit(0)
def bench(f, chain=8, n=20):
    for _ in range(3):
        mx.eval([f() for _ in range(chain)])
    ts = []
    for _ in range(n):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval([f() for _ in range(chain)]); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)) / chain
x10 = None
for N in (2000, 10000, 20000):
    q, k, v = qkv(N)
    m = selmask(N)
    t = {"fs4": bench(lambda: fatt16s(q, k, v, SC, m))}
    for ks in (4, 2, 8):
        t[f"fd KS {ks}"] = bench(lambda: fatt16d(q, k, v, SC, m, ks=ks))
    say(f"2. N {N}: " + ", ".join(f"{a} {b:.3f}" for a, b in t.items()) + f" ms per call; fd KS 4 {t['fs4'] / t['fd KS 4']:.2f}x vs fs4")
    if N == 10000:
        x10 = t["fs4"] / t["fd KS 4"]
        q8, k8, v8 = qkv(N, 8)
        m8 = selmask(N, 8)
        t8 = {"fs4": bench(lambda: fatt16s(q8, k8, v8, SC, m8)), "fd": bench(lambda: fatt16d(q8, k8, v8, SC, m8))}
        say(f"2. N 10000 R 8: fs4 {t8['fs4']:.3f}, fd KS 4 {t8['fd']:.3f} ms per call ({t8['fs4'] / t8['fd']:.2f}x)")
v_ = "PASS" if x10 >= 1.5 else "KILL" if x10 < 1.15 else "BETWEEN"
say(f"B6-D verdict (bits == fs4 {bits_ok}; fd KS 4 vs fs4 at 10K {x10:.2f}x; PASS >= 1.5x, KILL < 1.15x): {v_}")
