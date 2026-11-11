"""WIDE-NT bench (Leap, Fri 2026-11-06 08:5x; Fable B v2: "in_proj 210 vs q's 245 GB/s (try nt 2)"): qmm16n's (ks, nt) on the
two widest native calls of the check, in_proj qkv|z (N 16384, K 2560; 36 per check) and attention q (N 12288, K 2560; 12),
as dnks_bench: dependent chains over 12 distinct cold copies, round-robin over the variants, median of 9 rounds.
Bits vs (1, 1): max relative error. -> ~/Leap/data/wide_bench.txt. Share lock, MEM ~0.6 GB."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
import mlx.core as mx
import qmm26

OUT = os.path.expanduser("~/Leap/data/wide_bench.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nWIDE-NT bench {time.strftime('%a %H:%M:%S')} ({os.environ.get('WB_COND', 'share lock')}; MLX {mx.__version__})")
rng = np.random.default_rng(13)
clock = time.perf_counter
VAR = [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (4, 1), (5, 1), (8, 1), (10, 1)]


def make(N, K, n):
    out = []
    for _ in range(n):
        w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=64, bits=4)
        wn, sn, bn = qmm26.repack(q, s, b)
        mx.eval(wn, sn, bn)
        out.append((wn, sn, bn))
    return out


def chain(W, N, x0, ks, nt):
    tt = []
    for calls in (False, True):
        prev = mx.zeros((1,), mx.float32)
        for wn, sn, bn in W:
            x = x0 + (prev * 0).astype(x0.dtype)
            if calls:
                y = qmm26.qmm16n(x, wn, sn, bn, N, ks, nt)
                prev = y.reshape(-1)[:1].astype(mx.float32)
            else:
                prev = x.reshape(-1)[:1].astype(mx.float32)
        t0 = clock(); mx.eval(prev); tt.append(clock() - t0)
    return 1e6 * (tt[1] - tt[0]) / len(W)


for name, N, K, calls in (("in_proj qkv|z", 16384, 2560, 36), ("attention q", 12288, 2560, 12)):
    W = make(N, K, 12)
    x = mx.array(rng.standard_normal((2, K)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(x)
    g = K // 64
    vs = [(ks, nt) for ks, nt in VAR if g % ks == 0 and N % (32 * nt) == 0 and 32 * ks * nt <= 1024 and (ks - 1) * nt <= 10]
    y1 = qmm26.qmm16n(x, *W[0], N, 1, 1).astype(mx.float32); mx.eval(y1)
    err = {}
    for v in vs:
        yv = qmm26.qmm16n(x, *W[0], N, *v).astype(mx.float32)
        err[v] = float(mx.max(mx.abs(yv - y1)).item() / max(1e-30, mx.max(mx.abs(y1)).item()))
        chain(W, N, x, *v)
    acc = {v: [] for v in vs}
    for r in range(9):
        for v in (vs if r % 2 == 0 else vs[::-1]):
            acc[v].append(chain(W, N, x, *v))
    t = {v: float(np.median(acc[v])) for v in vs}
    best = min(t, key=t.get)
    mb = (N * K // 2 + 2 * N * (K // 64) * 2) / 1e6
    say(f"  {name} N {N} K {K} ({mb:.1f} MB, {calls} per check): " + "; ".join(
        f"ks {v[0]} nt {v[1]}: {t[v]:.1f} us ({mb * 1e-3 / max(1e-9, t[v] * 1e-6):.0f} GB/s, err {err[v]:.0e})" for v in vs)
        + f" -> best ks {best[0]} nt {best[1]}: {calls * (t[(1, 1)] - t[best]) / 1e3:+.2f} ms per check vs (1, 1)")
    del W
    mx.clear_cache()
