"""LOWROW (Leap, Sun 2026-11-08 00:1x; F1a item 1, chat's 1-6-row checks): dependent chains of n calls over n distinct
cold weights at the 125B's dense shapes, per call us and GB/s, M rows:
  ref   REF2's form today: x padded to 16 rows -> qmm16c (bitwise == the 16-row reference), sliced back
  mlx   MLX's own quantized_matmul at M rows (its qmv / small-M path): bit-changing vs ref
Chain link: each call's input adds 0 x the last output's first element (a true dependency, ~2-3 us).
usage: python lowrow_bench.py [M ...] -> ~/Leap/data/lowrow_bench.txt"""
import os
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
from qmm16 import qmm16c

OUT = os.path.expanduser("~/Leap/data/lowrow_bench.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def chain(fn, X, W, reps=5):
    n = len(W)

    def one():
        y = None
        for i in range(n):
            x = X if y is None else X + (0 * y[:1, :1]).astype(X.dtype)
            y = fn(x, *W[i])
        return y
    for _ in range(3):
        mx.eval(one())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(one())
        ts.append(1e6 * (time.perf_counter() - t0) / n)
    return sorted(ts)[len(ts) // 2]


def main(Ms):
    rng = np.random.default_rng(4)
    shapes = [("gdn_inproj", 2560, 16384, 4, 36), ("out_proj", 6144, 2560, 4, 48), ("att_q", 2560, 12288, 4, 12),
              ("shared_gu", 2560, 1280, 4, 48), ("shared_down", 640, 2560, 4, 48), ("gdn_ab", 2560, 96, 4, 36),
              ("att_kv", 2560, 512, 4, 24), ("router8", 2560, 512, 8, 48)]
    say(f"\nLOWROW {time.strftime('%a %H:%M:%S')} ({os.environ.get('GS_COND', 'quiet lock')}; MLX {mx.__version__}); M {Ms}")
    for name, K, N, bits, n in shapes:
        W = []
        for _ in range(n):
            w = mx.array((rng.standard_normal((N, K)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
            W.append(mx.quantize(w, group_size=64, bits=bits))
        mx.eval(W)
        mb = (W[0][0].nbytes + W[0][1].nbytes + W[0][2].nbytes) / 1e6
        line = []
        for M in Ms:
            X = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(X)
            tm = chain(lambda x, wq, s, b: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits), X, W)
            if bits == 4 and N % 64 == 0 and N >= 1024:
                tr = chain(lambda x, wq, s, b: qmm16c(mx.concatenate([x, mx.zeros((16 - x.shape[0], K), x.dtype)]) if x.shape[0] < 16 else x,
                                                      wq, s, b)[:x.shape[0]], X, W)
                line.append(f"M {M}: ref {tr:.1f} us ({mb / tr * 1e3:.0f} GB/s) mlx {tm:.1f} us ({mb / tm * 1e3:.0f} GB/s)")
            else:
                line.append(f"M {M}: mlx {tm:.1f} us ({mb / tm * 1e3:.0f} GB/s)")
        say(f"  {name:12s} K {K:5d} N {N:6d} {bits}-bit {mb:6.2f} MB x {n:2d} | " + " | ".join(line))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 4])
