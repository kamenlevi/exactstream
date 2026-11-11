"""L1-01b (a): can lm_head / in_proj_qkv / q_proj run as 16-row padded qmm for EVERY row count (decode included) and stay
exact? Needs each real row's bits to be independent of the padding rows' values and of how many rows are real.
For M = 1..16 real rows padded to 16 with zeros, with random rows, and with copies of row 0, plus natural (unpadded) M = 13..16:
every real row bitwise equal to the same row in the all-real 16-row call. Also 32 rows (rows 0..15 vs the 16-row call).
usage: qmm_pad.py   (GPU, ~1 min; quiet lock)"""
import time
import mlx.core as mx
import numpy as np

SHAPES = {"lm_head": (248320, 2560), "in_proj_qkv": (10240, 2560), "q_proj": (12288, 2560), "o_proj": (2560, 6144)}
for name, (N, K) in SHAPES.items():
    W = mx.quantize((mx.random.normal((N, K), key=mx.random.key(7)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
    mx.eval(W)
    f = lambda x: mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4)
    X = (mx.random.normal((32, K), key=mx.random.key(8)) * 0.5).astype(mx.bfloat16)
    ref = np.array(f(X[:16]).view(mx.uint16))                    # all 16 rows real
    bad = []
    for M in range(1, 17):
        for lab, pad in (("zeros", mx.zeros((16 - M, K), mx.bfloat16)), ("random", X[16:32 - M]),
                         ("copies", mx.broadcast_to(X[:1], (16 - M, K)))):
            Y = np.array(f(mx.concatenate([X[:M], pad]) if M < 16 else X[:16]).view(mx.uint16))
            if not np.array_equal(Y[:M], ref[:M]):
                bad.append(f"M={M} {lab}")
    nat = [M for M in range(13, 17) if np.array_equal(np.array(f(X[:M]).view(mx.uint16)), ref[:M])]
    y32 = np.array_equal(np.array(f(X[:32]).view(mx.uint16))[:16], ref)
    t = {}
    for M in (1, 8, 16):
        best = 1e9
        for rep in range(5):
            outs = [f(X[:M] if M == 16 else mx.concatenate([X[:M], mx.zeros((16 - M, K), mx.bfloat16)])) for _ in range(8)]
            t0 = time.perf_counter(); mx.eval(outs)
            if rep:
                best = min(best, time.perf_counter() - t0)
        t[M] = best / 8 * 1e6
    print(f"[qmm_pad] {name} {N}x{K}: padded-16 rows independent of padding (M 1..16 x zeros/random/copies): "
          f"{'YES' if not bad else 'NO ' + ', '.join(bad[:6])}; natural M equal to the 16-row call at M {nat}; "
          f"32-row call rows 0..15 equal: {y32}; us per padded-16 call (M=1/8/16 real, same weights, cached): "
          + " ".join(f"{k}:{v:.0f}" for k, v in t.items()), flush=True)
