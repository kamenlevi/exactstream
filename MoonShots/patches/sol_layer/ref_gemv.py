"""Reference GEMV speed for the speed-of-light layer: MLX's own kernels on the 125B's dense GDN-layer shapes, M = 1, 2, 4, 8.
4-bit g64 mx.quantized_matmul (in_proj 16480x2560, out_proj 2560x6144, router+shared 1796x2560) and bf16 matmul (HC down
324x10240, HC up 10240x320). 48 independent calls over 6 weight sets (DRAM, not cache), graph built before the clock:
GPU-side throughput. usage: ref_gemv.py   (GPU seconds; quiet lock)"""
import time
import mlx.core as mx

NLP, N, REPS = 6, 48, 5
SHAPES = [("in_proj 4b", 16480, 2560, 4), ("out_proj 4b", 2560, 6144, 4), ("router+sh 4b", 1796, 2560, 4),
          ("HC down bf16", 324, 10240, 16), ("HC up bf16", 10240, 320, 16)]
for name, n, k, bits in SHAPES:
    if bits == 4:
        W = [mx.quantize((mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(NLP)]
        nbytes = n * k // 2 + n * (k // 64) * 4
    else:
        W = [(mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16) for _ in range(NLP)]
        nbytes = n * k * 2
    mx.eval(W)
    line = f"{name:14s} {n}x{k} ({nbytes / 1e6:.1f} MB):"
    for M in (1, 2, 4, 8):
        X = [(mx.random.normal((M, k)) * 0.5).astype(mx.bfloat16) for _ in range(NLP)]
        mx.eval(X)
        if bits == 4:
            f = lambda i: mx.quantized_matmul(X[i % NLP], *W[i % NLP], transpose=True, group_size=64, bits=4)
        else:
            f = lambda i: X[i % NLP] @ W[i % NLP].T
        best = 1e9
        for rep in range(REPS + 1):
            outs = [f(i) for i in range(N)]
            t0 = time.perf_counter()
            mx.eval(outs)
            if rep:
                best = min(best, time.perf_counter() - t0)
        us = best / N * 1e6
        line += f"  M={M} {us:6.1f} us {nbytes / us / 1e3:4.0f} GB/s"
    print(line, flush=True)
