"""Dense part of a wide verify pass: MLX's 4-bit g64 quantized_matmul and bf16 matmul on the 125B's dense GDN-layer shapes
(patches/sol_layer/ref_gemv.py extended) at M = 4, 5, 8, 16, 32 rows, plus the chunked form BigMoE runs today for wide
passes (BIGMOE_DUPROW_CHUNK=5: ceil(M/5) calls of <= 5 rows). 48 independent calls over 6 weight sets (DRAM, not cache),
graph built before the clock. Shows whether the non-MoE half of a pass scales with S once the experts are read once.
usage: dense_scaling.py   (GPU, ~1 min; quiet lock)"""
import math, time
import mlx.core as mx

NLP, N, REPS = 6, 48, 4
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
    for M in (4, 5, 8, 16, 32):
        X = [(mx.random.normal((M, k)) * 0.5).astype(mx.bfloat16) for _ in range(NLP)]
        mx.eval(X)
        for chunked in ((False, True) if M > 5 else (False,)):
            def f(i):
                x = X[i % NLP]
                parts = [x[a:a + 5] for a in range(0, M, 5)] if chunked else [x]
                if bits == 4:
                    return [mx.quantized_matmul(p, *W[i % NLP], transpose=True, group_size=64, bits=4) for p in parts]
                return [p @ W[i % NLP].T for p in parts]
            best = 1e9
            for rep in range(REPS + 1):
                outs = [f(i) for i in range(N)]
                t0 = time.perf_counter()
                mx.eval(outs)
                if rep:
                    best = min(best, time.perf_counter() - t0)
            us = best / N * 1e6
            line += f"  M={M}{'c5' if chunked else ''} {us:6.1f} us"
    print(line, flush=True)
