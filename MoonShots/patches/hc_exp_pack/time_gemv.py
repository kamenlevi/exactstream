"""Locate the HC-pack slowdown: MLX x @ W.T vs the GemvWide clone (bf16 weights) vs the clone reading the pack,
per HC shape at M = 2 (dup-row S=1) and M = 3, as dependent chains over 16 distinct weight matrices.
Usage: time_gemv.py   (GPU, quiet lock)"""
import os, sys, time
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemv_probe import gemv_wide
from hc_pack import pack_bf16

NC = 16
for (R, K) in ((320, 10240), (10240, 320)):
    Ws = [(mx.random.normal((R, K)) * 0.02).astype(mx.bfloat16) for _ in range(NC)]
    Ps = [pack_bf16(W) for W in Ws]
    mx.eval(Ws, Ps)
    for M in (2, 3):
        x0 = mx.random.normal((M, K)).astype(mx.bfloat16)
        ref = x0 @ Ws[0].T
        kl = next(k for k in (32, 16, 8) if mx.array_equal(gemv_wide(Ws[0], x0, k), ref).item())

        def glue(y):
            return y[:, :K] if R >= K else mx.tile(y, (1, -(-K // R)))[:, :K]
        variants = {
            "MLX": lambda x, i: x @ Ws[i].T,
            "clone bf16": lambda x, i: gemv_wide(Ws[i], x, kl),
            "clone packed": lambda x, i: gemv_wide(None, x, kl, packed=(Ps[i], R)),
        }
        out = []
        for name, f in variants.items():
            def run():
                x = x0
                for i in range(NC):
                    x = glue(f(x, i))
                mx.eval(x)
            for _ in range(3):
                run()
            ts = []
            for _ in range(15):
                a = time.perf_counter(); run(); ts.append(time.perf_counter() - a)
            t = sorted(ts)[7] / NC * 1e6
            out.append(f"{name} {t:6.1f} µs")
        print(f"{R}x{K} M={M} k_lanes={kl}: " + " | ".join(out), flush=True)
