"""L1-15 check (Leap, Sat 2026-10-24): accuracy of the 16-row matmuls against an fp32-EXACT reference (codes dequantized
with fp32 scales / biases; the earlier check used mx.dequantize in bf16, which rounds the weights the same way qmm16 does,
so it could not see that error). RMS and max error relative to the RMS of the result, for MLX at M = 1 (qmv: scales applied
in fp32), MLX at M = 16, qmm16 (dequantized weights rounded to bf16, NAX matmul) and qmm16c (raw codes, scales after).
Activations: real-scale random normal; weights: random, 4-bit g64 (30 trials pooled).
Usage: [SpecDecode venv] test_qmm_acc.py  -> ~/Leap/data/test_qmm_acc.txt"""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_s5p/Leap_scripts"))
from qmm16 import qmm16, qmm16c
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
mx.random.seed(5)
for name, (N, K) in {"fused in_proj": (12352, 2048), "out_proj": (2048, 4096)}.items():
    errs = {}
    for trial in range(8):
        w, s, b = mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4)
        x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
        wf = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=4)   # fp32-exact weights
        ref = np.array((x.astype(mx.float32) @ wf.T).astype(mx.float32), dtype=np.float64)
        outs = {
            "MLX M=1 (qmv)": mx.concatenate([mx.quantized_matmul(x[t:t + 1], w, s, b, transpose=True, group_size=64, bits=4)
                                             for t in range(16)]),
            "MLX M=16": mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4),
            "qmm16 KS 2": qmm16(x, w, s, b, ks=2),
            "qmm16c KS 2": qmm16c(x, w, s, b, ks=2),
            "fp32-exact rounded to bf16 (floor)": mx.array(ref.astype(np.float32)).astype(mx.bfloat16),
        }
        for k, y in outs.items():
            e = np.array(y.astype(mx.float32), dtype=np.float64) - ref
            errs.setdefault(k, []).append((np.sqrt((e ** 2).mean()), np.abs(e).max(), np.sqrt((ref ** 2).mean())))
    say(f"{name} (N {N}, K {K}), error vs fp32-exact, relative to RMS(result):")
    for k, v in errs.items():
        rms = np.mean([a / c for a, _, c in v]); mxe = max(m / c for _, m, c in v)
        say(f"  {k:36s} RMS {rms:.2e}   max {mxe:.2e}")
open(os.path.expanduser("~/Leap/data/test_qmm_acc.txt"), "w").write("\n".join(lines) + "\n")
