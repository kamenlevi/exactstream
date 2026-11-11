"""L1-15 (Leap, Sat 2026-10-24): qmm16 (16-row NAX 4-bit matmul) vs a float reference and vs MLX's quantized_matmul:
accuracy, bitwise padding independence (row t alone == row t among 16, at every KS), and ms for the 35B's GDN projection
shapes (30 random weight instances round-robin, as the 30 layers).
Plus a diagnosis of prework35's rare mismatches: conv1d alone and silu alone vs MLX's ops on large-scale random data.
Usage: [SpecDecode venv] test_qmm16.py  -> ~/Leap/data/test_qmm16.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_e5/Leap_scripts"))
from qmm16 import qmm16
mx.set_cache_limit(2 << 30)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def timeit(fn, reps=21):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

def bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)

mx.random.seed(0)
SHAPES = {"fused in_proj qkv|z|b|a": (12352, 2048), "out_proj": (2048, 4096), "in_proj_qkv": (8192, 2048)}
for name, (N, K) in SHAPES.items():
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(30)]
    mx.eval(Ws)
    mb = sum(a.nbytes for a in Ws[0]) / 1e6
    fl = 30 * mb / 280e3 * 1e3
    x = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    wd = mx.dequantize(*Ws[0], group_size=64, bits=4).astype(mx.float32)
    ref = x.astype(mx.float32) @ wd.T
    y_mlx = mx.quantized_matmul(x, *Ws[0], transpose=True, group_size=64, bits=4)
    say(f"\n{name}: N {N} x K {K}, {mb:.2f} MB per instance; 30-instance floor {fl:.2f} ms")
    e_mlx = float(mx.abs(y_mlx.astype(mx.float32) - ref).max()); sc = float(mx.abs(ref).max())
    say(f"  MLX quantized_matmul M=16: max |err| vs fp32 ref {e_mlx:.4g} (|ref| max {sc:.3g}); "
        f"time M=1 {timeit(lambda: [mx.quantized_matmul(x[:1], *W, transpose=True, group_size=64, bits=4) for W in Ws]):.3f} ms, "
        f"M=16 {timeit(lambda: [mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4) for W in Ws]):.3f} ms")
    for ks in (1, 2, 4, 8):
        if K % (64 * ks):
            continue
        y = qmm16(x, *Ws[0], ks=ks)
        err = float(mx.abs(y.astype(mx.float32) - ref).max())
        b16 = bits(y)
        pad = sum(int((bits(qmm16(x[t:t + 1], *Ws[0], ks=ks)) != b16[t:t + 1]).sum()) for t in range(16))
        sub = int((bits(qmm16(x[:5], *Ws[0], ks=ks)) != b16[:5]).sum())
        t16 = timeit(lambda: [qmm16(x, *W, ks=ks) for W in Ws])
        say(f"  qmm16 KS={ks}: max |err| {err:.4g}; rows alone vs among 16: {pad} differ, 5 rows vs 16: {sub} differ; "
            f"{t16:.3f} ms ({fl / t16:4.0%} of floor)")
    del Ws

# ---------------------------------------------------------------- prework35 diagnosis: conv1d alone, silu alone
say("\nprework35 diagnosis (random data, x scale 3, w scale 0.5, S = 16):")
mx.random.seed(1)
conv = nn.Conv1d(8192, 8192, kernel_size=4, groups=8192, bias=False)
conv.weight = (mx.random.normal(conv.weight.shape) * 0.5).astype(mx.bfloat16)
ci = (mx.random.normal((1, 19, 8192)) * 3.0).astype(mx.bfloat16)
co = conv(ci)                                               # MLX conv1d, bf16
w4 = conv.weight.reshape(8192, 4).astype(mx.float32); cf = ci.astype(mx.float32)[0]
orders = {
    "t0+t1+t2+t3 (sequential)": lambda t: ((cf[t] * w4[:, 0] + cf[t + 1] * w4[:, 1]) + cf[t + 2] * w4[:, 2]) + cf[t + 3] * w4[:, 3],
    "(t0+t1)+(t2+t3) (pairs)": lambda t: (cf[t] * w4[:, 0] + cf[t + 1] * w4[:, 1]) + (cf[t + 2] * w4[:, 2] + cf[t + 3] * w4[:, 3]),
    "t3+t2+t1+t0 (reverse)": lambda t: ((cf[t + 3] * w4[:, 3] + cf[t + 2] * w4[:, 2]) + cf[t + 1] * w4[:, 1]) + cf[t] * w4[:, 0],
}
for nm, f in orders.items():
    mine = mx.stack([f(t) for t in range(16)]).astype(mx.bfloat16)
    say(f"  conv1d vs fp32 sum in order {nm}: {int((bits(mine) != bits(co[0])).sum())} of {co.size} differ")
sv = nn.silu(co)
say(f"  nn.silu output dtype {sv.dtype}; silu as Multiply(x, Sigmoid(x)) in bf16 ops: "
    f"{int((bits(co * mx.sigmoid(co)) != bits(sv)).sum())} of {sv.size} differ")
open(os.path.expanduser("~/Leap/data/test_qmm16.txt"), "w").write("\n".join(lines) + "\n")
