"""L1-15 step 1b (Leap, Sat 2026-10-24): which MLX quantized-matmul call shapes give the 35B's GDN projections a fast AND
batch-invariant 16-row kernel. 30 weight instances per shape (4-bit g64, random; round-robin as the 30 layers), x bf16.
For each row count M (the real rows are the first <= 16; the rest zero padding): ms for the 30 matmuls, and whether a
row's bits are the same at every M of the set (one kernel family + padding independence), vs M = 1 decode.
Also: the GPU's architecture string, and whether a custom kernel can include Metal 4's tensor ops (MPP).
Usage: [SpecDecode venv] qmm_bands35.py  -> ~/Leap/data/qmm_bands35.txt"""
import os, time
import numpy as np
import mlx.core as mx
mx.set_cache_limit(2 << 30)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

info = mx.metal.device_info()
say(f"device: {info.get('device_name')} {info.get('architecture')}, MLX {mx.__version__}")

def timeit(fn, reps=21):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

SHAPES = {"fused in_proj qkv|z|b|a": (12352, 2048), "in_proj_qkv": (8192, 2048), "in_proj_z": (4096, 2048),
          "out_proj": (2048, 4096)}
MS = {"fused in_proj qkv|z|b|a": (1, 2, 8, 12, 13, 16, 24, 32, 33),
      "in_proj_qkv": (1, 8, 12, 13, 16, 32),
      "in_proj_z": (1, 8, 16, 24, 25, 32, 33),
      "out_proj": (1, 8, 16, 24, 25, 32, 33, 48, 64)}
mx.random.seed(0)
for name, (N, K) in SHAPES.items():
    Ws = []
    for _ in range(30):
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        Ws.append(mx.quantize(w, group_size=64, bits=4))
    mx.eval(Ws)
    mb = sum(a.nbytes for a in Ws[0]) / 1e6
    x16 = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    say(f"\n{name}: N {N} x K {K}, {mb:.2f} MB per instance, 30-instance floor {30 * mb / 280e3 * 1e3:.2f} ms at 280 GB/s")
    ref1 = [mx.quantized_matmul(x16[t:t + 1], *Ws[0], transpose=True, group_size=64, bits=4) for t in range(16)]
    outs = {}
    for M in MS[name]:
        xs = mx.concatenate([x16[:min(M, 16)], mx.zeros((max(0, M - 16), K), mx.bfloat16)], 0)
        ms = timeit(lambda: [mx.quantized_matmul(xs, *W, transpose=True, group_size=64, bits=4) for W in Ws])
        o = mx.quantized_matmul(xs, *Ws[0], transpose=True, group_size=64, bits=4)[:min(M, 16)]
        outs[M] = np.array(o.astype(mx.float32))
        d1 = sum(int((np.array(ref1[t].astype(mx.float32)) != outs[M][t:t + 1]).sum()) for t in range(min(M, 16)))
        # padding independence: row 0 alone, padded with zeros to M rows
        x0 = mx.concatenate([x16[:1], mx.zeros((M - 1, K), mx.bfloat16)], 0)
        p0 = np.array(mx.quantized_matmul(x0, *Ws[0], transpose=True, group_size=64, bits=4)[:1].astype(mx.float32))
        dp = int((p0 != outs[M][:1]).sum())
        say(f"  M = {M:2d}: {ms:6.3f} ms ({30 * mb / 280e3 * 1e3 / ms:4.0%} of floor)   rows vs M=1 decode: {d1:6d} differ"
            f"   row 0 alone padded to M vs with real rows: {dp} differ")
    # which M values agree with each other on the first rows (same kernel family)
    Mk = [M for M in MS[name] if M >= 8]
    fam = []
    for M in Mk:
        same = [M2 for M2 in Mk if (outs[M2][:8] == outs[M][:8]).all()]
        fam.append(f"{M}:{same}")
    say("  rows 0-7 bitwise equal across M: " + "  ".join(fam))
    del Ws

# ---------------------------------------------------------------- can a custom kernel use Metal 4 tensor ops?
src_mpp = r"""
    uint i = thread_position_in_grid.x;
    out[i] = inp[i] * 2.0f;
"""
hdr_mpp = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
try:
    k = mx.fast.metal_kernel(name="leap_mpp_probe", input_names=["inp"], output_names=["out"], source=src_mpp, header=hdr_mpp)
    (o,) = k(inputs=[mx.ones((32,))], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(32,)], output_dtypes=[mx.float32])
    mx.eval(o)
    say(f"\nMPP header in a custom kernel: compiles and runs ({float(o[0])})")
except Exception as e:
    say(f"\nMPP header in a custom kernel: FAILS: {str(e)[:300]}")
open(os.path.expanduser("~/Leap/data/qmm_bands35.txt"), "w").write("\n".join(lines) + "\n")
