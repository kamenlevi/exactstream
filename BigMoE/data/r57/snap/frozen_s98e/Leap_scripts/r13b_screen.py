"""R-13b WIDE16 (R-13's exact half, family 3; Leap, pre-registered Tue 2026-10-27 20:08 EET in ~/Leap/TABLE_350.md
'R-13b'): the small 4-bit dense shapes of a 16-row check in ONE qmv_wide pass instead of four.
MLX 0.32.2 serves 4-bit affine x @ w.T at M = 16 with qmv_wide when K, N <= 4096 (vector limit 25 / 33 on gen 17):
vecs_per_tg = ceil(16 / ceil(16 / 5)) = 4, so grid.x = 4 vector groups, each dequantizing and reading every weight
row again. k_lanes is always 8 for affine, and vecs_per_tg never enters a vector's arithmetic: each vector
accumulates its own partials over the same lane-strided groups, 8-value sub-chunks and shuffle ladder (MTPLX's M4
route kernel transcribes qmv_wide_impl on the same ground). So qmv_wide_impl at vecs_per_tg = 16 should give the same
bits from one pass.
Kernels: MLX's own dequantize and qmv_wide_impl, text taken from the installed quantized.h (renamed; constant refs and
attributes dropped; SIMD_SIZE = 32), instantiated through mx.fast.metal_kernel at NV = 4 (must equal MLX: checks the
transcription), 8 and 16.
Shapes and per-check counts: kern23's (random 4-bit weights, group 64, cold): shared gate/up 640 x 2560 (96), shared
down 2560 x 640 (48), k/v 512 x 2560 (24), in_proj_a/b 48 x 2560 (72), shared-expert gate 1 x 2560 (48).
Lines: PASS >= 2 ms saved per check vs MLX's 16-row calls AND bitwise on every shape. KILL < 1 ms or not bitwise.
Usage: [BigMoE venv] r13b_screen.py -> ~/Leap/data/r13b_screen.txt"""
import os, re, time
from pathlib import Path
import numpy as np
import mlx.core as mx

D = os.path.expanduser("~/Leap/data")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def timeit(fn, reps=15):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        while os.path.exists(PAUSE):
            time.sleep(5)
        mx.synchronize(); t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return sorted(ts)[reps // 2] * 1e3

# ---------------------------------------------------------------- MLX's own functions, from the installed header
SRC = (Path(mx.__file__).resolve().parent / "include/mlx/backend/metal/kernels/quantized.h").read_text()
def block(anchor):
    i = SRC.index(anchor)
    t = SRC.rindex("template <", 0, i)                   # the template line right above
    j = SRC.index("{", i)
    depth = 0
    for k in range(j, len(SRC)):
        depth += {"{": 1, "}": -1}.get(SRC[k], 0)
        if depth == 0:
            return SRC[t:k + 1]
    raise ValueError(anchor)
deq = block("inline void dequantize(const device uint8_t* w, U scale, U bias, W w_local)")
wide = block("METAL_FUNC void qmv_wide_impl(")
wide = re.sub(r"\s*\[\[[a-z_]+\]\]", "", wide)
wide = wide.replace("const constant int& ", "const int ").replace("SIMD_SIZE", "32").replace("METAL_FUNC", "inline")
wide = wide.replace("qmv_wide_impl(", "leap_qmv_wide_impl(").replace("dequantize<", "leap_dequantize<")
deq = deq.replace("inline void dequantize(", "inline void leap_dequantize(")
HDR = deq + "\n\n" + wide + "\n"
KERN = {}
def wide_kernel(nv, K, N, M):   # integer constants in the header, as MTPLX's kernels do (int template args + a header
    key = (nv, K, N, M)         # did not build in R-09b's first run)
    if key not in KERN:
        consts = "".join(f"constant constexpr int {k} = {v};\n" for k, v in (("GNV", nv), ("GKK", K), ("GNN", N), ("GMM", M)))
        KERN[key] = mx.fast.metal_kernel(
            name=f"r13b_wide_{nv}_{K}_{N}_{M}", input_names=["w", "scales", "biases", "x"], output_names=["y"],
            header=HDR + consts, source="""
    leap_qmv_wide_impl<T, 64, 4, GNV, 8>(w, scales, biases, x, y, GKK, GNN, GMM, threadgroup_position_in_grid,
                                         simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
""")
    return KERN[key]

def wide_call(x, W, nv):
    w, s, b = W
    M, K = x.shape
    N = w.shape[0]
    (y,) = wide_kernel(nv, K, N, M)(inputs=[w, s, b, x], template=[("T", mx.bfloat16)],
                                    grid=(32 * ((M + nv - 1) // nv), 2 * ((N + 7) // 8), 1), threadgroup=(32, 2, 1),
                                    output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])
    return y

mlx_call = lambda x, W: mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4)
SHAPES = [("shared gate/up", 640, 2560, 96), ("shared down", 2560, 640, 48), ("k/v_proj", 512, 2560, 24),
          ("in_proj_a/b", 48, 2560, 72), ("shared_expert_gate", 1, 2560, 48)]
NVS = (4, 8, 16)
eq = lambda a, b: bool(mx.all(mx.view(a, mx.uint16) == mx.view(b, mx.uint16)).item())
mx.random.seed(13)
say("R-13b: small 4-bit dense shapes of a 16-row check (kern23's shapes / counts, random weights, cold), ms per check")
tot = {k: 0.0 for k in ("mlx",) + NVS}
allbit = True
for name, N, K, cnt in SHAPES:
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(cnt)]
    X = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(Ws, X)
    try:
        bit = {nv: all(eq(wide_call(X, W, nv), mlx_call(X, W)) for W in Ws[:8]) for nv in NVS}
    except Exception as exc:
        say(f"  {name}: kernel failed: {str(exc)[:600]}")
        say("R-13b verdict: BLOCKED (the transcribed qmv_wide does not build)")
        open(os.path.join(D, "r13b_screen.txt"), "w").write("\n".join(lines) + "\n")
        raise SystemExit(2)
    t = {"mlx": timeit(lambda: [mlx_call(X, W) for W in Ws])}
    for nv in NVS:
        t[nv] = timeit(lambda nv=nv: [wide_call(X, W, nv) for W in Ws])
    for k in tot:
        tot[k] += t[k]
    allbit &= bit[16]
    say(f"  {name:19s} {N:5d} x {K:4d} ({cnt:2d}): MLX 16-row {t['mlx']:5.2f} | " +
        " | ".join(f"NV{nv} {t[nv]:5.2f} ms ({'bitwise' if bit[nv] else 'DIFFERS'})" for nv in NVS))
    del Ws
saved = tot["mlx"] - tot[16]
say(f"  per check: MLX {tot['mlx']:.2f} ms; " + ", ".join(f"NV{nv} {tot[nv]:.2f}" for nv in NVS) +
    f"; NV16 saves {saved:.2f} ms")
v = "KILL (not bitwise)" if not allbit else "PASS" if saved >= 2 else "KILL" if saved < 1 else "BETWEEN"
say(f"R-13b verdict (PASS >= 2 ms saved per check AND bitwise on every shape; KILL < 1 ms or not bitwise): {v}")
open(os.path.join(D, "r13b_screen.txt"), "w").write("\n".join(lines) + "\n")
