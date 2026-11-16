"""R-13d SPK16 (R-13's exact half, family 2, the biggest piece; NewIdeas' lines; Leap, pre-registered Tue 2026-10-27
20:47 EET in ~/Leap/TABLE_350.md 'R-13d'): the 4-bit shapes MLX serves with its non-NAX split-K GEMM at M = 16
(in_proj_z 6144 x 2560 at split 2, o_proj / out_proj 2560 x 6144 at split 6), with MLX's own sums.
MLX 0.32.2 (quantized.cpp qmm_splitk; quantized.h affine_qmm_t_splitk -> qmm_t_impl): tiles of BM = BK = BN = 32,
2 x 2 simdgroups, split_k = max(1, 512 / tiles) partitions of K / split_k, each writing a bf16 partial
[split_k, M, N]; then a strided sum over axis 0 (a second launch). At M = 16 half of each 32-row tile is padding.
The per-output order is steel's BlockMMA: 8-wide simdgroup-MMA k-steps through each 32-wide k block, block by block,
independent of the tile layout.
Kernels (mx.fast.metal_kernel; MLX's steel gemm.h + quantized.h inlined, integer constants in the header so MLX's
own qmm_t_impl binds its constant refs to them), each followed by mx.sum(partials, axis=0) (MLX's own reduction):
  D1  MLX's qmm_t_impl at BM 32 (the replica: must equal MLX)
  D2  MLX's qmm_t_impl at BM 16 (no padding rows: half the MMA work and X loads)
  D3  D2's loop with double-buffered X / W tiles (the next tile loads while this one multiplies; one barrier per step)
Shapes and per-check counts as kern23 (random 4-bit weights, group 64, cold).
Lines (NewIdeas): PASS >= 4 ms saved per check AND bitwise on both shapes. KILL < 2 ms or not bitwise.
Usage: [BigMoE venv] r13d_screen.py -> ~/Leap/data/r13d_screen.txt"""
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

INC = Path(mx.__file__).resolve().parent / "include"
def inline(rel, seen):
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    for line in (INC / rel).read_text().splitlines():
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(inline(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "\n".join(out)
QH = (INC / "mlx/backend/metal/kernels/quantized.h").read_text().splitlines()
i0 = next(i for i, l in enumerate(QH) if l.startswith("#define MLX_MTL_CONST"))
i_impl = next(i for i, l in enumerate(QH) if l.startswith("METAL_FUNC void qmm_t_impl("))
depth, i1 = 0, None                                      # the end of qmm_t_impl's body
for i in range(i_impl, len(QH)):
    depth += QH[i].count("{") - QH[i].count("}")
    if "{" in "".join(QH[i_impl:i + 1]) and depth == 0:
        i1 = i + 1
        break
lib = [l for l in QH[i0:i1] if "function_constant" not in l]
HDR = inline("mlx/backend/metal/kernels/steel/gemm/gemm.h", set()) + "\nusing namespace metal;\n" + "\n".join(lib) + "\n"

SPLIT_MLX = """
    // affine_qmm_t_splitk's body with MLX's own qmm_t_impl (BM from the header)
    constexpr int BK_padded = (32 + 16 / sizeof(T));
    threadgroup T Xs[GBM * BK_padded];
    threadgroup T Ws[32 * BK_padded];
    const uint3 tid = threadgroup_position_in_grid;
    const int k_start = tid.z * GKP;
    constexpr int pack_factor = get_pack_factor<4, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<4>();
    auto wl = (const device uint8_t*)w + k_start * bytes_per_pack / pack_factor;
    qmm_t_impl<T, 64, 4, true, GBM, 32, 32>((const device uint32_t*)wl, scales + k_start / 64, biases + k_start / 64,
        x + k_start, y + tid.z * (GM * GN), Xs, Ws, GK, GN, GM, GKP, tid, thread_index_in_threadgroup,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
SPLIT_DB = """
    // the same per-output order as qmm_t_impl at BM = GBM, with double-buffered X / W tiles
    constexpr int BK = 32, BN = 32, WM = 2, WN = 2;
    constexpr int BK_padded = (BK + 16 / sizeof(T));
    threadgroup T Xs[2 * GBM * BK_padded];
    threadgroup T Ws[2 * BN * BK_padded];
    const uint3 tid = threadgroup_position_in_grid;
    const ushort sg = ushort(simdgroup_index_in_threadgroup), ln = ushort(thread_index_in_simdgroup);
    const int k_start = tid.z * GKP;
    constexpr int pack_factor = get_pack_factor<4, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<4>();
    using mma_t = mlx::steel::BlockMMA<T, T, GBM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
    using loader_x_t = mlx::steel::BlockLoader<T, GBM, BK, BK_padded, 1, WM * WN * 32>;
    using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, WM * WN * 32, 64, 4>;
    const int K_w = GK * bytes_per_pack / pack_factor;
    const int K_g = GK / 64;
    const int y_row = tid.y * GBM, y_col = tid.x * BN;
    const device T* xp = x + k_start + y_row * GK;
    auto wl = (const device uint8_t*)w + k_start * bytes_per_pack / pack_factor + y_col * K_w;
    const device T* sc = scales + k_start / 64 + y_col * K_g;
    const device T* bi = biases + k_start / 64 + y_col * K_g;
    device T* yp = y + tid.z * (GM * GN) + y_row * GN + y_col;
    loader_x_t lxa(xp, GK, Xs, sg, ln), lxb(xp, GK, Xs + GBM * BK_padded, sg, ln);
    loader_w_t lwa(wl, sc, bi, GK, Ws, sg, ln), lwb(wl, sc, bi, GK, Ws + BN * BK_padded, sg, ln);
    mma_t mma_op(sg, ln);
    const short num_els = min(GBM, GM - y_row);
    const int steps = GKP / BK;
    auto load = [&](thread loader_x_t& lx, thread loader_w_t& lw) {
        if (num_els < GBM) { lx.load_safe(short2(BK, num_els)); } else { lx.load_unsafe(); }
        lw.load_unsafe();
        lx.next(); lx.next(); lw.next(); lw.next();
    };
    load(lxa, lwa);                                   // tile 0 -> buffer A; A's loaders now at tile 2
    lxb.next(); lwb.next();                           // B's loaders at tile 1
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < steps; t++) {
        if (t + 1 < steps) {
            if ((t & 1) == 0) { load(lxb, lwb); } else { load(lxa, lwa); }
        }
        if ((t & 1) == 0) { mma_op.mma(Xs, Ws); } else { mma_op.mma(Xs + GBM * BK_padded, Ws + BN * BK_padded); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (num_els < GBM) {
        mma_op.store_result_safe(yp, GN, short2(BN, num_els));
    } else {
        mma_op.store_result(yp, GN);
    }
"""
CFG = {"D1": (32, SPLIT_MLX), "D2": (16, SPLIT_MLX), "D3": (16, SPLIT_DB)}
KERN = {}
def kernel(name, N, K, M, split):
    key = (name, N, K, M, split)
    if key not in KERN:
        bm, src = CFG[name]
        consts = "".join(f"constant constexpr int {k} = {v};\n" for k, v in
                         (("GBM", bm), ("GN", N), ("GK", K), ("GM", M), ("GKP", K // split)))
        KERN[key] = mx.fast.metal_kernel(name=f"r13d_{name}_{N}_{K}_{M}_{split}",
                                         input_names=["w", "scales", "biases", "x"], output_names=["y"],
                                         header=HDR + consts, source=src)
    return KERN[key]

def split_k_of(M, N, K):                                 # MLX 0.32.2 qmm_splitk
    split = max(1, 512 // (((N + 31) // 32) * ((M + 31) // 32)))
    split = min(split, K // 64)
    while split > 1 and K % (split * 64) != 0:
        split -= 1
    return split

def call(name, x, W):
    w, s, b = W
    M, K = x.shape
    N = w.shape[0]
    split = split_k_of(M, N, K)
    bm, _ = CFG[name]
    (part,) = kernel(name, N, K, M, split)(inputs=[w, s, b, x], template=[("T", mx.bfloat16)],
                                           grid=(32 * ((N + 31) // 32), 2 * ((M + bm - 1) // bm), 2 * split),
                                           threadgroup=(32, 2, 2), output_shapes=[(split, M, N)],
                                           output_dtypes=[mx.bfloat16])
    return mx.sum(part, axis=0)

mlx_call = lambda x, W: mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4)
SHAPES = [("in_proj_z", 6144, 2560, 36), ("o_proj+out_proj", 2560, 6144, 48)]
eq = lambda a, b: bool(mx.all(mx.view(a, mx.uint16) == mx.view(b, mx.uint16)).item())
mx.random.seed(13)
say("R-13d: the split-K family of a 16-row check (kern23's shapes / counts, random 4-bit weights, cold), ms per check")
tot = {k: 0.0 for k in ["mlx"] + list(CFG)}
bit_all = {k: True for k in CFG}
for name, N, K, cnt in SHAPES:
    Ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16), group_size=64, bits=4) for _ in range(cnt)]
    X = (mx.random.normal((16, K)) * 0.5).astype(mx.bfloat16)
    mx.eval(Ws, X)
    try:
        bit = {c: all(eq(call(c, X, W), mlx_call(X, W)) for W in Ws[:4]) for c in CFG}
    except Exception as exc:
        say(f"  {name}: kernel failed: {str(exc)[:900]}")
        say("R-13d verdict: BLOCKED (the replica does not build)")
        open(os.path.join(D, "r13d_screen.txt"), "w").write("\n".join(lines) + "\n")
        raise SystemExit(2)
    t = {"mlx": timeit(lambda: [mlx_call(X, W) for W in Ws])}
    for c in CFG:
        t[c] = timeit(lambda c=c: [call(c, X, W) for W in Ws])
    for k in tot:
        tot[k] += t[k]
    for c in CFG:
        bit_all[c] &= bit[c]
    say(f"  {name:16s} {N:5d} x {K} (split {split_k_of(16, N, K)}, {cnt}): MLX {t['mlx']:6.2f} | " +
        " | ".join(f"{c} {t[c]:6.2f} ({'bitwise' if bit[c] else 'DIFFERS'})" for c in CFG))
    del Ws
say("  per check: MLX " + f"{tot['mlx']:.2f} ms; " + ", ".join(f"{c} {tot[c]:.2f}" for c in CFG))
if not bit_all["D1"]:
    v, best, saved = "BLOCKED (the replica D1 + mx.sum is not MLX's sums: the reduction or the partials differ)", None, 0.0
else:
    ok = [c for c in CFG if bit_all[c]]
    best = min(ok, key=lambda c: tot[c])
    saved = tot["mlx"] - tot[best]
    v = "PASS" if saved >= 4 else "KILL" if saved < 2 else "BETWEEN"
say(f"R-13d verdict (PASS >= 4 ms saved per check AND bitwise on both shapes; KILL < 2 ms or not bitwise): {v} "
    f"(best bitwise {best}: {saved:.2f} ms saved per check)")
open(os.path.join(D, "r13d_screen.txt"), "w").write("\n".join(lines) + "\n")
