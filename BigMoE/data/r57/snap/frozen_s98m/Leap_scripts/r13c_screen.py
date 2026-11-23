"""R-13c NAXP (R-13's exact half, family 1; NewIdeas' lines; Leap, pre-registered Tue 2026-10-27 20:44 EET in
~/Leap/TABLE_350.md 'R-13c'): the 4-bit shapes MLX serves with its non-split NAX GEMM at M = 16 (qkv 10240 x 2560,
q_proj 12288 x 2560, lm_head 248320 x 2560: split_k = 1 at bm = bn = 32 tiles >= 512 / ...), with MLX's own sums.
MLX 0.32.2's qmm_t_nax (quantized_nax.h qmm_t_nax_tgp_impl; quantized.cpp qmm_nax: bm 32 for M <= 32, bn 64, bk 64,
wm 2, wn 2): per 64-wide k step, all 128 threads dequantize a 64 x 64 weight tile into threadgroup memory (two
barriers), then each simdgroup runs two 32-wide sub-steps of the fixed 16 x 32 x 16 NAX op. At M = 16 the second
simdgroup row (rows 16-31) has no rows but still runs. R-09b: MLX's NAX order replicates bit for bit and prefetching
keeps it.
Kernels (mx.fast.metal_kernel; MLX's nax.h + quantized_nax.h's loader inlined; integer constants in the header):
  C1  the replica (BM 32, BN 64, WM 2, WN 2): must equal MLX
  C2  BM 16, BN 64, WM 1, WN 2 (no idle simdgroups)       C4  C2 with double-buffered weight tiles
  C3  BM 16, BN 32, WM 1, WN 1 (2x the threadgroups)       C5  C3 with double-buffered weight tiles
  (double buffering: the next tile is dequantized while this one computes; one barrier per step; same op order)
Shapes and per-check counts as kern23 (random 4-bit weights, group 64, cold): qkv (36), q_proj (12), lm_head (1).
Lines (NewIdeas): PASS >= 1.5 ms saved per check AND bitwise on every shape. KILL < 0.5 ms or not bitwise.
Usage: [BigMoE venv] r13c_screen.py -> ~/Leap/data/r13c_screen.txt"""
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
QN = (INC / "mlx/backend/metal/kernels/quantized_nax.h").read_text().splitlines()
i0 = next(i for i, l in enumerate(QN) if l.startswith("#define MLX_MTL_CONST"))
i1 = next(i for i, l in enumerate(QN) if "METAL_FUNC void adjust_matrix_offsets(" in l) - 1   # its template line
lib = [l for l in QN[i0:i1] if "function_constant" not in l]
HDR = inline("mlx/backend/metal/kernels/steel/gemm/nax.h", set()) + "\nusing namespace mlx::steel;\n" + "\n".join(lib) + "\n"

BODY = """
    constexpr int BKP = GBK + 16 / sizeof(T);
    threadgroup T Ws[(GDB ? 2 : 1) * GBN * BKP];
    using loader_t = QuantizedBlockLoader<T, GBN, GBK, BKP, 1, GWM * GWN * 32, 64, 4>;
    constexpr int pack_factor = get_pack_factor<4, 8>();
    constexpr int bytes_per_pack = get_bytes_per_pack<4>();
    const int K_w = GK * bytes_per_pack / pack_factor;
    const int K_g = GK / 64;
    const uint3 tid = threadgroup_position_in_grid;
    const ushort sg = ushort(simdgroup_index_in_threadgroup);
    const ushort ln = ushort(thread_index_in_simdgroup);
    const int y_row = tid.y * GBM;
    const int y_col = tid.x * GBN;
    const device uint8_t* wl = (const device uint8_t*)w + y_col * K_w;
    const device T* sc = scales + y_col * K_g;
    const device T* bi = biases + y_col * K_g;
    const device T* xp = x + y_row * GK;
    device T* yp = y + y_row * GN + y_col;
    loader_t la(wl, sc, bi, GK, Ws, sg, ln);
    loader_t lb(wl, sc, bi, GK, Ws + (GDB ? GBN * BKP : 0), sg, ln);
    constexpr short SM = GBM / GWM, SN = GBN / GWN, SK = 32;
    constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
    const short tm = SM * (sg / GWN);
    const short tn = SN * (sg % GWN);
    const short sgp_sm = short(min(int(SM), GM - (y_row + tm)));
    NAXTile<float, TM, TN> Dtile;
    Dtile.clear();
    xp += tm * GK;
    auto compute = [&](const threadgroup T* Wt) {
        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < GBK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, TN, TK> Btile;
            volatile int compiler_barrier;
            if (sgp_sm == SM) {
                Atile.load(xp + kk1, GK);
            } else {
                Atile.load_safe(xp + kk1, GK, short2(SK, sgp_sm));
            }
            Btile.template load<T, BKP, 1>(Wt + tn * BKP + kk1);
            tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
            (void)compiler_barrier;
        }
        xp += GBK;
    };
    const int steps = GK / GBK;
    if (!GDB) {
        for (int t = 0; t < steps; t++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            la.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            compute(Ws);
            la.next();
        }
    } else {
        la.load_unsafe(); la.next(); la.next();      // tile 0 -> buffer A; la now at tile 2
        lb.next();                                    // lb at tile 1
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int t = 0; t < steps; t++) {
            if (t + 1 < steps) {
                if ((t & 1) == 0) { lb.load_unsafe(); lb.next(); lb.next(); }
                else { la.load_unsafe(); la.next(); la.next(); }
            }
            compute((t & 1) == 0 ? Ws : Ws + GBN * BKP);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgp_sm == SM) {
        Dtile.store(yp + tm * GN + tn, GN);
    } else {
        Dtile.store_safe(yp + tm * GN + tn, GN, short2(SN, sgp_sm));
    }
"""
CFG = {"C1": (32, 64, 2, 2, 0), "C2": (16, 64, 1, 2, 0), "C3": (16, 32, 1, 1, 0), "C4": (16, 64, 1, 2, 1),
       "C5": (16, 32, 1, 1, 1)}                         # BM, BN, WM, WN, double buffer
KERN = {}
def kernel(name, N, K, M):
    key = (name, N, K, M)
    if key not in KERN:
        bm, bn, wm, wn, db = CFG[name]
        consts = "".join(f"constant constexpr int {k} = {v};\n" for k, v in
                         (("GBM", bm), ("GBN", bn), ("GBK", 64), ("GWM", wm), ("GWN", wn), ("GDB", db), ("GN", N),
                          ("GK", K), ("GM", M)))
        KERN[key] = mx.fast.metal_kernel(name=f"r13c_{name}_{N}_{K}_{M}", input_names=["w", "scales", "biases", "x"],
                                         output_names=["y"], header=HDR + consts, source=BODY)
    return KERN[key]

def call(name, x, W):
    w, s, b = W
    M, K = x.shape
    N = w.shape[0]
    bm, bn, wm, wn, _ = CFG[name]
    (y,) = kernel(name, N, K, M)(inputs=[w, s, b, x], template=[("T", mx.bfloat16)],
                                 grid=(32 * (N // bn), wn * ((M + bm - 1) // bm), wm), threadgroup=(32, wn, wm),
                                 output_shapes=[(M, N)], output_dtypes=[mx.bfloat16])
    return y

mlx_call = lambda x, W: mx.quantized_matmul(x, *W, transpose=True, group_size=64, bits=4)
SHAPES = [("in_proj_qkv", 10240, 2560, 36), ("q_proj", 12288, 2560, 12), ("lm_head", 248320, 2560, 1)]
eq = lambda a, b: bool(mx.all(mx.view(a, mx.uint16) == mx.view(b, mx.uint16)).item())
mx.random.seed(13)
say("R-13c: the NAX qmm family of a 16-row check (kern23's shapes / counts, random 4-bit weights, cold), ms per check")
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
        say("R-13c verdict: BLOCKED (the replica does not build)")
        open(os.path.join(D, "r13c_screen.txt"), "w").write("\n".join(lines) + "\n")
        raise SystemExit(2)
    t = {"mlx": timeit(lambda: [mlx_call(X, W) for W in Ws])}
    for c in CFG:
        t[c] = timeit(lambda c=c: [call(c, X, W) for W in Ws])
    for k in tot:
        tot[k] += t[k]
    for c in CFG:
        bit_all[c] &= bit[c]
    say(f"  {name:12s} {N:6d} x {K} ({cnt:2d}): MLX {t['mlx']:6.2f} | " +
        " | ".join(f"{c} {t[c]:6.2f} ({'bitwise' if bit[c] else 'DIFFERS'})" for c in CFG))
    del Ws
say("  per check: MLX " + f"{tot['mlx']:.2f} ms; " + ", ".join(f"{c} {tot[c]:.2f}" for c in CFG))
ok = [c for c in CFG if bit_all[c]]
if not bit_all["C1"]:
    v, best, saved = "BLOCKED (the replica C1 is not MLX's sums)", None, 0.0
else:
    best = min(ok, key=lambda c: tot[c])
    saved = tot["mlx"] - tot[best]
    v = "PASS" if saved >= 1.5 else "KILL" if saved < 0.5 else "BETWEEN"
say(f"R-13c verdict (PASS >= 1.5 ms saved per check AND bitwise on every shape; KILL < 0.5 ms or not bitwise): {v} "
    f"(best bitwise {best}: {saved:.2f} ms saved per check)")
open(os.path.join(D, "r13c_screen.txt"), "w").write("\n".join(lines) + "\n")
