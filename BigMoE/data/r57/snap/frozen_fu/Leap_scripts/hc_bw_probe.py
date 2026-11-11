"""D3-0 (Leap, Wed 2026-11-04 17:1x; TABLE_350 'D3-0'; HC_READ.md D3): is the exact 16-row HC read's mix-down GEMM (leapk S4,
the merged down + inject weights, N 324 x K 10,240 bf16, 6.6 MB per read) bound by its weight bytes? D3 (the lossless
exponent-offset pack in the loaders, ~12.4 bits per weight, -22 %) pays only if it is.
Arms (48 distinct weight copies cycled, as 48 layers' reads in a check, so the bytes come from DRAM):
  S4          leapk.s4_partials as REF2 runs it;
  S4-cached   the same kernel with every tile's B rows pointed at tile 0's (the same arithmetic count, ~1/11 of the DRAM
              bytes): the time left when the weight bytes are (nearly) free.
time(S4) - time(S4-cached) = the byte-bound share; D3 can save at most ~22 % of it.
usage: [BigMoE venv] hc_bw_probe.py -> ~/Leap/data/hc_bw_probe.txt (GPU, MEM ~0.6 GB; share lock)"""
import os, sys, json, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
import leapk as LK

OUT = os.path.expanduser("~/Leap/data/hc_bw_probe.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
say(f"\nD3-0 {time.strftime('%a %H:%M:%S')}: MLX {mx.__version__}")
N, K, NL = 324, 10240, 48
mx.random.seed(0)
Ws = [(mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16) for _ in range(NL)]
x = (mx.random.normal((1, 16, K)) * 0.5).astype(mx.bfloat16)
mx.eval(Ws, x)
SRC_C = LK._S4_SRC.replace("const device T* Bp = w + size_t(c_col) * GK + k0;", "const device T* Bp = w + k0;")
assert SRC_C != LK._S4_SRC
PS = 4096
NP, NT = (K + PS - 1) // PS, (N + 31) // 32
kc = mx.fast.metal_kernel(name="leapk_s4_cached_probe", input_names=["x", "w"], output_names=["cs"], source=SRC_C,
                          header=LK._nax_header() + "\n" + LK.consts(GN=N, GK=K, GPS=PS, GNP=NP, GNT=NT, GPF=4))
def s4c(a, w):
    (cs,) = kc(inputs=[a, w], template=[("T", mx.bfloat16)], grid=(32 * NT * NP, 1, 1), threadgroup=(32, 1, 1),
               output_shapes=[(NP, 16, N)], output_dtypes=[mx.float32])
    return cs
ARMS = {"S4": lambda: [LK.s4_partials(x, w)[0] for w in Ws], "S4-cached": lambda: [s4c(x, w) for w in Ws]}
for f in ARMS.values():
    mx.eval(f())
T = {k: [] for k in ARMS}
for r in range(15):
    for k in (list(ARMS) if r % 2 == 0 else list(ARMS)[::-1]):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(ARMS[k]()); mx.synchronize()
        T[k].append((time.perf_counter() - t0) / NL)
a, c = np.median(T["S4"]), np.median(T["S4-cached"])
say(f"us per S4 call (48 distinct 6.6 MB weights, median of 15): S4 {1e6 * a:.1f} ({N * K * 2 / a / 1e9:.0f} GB/s), "
    f"S4-cached {1e6 * c:.1f}; byte-bound share {1e6 * (a - c):.1f} us ({100 * (a - c) / a:.0f} %); D3's ceiling (-22 % of "
    f"it) {1e6 * 0.22 * (a - c):.1f} us per read = {96 * 0.22 * (a - c) * 1e3:.2f} ms per check (96 reads, down GEMM only)")
