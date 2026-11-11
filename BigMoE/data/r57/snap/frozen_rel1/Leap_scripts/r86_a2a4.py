"""A2 + A4 (ROUTE_350 block A; DAY9 condition 5; Leap, Mon 2026-11-02; quiet GPU): are bmm16 (L1-29, the HC mix-down)
and SPK16 (R-13d's D2, the split-K shapes) still gains over what the 45 GB default runs today?
REF2-T runs the HC read's mix-down (+ its 4 gate rows) on S4 (leapk.s4_gemm: MLX's NAX split-K sums; R-09b / R-36) and
16-row 4-bit projections with N >= 1024 on qmm16c. Kernel level, real shapes, random weights, interleaved medians:
- A2: x [16, 10240] (hc 4 x 2560) times the mix-down: MLX x @ W.T (W [320, 10240]), bmm16 (ks 10, L1-29's best), S4 on
  wcat [324, 10240] (REF2's call; it includes the gate rows) -> per check x 97 HC reads.
- A4: in_proj_z [6144, 2560] (x 36) and o_proj / out_proj [2560, 6144] (x 48) at 16 rows: MLX's 16-row call (split-K), D2
  (SPK16, bitwise = MLX), qmm16c (REF2's) -> per check.
- bits: bmm16 vs S4's mix rows, D2 vs qmm16c.
Lines (DAY9 #5): A2 KILL if tokens differ or < -1.5 ms; A4 KILL if < -2 ms; both against the DEFAULT's kernels.
Usage: [BigMoE venv] r86_a2a4.py -> ~/Leap/data/r86_a2a4.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_rel1/Leap_scripts"))
from leapk import s4_gemm, splitk_d2
from bmm30 import bmm16
from qmm16 import qmm16c
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
def tm(f, reps=20):
    for _ in range(3):
        mx.eval(*[f() for _ in range(reps)])
    ts = []
    for _ in range(9):
        mx.synchronize(); t1 = time.perf_counter(); mx.eval(*[f() for _ in range(reps)]); ts.append(time.perf_counter() - t1)
    return 1e3 * float(np.median(ts)) / reps
eqb = lambda a, b: bool(mx.array_equal(a, b).item())
mx.random.seed(0)
BF = mx.bfloat16
# A2
x = (mx.random.normal((16, 10240)) * 0.5).astype(BF)
Wd = (mx.random.normal((320, 10240)) * 0.01).astype(BF)
Wg = (mx.random.normal((4, 10240)) * 0.01).astype(BF)
wcat = mx.concatenate([Wd, Wg], 0)
WdT = mx.contiguous(Wd.T)
mx.eval(x, Wd, wcat, WdT)
f_mlx = lambda: x @ Wd.T
f_bmm = lambda: bmm16(x, Wd, ks=10)
f_s4 = lambda: s4_gemm(x[None], wcat)
t = {k_: [] for k_ in ("mlx", "bmm16", "s4")}
for _ in range(3):
    t["mlx"].append(tm(f_mlx)); t["bmm16"].append(tm(f_bmm)); t["s4"].append(tm(f_s4))
m = {k_: float(np.median(v)) for k_, v in t.items()}
s4m = f_s4()[0, :, :320]
say(f"A2 HC mix-down (16 x 10240 -> 320; per call ms / per check x 97): MLX {m['mlx']:.4f} / {97 * m['mlx']:.2f}, bmm16 "
    f"{m['bmm16']:.4f} / {97 * m['bmm16']:.2f}, S4 (REF2's, + 4 gate rows) {m['s4']:.4f} / {97 * m['s4']:.2f}; bmm16 - S4 per check "
    f"{97 * (m['bmm16'] - m['s4']):+.2f} ms; bits bmm16 == S4's mix rows: {eqb(f_bmm(), s4m)}, S4 == MLX: {eqb(s4m, f_mlx())}")
# A4
for name, N, K, cnt in (("in_proj_z", 6144, 2560, 36), ("o_proj / out_proj", 2560, 6144, 48)):
    W = mx.quantize((mx.random.normal((N, K)) * 0.02).astype(BF), group_size=64, bits=4)
    xx = (mx.random.normal((16, K)) * 0.5).astype(BF)
    mx.eval(W, xx)
    fm = lambda: mx.quantized_matmul(xx, *W, transpose=True, group_size=64, bits=4)
    fd = lambda: splitk_d2(xx, W)
    fq = lambda: qmm16c(xx, *W)
    tt = {k_: [] for k_ in ("mlx", "d2", "q16")}
    for _ in range(3):
        tt["mlx"].append(tm(fm)); tt["d2"].append(tm(fd)); tt["q16"].append(tm(fq))
    mm = {k_: float(np.median(v)) for k_, v in tt.items()}
    say(f"A4 {name} [{N} x {K}] x {cnt} per check at 16 rows: MLX {cnt * mm['mlx']:.2f} ms, D2 (SPK16) {cnt * mm['d2']:.2f}, "
        f"qmm16c (REF2's) {cnt * mm['q16']:.2f}; D2 - qmm16c {cnt * (mm['d2'] - mm['q16']):+.2f} ms per check; bits D2 == MLX "
        f"{eqb(fd(), fm())}, D2 == qmm16c {eqb(fd(), fq())}")
open(os.path.expanduser("~/Leap/data/r86_a2a4.txt"), "w").write("\n".join(lines) + "\n")
