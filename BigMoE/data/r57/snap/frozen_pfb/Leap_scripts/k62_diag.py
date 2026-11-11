"""K-62 diag (Leap, Sun 2026-11-01; shared GPU, rough): where tier_glu_nax's time goes at M 16 U 70 -- full kernel vs
DIAG 1 (B codes replaced by constants: MMA + A loads + scale epilogue) vs DIAG 2 (no MMA: code loads + decode only)."""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
from tierglu import tier_glu
from tierglu_grouped import make_grp
from tierglu_nax import tier_glu_nax
H, I, CM, CL, NR, PLAN = 2560, 640, 128, 64, 10, "4433333333"
mx.random.seed(0)
def ru32(*shape): return mx.random.randint(0, 2 ** 31 - 1, shape).astype(mx.uint32)
def rbf(*shape, s=0.01): return (mx.random.normal(shape) * s).astype(mx.bfloat16)
def msb_part(o, k): return {"w3": ru32(CM, o, k * 3 // 32), "s3": rbf(CM, o, k // 64), "b3": rbf(CM, o, k // 64)}
def lsb_part(o, k): return {"l1": ru32(CL, o, k // 32), "b": rbf(CL, o, k // 64)}
msb = {"gate_proj": msb_part(I, H), "up_proj": msb_part(I, H), "down_proj": msb_part(H, I)}
lsb = {"gate_proj": lsb_part(I, H), "up_proj": lsb_part(I, H), "down_proj": lsb_part(H, I)}
mx.eval(msb, lsb)
rng = np.random.default_rng(1)
side_of = rng.permutation(CM) % CL
M = 16
for U in (70, 24):
    ms_np = np.stack([rng.choice(U, NR, replace=False) for _ in range(M)]).astype(np.int32)
    ls_np = side_of[ms_np[:, :2]].astype(np.int32)
    ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
    x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
    grp = make_grp(ms_, ls_, PLAN, M, ms_np, ls_np, 16)
    fns = [lambda: tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN)]
    fns += [lambda dg=dg: tier_glu_nax(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp, diag=dg) for dg in (0, 1, 2)]
    for f in fns:
        for _ in range(3): mx.eval(*[f() for _ in range(8)])
    ts = [[] for _ in fns]
    for _ in range(25):
        for i, f in enumerate(fns):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[f() for _ in range(8)]); ts[i].append(time.perf_counter() - t0)
    t = [1e3 * float(np.median(v)) / 8 for v in ts]
    print(f"U {U}: {grp[3]} groups; tier_glu {t[0]:.3f} ms; NAX full {t[1]:.3f}, const-B (MMA+A+epilogue) {t[2]:.3f}, "
          f"no-MMA (decode only) {t[3]:.3f}", flush=True)
