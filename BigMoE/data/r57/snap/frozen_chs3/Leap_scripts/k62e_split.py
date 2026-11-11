"""K-62e split (Leap, Sun 2026-11-01; shared GPU, rough): NAX4's kernel A (gate / up, 2/3 of the MACs) vs kernel B (down)
vs the two rowsum64 dispatches, at M 16 U 70; 8 chained calls per eval, median of 25."""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
import tierglu_nax4 as N4
from tierglu import PROJ, plan_masks
from tierglu_grouped import make_grp
from tierglu_nax import rowsum64
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
M, R = 16, 10
ms_np = np.stack([rng.choice(70, NR, replace=False) for _ in range(M)]).astype(np.int32)
ls_np = side_of[ms_np[:, :2]].astype(np.int32)
ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
pidx, seg, nsg_, nseg = make_grp(ms_, ls_, PLAN, M, ms_np, ls_np, 16)
m4, m2, r4 = plan_masks(PLAN)
g, u, d = (msb[p] for p in PROJ)
L = lsb
ms = ms_.reshape(-1); ls = ls_.reshape(-1)
tmpl = [("T", x.dtype), ("NR_C", R), ("MASK4_C", m4), ("MASK2_C", m2), ("R4_C", r4), ("DIAG_C", 0), ("GS_C", 1)]
xs = rowsum64(x); mx.eval(xs)
def ka():
    return N4._ka4()(inputs=[x, xs, g["w3"], u["w3"], g["s3"], u["s3"], g["b3"], u["b3"], g["b3"], u["b3"],
                             L["gate_proj"]["l1"], L["up_proj"]["l1"], L["gate_proj"]["b"], L["up_proj"]["b"], ms, ls, pidx, seg, nsg_],
                     template=tmpl + [("K_C", H), ("N_C", I)], grid=(I // 32 * 64, nseg, 1), threadgroup=(64, 1, 1),
                     output_shapes=[(M * R, I)], output_dtypes=[x.dtype])[0]
hh = ka(); hs = rowsum64(hh); mx.eval(hh, hs)
def kb():
    return N4._kb4()(inputs=[hh, hs, d["w3"], d["s3"], d["b3"], d["b3"], L["down_proj"]["l1"], L["down_proj"]["b"], ms, ls, pidx, seg, nsg_],
                     template=tmpl + [("N_C", I), ("D_C", H), ("NSG_C", 2)], grid=(H // 64 * 64, nseg, 1), threadgroup=(64, 1, 1),
                     output_shapes=[(M * R, H)], output_dtypes=[x.dtype])[0]
fns = {"A (gate/up)": ka, "B (down)": kb, "rowsum x": lambda: rowsum64(x), "rowsum h": lambda: rowsum64(hh),
       "whole": lambda: N4.tier_glu_nax4(x, msb, lsb, ms_, ls_, plan=PLAN, grp=(pidx, seg, nsg_, nseg))}
for f in fns.values():
    for _ in range(3): mx.eval(*[f() for _ in range(8)])
ts = {k: [] for k in fns}
for _ in range(25):
    for k, f in fns.items():
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[f() for _ in range(8)]); ts[k].append(time.perf_counter() - t0)
print(f"{nseg} groups: " + ", ".join(f"{k} {1e3 * float(np.median(v)) / 8:.3f} ms" for k, v in ts.items()), flush=True)
