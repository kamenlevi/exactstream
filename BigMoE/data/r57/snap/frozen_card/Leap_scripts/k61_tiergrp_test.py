"""K-61 TIERGRP screen (Leap, Sun 2026-11-01; shared GPU, rough timing): tierglu_grouped.tier_glu_grouped vs MoonShots' 3c
tier_glu (frozen copy) at the 125B's expert shapes, synthetic pools like K-60-T (base 128 slots, sidecar 64), but each base
expert has ONE sidecar slot (as in REF2-T: the LSB slot follows the expert), so tier-4 pairs of one expert can group.
1. bitwise: outputs as uint16, max |diff| must be 0 (M 1 / 4 / 16; routings uniform-100, uniform-12, all rows equal;
   plans 4433333333 and 3333333333; ld 0 / 1; caps 2 / 4 / 16)
2. timing at M 16: uniform over U slots (U 100 / 70 ~ natural S=16 union / 24 high sharing), tier_glu vs grouped caps
   2 / 4 / 16, interleaved rounds, 8 chained calls per eval, median of 25 rounds -> ms per call
Usage: [BigMoE venv] k61_tiergrp_test.py -> ~/Leap/data/k61_tiergrp.txt"""
import os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_card/Leap_scripts"))
from tierglu import tier_glu
from tierglu_grouped import tier_glu_grouped, make_grp

D_ = os.path.expanduser("~/Leap/data")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
H, I, CM, CL, NR = 2560, 640, 128, 64, 10
mx.random.seed(0)
def ru32(*shape):
    return mx.random.randint(0, 2 ** 31 - 1, shape).astype(mx.uint32)
def rbf(*shape, s=0.01):
    return (mx.random.normal(shape) * s).astype(mx.bfloat16)
def msb_part(o, k):
    return {"w3": ru32(CM, o, k * 3 // 32), "s3": rbf(CM, o, k // 64), "b3": rbf(CM, o, k // 64)}
def lsb_part(o, k):
    return {"l1": ru32(CL, o, k // 32), "b": rbf(CL, o, k // 64)}
msb = {"gate_proj": msb_part(I, H), "up_proj": msb_part(I, H), "down_proj": msb_part(H, I)}
lsb = {"gate_proj": lsb_part(I, H), "up_proj": lsb_part(I, H), "down_proj": lsb_part(H, I)}
mx.eval(msb, lsb)
rng = np.random.default_rng(1)
side_of = rng.permutation(CM) % CL          # base slot -> its sidecar slot

def route(M, U, mode="uniform"):
    if mode == "same":
        row = rng.choice(U, NR, replace=False)
        ms = np.stack([row] * M)
    else:
        ms = np.stack([rng.choice(U, NR, replace=False) for _ in range(M)])
    return ms.astype(np.int32), side_of[ms[:, :2]].astype(np.int32)

def bits(a):
    return np.array(a.view(mx.uint16)).astype(np.int64)

say(f"K-61 TIERGRP: MLX {mx.__version__}")
fails = 0
for plan in ("4433333333", "3333333333"):
    for M in (1, 4, 16):
        for U, mode in ((100, "uniform"), (12, "uniform"), (40, "same")):
            ms_np, ls_np = route(M, U, mode)
            x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
            ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
            for ld in (0, 1):
                ref = tier_glu(x, msb, lsb, ms_, ls_, plan=plan, ld=ld)
                for cap in (2, 4, 16):
                    out = tier_glu_grouped(x, msb, lsb, ms_, ls_, plan=plan, ld=ld, ms_np=ms_np, ls_np=ls_np, cap=cap)
                    mx.eval(ref, out)
                    d = int(np.abs(bits(ref) - bits(out)).max())
                    fails += d != 0
                    if d:
                        say(f"  BITWISE FAIL plan {plan} M {M} U {U} {mode} ld {ld} cap {cap}: max |diff| in bf16 bits {d}")
say(f"1. bitwise: {'PASS (all max |diff| 0)' if fails == 0 else f'FAIL ({fails} cases)'}")

def bench(fns, rounds=25, chain=8):
    for f in fns:
        for _ in range(3):
            mx.eval(*[f() for _ in range(chain)])
    ts = [[] for _ in fns]
    for _ in range(rounds):
        for i, f in enumerate(fns):
            mx.synchronize(); t0 = time.perf_counter(); mx.eval(*[f() for _ in range(chain)]); ts[i].append(time.perf_counter() - t0)
    return [float(np.median(t)) / chain for t in ts]

M = 16
for U in (100, 70, 24):
    ms_np, ls_np = route(M, U)
    ms_, ls_ = mx.array(ms_np), mx.array(ls_np)
    x = (mx.random.normal((M, H)) * 0.5).astype(mx.bfloat16)
    grps = {cap: make_grp(ms_, ls_, "4433333333", M, ms_np, ls_np, cap) for cap in (2, 4, 16)}
    fns = [lambda: tier_glu(x, msb, lsb, ms_, ls_, plan="4433333333")]
    fns += [lambda c=c: tier_glu_grouped(x, msb, lsb, ms_, ls_, plan="4433333333", cap=c, grp=grps[c]) for c in (2, 4, 16)]
    t = bench(fns)
    ng = {c: grps[c][3] for c in grps}
    say(f"2. M 16, U {U}: {len(np.unique(ms_np))} distinct experts, groups (cap 2/4/16) {ng[2]}/{ng[4]}/{ng[16]}; "
        f"tier_glu {1e3 * t[0]:.3f} ms; grouped cap 2 {1e3 * t[1]:.3f} ({t[0] / t[1]:.2f}x), cap 4 {1e3 * t[2]:.3f} "
        f"({t[0] / t[2]:.2f}x), cap 16 {1e3 * t[3]:.3f} ({t[0] / t[3]:.2f}x)")
say("K-61 done")
open(os.path.join(D_, "k61_tiergrp.txt"), "w").write("\n".join(lines) + "\n")
