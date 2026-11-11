"""C1 / K-62o (Leap, Mon 2026-11-02; shared GPU, rough timings): NAX6 inside BigMoE's OOO phases (ref2 CFG tier_nax_ooo).
1. Bitwise, one real 125B layer (records3 pools, slot = expert id) + real routing (Breakthrough's expA traces), S 1 / 3 / 8 /
   16, random resident shares 0.3 / 0.7 / 0.95 and LP1 pairs (4-bit rank, no LSB read: sidecar slot -2 -> tier 3):
   combine_phase(phase 1 on the post's slots, phase 2 on the spun slots) == combine_phase of ONE NAX6 call on the final
   slots (ref2's non-OOO C1 path), and each phase's rows == the one call's rows.
2. Timing at S 16 (and 8, 32): 3c phases + combine (today's OOO) vs NAX6 phases + combine, and the one-call forms.
Usage: [BigMoE venv] k62o_nax_ooo.py [layer=20] -> ~/Leap/data/k62o_nax_ooo_L<layer>.txt"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("~/MoonShots/patches/planes_engine"))   # BigMoE's tierglu, as the engine loads it
import tierglu as TG
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_chs3/Leap_scripts"))
import tierglu_nax6 as N6
from tierglu_nax import rowsum64
PROJ = TG.PROJ
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
PLAN = "4433333333"
R, R4 = 10, 2
lines = []
def say(s):
    print(s, flush=True); lines.append(s)
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
msb, lsb = {}, {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(raw[:, ow:ow + o * i * 3 // 8]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(raw[:, os_:os_ + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(raw[:, ob:ob + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(raw[:, lw:lw + o * i // 8]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(raw[:, lb:lb + o * i // 32]).view(np.uint16).reshape(E, o, i // 64)).view(mx.bfloat16)}
del raw
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
rng = np.random.default_rng(0)
mx.random.seed(0)
eqb = lambda a, b: bool(mx.array_equal(a, b, equal_nan=True).item())

def one_call(x, sm, sl, sc):
    M = x.shape[0]
    y = N6.tier_glu_nax6(x, msb, lsb, sm, sl, plan=PLAN, grp=N6.gpu_grp(sm, sl, PLAN)).reshape(M * R, -1)
    return y, TG.combine_phase(y, y, mx.ones((M, R), mx.int32), sc)
def two_phase(x, pre_m, pre_l, sm, sl, phase, sc):
    xs = rowsum64(x)
    sl1 = mx.maximum(pre_l, 0)
    y1 = N6.tier_glu_nax6(x, msb, lsb, pre_m, sl1, plan=PLAN, xs=xs, grp=N6.gpu_grp_phase(pre_m, sl1, PLAN, phase, 1)).reshape(-1, 2560)
    y2 = N6.tier_glu_nax6(x, msb, lsb, sm, sl, plan=PLAN, xs=xs, grp=N6.gpu_grp_phase(sm, sl, PLAN, phase, 2)).reshape(-1, 2560)
    return y1, y2, TG.combine_phase(y1, y2, phase, sc)
def c3_phase(x, pre_m, pre_l, sm, sl, phase, sc):
    y1 = TG.tier_glu_phase(x, msb, lsb, pre_m, mx.maximum(pre_l, 0), phase, 1, plan=PLAN)
    y2 = TG.tier_glu_phase(x, msb, lsb, sm, sl, phase, 2, plan=PLAN)
    return TG.combine_phase(y1, y2, phase, sc)
def c3_one(x, sm, sl, sc):
    return TG.tier_glu(x, msb, lsb, sm, sl, plan=PLAN, scores=sc)

def case(ids, share, lp1):
    """the post's view (pre_m / pre_l, -1 = not resident yet), the spun slots (sm / sl; -2 = LP1, no LSB read), phase"""
    M = ids.shape[0]
    sm = ids.astype(np.int32)
    sl = ids[:, :R4].astype(np.int32).copy()
    res = rng.random((M, R)) < share
    res_l = res[:, :R4] & (rng.random((M, R4)) < 0.9)
    if lp1:
        late4 = ~res_l
        sl[late4 & (rng.random((M, R4)) < 0.5)] = -2
    pre_m = np.where(res, sm, -1)
    pre_l = np.where(res_l, sl, -1)
    ok = (pre_m >= 0) & np.concatenate([pre_l >= 0, np.ones((M, R - R4), bool)], axis=1)
    phase = np.where(ok, 1, 2).astype(np.int32)
    return [mx.array(a) for a in (pre_m, pre_l, sm, sl, phase)]

bad = n = 0
for trace in ("code", "chat", "math", "bulgarian"):
    tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
    for S in (1, 3, 8, 16):
        for share in (0.3, 0.7, 0.95):
            for lp1 in (False, True):
                t = int(rng.integers(0, tr.shape[0] - S))
                ids = tr[t:t + S, :R]
                pre_m, pre_l, sm, sl, phase = case(ids, share, lp1)
                x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
                sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
                y, a = one_call(x, sm, sl, sc)
                y1, y2, b = two_phase(x, pre_m, pre_l, sm, sl, phase, sc)
                ph = np.array(phase).reshape(-1)
                i1, i2 = mx.array(np.nonzero(ph == 1)[0]), mx.array(np.nonzero(ph == 2)[0])
                ok = eqb(a, b) and (i1.size == 0 or eqb(y1[i1], y[i1])) and (i2.size == 0 or eqb(y2[i2], y[i2]))
                bad += not ok; n += 1
say(f"1. bitwise, L{LAYER}, 4 traces x S 1 / 3 / 8 / 16 x resident 0.3 / 0.7 / 0.95 x LP1 off / on ({n} cases): two phases "
    f"== one NAX6 call (combined and per phase) in {n - bad} / {n} -> {'PASS' if bad == 0 else 'FAIL'}")

def tm(f, reps=8):
    for _ in range(3):
        mx.eval(*[f() for _ in range(reps)])
    ts = []
    for _ in range(7):
        mx.synchronize(); t1 = time.perf_counter(); mx.eval(*[f() for _ in range(reps)]); ts.append(time.perf_counter() - t1)
    return 1e3 * float(np.median(ts)) / reps
for S in (8, 16, 32):
    rows = []
    for trace in ("code", "chat", "math", "bulgarian"):
        tr = np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{trace}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
        t = int(rng.integers(0, tr.shape[0] - S))
        ids = tr[t:t + S, :R]
        pre_m, pre_l, sm, sl, phase = case(ids, 0.8, False)
        x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
        rows.append([tm(lambda: c3_phase(x, pre_m, pre_l, sm, sl, phase, sc)), tm(lambda: two_phase(x, pre_m, pre_l, sm, sl, phase, sc)[2]),
                     tm(lambda: c3_one(x, sm, sl, sc)), tm(lambda: one_call(x, sm, sl, sc)[1]), len(np.unique(ids))])
    r = np.array(rows)
    say(f"2. S {S} (resident 0.8, {r[:, 4].min():.0f}-{r[:, 4].max():.0f} distinct experts): 3c phases + combine {r[:, 0].mean():.3f} ms, "
        f"NAX6 phases + combine {r[:, 1].mean():.3f} ({r[:, 0].mean() / r[:, 1].mean():.2f}x); one call: 3c {r[:, 2].mean():.3f}, "
        f"NAX6 {r[:, 3].mean():.3f} ({r[:, 2].mean() / r[:, 3].mean():.2f}x) [shared GPU, rough]")
open(os.path.expanduser(f"~/Leap/data/k62o_nax_ooo_L{LAYER}.txt"), "w").write("\n".join(lines) + "\n")
