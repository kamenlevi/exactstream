"""NAX-C screen (Leap, Mon 2026-11-02 22:5x; pre-registered in ~/Leap/TABLE_350.md 'NAX-C'): design C (tierglu_naxc.py,
int8 x int4 -> int32 matrix ops, FAST) vs NAX6 on the real layer-20 pools with real routing (Breakthrough's expA traces):
1. Quality, one S 16 pass per trace: per (row, rank) output, |y - ref| / RMS(ref) against an fp64 reference from the same
   dequantized weights (K-62g's deq: s q + b, tier 4 s / 2 (2 q + l) + b_lsb; gate / up, swiglu, down in fp64), for
   NAX-C, NAX6 and 3c; max, mean, and the share of NAX-C's errors above NAX6's.
2. Speed: the whole call (+ combine_phase) in DEPENDENT chains of 16, NAX6 vs NAX-C, S 16 / 8 / 32, 4 traces x 4 passes,
   median of 9 interleaved rounds.
Lines (FAST, speed): PASS NAX-C >= 15 % faster than NAX6 at S 16; KILL < 5 %. Quality reported for EvalSuite / the map.
Usage: [BigMoE venv] naxc_screen.py -> ~/Leap/data/naxc_screen.txt"""
import json, os, sys, time
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_fu/Leap_scripts"))
from tierglu import tier_glu, PROJ
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_naxc as NC

OUT = os.path.expanduser("~/Leap/data/naxc_screen.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, PLAN = 20, "4433333333"
RD = os.path.expanduser("~/BigMoE/models/bare/records3")
idx = json.load(open(f"{RD}/index.json"))["layers"][str(LAYER)]
E, REC = idx["E"], idx["recsz"]
SH = {"gate_proj": (640, 2560), "up_proj": (640, 2560), "down_proj": (2560, 640)}
raw = np.fromfile(f"{RD}/{idx['file']}", dtype=np.uint8).reshape(E, REC)
msb, lsb, npc = {}, {}, {}
for p, pr in enumerate(PROJ):
    o, i = SH[pr]
    ow, os_, ob = idx["msb_piece_off"][3 * p:3 * p + 3]
    lw, lb = idx["lsb_piece_off"][2 * p:2 * p + 2]
    npc[pr] = (raw[:, ow:ow + o * i * 3 // 8], raw[:, os_:os_ + o * i // 32].view(np.uint16), raw[:, ob:ob + o * i // 32].view(np.uint16),
               raw[:, lw:lw + o * i // 8], raw[:, lb:lb + o * i // 32].view(np.uint16))
    msb[pr] = {"w3": mx.array(np.ascontiguousarray(npc[pr][0]).view(np.uint32).reshape(E, o, i * 3 // 32)),
               "s3": mx.array(np.ascontiguousarray(npc[pr][1]).reshape(E, o, i // 64)).view(mx.bfloat16),
               "b3": mx.array(np.ascontiguousarray(npc[pr][2]).reshape(E, o, i // 64)).view(mx.bfloat16)}
    lsb[pr] = {"l1": mx.array(np.ascontiguousarray(npc[pr][3]).view(np.uint32).reshape(E, o, i // 32)),
               "b": mx.array(np.ascontiguousarray(npc[pr][4]).reshape(E, o, i // 64)).view(mx.bfloat16)}
mx.eval([v for d in (msb, lsb) for pr in d for v in d[pr].values()])
say(f"NAX-C screen {time.strftime('%a %H:%M')} ({os.environ.get('NAXC_COND', 'shared GPU, rough')}); layer {LAYER}, plan {PLAN}")
def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
def deq(pr, e, tier):
    """K-62g's fp64 dequant of one expert's projection"""
    o, i = SH[pr]
    w3, s3, b3, l1, bl = (a[e] for a in npc[pr])
    q = np.unpackbits(np.ascontiguousarray(w3).reshape(o, -1), axis=1, bitorder="little").reshape(o, i, 3)
    q = (q[..., 0] + 2 * q[..., 1] + 4 * q[..., 2]).astype(np.float64)
    s = np.repeat(bf16(np.ascontiguousarray(s3).reshape(o, -1)), 64, axis=1)
    if tier == 4:
        l = np.unpackbits(np.ascontiguousarray(l1).reshape(o, -1), axis=1, bitorder="little").astype(np.float64)
        b = np.repeat(bf16(np.ascontiguousarray(bl).reshape(o, -1)), 64, axis=1)
        return s * 0.5 * (2 * q + l) + b
    b = np.repeat(bf16(np.ascontiguousarray(b3).reshape(o, -1)), 64, axis=1)
    return s * q + b
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
R = len(PLAN)
mx.random.seed(0)
# ---------------------------------------------------------------- 1. quality vs fp64
S = 16
for trace in TR:
    ids = TR[trace][300:300 + S, :10].copy()
    x = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
    ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
    grp = N6.gpu_grp(ms_, ls_, PLAN)
    outs = {"NAX-C": NC.tier_glu_naxc(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp),
            "NAX6": N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=grp),
            "3c": tier_glu(x, msb, lsb, ms_, ls_, plan=PLAN)}
    mx.eval(list(outs.values()))
    outs = {k: np.array(v.astype(mx.float32)).astype(np.float64) for k, v in outs.items()}
    xf = np.array(x.astype(mx.float32)).astype(np.float64)
    ref = np.zeros_like(outs["NAX6"])
    for m in range(S):
        for r in range(R):
            t = 4 if PLAN[r] == "4" else 3
            g = deq("gate_proj", ids[m, r], t) @ xf[m]
            u = deq("up_proj", ids[m, r], t) @ xf[m]
            ref[m, r] = deq("down_proj", ids[m, r], t) @ (g / (1 + np.exp(-g)) * u)
    rms = float(np.sqrt((ref ** 2).mean()))
    err = {k: np.abs(v - ref) for k, v in outs.items()}
    say(f"1. {trace:9s}: " + "; ".join(f"{k} max {e.max() / rms:.2e} mean {e.mean() / rms:.2e}" for k, e in err.items())
        + f"; NAX-C err > NAX6 err {100 * (err['NAX-C'] > err['NAX6']).mean():.0f} %")
# ---------------------------------------------------------------- 2. speed
NCH = 16
def call6(x, ms_, ls_, sc, one):
    y = N6.tier_glu_nax6(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
    return TG.combine_phase(y, y, one, sc)
def callc(x, ms_, ls_, sc, one):
    y = NC.tier_glu_naxc(x, msb, lsb, ms_, ls_, plan=PLAN, grp=N6.gpu_grp(ms_, ls_, PLAN)).reshape(-1, 2560)
    return TG.combine_phase(y, y, one, sc)
for S in (16, 8, 32):
    acc = {}
    for trace in TR:
        for t in np.linspace(0, TR[trace].shape[0] - S, 4).astype(int):
            ids = TR[trace][t:t + S, :10].copy()
            ms_, ls_ = mx.array(ids), mx.array(ids[:, :2].copy())
            x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
            sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
            one = mx.ones((S, R), mx.int32)
            mx.eval(x0, sc, one)
            def chain(f):
                out = None
                for _ in range(NCH):
                    out = f(x0 if out is None else mx.depends(x0, out), ms_, ls_, sc, one)
                return out
            fns = {"NAX6": lambda: chain(call6), "NAX-C": lambda: chain(callc)}
            for f in fns.values():
                mx.eval(f())
            ts = {k: [] for k in fns}
            for _ in range(9):
                for k, f in fns.items():
                    mx.synchronize()
                    t1 = time.perf_counter()
                    mx.eval(f())
                    ts[k].append((time.perf_counter() - t1) / NCH)
            for k, v in ts.items():
                acc.setdefault(k, []).append(float(np.median(v)))
    m = {k: 1e3 * float(np.mean(v)) for k, v in acc.items()}
    say(f"2. S {S}: per call NAX6 {m['NAX6']:.3f} -> NAX-C {m['NAX-C']:.3f} ms ({100 * (1 - m['NAX-C'] / m['NAX6']):+.1f} %)")
    if S == 16:
        g16 = 1 - m["NAX-C"] / m["NAX6"]
v = "PASS" if g16 >= 0.15 else "KILL" if g16 < 0.05 else "BETWEEN"
say(f"NAX-C verdict (FAST speed line: PASS >= 15 % faster than NAX6 at S 16, KILL < 5 %): {v} ({100 * g16:.1f} %)")
