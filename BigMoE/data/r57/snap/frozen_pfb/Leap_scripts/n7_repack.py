"""N7-15 screen (Leap, Tue 2026-11-03 00:1x; pre-registered in ~/Leap/TABLE_350.md 'N7-15'): what the in-place GPU repack of
freshly loaded slots (tierglu_nax6n.repack_slots, BigMoE's in-service demand loads) costs against NAX6-N's gain, on the
real layer-20 pools with real routing. Dependent chains of 16 S 16 calls (+ combine_phase), call i on its own routing
window (rows t0 + 16 i ...), so the repacked slots change from call to call (cold). Arms, interleaved, 9 rounds:
  NAX6 (row-major pools) | NAX6-N (native pools) | R(nm, nl): repack_slots over nm of the call's base slots + nl of its
  sidecar slots (fresh flags per call and round), then NAX6-N on the returned slots | RO(nm, nl): the repack alone.
kept(n) = (NAX6 - R(n)) / (NAX6 - NAX6-N). Lines (the orchestrator's): PASS if kept >= 80 % at the 45 GB default's miss
rate (3 base + 1 sidecar slots per layer call ~ 192 per check; C 256 measured 171-219); KILL < 50 %.
Usage: [BigMoE venv] n7_repack.py -> ~/Leap/data/n7_repack.txt (GPU ~3.5 GB; share lock, quiet for the final number)"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_third_party/ms_gdn"))
sys.path.insert(0, os.path.expanduser("/Users/kamen/BigMoE/data/r57/snap/frozen_pfb/Leap_scripts"))
from tierglu import PROJ
import tierglu as TG
import tierglu_nax6 as N6
import tierglu_nax6n as NN

OUT = os.path.expanduser("~/Leap/data/n7_repack.txt")
def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")
LAYER, PLAN = 20, "4433333333"
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
msb_n, lsb_n = NN.repack(msb, lsb)
mx.eval([v for d in (msb_n, lsb_n) for pr in d for v in d[pr].values()])
say(f"N7-15 repack-at-load screen {time.strftime('%a %H:%M')} ({os.environ.get('N7_COND', 'shared GPU, rough')}); layer {LAYER}")
TR = {t: np.load(os.path.expanduser(f"~/Breakthrough/data/expA/traces125/{t}_800_g.npz"))[f"L{LAYER}"].astype(np.int32)
      for t in ("code", "chat", "math", "bulgarian")}
R, S, NCH = len(PLAN), 16, 16
RARMS = [(0, 0), (1, 0), (2, 1), (3, 1), (6, 2), (12, 4)]
ROARMS = [(0, 0), (1, 0), (4, 0), (8, 0), (16, 0), (0, 4), (0, 16)]
rng = np.random.default_rng(0)
mx.random.seed(0)
def flags(ids, nm, nl):
    """fresh per-call flag arrays: nm of the call's unique base slots, nl of its unique sidecar (4-bit rank) slots"""
    dm, dl = np.zeros(E, np.int32), np.zeros(E, np.int32)
    ub, us = np.unique(ids), np.unique(ids[:, :2])
    dm[rng.choice(ub, min(nm, len(ub)), replace=False)] = 1
    dl[rng.choice(us, min(nl, len(us)), replace=False)] = 1
    return mx.array(dm), mx.array(dl)
acc = {}
for trace in TR:
    for t0 in (0, TR[trace].shape[0] - S * NCH):                       # chat's trace has 559 rows
        win = [TR[trace][t0 + S * i:t0 + S * i + S, :10].copy() for i in range(NCH)]
        ms = [mx.array(w) for w in win]
        ls = [mx.array(w[:, :2].copy()) for w in win]
        x0 = (mx.random.normal((S, 2560)) * 0.5).astype(mx.bfloat16)
        sc = mx.softmax(mx.random.normal((S, R)), axis=-1).astype(mx.bfloat16)
        one = mx.ones((S, R), mx.int32)
        mx.eval(x0, sc, one, *ms, *ls)
        def chain(kind, fl=None):
            out = None
            for i in range(NCH):
                x = x0 if out is None else mx.depends(x0, out)
                if kind == "NAX6":
                    y = N6.tier_glu_nax6(x, msb, lsb, ms[i], ls[i], plan=PLAN, grp=N6.gpu_grp(ms[i], ls[i], PLAN))
                elif kind == "RO":
                    s2, t2 = NN.repack_slots(msb_n, lsb_n, ms[i] if out is None else mx.depends(ms[i], out), ls[i], *fl[i])
                    out = s2 + t2.sum()
                    continue
                else:
                    s2, t2 = ms[i], ls[i]
                    if kind == "R":
                        s2, t2 = NN.repack_slots(msb_n, lsb_n, s2 if out is None else mx.depends(s2, out), t2, *fl[i])
                    y = NN.tier_glu_nax6n(x, msb_n, lsb_n, s2, t2, plan=PLAN, grp=N6.gpu_grp(s2, t2, PLAN))
                y = y.reshape(-1, 2560)
                out = TG.combine_phase(y, y, one, sc)
            return out
        arms = [("NAX6", None), ("NAX6-N", None)] + [(("R", n)) for n in RARMS] + [("RO", n) for n in ROARMS]
        name = lambda k, n: k if n is None else f"{k}{n}"
        def run(k, n):
            fl = [flags(win[i], *n) for i in range(NCH)] if n is not None else None
            if fl:
                mx.eval([a for f in fl for a in f])
            out = chain(k, fl)                                                # graph built outside the timing
            mx.synchronize()
            t1 = time.perf_counter()
            mx.eval(out)
            return (time.perf_counter() - t1) / NCH
        for k, n in arms:
            run(k, n)                                                         # warm-up / compile
        ts = {name(k, n): [] for k, n in arms}
        for _ in range(9):
            for k, n in arms:
                ts[name(k, n)].append(run(k, n))
        for k, v in ts.items():
            acc.setdefault(k, []).append(float(np.median(v)))
m = {k: 1e3 * float(np.mean(v)) for k, v in acc.items()}
gain = m["NAX6"] - m["NAX6-N"]
say(f"1. per S 16 call (+ combine): NAX6 {m['NAX6']:.3f} ms, NAX6-N {m['NAX6-N']:.3f} ms "
    f"({100 * gain / m['NAX6']:+.1f} %, gain {1e3 * gain:.0f} us)")
for n in RARMS:
    r = m[f"R{n}"]
    say(f"   R{n}: {r:.3f} ms; kept {100 * (m['NAX6'] - r) / gain:.0f} % of the gain")
say("2. the repack alone, per call: " + "; ".join(f"RO{n} {1e3 * m[f'RO{n}']:.1f} us" for n in ROARMS))
base_us = 1e3 * (m["RO(16, 0)"] - m["RO(0, 0)"]) / 16
side_us = 1e3 * (m["RO(0, 16)"] - m["RO(0, 0)"]) / 16
say(f"   ~{1e3 * m['RO(0, 0)']:.1f} us per call with nothing dirty, ~{base_us:.1f} us per base slot, ~{side_us:.1f} us per "
    f"sidecar slot; break-even (kept 0 %) at ~{1e3 * gain / max(base_us, 1e-9):.0f} base slots per call")
k31 = (m["NAX6"] - m["R(3, 1)"]) / gain
v = "PASS" if k31 >= 0.80 else "KILL" if k31 < 0.50 else "BETWEEN"
say(f"N7-15 verdict (kept at 3 base + 1 sidecar slots per call; PASS >= 80 %, KILL < 50 %): {v} ({100 * k31:.0f} %)")
