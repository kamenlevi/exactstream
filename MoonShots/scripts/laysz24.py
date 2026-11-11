"""R-27 LAYSZ (MoonShots Tue 20:2x; exact, CPU replay): static NON-uniform per-layer pool sizes at 24 GB. Today every
layer gets the same C_m / C_l; layers whose routing is more spread may gain more per slot. Per layer and per trace set:
LFU miss curves over a C_m / C_l grid (wall24 replay, 16-row checks accepting 10, tokens 400+); a Lagrangian picks per-layer
(C_m, C_l) on CALIBRATION sets within the exact stack's RAM (48 x (159 x 2,162,688 + 44 x 770,048): slots hold today's
decoded layout), scored on HELD-OUT sets with SSD miss bytes 1,900,544 / 720,896 (W24-ec + 11-bit at load).
PASS (written first): >= -8 % bytes / token on held-out sets vs uniform 159 / 44; KILL: < -3 %. Held-out = leave one
trace set out (calibrate on the other 3), mean over the 4 sets; grid C_m 59-339, C_l 8-104.
usage: [BigMoE venv] taskpolicy -b python laysz24.py -> stdout + data/laysz24.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wall24 as W

R, A = int(os.environ.get("LAYSZ_R", "16")), int(os.environ.get("LAYSZ_A", "10"))
POL = os.environ.get("LAYSZ_POL", "lfu")


def run_strict(seq, C, d=0.99):
    """wall24's LFU (decayed counts, forced admission) but the pool is trimmed back to C after every check: experts staged
    beyond C for the current check are released (an engine's staging buffer), not kept as cache for free"""
    miss, R, score, w = [], set(), {}, 1.0
    for N in seq:
        w /= d
        for e in N:
            score[e] = score.get(e, 0.0) + w
        m = [e for e in N if e not in R]
        miss.append(len(m))
        for e in m:
            if len(R) >= C:
                vic = [v for v in R if v not in N]
                if vic:
                    R.discard(min(vic, key=lambda v: score[v]))
            R.add(e)
        if len(R) > C:
            R = set(sorted(R, key=lambda v: -score[v])[:C])
    return miss


POOL = (lambda seq, c: run_strict(seq, c)) if POL == "lfus" else (lambda seq, c: W.run_pool(seq, c, POL))
MR, LR = 2162688, 770048                                  # RAM per slot (decoded layout)
MS, LS = int(os.environ.get("LAYSZ_MS", "1900544")), int(os.environ.get("LAYSZ_LS", "720896"))   # SSD bytes per miss
BM, BL = int(os.environ.get("LAYSZ_CM", "159")), int(os.environ.get("LAYSZ_CL", "44"))   # uniform base
LO = int(os.environ.get("LAYSZ_LO", "10"))             # grid steps below the base
CM = list(range(BM - 10 * min(LO, (BM - 1) // 10), BM + 181, 10))
CL = list(range(BL - 6 * min(int(os.environ.get("LAYSZ_LOL", "6")), BL // 6), BL + 61, 6))
BUDGET = W.NL * (BM * MR + BL * LR)
t0 = time.time()
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


cur = {}                                                  # set -> (curves_m [NL, len(CM)], curves_l [NL, len(CL)]) per token
for k, ids in W.IDS.items():
    ch = W.needs(ids, R, A)
    meas = [m for m, _ in ch]
    ntok = A * sum(meas)
    cm = np.zeros((W.NL, len(CM))); cl = np.zeros((W.NL, len(CL)))
    for l in range(W.NL):
        sm = [lay[l][0] for _, lay in ch]
        sl = [lay[l][1] for _, lay in ch]
        for i, c in enumerate(CM):
            cm[l, i] = sum(a for a, f in zip(POOL(sm, c), meas) if f) / ntok
        for i, c in enumerate(CL):
            cl[l, i] = sum(a for a, f in zip(POOL(sl, c), meas) if f) / ntok
    cur[k] = (cm, cl)
    say(f"{k}: curves done ({time.time() - t0:.0f} s); uniform {BM} / {BL}: "
        f"{(cm[:, CM.index(BM)].sum() * MS + cl[:, CL.index(BL)].sum() * LS) / 1e6:.1f} MB / token")


def alloc(sets):
    """per-layer (i, j) grid indices minimizing calibration bytes within BUDGET (Lagrangian, binary search on lambda)"""
    cm = sum(cur[k][0] for k in sets) / len(sets) * MS
    cl = sum(cur[k][1] for k in sets) / len(sets) * LS
    ram_m = np.array(CM) * MR; ram_l = np.array(CL) * LR

    def pick(lam):
        i = np.argmin(cm + lam * ram_m[None, :], 1)
        j = np.argmin(cl + lam * ram_l[None, :], 1)
        return i, j, (ram_m[i].sum() + ram_l[j].sum())
    lo, hi = 0.0, 1.0
    assert pick(hi)[2] <= BUDGET
    for _ in range(100):
        mid = (lo + hi) / 2
        if pick(mid)[2] > BUDGET:
            lo = mid
        else:
            hi = mid
    return pick(hi)


def score(sets, i, j):
    return float(np.mean([(cur[k][0][np.arange(W.NL), i].sum() * MS + cur[k][1][np.arange(W.NL), j].sum() * LS) for k in sets]))


ks = list(cur)
uni_i, uni_j = np.full(W.NL, CM.index(BM)), np.full(W.NL, CL.index(BL))
loo = []
for cal, ev in tuple((tuple(x for x in ks if x != h), (h,)) for h in ks) + tuple(((h,), (h,)) for h in ks) + (((ks[0], ks[1]), (ks[2], ks[3])), ((ks[2], ks[3]), (ks[0], ks[1])), (tuple(ks), tuple(ks))):
    i, j, ram = alloc(cal)
    u, b = score(ev, uni_i, uni_j), score(ev, i, j)
    say(f"calibrate on {'+'.join(cal)}, score on {'+'.join(ev)}: uniform {u / 1e6:.1f} -> per-layer {b / 1e6:.1f} MB / token "
        f"({100 * (b / u - 1):+.1f} %), RAM {ram / 1e9:.3f} of {BUDGET / 1e9:.3f} GB; C_m range {CM[i.min()]}-{CM[i.max()]}, "
        f"C_l range {CL[j.min()]}-{CL[j.max()]}")
    if len(ev) == 1 and cal != ev:
        loo.append(b / u - 1)
say(f"leave-one-set-out mean: {100 * np.mean(loo):+.1f} % (per set {', '.join(f'{100 * v:+.1f}' for v in loo)})")
ia, ja, ra = alloc(tuple(ks))
say("per-layer C_m (all-sets calibration): " + " ".join(str(CM[x]) for x in ia))
say("per-layer C_l (all-sets calibration): " + " ".join(str(CL[x]) for x in ja))
if os.environ.get("LAYSZ_JSON"):
    import json
    json.dump({"C_m": [CM[x] for x in ia], "C_l": [CL[x] for x in ja], "budget_bytes": int(BUDGET), "used_bytes": int(ra),
               "uniform": [BM, BL], "checks": [R, A], "policy": POL, "miss_bytes": [MS, LS],
               "loo_mean": float(np.mean(loo)), "loo_per_set": dict(zip(ks, [float(v) for v in loo]))},
              open(os.path.expanduser(os.environ["LAYSZ_JSON"]), "w"), indent=1)
    say(f"wrote {os.environ['LAYSZ_JSON']}")
say(f"({time.time() - t0:.0f} s)")
open(os.path.expanduser("~/MoonShots/data/laysz24.txt" if (BM, LO, R, A, POL) == (159, 10, 16, 10, "lfu") else f"~/MoonShots/data/laysz_{BM}_{BL}_lo{LO}_r{R}a{A}_{POL}.txt"), "w").write("\n".join(lines) + "\n")
