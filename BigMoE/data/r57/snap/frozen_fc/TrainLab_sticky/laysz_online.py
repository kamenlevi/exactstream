"""R-29 LAYSZ-ON (NewIdeas Tue ~20:30, QUEUE R-29; follow-on of MoonShots' R-27): ONLINE per-layer pool sizes at 24 GB.
MoonShots' R-27 picks static per-layer C_m / C_l (a Lagrangian on calibration sets' LFU miss curves, the RAM of uniform
159 / 44) and scores held-out sets. Under the STRICT replay (pool trimmed back to C after every check; laysz24.py
LAYSZ_POL=lfus, LAYSZ_LO=15) static is -10.3 % (16/10) / -8.9 % (4/3) held-out, and self-calibration on the scored set
itself (the in-sample ceiling of a static allocation) -14.7 % / -13.6 % (MoonShots 20:3x).
Online: the pools start uniform; at token 400 and every 200 tokens after, each layer's (C_m, C_l) is re-fit by the same
Lagrangian on the trace's OWN miss curves seen so far (fixed-size strict-LFU replays at every grid size, checks fully
inside [200, t): tokens 0-199 fill the pools), same RAM; the pools resize at check boundaries (a shrink drops the lowest
LFU scores, the strict rule's own trim). Scored like R-27: SSD bytes per token over the checks starting in [400, 800),
against uniform, static held-out (calibrated on the other 3 sets) and the self-calibrated ceiling; per set and mean;
16-row checks accepting 10, and 4 / 3.
Replay = MoonShots' (read-only import of ~/MoonShots/scripts/wall24.py for the expA traces and the checks; laysz24.py
runs and writes at import, so its run_strict, constants, grid and Lagrangian are copied here). Check: static held-out and
self-calibrated must reproduce MoonShots' per-set numbers before the online line is read.
PASS (NewIdeas, written first): online >= 4 points beyond static held-out (mean of the 4 sets) in both shapes; KILL:
< 2 points in both; else between.
usage: [BigMoE venv] taskpolicy -b python sticky/laysz_online.py -> data/laysz_online.txt"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/MoonShots/scripts"))
import wall24 as W  # noqa: E402  (read-only: traces, needs())

MR, LR = 2162688, 770048                                  # RAM per slot (decoded layout)      (laysz24.py)
MS, LS = 1900544, 720896                                  # SSD bytes per miss (W24-ec + 11-bit)
BM, BL, LO = 159, 44, 15                                  # uniform base, grid steps below it (MoonShots' LAYSZ_LO=15)
CM = list(range(BM - 10 * min(LO, (BM - 1) // 10), BM + 181, 10))
CL = list(range(BL - 6 * min(6, BL // 6), BL + 61, 6))
BUDGET = W.NL * (BM * MR + BL * LR)
W0, T1, EVERY = 200, 400, 200                             # curve window start, first re-fit, re-fit period (tokens)
SHAPES = [(int(r), int(a)) for r, a in (s.split("/") for s in os.environ.get("LAYON_SHAPES", "16/10,4/3").split(","))]
t0, lines = time.time(), []


def say(s):
    print(s, flush=True); lines.append(s)


def run_strict(seq, caps, d=0.99):
    """laysz24.run_strict (LFU: decayed counts, forced admission, the pool trimmed back to C after every check) with a
    capacity per check; when a check's capacity is below the pool's size the pool is first trimmed to it (same rule)"""
    miss, R, score, w = [], set(), {}, 1.0
    for N, C in zip(seq, caps):
        if len(R) > C:
            R = set(sorted(R, key=lambda v: -score[v])[:C])
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


def alloc(cm, cl):
    """laysz24.alloc: per-layer grid indices minimizing curve bytes (cm [NL, |CM|], cl [NL, |CL|], misses per token)
    within BUDGET (Lagrangian, binary search on lambda)"""
    cm, cl = cm * MS, cl * LS
    ram_m, ram_l = np.array(CM) * MR, np.array(CL) * LR

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


res = {}
for R, A in SHAPES:
    S = {}                                                # set -> per-check data
    for k, ids in W.IDS.items():
        ch = W.needs(ids, R, A)
        pos = np.arange(len(ch)) * A                      # each check's first token
        meas = np.array([m for m, _ in ch])
        sm = [[lay[l][0] for _, lay in ch] for l in range(W.NL)]
        sl = [[lay[l][1] for _, lay in ch] for l in range(W.NL)]
        n = len(ch)
        mm = np.array([[run_strict(sm[l], [c] * n) for c in CM] for l in range(W.NL)], np.float64)   # [NL, |CM|, n]
        ml = np.array([[run_strict(sl[l], [c] * n) for c in CL] for l in range(W.NL)], np.float64)
        ntok = A * meas.sum()
        S[k] = dict(pos=pos, meas=meas, sm=sm, sl=sl, mm=mm, ml=ml, ntok=ntok,
                    cm=mm[:, :, meas].sum(2) / ntok, cl=ml[:, :, meas].sum(2) / ntok)
        say(f"R {R} A {A} {k}: {n} checks ({meas.sum()} measured), curves done ({time.time() - t0:.0f} s)")
    ks = list(S)
    bpt = lambda k, i, j: (S[k]["cm"][np.arange(W.NL), i].sum() * MS + S[k]["cl"][np.arange(W.NL), j].sum() * LS)
    ui, uj = np.full(W.NL, CM.index(BM)), np.full(W.NL, CL.index(BL))
    rows = {}
    for k in ks:
        u = bpt(k, ui, uj)
        oth = [x for x in ks if x != k]
        i_s, j_s, _ = alloc(sum(S[x]["cm"] for x in oth) / 3, sum(S[x]["cl"] for x in oth) / 3)
        i_c, j_c, _ = alloc(S[k]["cm"], S[k]["cl"])
        # online: uniform until T1, then re-fit every EVERY tokens on the checks fully inside [W0, t)
        s, n = S[k], len(S[k]["pos"])
        cap_m, cap_l = np.full((W.NL, n), BM), np.full((W.NL, n), BL)
        fits = []
        for t in range(T1, 800, EVERY):
            seen = (s["pos"] >= W0) & (s["pos"] + R <= t)
            ntk = A * seen.sum()
            i_o, j_o, ram = alloc(s["mm"][:, :, seen].sum(2) / ntk, s["ml"][:, :, seen].sum(2) / ntk)
            nxt = s["pos"] >= t
            cap_m[:, nxt] = np.array(CM)[i_o][:, None]
            cap_l[:, nxt] = np.array(CL)[j_o][:, None]
            fits.append((t, int(seen.sum()), ram))
        mo = sum(np.array(run_strict(s["sm"][l], cap_m[l]))[s["meas"]].sum() for l in range(W.NL))
        lo_ = sum(np.array(run_strict(s["sl"][l], cap_l[l]))[s["meas"]].sum() for l in range(W.NL))
        on = (mo * MS + lo_ * LS) / s["ntok"]
        rows[k] = dict(uniform=u, static=bpt(k, i_s, j_s), selfcal=bpt(k, i_c, j_c), online=on)
        say(f"  {k}: uniform {u / 1e6:.1f} MB / token; static held-out {100 * (rows[k]['static'] / u - 1):+.1f} %, "
            f"self-calibrated {100 * (rows[k]['selfcal'] / u - 1):+.1f} %, ONLINE {100 * (on / u - 1):+.1f} % (re-fits at "
            + ", ".join(f"{t} on {c} checks, RAM {r / 1e9:.3f} GB" for t, c, r in fits) + f" of {BUDGET / 1e9:.3f})")
    mean = {m: float(np.mean([rows[k][m] / rows[k]["uniform"] - 1 for k in ks])) for m in ("static", "selfcal", "online")}
    gain = 100 * (mean["static"] - mean["online"])
    res[R, A] = dict(rows=rows, mean=mean, gain=gain)
    say(f"R {R} A {A} mean of the 4 sets: static held-out {100 * mean['static']:+.1f} %, self-calibrated "
        f"{100 * mean['selfcal']:+.1f} %, ONLINE {100 * mean['online']:+.1f} % -> {gain:+.1f} points beyond static "
        f"(ceiling {100 * (mean['static'] - mean['selfcal']):+.1f}); per set beyond static "
        + ", ".join(f"{k} {100 * (rows[k]['static'] - rows[k]['online']) / rows[k]['uniform']:+.1f}" for k in ks))
g = [res[s]["gain"] for s in SHAPES]
verdict = ("PASS (>= 4 points beyond static in both shapes)" if min(g) >= 4 else
           "KILL (< 2 points beyond static in both shapes)" if max(g) < 2 else "between (NewIdeas' call)")
say(f"VERDICT R-29 (pre-registered; NewIdeas): online {', '.join(f'{x:+.1f}' for x in g)} points beyond static "
    f"({', '.join(f'{r}/{a}' for r, a in SHAPES)}) -> {verdict}; ({time.time() - t0:.0f} s)")
if len(SHAPES) == 2:
    open(os.path.expanduser("~/TrainLab/data/laysz_online.txt"), "w").write("\n".join(lines) + "\n")
