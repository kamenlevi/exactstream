#!/usr/bin/env python3 -I
"""replay_c3.py (Fable C, mission C3, Tue 3 Nov 2026, 00:0x EET). CPU only (numpy, nice 15, 4 threads), recorded data only.
The SSD bytes at the 50 GB route's cache sizes and on the way back to 45 GB, priced on the union per 16-row block with
replay_c2's method (cap45's real check rows incl. the rejected ones, D-23 cf vectors transplanted per check as band45
does, the t 0.3 cut modelled, per-layer LRU + 30 x popularity, strict pools, prompts chained with res0 forced, prompts
2-6 counted, 2 seeds). Size files: M-17 (laysz45_{256,320}.json) and MoonShots' M-25b joint fits (joint45_{C}_{extra}.json,
extra = +0 / +2070 / +4140 MB over the uniform pools). Rules: EXACT, Q1 theta 0.99 / 0.9, and each with the sidecar
DROPPED (plan 3333333333: ranks 0-1 read no LSB part; sidecar misses 0, the freed slots NOT re-spent: a lower bound).
RAM per arm = pool bytes (C_m x 2,162,688 + C_l x 770,048 over 48 layers) + 8.1 GB of everything else (the 45 GB
default: 41.7 GB whole process at uniform C 256 = 33.9 GB of pools + ~8 GB; cachesize45: 38.0 GB at C 224 + 0.13 per unit).
usage: nice -n 15 ~/BigMoE/.venv/bin/python -I replay_c3.py -> stdout (tee to replay_c3.out)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, pickle, time
import numpy as np

T0 = time.time()
NL, E, R4 = 48, 512, 2
MR, LR = 2162688, 770048
OTHER_GB = 8.1
PKL = os.path.expanduser("~/MoonShots/data/band45_cap45.pkl")   # MoonShots' own laysz45.load() cache (a sibling session), not a download
D23 = os.path.expanduser("~/TrainLab/data/d23_runs.jsonl")
MS = os.path.expanduser("~/MoonShots/data")
SIZES = [("M-17 C 256", "laysz45_256.json"), ("joint C 256 +0", "joint45_256_0.json"), ("joint C 256 +2.07", "joint45_256_2070.json"),
         ("joint C 256 +4.14", "joint45_256_4140.json"), ("M-17 C 320", "laysz45_320.json"), ("joint C 320 +0", "joint45_320_0.json"),
         ("joint C 320 +2.07", "joint45_320_2070.json"), ("joint C 320 +4.14", "joint45_320_4140.json")]
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
TYPES = {"humaneval": "code", "math500": "math", "gsm8k": "math"}
H = (1, 2, 3, 4, 5)
SEEDS = 2
B_PRIOR = 30.0


def say(s=""):
    print(s, flush=True)


P, agree = pickle.load(open(PKL, "rb"))
assert [p[0] for p in P] == list(ORDER)
KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float)
kept_counted = KEPT[list(H)].sum()
D = {}
for r in map(json.loads, open(D23)):
    if r["arm"] == "dfl" and not r.get("invalid") and not r.get("paused"):
        for pc, cf in zip(r["pc"], r["cf"]):
            D.setdefault(r["type"], []).append((int(pc[0]), int(pc[1]), np.asarray(cf, float)))


def transplant(seed):
    rng = np.random.default_rng(seed)
    T = []
    for name, cks, _ in P:
        pool = D[TYPES[name.split("_")[0]]]
        kk = np.array([k for _, k, _ in pool])
        out = []
        for c in cks:
            n, k = len(c["rid"][0]), c["kept"]
            d = np.abs(kk - k)
            r2, k2, cf = pool[rng.choice(np.nonzero(d == d.min())[0])]
            conf = np.full(16, -1.0); conf[:15] = cf
            out.append((int(min(n, max(k, k + r2 - k2))), conf))
        T.append(out)
    return T


def needs(T, rule):
    Q = []
    for (name, cks, res0), tr in zip(P, T):
        out = []
        for c, (n, conf) in zip(cks, tr):
            kr = [rule(conf[j]) for j in range(n)]
            ms, ls = [], []
            for l in range(NL):
                rid = c["rid"][l]
                sm, sl = set(), set()
                for j, (k, r4) in enumerate(kr):
                    sm.update(rid[j, :k].tolist()); sl.update(rid[j, :r4].tolist())
                ms.append(frozenset(sm)); ls.append(frozenset(sl))
            out.append((ms, ls))
        Q.append(out)
    return Q


def popularity(Q, key):
    cnt = np.zeros((len(P), NL, E)); n = np.zeros(len(P))
    for p in range(len(P)):
        for t in range(len(P[p][1])):
            n[p] += 1
            for l in range(NL):
                cnt[p, l, list(Q[p][t][key][l])] += 1
    return [(cnt.sum(0) - cnt[p]) / max(n.sum() - n[p], 1) for p in range(len(P))]


def replay_layer(Q, l, Km, Kl, popm, popl):
    mm = np.zeros(len(P)); ml = np.zeros(len(P)); stm, stl, i = {}, {}, 0
    for p, (_, cks, res0) in enumerate(P):
        pm, pl = popm[p][l], popl[p][l]
        calls = [(-1, res0[l], frozenset())] + [(t, Q[p][t][0][l], Q[p][t][1][l]) for t in range(len(cks))]
        for t, nm, nl in calls:
            i += 1
            if t >= 0:
                mm[p] += sum(1 for e in nm if e not in stm); ml[p] += sum(1 for e in nl if e not in stl)
            for st, need, K, pop in ((stm, nm, Km, pm), (stl, nl, Kl, pl)):
                for e in need:
                    st[e] = float(i)
                if len(st) > K:
                    kf = lambda e, st=st, pop=pop: st[e] + B_PRIOR * pop[e]
                    for e in heapq.nsmallest(len(st) - K, (e for e in st if e not in need), key=kf):
                        del st[e]
                    if len(st) > K:
                        for e in heapq.nsmallest(len(st) - K, list(st), key=kf):
                            del st[e]
    return mm, ml


RULES = [("EXACT", lambda cf: (10, R4)), ("Q1 0.99", lambda cf: (2, R4) if cf >= 0.99 else (10, R4)),
         ("Q1 0.9", lambda cf: (2, R4) if cf >= 0.9 else (10, R4))]
res = {}
for seed in range(SEEDS):
    T = transplant(seed)
    for rname, rule in RULES:
        Q = needs(T, rule)
        popm, popl = popularity(Q, 0), popularity(Q, 1)
        for sname, f in SIZES:
            sz = json.load(open(os.path.join(MS, f)))
            Cm, Cl = sz["C_m"], sz["C_l"]
            mm = np.zeros(len(P)); ml = np.zeros(len(P))
            for l in range(NL):
                a, b = replay_layer(Q, l, Cm[l], Cl[l], popm, popl)
                mm += a; ml += b
            bm, bl = mm[list(H)].sum() / kept_counted, ml[list(H)].sum() / kept_counted
            ram = (sum(Cm) * MR + sum(Cl) * LR) / 1e9 + OTHER_GB
            res.setdefault((rname, sname), []).append((bm, bl, ram))
    say(f"  seed {seed} done, {time.time() - T0:.0f} s")

say("\n== SSD MB per kept token on the 50 GB route and back (cap45 + cf transplant, t 0.3 checks, LRU + P30; mean of 2 seeds) ==")
say("  sizes               | RAM GB (pools + 8.1) | EXACT: base + LSB = MB   | Q1 0.99: MB   | Q1 0.9: MB    | sidecar DROPPED (base only): EXACT / Q1 0.99 / Q1 0.9 MB")
for sname, _ in SIZES:
    ex = np.mean(np.array(res[("EXACT", sname)]), 0); q99 = np.mean(np.array(res[("Q1 0.99", sname)]), 0); q9 = np.mean(np.array(res[("Q1 0.9", sname)]), 0)
    mb = lambda a: (a[0] * MR + a[1] * LR) / 1e6
    say(f"  {sname:19s} | {ex[2]:5.1f}                | {ex[0]:5.2f} + {ex[1]:5.2f} = {mb(ex):5.1f} | {mb(q99):5.1f} (x{mb(q99) / mb(ex):.2f}) | {mb(q9):5.1f} (x{mb(q9) / mb(ex):.2f}) |"
        f" {ex[0] * MR / 1e6:5.1f} / {q99[0] * MR / 1e6:5.1f} / {q9[0] * MR / 1e6:5.1f}")
say(f"\ndone in {time.time() - T0:.0f} s")
