#!/usr/bin/env python3 -I
"""replay_c2.py (Fable C, mission C2, Mon 2 Nov 2026, 23:3x EET). CPU only (numpy, nice 15, 4 threads), recorded data only.
Q1 (top-2 on drafter-sure rows) at the TARGET cache sizes, priced on the union per 16-row block, with band45's method:
cap45's real check rows (incl. rejected), D-23 cf vectors transplanted per check (same type, nearest kept; 3 seeds), the
t 0.3 cut modelled as band45's SHORT (rows = kept + the D-23 check's rows - its kept), per-layer LRU + 30 x popularity,
strict pools, M-17 sizes from laysz45_{256,320}.json, prompts chained with res0 forced, prompts 2-6 counted.
Reported per rule (EXACT, Q1 0.99, Q1 0.9) and C (256, 320):
  SSD MB per kept token (base misses x 2,162,688 + sidecar misses x 770,048), union GB per check (distinct experts' MSB
  + the 4-bit ones' LSB), (expert, tier) groups per layer (the expert kernel's cost unit: 10.15 us each, EXPERT_KERNEL fit),
  and for the new idea Q1-UNION: of the pairs Q1 drops on sure rows (ranks 2-9), the share whose expert is in the check's
  union anyway (so keeping them costs no bytes and no group).
usage: nice -n 15 ~/BigMoE/.venv/bin/python -I replay_c2.py -> stdout (tee to replay_c2.out)"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, pickle, time
import numpy as np

T0 = time.time()
NL, E, R4 = 48, 512, 2
MR, LR = 2162688, 770048
PKL = os.path.expanduser("~/MoonShots/data/band45_cap45.pkl")   # MoonShots' own laysz45.load() cache (sibling session), not a download
D23 = os.path.expanduser("~/TrainLab/data/d23_runs.jsonl")
SZ = {c: json.load(open(os.path.expanduser(f"~/MoonShots/data/laysz45_{c}.json"))) for c in (256, 320)}
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
TYPES = {"humaneval": "code", "math500": "math", "gsm8k": "math"}
H = (1, 2, 3, 4, 5)
SEEDS = 3
B_PRIOR = 30.0


def say(s=""):
    print(s, flush=True)


P, agree = pickle.load(open(PKL, "rb"))
assert [p[0] for p in P] == list(ORDER)
KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float)
kept_counted = KEPT[list(H)].sum()
say(f"cap45: checks {[len(c) for _, c, _ in P]}, counted kept tokens {kept_counted:.0f}")

# D-23 cf vectors (band45.d23): type -> [(rows, kept, cf[15])]
D = {}
for r in map(json.loads, open(D23)):
    if r["arm"] == "dfl" and not r.get("invalid") and not r.get("paused"):
        for pc, cf in zip(r["pc"], r["cf"]):
            D.setdefault(r["type"], []).append((int(pc[0]), int(pc[1]), np.asarray(cf, float)))
say(f"D-23 dfl checks: { {k: len(v) for k, v in D.items()} }")


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


def rule_q1(theta):
    return lambda cf: (2, R4) if cf >= theta else (10, R4)


def needs(T, rule, stats=None):
    """per prompt / check: per-layer (base need, sidecar need), plus union-rule statistics"""
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
                if stats is not None:
                    for j, (k, r4) in enumerate(kr):
                        if k < 10:                      # a sure row: ranks k..9 dropped by Q1
                            for e in rid[j, k:10].tolist():
                                stats["dropped"] += 1
                                stats["in_union"] += e in sm
            out.append((ms, ls, c["kept"], n))
        Q.append(out)
    return Q


def replay_layer(Q, l, Km, Kl, popm, popl):
    """both pools, prompts chained, res0 forced into the base pool per prompt; returns per-prompt (miss_m, miss_l)"""
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


def popularity(Q, key):
    cnt = np.zeros((len(P), NL, E)); n = np.zeros(len(P))
    for p in range(len(P)):
        for t in range(len(P[p][1])):
            n[p] += 1
            for l in range(NL):
                cnt[p, l, list(Q[p][t][key][l])] += 1
    return [(cnt.sum(0) - cnt[p]) / max(n.sum() - n[p], 1) for p in range(len(P))]


RULES = [("EXACT", lambda cf: (10, R4)), ("Q1 0.99", rule_q1(0.99)), ("Q1 0.9", rule_q1(0.9))]
res = {}
for seed in range(SEEDS):
    T = transplant(seed)
    rows = np.mean([n for tr in T for n, _ in tr])
    for rname, rule in RULES:
        st = {"dropped": 0, "in_union": 0}
        Q = needs(T, rule, st)
        popm, popl = popularity(Q, 0), popularity(Q, 1)
        # union per 16-row block (bytes on the union, never per row): distinct experts' MSB + the 4-bit ones' LSB; groups
        ub = 0.0; groups = 0.0; nchk = 0
        for p in H:
            for ms, ls, kept, n in Q[p]:
                nchk += 1
                ub += sum(len(m) * MR + len(s) * LR for m, s in zip(ms, ls))
                groups += sum(len(m) + len(s) for m, s in zip(ms, ls))     # (expert, 3-bit) + (expert, 4-bit) groups
        for C in (256, 320):
            Cm, Cl = SZ[C]["C_m"], SZ[C]["C_l"]
            mm = np.zeros(len(P)); ml = np.zeros(len(P))
            for l in range(NL):
                a, b = replay_layer(Q, l, Cm[l], Cl[l], popm, popl)
                mm += a; ml += b
            bm, bl = mm[list(H)].sum() / kept_counted, ml[list(H)].sum() / kept_counted
            res.setdefault((rname, C), []).append((bm, bl, (bm * MR + bl * LR) / 1e6, ub / nchk / 1e9, groups / nchk / NL,
                                                   rows, st["in_union"] / max(st["dropped"], 1), st["dropped"] / nchk))
    say(f"  seed {seed} done, {time.time() - T0:.0f} s")

say("\n== Q1 at the target cache sizes (cap45 + D-23 cf transplant, SHORT checks, M-17 sizes, LRU + P30; mean over seeds) ==")
say("  rule      C   | base + sidecar misses per kept | SSD MB per kept | union GB per check | groups per layer | rows | Q1-UNION: dropped pairs per check, share in the union")
base = {}
for (rname, C), v in res.items():
    a = np.mean(np.array(v), 0)
    if rname == "EXACT":
        base[C] = a
    b = base[C]
    say(f"  {rname:8s} {C} | {a[0]:5.2f} + {a[1]:5.2f} = {a[0] + a[1]:5.2f}      | {a[2]:6.1f} (x{a[2] / b[2]:.2f}) | {a[3]:5.2f} (x{a[3] / b[3]:.2f})       | {a[4]:5.1f} (x{a[4] / b[4]:.2f})    | {a[5]:4.1f} | {a[7]:6.0f}, {100 * a[6]:.0f} %")
say(f"\ndone in {time.time() - T0:.0f} s")
