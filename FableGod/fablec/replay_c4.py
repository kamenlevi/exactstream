#!/usr/bin/env python3 -I
"""replay_c4.py (Fable C, mission C4 / BREAKTHROUGH PUSH, Tue 3 Nov 2026 17:1x EET). CPU only (numpy, nice 15, 4 threads).
The union arms priced on the union per LIVE check: cap45's real check rows (incl. the rejected ones), D-23 cf vectors
transplanted per check as band45 does, the t 0.3 cut modelled (rows = min(n, max(k, k + r2 - k2))), per-layer LRU + 30 x
popularity, strict pools, prompts chained with res0 forced, prompts 2-6 counted (code = HumanEval_91, math = the four
math prompts), 2 seeds. RAM points: M-17 C 256 (42.0 GB), joint C 256 +2.07 (44.1 GB = the 45 GB point), M-17 C 320
(50.5 GB = 45 GB resident + the volatile tier), joint C 320 +2.07 (52.6 GB).
Per arm: experts used per row-layer, (expert, tier) groups per layer (the expert kernel's cost unit), SSD MB per kept
token (base MSB-3 + LSB-1 sidecar), rows per check.
Arms: EXACT; Q1-UNION 0.99 / 0.9 (bytes and groups = Q1's by construction); U3 = Q1-UNION 0.9 + 3-bit sure rows; OEA k8 /
k6; HYB 0.9 u8 (+ t3) / u6; REACH-CUT 0.9 / 0.5 (+ t3); and Fable C's own: MASS-CUT tau (keep ranks until the renormalised
router mass reaches tau, min 2, + union), MISS-CUT r (unsure rows drop ranks >= r only when the expert would MISS; cache
state dependent), COLD-CUT r / p (unsure rows drop ranks >= r when the expert's leave-one-prompt-out popularity < p;
deterministic). Router mass is recomputed from cap45's router inputs (bf16 x) and routers.npz (softmax over 512, the
recorded top-10 renormalised), validated against the recorded ids.
usage: nice -n 15 ~/BigMoE/.venv/bin/python -I replay_c4.py | tee replay_c4.out"""
import os
for v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "4"
import heapq, json, pickle, time
import numpy as np

T0 = time.time()
NL, E = 48, 512
MR, LR = 2162688, 770048
OTHER_GB = 8.1
PKL = os.path.expanduser("~/MoonShots/data/band45_cap45.pkl")   # MoonShots' own laysz45.load() cache (a sibling session), not a download
D23 = os.path.expanduser("~/TrainLab/data/d23_runs.jsonl")
CAP = os.path.expanduser("~/BigMoE/data/cap45")
MS = os.path.expanduser("~/MoonShots/data")
SIZES = [("M-17 C256 42.0GB", "laysz45_256.json"), ("joint C256+2 44.1GB", "joint45_256_2070.json"),
         ("M-17 C320 50.5GB", "laysz45_320.json"), ("joint C320+2 52.6GB", "joint45_320_2070.json")]
ORDER = ("humaneval_HumanEval_68", "math500_test_prealgebra_1640", "humaneval_HumanEval_91", "math500_test_algebra_694",
         "math500_test_algebra_1282", "gsm8k_708")
TYPES = {"humaneval": "code", "math500": "math", "gsm8k": "math"}
DOM = {"code": (2,), "math": (1, 3, 4, 5), "all": (1, 2, 3, 4, 5)}
SEEDS = 2
B_PRIOR = 30.0


def say(s=""):
    print(s, flush=True)


P, agree = pickle.load(open(PKL, "rb"))
assert [p[0] for p in P] == list(ORDER)
KEPT = np.array([sum(c["kept"] for c in cks) for _, cks, _ in P], float)
NCHK = np.array([len(cks) for _, cks, _ in P], float)
D = {}
for r in map(json.loads, open(D23)):
    if r["arm"] == "dfl" and not r.get("invalid") and not r.get("paused"):
        for pc, cf in zip(r["pc"], r["cf"]):
            D.setdefault(r["type"], []).append((int(pc[0]), int(pc[1]), np.asarray(cf, float)))

# ---- router mass per (prompt, check, layer, row, rank): recomputed from cap45's x and routers.npz
RT = np.load(os.path.join(CAP, "routers.npz"))
MASS = []          # MASS[p][t][l] = cumulative renormalised mass over the recorded ranks, [rows, 10]
agree_ids = []
for name, cks, _ in P:
    z = np.load(os.path.join(CAP, f"{name}_gate_0.npz"))
    seq, L, rows, roff, xoff, ids, x = z["seq"], z["L"], z["rows"], z["row_off"], z["x_off"], z["ids"], z["x"]
    key = {}
    s0 = int(seq.min())              # seq is a global check index; the pickle's checks are 0-based per prompt
    for i in range(len(seq)):
        key.setdefault((int(seq[i]) - s0, int(L[i])), i)
    per = [[None] * NL for _ in cks]
    for t, c in enumerate(cks):
        for l in range(NL):
            i = key.get((t, l))
            if i is None:
                continue
            n = int(rows[i])
            rid = c["rid"][l]            # the pickle's rank-sorted ids (band45's loader); the mass is evaluated on them
            if n != rid.shape[0]:
                continue
            wk = f"w{l}" if f"w{l}" in RT.files else None
            if wk is None:
                continue
            xr = (x[xoff[i]:xoff[i] + n].astype(np.uint32) << 16).view(np.float32)
            lg = xr @ RT[wk].astype(np.float32).T
            lg -= lg.max(1, keepdims=True)
            pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
            top = np.argsort(-pr, 1)[:, :10]
            agree_ids.append(np.mean([len(set(top[j].tolist()) & set(rid[j].tolist())) / 10 for j in range(n)]))
            pm = pr[np.arange(n)[:, None], rid.astype(np.int64)]
            pm /= pm.sum(1, keepdims=True)
            per[t][l] = np.cumsum(pm, 1)
    MASS.append(per)
have_mass = sum(1 for per in MASS for row in per for m in row if m is not None)
tot_mass = sum(len(per) * NL for per in MASS)
say(f"router mass: {have_mass}/{tot_mass} (check, layer) entries recomputed; recorded top-10 vs recomputed top-10 overlap "
    f"{np.mean(agree_ids) if agree_ids else 0:.3f}; routers keys {len(RT.files)} ({'w0' in RT.files and 'w0 present' or 'no w0'}); {time.time() - T0:.0f} s")
# mass per rank for the record
mr = np.zeros(10); nmr = 0
for per in MASS:
    for row in per:
        for m in row:
            if m is not None:
                mr += (m - np.concatenate([np.zeros((len(m), 1)), m[:, :-1]], 1)).sum(0); nmr += len(m)
say(f"mean renormalised mass by rank: {np.round(mr / max(nmr, 1), 3).tolist()}")


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


def reach_of(conf, n):
    r = np.ones(n); acc = 1.0
    for j in range(n):
        r[j] = acc
        acc *= conf[j] if conf[j] >= 0 else 0.5
    return r


# a rule maps (cf, reach, cum mass [10] or None, l, pop[l]) -> (kmsb, r4, opt_from) : the row keeps ranks < kmsb as base,
# ranks < r4 read the LSB sidecar, and ranks >= opt_from are optional (kept only if resident at check time: MISS-CUT).
# For COLD-CUT the rule returns a callable on the row's ids instead (kept ranks list).
def mass_k(cum, tau, kmin=2):
    if cum is None:
        return 10
    k = int(np.searchsorted(cum, tau - 1e-9) + 1)
    return max(kmin, min(10, k))


def R_exact(cf, rc, cum, ids, pop): return (10, 2, 10)
def R_q1u(th): return lambda cf, rc, cum, ids, pop: (2, 2, 10) if cf >= th else (10, 2, 10)
def R_u3(cf, rc, cum, ids, pop): return (2, 0, 10) if cf >= 0.9 else (10, 2, 10)
def R_oea(k): return lambda cf, rc, cum, ids, pop: (k, 2, 10)
def R_hyb(u, t3=False): return lambda cf, rc, cum, ids, pop: (2, 0 if t3 else 2, 10) if cf >= 0.9 else (u, 2, 10)
def R_reach(t3=False): return lambda cf, rc, cum, ids, pop: (2, 0 if t3 else 2, 10) if cf >= 0.9 else ((2, 2, 10) if rc < 0.5 else (10, 2, 10))
def R_mass(tau): return lambda cf, rc, cum, ids, pop: (mass_k(cum, tau), 2, 10)
def R_hybmass(tau, t3=True): return lambda cf, rc, cum, ids, pop: (2, 0 if t3 else 2, 10) if cf >= 0.9 else (mass_k(cum, tau), 2, 10)
def R_miss(r, t3=True): return lambda cf, rc, cum, ids, pop: (2, 0 if t3 else 2, 10) if cf >= 0.9 else (r, 2, r)
def R_missall(r): return lambda cf, rc, cum, ids, pop: (r, 2, r)
def R_cold(r, pth, t3=True):
    def f(cf, rc, cum, ids, pop):
        if cf >= 0.9:
            return (2, 0 if t3 else 2, 10)
        keep = [j for j in range(10) if j < r or pop[ids[j]] >= pth]
        return (keep, 2, 10)
    return f


RULES = [("EXACT", R_exact), ("Q1-UNION 0.99", R_q1u(0.99)), ("Q1-UNION 0.9", R_q1u(0.9)), ("U3 = Q1U 0.9 + t3 sure", R_u3),
         ("OEA k8", R_oea(8)), ("OEA k6", R_oea(6)), ("HYB 0.9 u8", R_hyb(8)), ("HYB 0.9 u8 + t3", R_hyb(8, True)),
         ("HYB 0.9 u6", R_hyb(6)), ("REACH 0.9/0.5", R_reach()), ("REACH + t3 sure", R_reach(True)),
         ("MASS 0.9 all rows", R_mass(0.9)), ("MASS 0.8 all rows", R_mass(0.8)), ("HYB-MASS 0.9 + t3", R_hybmass(0.9)),
         ("HYB-MASS 0.8 + t3", R_hybmass(0.8)), ("MISS-CUT 6 all rows", R_missall(6)), ("HYB-MISS 6 + t3", R_miss(6)),
         ("HYB-MISS 4 + t3", R_miss(4)), ("HYB-COLD 4/0.05 + t3", R_cold(4, 0.05)), ("HYB-COLD 2/0.05 + t3", R_cold(2, 0.05)),
         ("HYB-COLD 4/0.10 + t3", R_cold(4, 0.10)), ("COLD 6/0.05 all rows", R_cold(6, 0.05, False) if False else (lambda cf, rc, cum, ids, pop: ([j for j in range(10) if j < 6 or pop[ids[j]] >= 0.05], 2, 10))),
         ("COLD 8/0.05 all rows", lambda cf, rc, cum, ids, pop: ([j for j in range(10) if j < 8 or pop[ids[j]] >= 0.05], 2, 10))]


def needs(T, rule, POP):
    """per check and layer: base MSB set, LSB set, optional rows (lists of ids), base pairs"""
    Q = []
    for p, ((name, cks, res0), tr) in enumerate(zip(P, T)):
        out = []
        for t, (c, (n, conf)) in enumerate(zip(cks, tr)):
            rc = reach_of(conf, n)
            per = []
            for l in range(NL):
                rid = c["rid"][l]; cum = MASS[p][t][l]
                sm, sl, opt, bp = set(), set(), [], 0
                for j in range(n):
                    k, r4, of = rule(conf[j], rc[j], None if cum is None else cum[j], rid[j], POP[p][l])
                    row = rid[j].tolist()
                    if isinstance(k, list):
                        sm.update(row[i] for i in k); bp += len(k)
                    else:
                        sm.update(row[:k]); bp += k
                        if of < 10:
                            opt.append(row[of:])
                    sl.update(row[:r4])
                per.append((frozenset(sm), frozenset(sl), opt, bp))
            out.append(per)
        Q.append(out)
    return Q


def popularity_exact(T):
    Q = needs(T, R_exact, [[None] * NL] * len(P))
    cnt = np.zeros((len(P), NL, E)); n = np.zeros(len(P))
    for p in range(len(P)):
        for t in range(len(P[p][1])):
            n[p] += 1
            for l in range(NL):
                cnt[p, l, list(Q[p][t][l][0])] += 1
    return [(cnt.sum(0) - cnt[p]) / max(n.sum() - n[p], 1) for p in range(len(P))]


def popularity(Q, key):
    cnt = np.zeros((len(P), NL, E)); n = np.zeros(len(P))
    for p in range(len(P)):
        for t in range(len(P[p][1])):
            n[p] += 1
            for l in range(NL):
                cnt[p, l, list(Q[p][t][l][key])] += 1
    return [(cnt.sum(0) - cnt[p]) / max(n.sum() - n[p], 1) for p in range(len(P))]


def replay_layer(Q, l, Km, Kl, popm, popl):
    mm = np.zeros(len(P)); ml = np.zeros(len(P)); gr = np.zeros(len(P)); pairs = np.zeros(len(P))
    stm, stl, i = {}, {}, 0
    for p, (_, cks, res0) in enumerate(P):
        pm, pl = popm[p][l], popl[p][l]
        calls = [(-1, res0[l], frozenset(), [], 0)] + [(t,) + Q[p][t][l] for t in range(len(cks))]
        for t, nm, nl, opt, bp in calls:
            i += 1
            if t >= 0:
                if opt:
                    extra = set(e for row in opt for e in row if e in stm or e in nm)
                    pairs[p] += bp + sum(1 for row in opt for e in row if e in stm or e in nm)
                    nm = frozenset(nm | extra)
                else:
                    pairs[p] += bp
                mm[p] += sum(1 for e in nm if e not in stm); ml[p] += sum(1 for e in nl if e not in stl)
                gr[p] += len(nm) + len(nl)
            else:
                nm = frozenset(nm)
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
    return mm, ml, gr, pairs


res = {}
rows_dom = {}
for seed in range(SEEDS):
    T = transplant(seed)
    POP = popularity_exact(T)
    for dom, idx in DOM.items():
        rows_dom.setdefault(dom, []).append(sum(T[p][t][0] for p in idx for t in range(len(T[p]))) / sum(len(T[p]) for p in idx))
    for rname, rule in RULES:
        Q = needs(T, rule, POP)
        popm, popl = popularity(Q, 0), popularity(Q, 1)
        for sname, f in SIZES:
            sz = json.load(open(os.path.join(MS, f)))
            Cm, Cl = sz["C_m"], sz["C_l"]
            mm = np.zeros(len(P)); ml = np.zeros(len(P)); gr = np.zeros(len(P)); pairs = np.zeros(len(P))
            for l in range(NL):
                a, b, g, q = replay_layer(Q, l, Cm[l], Cl[l], popm, popl)
                mm += a; ml += b; gr += g; pairs += q
            ram = (sum(Cm) * MR + sum(Cl) * LR) / 1e9 + OTHER_GB
            for dom, idx in DOM.items():
                idx = list(idx)
                kept = KEPT[idx].sum(); nchk = NCHK[idx].sum()
                rows = sum(T[p][t][0] for p in idx for t in range(len(T[p])))
                res.setdefault((rname, sname, dom), []).append((
                    (mm[idx].sum() * MR + ml[idx].sum() * LR) / 1e6 / kept,      # MB per kept token
                    gr[idx].sum() / nchk / NL,                                   # groups per layer per check
                    pairs[idx].sum() / rows / NL,                                # experts used per row-layer
                    mm[idx].sum() / kept, ml[idx].sum() / kept, ram))
    say(f"  seed {seed} done, {time.time() - T0:.0f} s")

say(f"\nrows per live check (t 0.3 transplant): " + ", ".join(f"{d} {np.mean(v):.1f}" for d, v in rows_dom.items())
    + f"; kept per check: code {KEPT[2] / NCHK[2]:.1f}, math {KEPT[[1, 3, 4, 5]].sum() / NCHK[[1, 3, 4, 5]].sum():.1f}, all {KEPT[1:].sum() / NCHK[1:].sum():.1f}")
say("\n== union arms at live check lengths (cap45 + cf transplant; mean of 2 seeds). Per arm and RAM point: SSD MB per kept token"
    " (base + LSB misses per kept), (expert, tier) groups per layer per check, experts used per row-layer; code / math / all ==")
ex = {}
for rname, _ in RULES:
    say(f"-- {rname}")
    for sname, _ in SIZES:
        line = f"   {sname:20s}|"
        for dom in ("code", "math", "all"):
            a = np.mean(np.array(res[(rname, sname, dom)]), 0)
            if rname == "EXACT":
                ex[(sname, dom)] = a
            e = ex[(sname, dom)]
            line += f" {dom} {a[0]:5.1f} MB (x{a[0] / e[0]:.2f}; {a[3]:4.1f}+{a[4]:4.1f}) g {a[1]:4.1f} (x{a[1] / e[1]:.2f}) e/row {a[2]:4.2f} |"
        say(line)
say(f"\ndone in {time.time() - T0:.0f} s")
