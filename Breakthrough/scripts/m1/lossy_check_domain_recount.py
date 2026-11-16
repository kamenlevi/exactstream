#!/usr/bin/env python3
"""Per-domain byte recount of LOSSY_CHECK_PLAN's levers (Breakthrough, Mon 16 Nov 2026 08:3x; an erratum check after
Fit12 08:29; pre-registered in docs/LOSSY_CHECK_PRICE.md "Erratum check").
lossy_check_price.py priced bytes as keq = size.sum() / every block-layer of all 36 map windows (code + IFEval + multi),
while its code-scoped reading used code rows' flips. This rebuilds the same 32-row blocks and the same tier rule
(copied from lossy_check_price.py) and prints, per domain and pooled:
- the exact union per layer;
- tier counts and 4-bit-eq experts per layer for 1a / 1b / 1c / 1d, and 1c over a lambda sweep;
- GB and ms with the plan's dense parts.
Bytes only: the flips per lambda are in data/m1/lossy_check_price.out (they were per domain already).
CPU only. Usage: nice -n 15 ~/BigMoE/.venv/bin/python -I -B lossy_check_domain_recount.py
Output: ~/Breakthrough/data/m1/lossy_check_domain_recount.out
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
OUT = os.path.expanduser("~/Breakthrough/data/m1/lossy_check_domain_recount.out")
EXP_MB, BW, EFF, ROWS, NL = 2.135, 280.0, 0.84, 32, 48
D_EXACT, D_PLAN = 3.44, 2.14            # dense GB: exact; mixers at 4-bit + the 16K shortlist (dense_cut_price.py)
TIERS4 = [("4-bit", 0.0, 1.0), ("MSB-3", 0.08, 0.778), ("MSB-2", 0.32, 0.556), ("drop", 1.0, 0.0)]
TIERS3 = [t for t in TIERS4 if t[0] != "MSB-2"]
DOMS = ["code", "ifeval", "multi"]
PLAN = {"union": 99.9, "1a": 29.2, "1b": 33.5, "1c": 36.4, "1d": 54.8}   # LOSSY_CHECK_PLAN.md (pooled, as printed)
FIT12_CODE = {"union": 114.2, "1c": 39.2}
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


base = json.load(open(os.path.join(M, "baseline.json")))
W = {}
for f in sorted(os.listdir(os.path.join(M, "xcap"))):
    z = np.load(os.path.join(M, "xcap", f))
    W[f] = (base[f]["dom"], z["ids"].astype(np.int32), z["w"].astype(np.float32))

# ---- the same 32-row blocks as lossy_check_price.py, plus the domain of each block-layer ----
GIDX, W2, RD, BLDOM, G_of, distinct = [], [], [], [], [], []
g0 = 0
for nm, (dom, ids, w) in W.items():
    for bi in range(ids.shape[0] // ROWS):
        sl = slice(bi * ROWS, (bi + 1) * ROWS)
        I, w2 = ids[sl], w[sl] ** 2
        gi = np.empty((ROWS, NL, 10), np.int64)
        for l in range(NL):
            u, inv = np.unique(I[:, l].ravel(), return_inverse=True)
            gi[:, l, :] = (inv + g0).reshape(ROWS, 10)
            G_of.append(np.full(len(u), len(distinct))); distinct.append(len(u)); BLDOM.append(DOMS.index(dom)); g0 += len(u)
        GIDX.append(gi); W2.append(w2); RD.append(np.arange(ROWS))
GIDX = np.concatenate(GIDX); W2 = np.concatenate(W2); RD = np.concatenate(RD)
GROUP = np.concatenate(G_of); NG = g0
distinct = np.array(distinct); BLDOM = np.array(BLDOM); GDOM = BLDOM[GROUP]
NBL = {d: int((BLDOM == i).sum()) for i, d in enumerate(DOMS)}; NBL["all"] = len(distinct)
say(f"Per-domain byte recount: {len(W)} windows, {len(RD)} rows in {len(distinct) // NL} full 32-row blocks; "
    f"block-layers " + ", ".join(f"{d} {NBL[d]}" for d in DOMS + ["all"]))


def sel(d):
    return (GDOM >= 0) if d == "all" else (GDOM == DOMS.index(d))


def S_of(s):
    rw = np.ones(len(RD)) if s is None else s ** RD
    return np.bincount(GIDX.ravel(), weights=(W2 * rw[:, None, None]).ravel(), minlength=NG)


def tiers(lam, S, T):
    err = np.array([t[1] for t in T]); size = np.array([t[2] for t in T])
    t = (err[None, :] * S[:, None] + lam * size[None, :]).argmin(1)
    return t, size[t]


def gb(dense, keq):
    return dense + EXP_MB * NL * keq / 1000


union = {d: float(distinct[BLDOM == DOMS.index(d)].mean()) for d in DOMS}; union["all"] = float(distinct.mean())
say("")
say("EXACT UNION (distinct experts per layer per 32-row block) and the exact read (dense 3.44 GB, all experts at 4 bits):")
say("| domain | union / layer | GB | ms floor | ms at 84 % |")
say("|---|---|---|---|---|")
for d in DOMS + ["all"]:
    g = gb(D_EXACT, union[d])
    say(f"| {d} | {union[d]:.1f} | {g:.2f} | {1000 * g / BW:.1f} | {1000 * g / (BW * EFF):.1f} |")

KEY = [("1a acc-tiers s 0.935 lam 1.2e-2", "1a", 0.935, 1.2e-2, TIERS4), ("1b acc-tiers s 0.98 lam 1.8e-2", "1b", 0.98, 1.8e-2, TIERS4),
       ("1c acc-3tiers s 0.98 lam 1.6e-2", "1c", 0.98, 1.6e-2, TIERS3), ("1d acc-tiers s 0.98 lam 8e-3", "1d", 0.98, 8e-3, TIERS4)]
say("")
say("LEVERS, 4-bit-eq experts per layer by domain (tier counts per layer: 4-bit / MSB-3 / MSB-2 / drop):")
say("| lever | domain | 4-bit | MSB-3 | MSB-2 | drop | 4-bit-eq / layer | GB (exact dense) | GB (plan dense 2.14) | ms floor / 84 % (plan dense) |")
say("|---|---|---|---|---|---|---|---|---|---|")
keqs = {}
for label, k, s, lam, T in KEY:
    t, size = tiers(lam, S_of(s), T)
    names = [x[0] for x in T]
    for d in DOMS + ["all"]:
        m = sel(d)
        cnt = np.bincount(t[m], minlength=len(T)) / NBL[d]
        c = {n: cnt[i] for i, n in enumerate(names)}
        keq = float(size[m].sum() / NBL[d]); keqs[(k, d)] = keq
        g = gb(D_PLAN, keq)
        say(f"| {label} | {d} | {c.get('4-bit', 0):.1f} | {c.get('MSB-3', 0):.1f} | {c.get('MSB-2', 0):.1f} | {c.get('drop', 0):.1f} | "
            f"{keq:.1f} | {gb(D_EXACT, keq):.2f} | {g:.2f} | {1000 * g / BW:.1f} / {1000 * g / (BW * EFF):.1f} |")

say("")
ok = abs(round(union["all"], 1) - PLAN["union"]) < 0.051 and all(abs(round(keqs[(k, "all")], 1) - PLAN[k]) < 0.051 for k in ("1a", "1b", "1c", "1d"))
say(f"SELF-CHECK (pooled reproduces the plan to 0.1: union {union['all']:.2f} vs {PLAN['union']}; "
    + ", ".join(f"{k} {keqs[(k, 'all')]:.2f} vs {PLAN[k]}" for k in ("1a", "1b", "1c", "1d")) + f"): {'PASS' if ok else 'FAIL'}")
fu = 100 * (union["code"] / FIT12_CODE["union"] - 1); fc = 100 * (keqs[("1c", "code")] / FIT12_CODE["1c"] - 1)
say(f"vs Fit12's code recount: union {union['code']:.2f} vs {FIT12_CODE['union']} ({fu:+.2f} %); 1c {keqs[('1c', 'code')]:.2f} vs "
    f"{FIT12_CODE['1c']} ({fc:+.2f} %): {'WITHIN 1 %' if abs(fu) <= 1 and abs(fc) <= 1 else 'OFF BY MORE THAN 1 %'}")

say("")
say("THE CODE PLAN RESTATED (1c experts + mixers at 4-bit + the 16K shortlist; dense 2.14 GB):")
gp, gc = gb(D_PLAN, keqs[("1c", "all")]), gb(D_PLAN, keqs[("1c", "code")])
for nm, g in (("as published (pooled bytes)", gp), ("code blocks' bytes", gc)):
    say(f"- {nm}: {g:.2f} GB, {1000 * g / BW:.1f} ms at the floor, {1000 * g / (BW * EFF):.1f} ms at 84 %; 25 ms needs "
        f"{100 * g / (0.025 * BW):.1f} % of {BW:.0f} GB/s; tok/s HE+ at 84 % (today 13.8 / 98 % 23.3 kept, +7 ms drafting): "
        f"{13.8 / ((1000 * g / (BW * EFF) + 7) / 1000):.0f} / {23.3 / ((1000 * g / (BW * EFF) + 7) / 1000):.0f}")

say("")
say("1c (3 tiers, s 0.98) OVER LAMBDA, code blocks vs pooled, with the plan's dense 2.14 GB (25 ms at 84 % = 5.88 GB):")
say("| lambda | 4-bit-eq code | GB code | ms floor / 84 % code | 4-bit-eq pooled | GB pooled | ms floor / 84 % pooled |")
say("|---|---|---|---|---|---|---|")
S98 = S_of(0.98)
for lam in (1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 2.0e-2, 2.2e-2):
    t, size = tiers(lam, S98, TIERS3)
    kc, ka = float(size[sel("code")].sum() / NBL["code"]), float(size.sum() / NBL["all"])
    g1, g2 = gb(D_PLAN, kc), gb(D_PLAN, ka)
    say(f"| {lam:.1e} | {kc:.1f} | {g1:.2f} | {1000 * g1 / BW:.1f} / {1000 * g1 / (BW * EFF):.1f} | {ka:.1f} | {g2:.2f} | "
        f"{1000 * g2 / BW:.1f} / {1000 * g2 / (BW * EFF):.1f} |")

# ---- 08:3x: the plan's rows restated on code blocks (dense parts as in dense_cut_price.py: 3.437 GB from the weight
# headers minus the cuts 0.968 (HC + PLE 4-bit) / 0.631 (8-bit) and 0.334 (16K shortlist); the ranked levers use 3.44)
say("")
say("RESTATED FOR THE PLAN (pooled -> code blocks; tok/s HE+ at 84 %: kept 13.8 today / 23.3 98 %, +7 ms drafting):")
say("| plan row | dense GB | 4-bit-eq pooled -> code | GB pooled -> code | ms floor pooled -> code | ms 84 % pooled -> code | tok/s 84 % code (today / 98 %) |")
say("|---|---|---|---|---|---|---|")
D4, D8 = 3.437 - 0.968 - 0.334, 3.437 - 0.631 - 0.334
R = [("1a (ranked levers)", D_EXACT, "1a", None), ("1b (ranked levers)", D_EXACT, "1b", None), ("1c (ranked levers)", D_EXACT, "1c", None),
     ("lam 1.2e-2 + mixers 8-bit + shortlist", D8, None, 1.2e-2), ("lam 1.2e-2 + mixers 4-bit + shortlist", D4, None, 1.2e-2),
     ("lam 1.6e-2 (1c) + mixers 8-bit + shortlist", D8, None, 1.6e-2), ("lam 1.6e-2 (1c) + mixers 4-bit + shortlist (THE CODE PLAN)", D4, None, 1.6e-2),
     ("lam 1.8e-2 + mixers 4-bit + shortlist", D4, None, 1.8e-2)]
for label, dn, k, lam in R:
    if k:
        ka, kc = keqs[(k, "all")], keqs[(k, "code")]
    else:
        t, size = tiers(lam, S98, TIERS3)
        kc, ka = float(size[sel("code")].sum() / NBL["code"]), float(size.sum() / NBL["all"])
    ga, gc = gb(dn, ka), gb(dn, kc)
    fa_, fc_ = 1000 * ga / (BW * EFF), 1000 * gc / (BW * EFF)
    say(f"| {label} | {dn:.3f} | {ka:.1f} -> {kc:.1f} | {ga:.2f} -> {gc:.2f} | {1000 * ga / BW:.1f} -> {1000 * gc / BW:.1f} | "
        f"{fa_:.1f} -> {fc_:.1f} | {13.8 / ((fc_ + 7) / 1000):.0f} / {23.3 / ((fc_ + 7) / 1000):.0f} |")

with open(OUT, "w") as fh:
    fh.write("\n".join(lines) + "\n")
print("wrote", OUT)
