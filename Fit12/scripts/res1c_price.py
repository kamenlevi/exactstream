"""res1c_price.py (Fit12, Mon 16 Nov 2026 13:1x; pre-registered in /Users/kamen/Fit12/PLAN_12GB_90_fit12.md "RES-1c PRICE
+ SCOREBOARD v3" before this ran). Desk only, CPU, python -I.

The resident-only MSB-3 rule (RES) on top of 1c-tiers: an MSB-3-tier expert is read only when resident in the C 256 base
pool at the check's start; else it is dropped. The 4-bit tier is always read (it may miss).
Inputs (read-only): the decision map's 12 code windows (xcap ids + normalised w), Breakthrough's tiers_1c masks (32-row
blocks), Breakthrough's res_c256 residency masks (dm_res_masks.py "res": LRU per layer at laysz45_256's C_m, resident at
each 16-row live check's start), ~/FableGod/floor_rows.json (tie rows).
Quality: Breakthrough's calibrated code law (lossy_check_price.out): all-row flips % = 41.5 E^0.932, non-tie
143 E^1.446; E per row = sum over layers and ranks of err x w^2 / 48, err 0 / 0.08 / 1 (4-bit / MSB-3 / drop); reach
weight s^d (s 0.98) within each 32-row block, as Breakthrough priced 1c.
Approximations [est]: residency at a 16-row live check's start stands in for the 32-row check's start (an MSB-3-tier
expert is non-resident if ANY of its block slots is non-resident); the pool replay loads every expert (exact pass), so
1c's own drops would leave the real pool a little different.
"""
import json
import os

import numpy as np

M = os.path.expanduser("~/FableGod/decision_map/out")
ROWS, NL, S, MS_PER_MISS = 32, 48, 0.98, 0.250
FIT = {"all": (41.5, 0.932), "nt": (143.0, 1.446)}
ERR = np.array([0.0, 0.08, 1.0])
base = json.load(open(os.path.join(M, "baseline.json")))
TIE = {}
for r in json.load(open(os.path.expanduser("~/FableGod/floor_rows.json"))):
    TIE[(r["nm"], int(r["i"]))] = r["cls"] == "tie"
code = sorted(f for f in os.listdir(os.path.join(M, "xcap")) if base[f]["dom"] == "code")

E1, E2, RW, TI = [], [], [], []                       # per row: E under 1c, E under 1c + RES, reach weight, tie
miss1, miss2, n3, n3nr, nunion, nunion_nr, incons = [], [], 0, 0, 0, 0, 0
for f in code:
    z = np.load(os.path.join(M, "xcap", f))
    ids, w, a = z["ids"].astype(np.int64), z["w"].astype(np.float64), int(z["a"])
    m = ids.shape[0]
    t = np.load(os.path.join(M, "bt", "tiers_1c", f))
    k4 = t["keep"].astype(bool)[a:a + m]
    k3 = t["keep3"].astype(bool)[a:a + m]
    assert not (k4 & k3).any(), f
    res = np.load(os.path.join(M, "res_c256", f))["keep"].astype(bool)[a:a + m]
    tie = np.array([TIE.get((f, a + j), False) for j in range(m)])
    tier = np.where(k4, 0, np.where(k3, 1, 2))
    for bi in range(m // ROWS):
        sl = slice(bi * ROWS, (bi + 1) * ROWS)
        e1 = np.zeros(ROWS); e2 = np.zeros(ROWS)
        mb1 = mb2 = 0
        for ly in range(NL):
            I, W2, T, R = ids[sl, ly], w[sl, ly] ** 2, tier[sl, ly], res[sl, ly]
            T2 = T.copy()
            for e in np.unique(I):
                sel = I == e
                te = np.unique(T[sel])
                if len(te) != 1:
                    incons += 1
                te = int(te.min())
                nr = not R[sel].all()                     # non-resident at some check start in the block
                nunion += 1; nunion_nr += nr
                if te == 1:
                    n3 += 1; n3nr += nr
                    if nr:
                        T2[sel] = 2                       # RES: a non-resident MSB-3-tier expert is dropped
                if nr and te in (0, 1):
                    mb1 += 1                              # 1c: every kept expert is read, so a non-resident one misses
                if nr and te == 0:
                    mb2 += 1                              # RES: only the 4-bit tier may miss
            e1 += (ERR[T] * W2).sum(1)
            e2 += (ERR[T2] * W2).sum(1)
        E1.extend(e1 / NL); E2.extend(e2 / NL)
        RW.extend(S ** np.arange(ROWS)); TI.extend(tie[sl])
        miss1.append(mb1); miss2.append(mb2)
E1, E2, RW, TI = map(np.asarray, (E1, E2, RW, TI))


def stats(E):
    return dict(all=float(E.mean()), nontie=float(E[~TI].mean()), reach=float((E * RW).sum() / RW.sum()))


def pred(kind, E):
    a_, b_ = FIT[kind]
    return a_ * E ** b_ if E > 0 else 0.0


s1, s2 = stats(E1), stats(E2)
L = []


def p(x=""):
    print(x); L.append(x)


p(f"# RES-1c price (Fit12 res1c_price.py): {len(code)} code windows, {len(miss1)} 32-row blocks, {len(E1)} rows "
  f"({int(TI.sum())} tie rows); tier inconsistencies within an expert: {incons}")
E_ref = (1.08 / 41.5) ** (1 / 0.932)
sc = abs(s1["reach"] / E_ref - 1) <= 0.03
p(f"SELF-CHECK: 1c reach-weighted code E {s1['reach']:.5f} vs {E_ref:.5f} behind Breakthrough's 1.08 / 0.50 "
  f"(within 3 %): {'PASS' if sc else 'FAIL'} ({100 * (s1['reach'] / E_ref - 1):+.1f} %)")
p(f"residency (C 256 base pool, res_c256): union experts per block-layer non-resident {100 * nunion_nr / nunion:.2f} %; "
  f"MSB-3-tier experts non-resident {100 * n3nr / n3:.2f} % ({n3nr / len(miss1):.1f} per 32-row block, dropped by RES)")
p(f"misses per 32-row check (48 layers): 1c {np.mean(miss1):.1f} (median {np.median(miss1):.0f}) -> 1c + RES "
  f"{np.mean(miss2):.1f} (median {np.median(miss2):.0f}); x {MS_PER_MISS} ms = {MS_PER_MISS * np.mean(miss1):.1f} -> "
  f"{MS_PER_MISS * np.mean(miss2):.1f} ms (v2's scaled model: 5.3 -> 1.6)")
p()
p("| measure | E 1c | E 1c + RES | dE | pred flips 1c | pred 1c + RES | increment | 1c measured (map) | 1c + RES [P] | line |")
p("|---|---|---|---|---|---|---|---|---|---|")
out = {}
for lab, k_e, kind, meas, room in (("all-row, reach-weighted (Breakthrough's form)", "reach", "all", 1.13, 0.10),
                                    ("non-tie law on the reach-weighted E (Breakthrough's form)", "reach", "nt", 0.48, 0.05),
                                    ("all-row, unweighted", "all", "all", 1.13, 0.10),
                                    ("non-tie rows, unweighted", "nontie", "nt", 0.48, 0.05)):
    p1, p2 = pred(kind, s1[k_e]), pred(kind, s2[k_e])
    inc = p2 - p1
    out[lab] = dict(E1=s1[k_e], E2=s2[k_e], pred1=p1, pred2=p2, inc=inc)
    p(f"| {lab} | {s1[k_e]:.5f} | {s2[k_e]:.5f} | {s2[k_e] - s1[k_e]:+.5f} | {p1:.2f} | {p2:.2f} | {inc:+.3f} | {meas:.2f} | "
      f"{meas + inc:.2f} | {'within' if inc <= room else 'OVER'} +{room:.2f} |")
inc_a = out["all-row, reach-weighted (Breakthrough's form)"]["inc"]
inc_n = out["non-tie law on the reach-weighted E (Breakthrough's form)"]["inc"]
verdict = "CHEAP" if (inc_a <= 0.10 and inc_n <= 0.05) else "MATERIAL"
p()
p(f"VERDICT (pre-registered line, Breakthrough's form): {verdict} (increment all-row {inc_a:+.3f}, non-tie {inc_n:+.3f}); "
  f"a real map arm comes before any gate")
os.makedirs("/Users/kamen/Fit12/data/lossy32", exist_ok=True)
json.dump(dict(s1c=s1, sres=s2, out=out, miss_1c=float(np.mean(miss1)), miss_res=float(np.mean(miss2)),
               msb3_nonres_share=n3nr / n3, union_nonres_share=nunion_nr / nunion, selfcheck=bool(sc), verdict=verdict),
          open("/Users/kamen/Fit12/data/lossy32/res1c_price.json", "w"), indent=1)
open("/Users/kamen/Fit12/data/lossy32/res1c_price.txt", "w").write("\n".join(L) + "\n")
