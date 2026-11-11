"""Three charges for the ranks 2-9 store term against REF2-T, with M-36's own KLs / fit (exploratory, for NewIdeas' frame
question): credit (r - 1)(T01 + E29 k) [NewIdeas' in-frame form], v1.1 (r - 1) T01 + r E29 k, strict independent-quantizer
divergence (r - 1) T01 + (r + 1) E29 k (an error not nested in MSB-3's: MSE(e_s - e_3) = (r + 1) MSE_3)."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import sweep_quality_check as Q

base = Q.load("baseline"); K = {}
for tag, name in Q.ARMS.items():
    arm = Q.load(name); acc = {d: [0.0, 0, 0] for d in Q.DOMS}
    for w, b in base.items():
        a = arm[w]
        for tp, lp, tq, lq in zip(b["tk"], b["lp"], a["tk"], a["lp"]):
            acc[b["dom"]][0] += Q.row_kl(tp, np.array(lp), tq, np.array(lq), "mine"); acc[b["dom"]][1] += tq[0] != tp[0]; acc[b["dom"]][2] += 1
    K[tag] = {d: (acc[d][0] / acc[d][2], 100 * acc[d][1] / acc[d][2]) for d in Q.DOMS}
AB = {}
for d in Q.DOMS:
    x = np.array([math.sqrt(K[t][d][0]) for t in Q.ARMS]); y = np.array([K[t][d][1] for t in Q.ARMS]); b_, a_ = np.polyfit(x, y, 1); AB[d] = (a_, b_)
print("store (r, slots): flips % code / IFEval / multi under  credit (r-1) | v1.1 r | strict (r+1)")
for name, r, c in Q.ROWS:
    out = {}
    for view in ("credit", "v1.1", "strict"):
        fl = []
        for di, d in enumerate(Q.DOMS):
            kl = lambda t: K[t][d][0]
            E29, T01 = kl("s7_ref3") - kl("s7_ref"), kl("bq_t3") - kl("bq")
            D17 = kl("screen6") - kl("C") if d == "code" else 0.0
            k0, kr, k1 = kl("screen6"), kl("s7_ref"), kl("bq_t3") + D17
            g, gr = Q.GAP[c][di], Q.GAP[239][di]
            mp = k0 + (kr - k0) * g / gr if g <= gr else kr + (k1 - kr) * (g - gr) / (1 - gr)
            ke = Q.KEEP[c][di] / Q.KEEP[239][di]
            e = {"credit": r - 1, "v1.1": r, "strict": r + 1}[view]
            klv = mp + (r - 1) * T01 + e * E29 * ke
            fl.append(AB[d][0] + AB[d][1] * math.sqrt(max(klv, 0.0)))
        out[view] = fl
    print(f"  {name:12s} ({r}, {c}): " + " | ".join(" / ".join(f"{v:.2f}" for v in out[k]) for k in ("credit", "v1.1", "strict")))
