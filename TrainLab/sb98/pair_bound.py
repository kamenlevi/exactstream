"""WEEK98 S21 PAIR stage 0 (MTP Head; REPORT 'WEEK98 S21 PAIR'): the upper bound on levels saved by a PERFECT level pairing,
from s30's stored S9 cells (data/sb98/tree_s30_{mbpp,he}.npz: right / hlp per anchor and depth, teacher-forced).
Walk (cc_readout's): from row max(0, gen - 1 - a0), kept = leading right run + 1. Per check, along the true path, greedy:
at depth j (1-based) inside the accepted run with p1(j) >= tau and j + 1 <= D, pair (skip level j + 1); else one level.
Levels to reach D = D - pairs (pairs only inside the accepted run: after a miss nothing pairs). Reports mean levels saved
/ D per check, by set, D in (12, 20, 31), tau in (0.95, 0.9). Usage: python3 pair_bound.py OUT.json"""
import sys, json
import numpy as np

OUT = sys.argv[1]
import os
OPT = os.environ.get("OPTIMISTIC") == "1"   # added after the first run: pairs at every depth (teacher-forced p1), not only inside the accepted run
R = {}
for st in ("mbpp", "he"):
    z = np.load(f"/Users/kamen/TrainLab/data/sb98/tree_s30_{st}.npz")
    n = len(z["gen"])
    for tau in (0.95, 0.9):
        for D in (12, 20, 31):
            saved, checks = [], 0
            for w in range(n):
                rt, hlp = z[f"{w}_right"], z[f"{w}_hlp"]
                gen, a0 = int(z["gen"][w]), int(z["a0"][w])
                p = max(0, gen - 1 - a0)
                while p < rt.shape[0]:
                    row = rt[p]
                    acc = int(np.argmin(np.append(row == 1, False)))
                    p1 = np.exp(hlp[p, :, 0].astype(np.float64))
                    j, pairs = 1, 0
                    while j <= D:
                        if (OPT or j <= acc) and j + 1 <= D and p1[j - 1] >= tau:
                            pairs += 1; j += 2
                        else:
                            j += 1
                    saved.append(pairs / D); checks += 1; p += acc + 1
            R[f"{st}|tau{tau}|D{D}"] = {"saved_frac": float(np.mean(saved)), "checks": checks}
            print(f"{st} tau {tau} D {D}: perfect pairing saves {100 * np.mean(saved):.1f} % of levels ({checks} checks)")
ok = all(R[f"{st}|tau0.95|D20"]["saved_frac"] >= 0.20 for st in ("mbpp", "he"))
print("VERDICT S21 stage 0: " + ("GO to stage 1" if ok else "KILL (too few sure runs)"))
json.dump({"res": R, "go": ok}, open(OUT, "w"), indent=1)
