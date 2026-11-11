import json, os
import numpy as np
R = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
R = [r for r in R if r["arm"] == "dfl" and not r["paused"] and not r["invalid"]]
C0 = 42.2 - 12.1            # Breakthrough FIXED-PART: constant (dense + draft + syncs) minus the draft
PER_MISS = 0.149            # ms per miss at the SSD's marginal rate (io = 13.3 + 0.149 x misses)
def run(beta, bub):
    out = {}
    for r in R:
        pc = np.array(r["pc"], float)          # rows, kept, check ms, draft ms
        rows, kept, ck, dr = pc.T
        mr = np.median(rows)
        miss = r["miss"] * (rows / mr) ** 0.67
        today = (ck + dr).sum()
        prevA = np.concatenate([[0], kept[:-1]])        # A = the previous check's kept rows (accepted + bonus)
        comp = C0 + beta * (rows + prevA)
        ssd = PER_MISS * miss
        p = dr + np.maximum(comp, ssd) + bub * (dr + C0 + beta * rows)
        out.setdefault(r["type"], [0, 0, 0])
        o = out[r["type"]]; o[0] += kept.sum(); o[1] += today; o[2] += p.sum()
    return {t: (o[0] / o[1] * 1e3, o[0] / o[2] * 1e3) for t, o in out.items()}
print("D-23 dfl arm (45 GB default, C 224, t 0.3, 18 prompts); RUNAHEAD model: pass = draft + max(30.1 + beta x (rows_B + rows_A),"
      " 0.149 ms x misses_B) + bubble share; today = logged check + draft")
for bub in (0.05, 0.15):
    for beta in (1.5, 2.5, 3.5, 5.0, 7.82):
        res = run(beta, bub)
        tot_k = sum(sum(np.array(r["pc"])[:, 1]) for r in R)
        s = "  ".join(f"{t} {a:.1f}->{b:.1f} ({b / a - 1:+.0%})" for t, (a, b) in sorted(res.items()))
        print(f"  bubble {bub:.2f} beta {beta:4.2f} ms/row: {s}")
