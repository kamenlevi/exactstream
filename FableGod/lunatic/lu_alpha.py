#!/usr/bin/env python3 -I
"""lu_alpha.py (Lunatic, round 1). CPU only, recorded data. What per-token agreement alpha can FAST reach, and which
rows stop long blocks? Joins DFlash2's HPD-0 code walk (~/TrainLab/data/d02b_walk_hpd0_code.npz: per check k accepted,
m drafted, A start; drafted token + cf per block position j) with the decision map's 125B top-k (baseline.json), as
~/FableGod/knowledge_probe.py does. Classes by the 125B: sure p1 >= 0.99, tie p1 - p2 <= 0.1, else mid.
Relaxed rules (FAST): accept the draft at a tie / mid row if it is the 125B's top-2 with p >= 0.2 (R1) or top-3 with
p >= 0.1 (R2). Sure-row misses are never relaxable (the 125B gives the draft < 0.01). First-order: ignores that a
relaxed token changes the later text. Prices kept and GB per kept token on lu_union's code union curve."""
import json, os, math, numpy as np
TD = os.path.expanduser("~/TrainLab/data"); OUT = os.path.expanduser("~/FableGod/decision_map/out")
base = json.load(open(f"{OUT}/baseline.json"))
W = np.load(f"{TD}/d02b_walk_hpd0_code.npz", allow_pickle=False); names = [str(x) for x in W["names"]]
R = []
for i, nm in enumerate(names):
    if nm not in base or base[nm].get("dom") != "code" or f"{i}_dtok" not in W.files: continue
    ch = W[f"{i}_checks"].ravel(); dt = W[f"{i}_dtok"].ravel(); dc = W[f"{i}_dcf"].ravel() if f"{i}_dcf" in W.files else None
    nc = len(ch) // 3; per = len(dt) // max(nc, 1); b = base[nm]; a = b["a"]
    for c in range(nc):
        k, m, A = int(ch[3*c]), int(ch[3*c+1]), int(ch[3*c+2])
        for j in range(min(k + 1, m)):
            r = A + j - a
            if r < 0 or r >= len(b["tk"]): continue
            d = int(dt[c*per + j]); cf = float(dc[c*per + j]) if dc is not None else float("nan")
            pr = [math.exp(x) for x in b["lp"][r]]; tk = [int(t) for t in b["tk"][r]]
            cls = "sure" if pr[0] >= 0.99 else ("tie" if pr[0] - pr[1] <= 0.1 else "mid")
            rk = tk.index(d) if d in tk else 99; pd = pr[rk] if rk < 99 else 0.0
            R.append((cls, j, j < k, cf, rk, pd))
print(f"tested code positions: {len(R)}; agreement of the walk's accept with the map's top-1: {np.mean([(x[4]==0)==x[2] for x in R])*100:.1f} %")
for cls in ["sure", "tie", "mid"]:
    s = [x for x in R if x[0] == cls]
    if not s: continue
    print(f"{cls:5s} n {len(s):5d} ({100*len(s)/len(R):4.1f} %)  drafter right {100*np.mean([x[2] for x in s]):5.1f} %"
          f"  R1-ok {100*np.mean([x[2] or (x[4]<=1 and x[5]>=0.2) for x in s]):5.1f} %  R2-ok {100*np.mean([x[2] or (x[4]<=2 and x[5]>=0.1) for x in s]):5.1f} %")
s = [x for x in R if x[0] == "sure"]
print("sure rows, drafter right by block position j: " + "  ".join(f"j{lo}-{hi}: {100*np.mean([x[2] for x in s if lo<=x[1]<=hi]):.1f} % (n {sum(lo<=x[1]<=hi for x in s)})" for lo, hi in [(0,0),(1,3),(4,7),(8,11),(12,15)]))
mis = [x for x in s if not x[2]]
print(f"sure-row misses: {len(mis)} = {100*len(mis)/len(R):.2f} % of all tested rows; drafter cf < 0.5 on {100*np.mean([x[3]<0.5 for x in mis]):.0f} % of them, cf >= 0.9 on {100*np.mean([x[3]>=0.9 for x in mis]):.0f} %; in the 125B's top-8 {100*np.mean([x[4]<99 for x in mis]):.0f} %")
U = {16: 76.6, 32: 115.1, 64: 165.6, 128: 228.0}
def price(al):
    out = []
    for Rr, u in U.items():
        kk = (1 - al ** (Rr + 1)) / (1 - al); out.append(f"R{Rr} k {kk:5.1f} {u*48*1.9e-3/kk + 3.44/kk:4.2f} GB")
    return " | ".join(out)
A = {"strict (EXACT today)": np.mean([x[2] for x in R]),
     "R1 relaxed tie/mid": np.mean([x[2] or (x[0] != "sure" and x[4] <= 1 and x[5] >= 0.2) for x in R]),
     "R2 relaxed tie/mid": np.mean([x[2] or (x[0] != "sure" and x[4] <= 2 and x[5] >= 0.1) for x in R]),
     "strict + sure rows perfect": np.mean([x[2] or x[0] == "sure" for x in R]),
     "R1 + sure rows perfect": np.mean([x[2] or x[0] == "sure" or (x[4] <= 1 and x[5] >= 0.2) for x in R]),
     "R2 + sure rows perfect": np.mean([x[2] or x[0] == "sure" or (x[4] <= 2 and x[5] >= 0.1) for x in R])}
for k, v in A.items(): print(f"alpha {v:.4f}  {k:28s} {price(min(v, 0.9995))}")
