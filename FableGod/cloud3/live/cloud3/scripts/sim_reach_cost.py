"""Direct-flip cost of REACH-CUT at tau on top of c3_b2d: the cut rows (cf < 0.99, reach < tau) that are REACHED (i < kept)
flip at their band's topk2 rate instead of the table's k; cost per realized row = sum over those rows of (topk2 - table-k rate)."""
import json, numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load(); fr = json.load(open("/root/350-tps/FableGod/floor_rows.json")); F = {(r["nm"], r["j"]): r for r in fr}
E = [0.0, 0.3, 0.9, 0.99, 1.01]
def band(c): return int(np.searchsorted(E, c, side="right") - 1) if c >= 0 else 4
for d in ["code", "ifeval", "multi"]:
    rate = {b: {"topk2": [0, 0], "topk5": [0, 0]} for b in range(4)}
    for w in W:
        if w["dom"] != d: continue
        for ch in checks_of(w):
            for i in range(ch["kept"]):
                rec = F.get((w["nm"], int(ch["rows"][i]) - w["gen"])); c = ch["cf"][i]
                if rec is None or not (c == c): continue
                b = band(c)
                for a in ("topk2", "topk5"): rate[b][a][0] += int(rec["flip"][a]); rate[b][a][1] += 1
    r2 = {b: rate[b]["topk2"][0] / max(1, rate[b]["topk2"][1]) for b in range(4)}; r5 = {b: rate[b]["topk5"][0] / max(1, rate[b]["topk5"][1]) for b in range(4)}
    base = {1: 0.0, 2: r5[2], 3: r2[3]}        # the table's own direct rate by band (k8 -> 0, k5 -> top-5 rate, k2 -> top-2 rate)
    for tau in [0.5, 0.7]:
        extra = 0.0; n = 0; cutr = 0
        for w in W:
            if w["dom"] != d: continue
            for ch in checks_of(w):
                R = ch["R"]; L = ch["L"]; cf = np.nan_to_num(ch["cf"], nan=-1.0); reach = np.ones(R); run = np.cumprod(np.where(cf[:L] >= 0, cf[:L], 1.0)); reach[1:L + 1] = run[:L]
                for i in range(ch["kept"]):
                    n += 1; b = band(cf[i])
                    if cf[i] >= 0 and reach[i] < tau and b in (1, 2): extra += r2[b] - base[b]; cutr += 1
        print(f"{d} tau {tau}: reached cut rows {cutr/n:.4f} of realized rows; extra direct flips {100*extra/n:.2f} points")
