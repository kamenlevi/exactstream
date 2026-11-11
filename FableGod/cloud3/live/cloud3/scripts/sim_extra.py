"""(a) uncut 16-row (walk) unions for comparability with the mission's 78-80; (b) per-layer profile of the live-cut union."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, distinct
W = load()
for dom in ["code", "ifeval", "multi"]:
    tot = np.zeros(48); n = 0; kept = 0; rows = 0; totk = np.zeros(48)
    for w in W:
        if w["dom"] != dom: continue
        for (A, R, k) in w["walk"]:
            if A + R > w["ids"].shape[0]: continue
            ids_rows = w["ids"][A:A + R]
            u = distinct(ids_rows, np.full(R, 10)); tot += u; n += 1; kept += k; rows += R
            uk = distinct(ids_rows[:k], np.full(k, 10)); totk += uk
    print(f"{dom} UNCUT walk checks: {n}, rows/check {rows/n:.2f}, kept/check {kept/n:.2f}, union per layer {tot.sum()/48/n:.1f} (kept rows only {totk.sum()/48/n:.1f}, x{totk.sum()/tot.sum():.3f}), distinct per kept token {tot.sum()/kept:.1f}")
    print("   per-layer union (uncut), layers 0..47:", " ".join(f"{x/n:.0f}" for x in tot))
