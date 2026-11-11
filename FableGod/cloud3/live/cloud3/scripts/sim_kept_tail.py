"""Order 8 (a): the kept rows' rank 2-9 tail beyond the band tables: per-layer cuts on the SURE bands of c3_b2d
(top-2 >= 0.99, top-5 on [0.9, 0.99), top-8 below). Quality bound for an in-band cut = the band's measured flip rate under FULL
MoE-output reuse in those layers (floor_rows, realized rows), harsher than keeping the row's own top expert [G].
(b): of the rejected rows, the share with reach < 0.5 (cut by REACH-CUT) vs reach >= 0.5."""
import json, numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import union_counts
W = load(); doms = ["code", "ifeval", "multi"]
E = [0.0, 0.3, 0.9, 0.99, 1.01]; L1235 = np.array([12 <= l < 36 for l in range(48)]); L835 = np.array([8 <= l < 36 for l in range(48)])
def band(c): return int(np.searchsorted(E, c, side="right") - 1) if c >= 0 else 4
def kk_for(ch, var):
    R = ch["R"]; cf = np.nan_to_num(ch["cf"], nan=-1.0); kk = np.full((R, 48), 10, np.int64)
    for i in range(R):
        b = band(cf[i]); k = [10, 8, 5, 2, 10][b]; kk[i, :] = k
        if var in ("D1", "D3") and b == 3: kk[i, L1235] = 1
        if var in ("D2", "D3") and b == 2: kk[i, L1235] = 3
        if var == "D4" and b == 3: kk[i, L835] = 1
        if var == "D4" and b == 2: kk[i, L835] = 3
        if var == "D5" and b == 3: kk[i, L1235] = 0          # shared-expert fallback: no routed experts for the >= 0.99 rows in 12-35
    return kk
print("tiles ratio vs exact (live-cut, strict): c3_b2d and its per-layer sure-band cuts")
for var in ["b2d", "D1", "D2", "D3", "D4", "D5"]:
    line = f"  {var:4s}"
    for d in doms:
        tot = 0; ex = 0
        for w in W:
            if w["dom"] != d: continue
            for ch in checks_of(w):
                I = w["ids"][ch["rows"]]; tot += union_counts(I, kk_for(ch, var)).sum(); ex += union_counts(I, np.full((ch["R"], 48), 10)).sum()
        line += f" | {d}: x{tot/ex:.3f}"
    print(line, flush=True)
print("\n(b) rejected rows (after the first miss, within the cut): share with reach < 0.5 / < 0.7 (cut by REACH-CUT at that tau), and their share of rows")
for d in doms:
    rej = 0; lo5 = 0; lo7 = 0; rows = 0
    for w in W:
        if w["dom"] != d: continue
        for ch in checks_of(w):
            R = ch["R"]; L = ch["L"]; cf = ch["cf"]; reach = np.ones(R); run = np.cumprod(np.nan_to_num(cf[:L], nan=1.0)); reach[1:L + 1] = run[:L]
            for i in range(ch["kept"], R): rej += 1; lo5 += reach[i] < 0.5; lo7 += reach[i] < 0.7
            rows += R
    print(f"  {d}: rejected rows {rej/rows:.3f} of rows; of them reach < 0.5: {lo5/max(1,rej):.3f}, < 0.7: {lo7/max(1,rej):.3f}")
