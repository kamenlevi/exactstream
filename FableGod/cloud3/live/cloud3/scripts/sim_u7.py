import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import union_counts
W = load(); E = [0.0, 0.3, 0.9, 0.99, 1.01]
def band(c): return int(np.searchsorted(E, c, side="right") - 1) if c >= 0 else 4
def kk_for(ch, tab, tau=0.7):
    R = ch["R"]; L = ch["L"]; cf = np.nan_to_num(ch["cf"], nan=-1.0); kk = np.full((R, 48), 10, np.int64)
    reach = np.ones(R); run = np.cumprod(np.where(cf[:L] >= 0, cf[:L], 1.0)); reach[1:L + 1] = run[:L]
    for i in range(R):
        b = band(cf[i]); kk[i, :] = tab[b]
        if cf[i] >= 0 and reach[i] < tau and b <= 2: kk[i, :] = np.minimum(kk[i, :], 2)
    return kk
for name, tab in [("stack (u8, mid 5)", [10, 8, 5, 2, 10]), ("u7", [10, 7, 5, 2, 10]), ("u6", [10, 6, 5, 2, 10]), ("mid 4", [10, 8, 4, 2, 10]), ("u7 + mid 4", [10, 7, 4, 2, 10])]:
    line = f"  {name:18s}"
    for d in ["code", "ifeval", "multi"]:
        tot = 0; ex = 0
        for w in W:
            if w["dom"] != d: continue
            for ch in checks_of(w):
                I = w["ids"][ch["rows"]]; tot += union_counts(I, kk_for(ch, tab)).sum(); ex += union_counts(I, np.full((ch["R"], 48), 10)).sum()
        line += f" | {d}: x{tot/ex:.3f}"
    print(line, flush=True)
