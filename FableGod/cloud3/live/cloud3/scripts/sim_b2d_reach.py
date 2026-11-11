import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import union_counts
W = load(); doms = ["code", "ifeval", "multi"]; E = [0.0, 0.3, 0.9, 0.99, 1.01]
def band(c): return int(np.searchsorted(E, c, side="right") - 1) if c >= 0 else 4
def kk_for(ch, tau, kr=2):
    R = ch["R"]; L = ch["L"]; cf = np.nan_to_num(ch["cf"], nan=-1.0); kk = np.full((R, 48), 10, np.int64)
    reach = np.ones(R); run = np.cumprod(np.where(cf[:L] >= 0, cf[:L], 1.0)); reach[1:L + 1] = run[:L]
    for i in range(R):
        b = band(cf[i]); kk[i, :] = [10, 8, 5, 2, 10][b]
        if tau is not None and reach[i] < tau and b <= 2 and cf[i] >= 0: kk[i, :] = np.minimum(kk[i, :], kr)
    return kk
for tau in [None, 0.5, 0.7]:
    line = f"  b2d + reach<{tau} top-2 on cf<0.99 rows"
    for d in doms:
        tot = 0; ex = 0
        for w in W:
            if w["dom"] != d: continue
            for ch in checks_of(w):
                I = w["ids"][ch["rows"]]; tot += union_counts(I, kk_for(ch, tau)).sum(); ex += union_counts(I, np.full((ch["R"], 48), 10)).sum()
        line += f" | {d}: x{tot/ex:.3f}"
    print(line, flush=True)
