"""Stretch check: union growth per added row under exact vs the BANDED-K stack (does the per-row increment fall?), on the
uncut 16-row walk checks (so every check has 16 rows), code / IFEval / multi; rows classed by the drafter's cf as in the live rule."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load
W = load(); MID = np.array([8 <= l < 36 for l in range(48)])
def kk_for(cf, R, rule):
    c = np.nan_to_num(cf, nan=-1); has = cf == cf; sure = has & (c >= 0.5)
    kk = np.full((R, 48), 10, np.int64)
    if rule == "exact": return kk
    kk[sure, :] = 2
    if rule == "banded":
        kk[np.ix_(~sure, MID)] = 5
    return kk
for dom in ["code", "ifeval", "multi"]:
    for rule in ["exact", "q1", "banded"]:
        g = np.zeros(16); n = 0
        for w in W:
            if w["dom"] != dom: continue
            for (A, R, k), dc in zip(w["walk"], w["dcf"]):
                if A + 16 > w["ids"].shape[0] or R < 16: continue
                cf = np.full(16, np.nan, np.float32); cf[:15] = dc[:15]
                kk = kk_for(cf, 16, rule); I = w["ids"][A:A + 16]
                for l in range(48):
                    s = set()
                    for r in range(16):
                        s.update(I[r, l, :int(kk[r, l])].tolist()); g[r] += len(s)
                n += 1
        g /= (n * 48)
        inc = np.diff(g)
        print(f"{dom:6s} {rule:7s} union after r rows: " + " ".join(f"{x:4.1f}" for x in g) + f" | mean increment rows 2-8: {inc[:8].mean():.2f}, rows 9-16: {inc[8:].mean():.2f}, exponent fit {np.polyfit(np.log(np.arange(1,17)), np.log(g), 1)[0]:.2f}")
