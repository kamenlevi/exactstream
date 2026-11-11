"""P3 candidates on the 36 windows (live-cut, strict): (1) BANDED-K: k by (row class, layer band); (2) P-frame for tiles:
sure rows post nothing in one 12-layer group, non-chained (a sure row reuses only if the previous row computed that group);
(3) depth / reach-decaying k; (4) combinations with the gated bonus row."""
import numpy as np, sys, itertools
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load(); doms = ["code", "ifeval", "multi"]
def union_counts(ids_rows, kk):
    out = np.zeros(48, np.int64)
    for l in range(48):
        s = set()
        for r in range(ids_rows.shape[0]):
            k = int(kk[r, l])
            if k > 0: s.update(ids_rows[r, l, :k].tolist())
        out[l] = len(s)
    return out
def banded(ch, thr, k_sure, k_unsure, bands, bq=True, k_sure_out=None, reach_tau=None, k_reach=2):
    """bands: dict band_name -> (layers list, (k for sure rows, k for unsure rows)); rows outside listed bands keep 10 / k_sure"""
    R = ch["R"]; cf = ch["cf"]; has = cf == cf; c = np.nan_to_num(cf, nan=-1)
    sure = has & (c >= thr)
    if not bq: sure[ch["L"]] = False                    # bonus row exact unless gated
    kk = np.full((R, 48), 10, np.int64)
    kk[sure, :] = k_sure if k_sure_out is None else k_sure_out
    for name, (layers, (ks, ku)) in bands.items():
        L = np.array(layers)
        kk[np.ix_(sure, L)] = ks; kk[np.ix_(~sure, L)] = ku
    if reach_tau is not None:
        reach = np.ones(R); run = np.cumprod(np.nan_to_num(cf[:ch["L"]], nan=1.0)); reach[1:ch["L"] + 1] = run[:ch["L"]]
        low = (reach < reach_tau) & ~sure; kk[low, :] = np.minimum(kk[low, :], k_reach)
    return kk
def pframe(ch, thr, k_sure, group, chained=False):
    """sure rows post nothing in `group` layers if the previous row computed them (non-chained) or always (chained)."""
    R = ch["R"]; cf = ch["cf"]; has = cf == cf; c = np.nan_to_num(cf, nan=-1); sure = has & (c >= thr)
    kk = np.full((R, 48), 10, np.int64); kk[sure, :] = k_sure
    G = np.array(group); prev_computed = True
    for r in range(R):
        if sure[r] and (chained or prev_computed) and r > 0:
            kk[r, G] = 0; prev_computed = False
        else: prev_computed = True
    return kk
MID = list(range(8, 36)); EARLY = list(range(0, 8)); LATE = list(range(36, 48))
specs = {
 "exact": lambda ch: np.full((ch["R"], 48), 10),
 "Q1 0.5 strict (bonus exact)": lambda ch: banded(ch, 0.5, 2, 10, {}, bq=False),
 "Q1 0.5 + BQ": lambda ch: banded(ch, 0.5, 2, 10, {}),
 "Q1 0.5 + BQ + unsure k6 in 8-35": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (2, 6))}),
 "Q1 0.5 + BQ + unsure k5 in 8-35": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (2, 5))}),
 "Q1 0.5 + BQ + unsure k8 in 0-7/36-47, k5 in 8-35": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (2, 5)), "early": (EARLY, (2, 8)), "late": (LATE, (2, 8))}),
 "Q1 0.5 + BQ + sure k1 in 8-35, unsure k6 in 8-35": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (1, 6))}),
 "Q1 0.5 + BQ + sure k1 in 8-35, unsure k5 in 8-35": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (1, 5))}),
 "Q1 0.5 + BQ + unsure k6 in 8-35 + reach<0.5 top-2": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (2, 6))}, reach_tau=0.5),
 "Q1 0.5 + BQ + unsure k6 in 8-35 + reach<0.3 top-2": lambda ch: banded(ch, 0.5, 2, 10, {"mid": (MID, (2, 6))}, reach_tau=0.3),
 "Q1 0.9 strict + BQ": lambda ch: banded(ch, 0.9, 2, 10, {}),
 "Q1 0.9 + BQ + unsure k6 in 8-35": lambda ch: banded(ch, 0.9, 2, 10, {"mid": (MID, (2, 6))}),
 "Q1 0.9 + BQ + unsure k8 in 8-35": lambda ch: banded(ch, 0.9, 2, 10, {"mid": (MID, (2, 8))}),
 "Q1 0.9 + BQ + unsure k6 in 8-35 + reach<0.5": lambda ch: banded(ch, 0.9, 2, 10, {"mid": (MID, (2, 6))}, reach_tau=0.5),
 "Q1 0.5 + BQ + P-frame sure rows in 12-23 (non-chained)": lambda ch: pframe(ch, 0.5, 2, list(range(12, 24))),
 "Q1 0.5 + BQ + P-frame sure rows in 12-35 (non-chained)": lambda ch: pframe(ch, 0.5, 2, list(range(12, 36))),
 "Q1 0.5 + BQ + P-frame sure rows in 8-35 (non-chained)": lambda ch: pframe(ch, 0.5, 2, MID),
 "Q1 0.5 + BQ + P-frame sure rows in 8-35 (chained)": lambda ch: pframe(ch, 0.5, 2, MID, chained=True),
 "Q1 0.9 + BQ + P-frame sure rows in 12-35 (non-chained)": lambda ch: pframe(ch, 0.9, 2, list(range(12, 36))),
}
res = {}
for name, fn in specs.items():
    tot = {d: [0, 0] for d in doms}
    for w in W:
        for ch in checks_of(w):
            kk = fn(ch); u = union_counts(w["ids"][ch["rows"]], kk)
            tot[w["dom"]][0] += int(u.sum()); tot[w["dom"]][1] += ch["kept"]
    res[name] = {d: tot[d][0] / tot[d][1] for d in doms}
base = res["exact"]
print(f"{'arm':62s} " + " ".join(f"{d:>14s}" for d in doms))
for name, r in res.items(): print(f"{name:62s} " + " ".join(f"{r[d]:7.1f} x{r[d]/base[d]:.3f}" for d in doms))
