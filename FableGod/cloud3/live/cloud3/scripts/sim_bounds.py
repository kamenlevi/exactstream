"""P2 upper bounds per policy kind on the 36 windows (live-cut): reuse across positions (P-frame: sure rows post nothing in a
band), sure-row cuts at the oracle (p1) vs the drafter (cf), rejected-row cuts, and their combinations."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of, rule_k, MID_L
W = load(); doms = ["code", "ifeval", "multi"]
def union_counts(ids_rows, kk):   # kk [R, 48] ranks per row per layer
    out = np.zeros(48, np.int64)
    for l in range(48):
        s = set()
        for r in range(ids_rows.shape[0]):
            k = int(kk[r, l])
            if k > 0: s.update(ids_rows[r, l, :k].tolist())
        out[l] = len(s)
    return out
arms = {}
def A(name, fn): arms[name] = fn
band = np.array([l in MID_L for l in range(48)])
def mk(ch, base_rule, p, mod):
    R = ch["R"]; k = rule_k(ch, base_rule, **p); kk = np.repeat(k[:, None], 48, 1)
    cf = np.nan_to_num(ch["cf"], nan=-1); has = ch["cf"] == ch["cf"]
    if mod == "sure_post_nothing_band":           # P-frame reuse on sure rows in 8-35 (the row's MoE output copied from the previous row)
        sure = has & (cf >= p["thr"]); kk[np.ix_(sure, band)] = 0
    elif mod == "all_post_nothing_band_alt":       # every other row reuses in 8-35 (rows 1,3,5.. post nothing)
        alt = (np.arange(R) % 2 == 1); kk[np.ix_(alt, band)] = 0
    elif mod == "kept_only":
        kk[ch["kept"]:, :] = 0
    elif mod == "p1_sure_top2":
        p1 = np.nan_to_num(ch["p1"], nan=0); kk[p1 >= 0.99, :] = np.minimum(kk[p1 >= 0.99, :], 2)
    return kk
specs = {
 "exact": ("exact", {}, None),
 "ORACLE kept rows only": ("exact", {}, "kept_only"),
 "Q1 0.5 strict": ("q1", dict(thr=0.5, ks=2), None),
 "Q1 0.5 + sure rows post nothing in L8-35 (P-frame reuse bound)": ("q1", dict(thr=0.5, ks=2), "sure_post_nothing_band"),
 "exact + every other row posts nothing in L8-35 (reuse on all, alternating)": ("exact", {}, "all_post_nothing_band_alt"),
 "Q1 0.5 + kept rows only (sure cut + rejected oracle)": ("q1", dict(thr=0.5, ks=2), "kept_only"),
 "Q1 0.5 + 125B-sure rows top-2 (drafter + oracle sureness)": ("q1", dict(thr=0.5, ks=2), "p1_sure_top2"),
 "Q1 0.9 + 125B-sure rows top-2": ("q1", dict(thr=0.9, ks=2), "p1_sure_top2"),
 "125B-sure top-2 + kept only (both oracles)": ("exact", {}, "both"),
}
res = {}
for name, (rule, p, mod) in specs.items():
    tot = {d: [0, 0] for d in doms}
    for w in W:
        for ch in checks_of(w):
            ids_rows = w["ids"][ch["rows"]]
            if mod == "both":
                kk = mk(ch, rule, p, "kept_only"); p1 = np.nan_to_num(ch["p1"], nan=0); kk[p1 >= 0.99, :] = np.minimum(kk[p1 >= 0.99, :], 2)
            else: kk = mk(ch, rule, p, mod)
            u = union_counts(ids_rows, kk); tot[w["dom"]][0] += int(u.sum()); tot[w["dom"]][1] += ch["kept"]
    res[name] = {d: tot[d][0] / tot[d][1] for d in doms}
base = res["exact"]
print(f"{'arm':78s} " + " ".join(f"{d:>14s}" for d in doms))
for name, r in res.items(): print(f"{name:78s} " + " ".join(f"{r[d]:7.1f} x{r[d]/base[d]:.3f}" for d in doms))
