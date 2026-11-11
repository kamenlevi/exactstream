"""REACH-CUT-X: rows with reach < tau (and not sure) run cheap (top-2 + union); if the check's FIRST MISS lands on a cheap row,
its correction is not emitted (the check ends at the previous row; the next check re-verifies from there exactly), so the only
deviation left is a false accept on a cheap row. Price: the kept loss (1 token when the first miss is on a cheap row) and
the union saving (REACH-CUT's). Also: per-window bootstrap CIs for the headline strict rules."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of, distinct, rule_k
W = load(); doms = ["code", "ifeval", "multi"]
print("REACH-CUT-X kept loss, per domain, thr 0.5 sureness gate:")
for dom in doms:
    for tau in [0.3, 0.5, 0.7]:
        n = 0; fm_cheap = 0; kept = 0; kept_x = 0; cheap_rows = 0; rows = 0; reached_cheap = 0
        for w in W:
            if w["dom"] != dom: continue
            for ch in checks_of(w):
                R = ch["R"]; L = ch["L"]; cf = ch["cf"]; c = np.nan_to_num(cf, nan=-1); has = cf == cf
                sure = has & (c >= 0.5)
                reach = np.ones(R); run = np.cumprod(np.nan_to_num(cf[:L], nan=1.0)); reach[1:L + 1] = run[:L]
                cheap = (reach < tau) & ~sure
                k_acc = ch["kept"] - 1                      # accepted drafts within the cut
                first_miss = k_acc if k_acc < L else None    # row index of the first miss (None: all L accepted, bonus row)
                n += 1; kept += ch["kept"]; cheap_rows += cheap.sum(); rows += R
                loss = 1 if (first_miss is not None and cheap[first_miss]) else 0
                fm_cheap += loss; kept_x += ch["kept"] - loss
                reached_cheap += cheap[:ch["kept"]].sum()    # cheap rows that were actually reached (kept rows)
        print(f"  {dom:6s} tau {tau}: cheap rows {cheap_rows/rows:.3f} of rows; first miss on a cheap row {fm_cheap/n:.3f} of checks; kept {kept/n:.2f} -> {kept_x/n:.2f} (x{kept_x/kept:.3f}); reached cheap rows per check {reached_cheap/n:.2f}")
print("\nper-window bootstrap (12 windows per domain, 2000 resamples) of the distinct-per-kept ratio vs exact:")
rules = {"Q1 0.5 strict + BQ (T1)": ("q1_bq", dict(thr=0.5, ks=2), None), "Q1 0.9 strict + BQ": ("q1_bq", dict(thr=0.9, ks=2), None),
         "uniform top-8": ("topk", dict(k=8), None)}
rng = np.random.default_rng(0)
for name, (rule, p, _) in rules.items():
    line = f"  {name:28s}"
    for dom in doms:
        per = []
        for w in W:
            if w["dom"] != dom: continue
            a = 0; b = 0
            for ch in checks_of(w):
                I = w["ids"][ch["rows"]]
                a += distinct(I, rule_k(ch, rule, **p)).sum(); b += distinct(I, np.full(ch["R"], 10)).sum()
            per.append((a, b))
        per = np.array(per, float); point = per[:, 0].sum() / per[:, 1].sum()
        bs = [per[idx, 0].sum() / per[idx, 1].sum() for idx in (rng.integers(0, len(per), len(per)) for _ in range(2000))]
        line += f" {dom}: x{point:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] per-window range {per[:,0].sum()/per[:,1].sum():.3f}"
        line = line.rsplit(" per-window", 1)[0] + f" (windows {np.min(per[:,0]/per[:,1]):.2f}-{np.max(per[:,0]/per[:,1]):.2f})"
    print(line)
