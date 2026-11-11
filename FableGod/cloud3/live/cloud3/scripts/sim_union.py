"""Union simulation on the 36 decision-map windows (MoonShots export, ~/c3mac/data/decision_map_moonshots).
Per live-cut check: rows A..A+L (L drafted rows + the bonus row), kept = min(k, L) + 1; row i (< L) verifies draft i+1 with
cf dcf[i]; the bonus row has 'next draft' cf dcf[L] when L < 15 (the QOM-BQ gate), else none.
Metric: distinct (expert, layer) pairs per kept token = sum_l |U_l| / kept, totals per domain; and per layer per check.
A rule gives each row a k (ranks kept, 0 = nothing): distinct = union over rows of ids[row, l, :k]. Union members
(OEA / Q1-UNION piggybacking) add no distinct experts, so they are not modelled here (quality only)."""
import numpy as np, glob, os, sys, json
D = "/root/c3mac/data/decision_map_moonshots"
MID_L = set(range(8, 36))      # the band DM P-frame found tolerant on sure rows

def load():
    W = []
    for f in sorted(glob.glob(os.path.join(D, "*.npz"))):
        z = np.load(f)
        W.append(dict(nm=os.path.basename(f), dom=str(z["domain"]), ids=z["ids"].astype(np.int16), conf=z["conf"],
                      checks=z["checks"], walk=z["checks_walk"], dcf=z["dcf"], p1=z["p1"], gen=int(z["gen"])))
    return W

def checks_of(w):
    """-> list of dicts: rows (array of row indices), kept, cf per row (nan = none), p1 per row, L"""
    out = []
    for (A, R, kept), dc in zip(w["checks"], w["dcf"]):
        L = R - 1
        rows = np.arange(A, A + R)
        if rows[-1] >= w["ids"].shape[0]: continue
        cf = np.full(R, np.nan, np.float32)
        cf[:L] = dc[:L]
        if L < 15: cf[L] = dc[L]          # the bonus row's next draft (QOM-BQ gate)
        out.append(dict(rows=rows, kept=int(kept), cf=cf, p1=w["p1"][rows], L=L, R=R))
    return out

def distinct(ids_rows, ks, layers=None):
    """ids_rows [R, 48, 10]; ks [R] ranks kept per row -> per-layer distinct counts [48]"""
    R = ids_rows.shape[0]; out = np.zeros(48, np.int64)
    for l in range(48):
        s = set()
        for r in range(R):
            k = int(ks[r])
            if k > 0: s.update(ids_rows[r, l, :k].tolist())
        out[l] = len(s)
    return out

def rule_k(ch, rule, **p):
    R = ch["R"]; L = ch["L"]; cf = ch["cf"]; kept = ch["kept"]
    k = np.full(R, 10, np.int64)
    drafted = np.arange(R) < L
    bonus = np.arange(R) == L
    has_draft = ~np.isnan(cf)
    if rule == "exact": pass
    elif rule == "oracle_kept":                      # only the kept rows exist (rejected rows cost nothing)
        k[kept:] = 0
    elif rule == "q1":                                # drafted rows with cf >= thr keep ks ranks; bonus row exact
        k[drafted & (np.nan_to_num(cf, nan=-1) >= p["thr"])] = p.get("ks", 2)
    elif rule == "q1_bq":                             # + the bonus row gated by its next draft's cf (QOM-BQ / DRAFT17)
        sure = has_draft & (np.nan_to_num(cf, nan=-1) >= p["thr"])
        k[sure] = p.get("ks", 2)
    elif rule == "topk":                              # uniform k on every row (Expert-Sample / Don't-skip-the-baseline)
        k[:] = p["k"]
    elif rule == "sure_unsure":                       # sure rows ks, every other row ku (UNION-STACK's form; bonus row ku)
        sure = has_draft & (np.nan_to_num(cf, nan=-1) >= p["thr"])
        k[:] = p["ku"]; k[sure] = p["ks"]
    elif rule == "reach":                             # REACH-CUT: unsure rows with reach < tau keep ks; sure rows ks; bonus row by reach
        sure = has_draft & (np.nan_to_num(cf, nan=-1) >= p["thr"])
        reach = np.ones(R)
        run = np.cumprod(np.nan_to_num(ch["cf"][:L], nan=1.0))
        reach[1:L + 1] = run[:L]                      # row i is reached iff drafts 1..i were accepted
        k[sure] = p["ks"]; k[(reach < p["tau"]) & ~sure] = p["ks"]
    elif rule == "reach_only":                        # only the reach rule, no sureness gate
        reach = np.ones(R); run = np.cumprod(np.nan_to_num(ch["cf"][:L], nan=1.0)); reach[1:L + 1] = run[:L]
        k[reach < p["tau"]] = p["ks"]
    elif rule == "p1_oracle":                         # the 125B's own class: rows with p1 >= thr keep ks (not realizable)
        k[np.nan_to_num(ch["p1"], nan=0) >= p["thr"]] = p["ks"]
    else: raise ValueError(rule)
    return k

def per_layer_k(ch, base_k, kmid, band=MID_L):
    """ranks per (row, layer): base_k outside the band, min(base_k, kmid) inside it, on rows selected by sel"""
    pass

def run(W, arms, dom_filter=None):
    res = {}
    for name, (rule, p, layer_rule) in arms.items():
        tot = {}
        for w in W:
            if dom_filter and w["dom"] != dom_filter: continue
            d = tot.setdefault(w["dom"], dict(pairs=0, kept=0, checks=0, rows=0, per_layer=np.zeros(48, np.int64), sure_rows=0))
            for ch in checks_of(w):
                ids_rows = w["ids"][ch["rows"]]
                k = rule_k(ch, rule, **p)
                if layer_rule is None:
                    u = distinct(ids_rows, k)
                else:
                    kmid, sel = layer_rule          # kmid applied in MID_L on rows where sel(ch) is true
                    m = sel(ch, k)
                    u = np.zeros(48, np.int64)
                    for l in range(48):
                        kk = k.copy()
                        if l in MID_L: kk[m] = np.minimum(kk[m], kmid)
                        s = set()
                        for r in range(ch["R"]):
                            if kk[r] > 0: s.update(ids_rows[r, l, :int(kk[r])].tolist())
                        u[l] = len(s)
                d["pairs"] += int(u.sum()); d["kept"] += ch["kept"]; d["checks"] += 1; d["rows"] += ch["R"]; d["per_layer"] += u
        res[name] = {dom: dict(dpk=v["pairs"] / v["kept"], dpl=v["pairs"] / (48 * v["checks"]), kept=v["kept"] / v["checks"],
                               rows=v["rows"] / v["checks"], checks=v["checks"]) for dom, v in tot.items()}
    return res

if __name__ == "__main__":
    W = load()
    doms = ["code", "ifeval", "multi"]
    print("windows:", len(W), {d: sum(1 for w in W if w["dom"] == d) for d in doms})
    # descriptive: rows, kept, cf bands of drafted rows, bonus-row draft availability
    for dom in doms:
        R = []; K = []; cfs = []; bonus_has = []; p1s = []
        for w in W:
            if w["dom"] != dom: continue
            for ch in checks_of(w):
                R.append(ch["R"]); K.append(ch["kept"]); cfs.extend(ch["cf"][:ch["L"]].tolist()); bonus_has.append(not np.isnan(ch["cf"][ch["L"]]))
                p1s.extend(np.nan_to_num(ch["p1"], nan=0).tolist())
        cfs = np.array(cfs); p1s = np.array(p1s)
        print(f"{dom}: checks {len(R)}, rows/check {np.mean(R):.2f}, kept/check {np.mean(K):.2f}, wasted rows/check {np.mean(R)-np.mean(K):.2f} "
              f"({100*(np.mean(R)-np.mean(K))/np.mean(R):.0f} %), drafted rows cf>=0.5/0.9/0.99: {np.mean(cfs>=0.5):.2f}/{np.mean(cfs>=0.9):.2f}/{np.mean(cfs>=0.99):.2f}, "
              f"bonus row has a draft: {np.mean(bonus_has):.2f}, rows with p1>=0.99: {np.mean(p1s>=0.99):.2f}")
    sel_unsure = lambda ch, k: (k == 10)
    sel_all = lambda ch, k: np.ones(ch["R"], bool)
    arms = {
        "exact (10 on every row)": ("exact", {}, None),
        "ORACLE kept rows only": ("oracle_kept", {}, None),
        "Q1 cf>=0.99 top-2": ("q1", dict(thr=0.99, ks=2), None),
        "Q1 cf>=0.9 top-2": ("q1", dict(thr=0.9, ks=2), None),
        "Q1 cf>=0.5 top-2": ("q1", dict(thr=0.5, ks=2), None),
        "Q1 cf>=0 top-2 (QOM cf0, bonus exact)": ("q1", dict(thr=0.0, ks=2), None),
        "Q1 cf>=0 top-2 + bonus gated (BQ)": ("q1_bq", dict(thr=0.0, ks=2), None),
        "Q1 cf>=0 top-1 + bonus gated": ("q1_bq", dict(thr=0.0, ks=1), None),
        "Q1 cf>=0.5 top-2 + bonus gated": ("q1_bq", dict(thr=0.5, ks=2), None),
        "Q1 cf>=0.9 top-2 + bonus gated": ("q1_bq", dict(thr=0.9, ks=2), None),
        "uniform top-8": ("topk", dict(k=8), None),
        "uniform top-5": ("topk", dict(k=5), None),
        "uniform top-2": ("topk", dict(k=2), None),
        "sure(0.9) 2 / unsure 8": ("sure_unsure", dict(thr=0.9, ks=2, ku=8), None),
        "sure(0.9) 2 / unsure 6": ("sure_unsure", dict(thr=0.9, ks=2, ku=6), None),
        "sure(0.5) 2 / unsure 8": ("sure_unsure", dict(thr=0.5, ks=2, ku=8), None),
        "sure(0.5) 2 / unsure 6": ("sure_unsure", dict(thr=0.5, ks=2, ku=6), None),
        "REACH-CUT: Q1 0.9 + reach<0.5 top-2": ("reach", dict(thr=0.9, ks=2, tau=0.5), None),
        "REACH-CUT: Q1 0.5 + reach<0.5 top-2": ("reach", dict(thr=0.5, ks=2, tau=0.5), None),
        "reach<0.5 top-2 only": ("reach_only", dict(ks=2, tau=0.5), None),
        "reach<0.3 top-2 only": ("reach_only", dict(ks=2, tau=0.3), None),
        "125B-sure oracle (p1>=0.99) top-2": ("p1_oracle", dict(thr=0.99, ks=2), None),
        "Q1 0.5 + unsure rows k 5 in L8-35": ("q1", dict(thr=0.5, ks=2), (5, sel_unsure)),
        "Q1 0.5 + unsure rows k 6 in L8-35": ("q1", dict(thr=0.5, ks=2), (6, sel_unsure)),
        "Q1 0.5 + unsure rows k 8 in L8-35": ("q1", dict(thr=0.5, ks=2), (8, sel_unsure)),
        "Q1 0.5 + ALL rows k 1 in L8-35": ("q1", dict(thr=0.5, ks=2), (1, sel_all)),
        "exact + all rows k 5 in L8-35": ("exact", {}, (5, sel_all)),
    }
    res = run(W, arms)
    base = res["exact (10 on every row)"]
    print(f"\n{'arm':44s} " + " ".join(f"{d:>22s}" for d in doms) + "   (distinct per layer per check; distinct per kept token; ratio vs exact)")
    for name, r in res.items():
        print(f"{name:44s} " + " ".join(f"{r[d]['dpl']:6.1f} {r[d]['dpk']:6.1f} x{r[d]['dpk']/base[d]['dpk']:.3f}" for d in doms))
    json.dump({k: {d: v for d, v in r.items()} for k, r in res.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_union_out.json"), "w"), indent=1)
