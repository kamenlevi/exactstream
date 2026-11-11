"""LRU replay on the 36 windows (ledger conventions: plain LRU per layer for the MSB base pool, warmed by the prompt rows,
one session per domain carried across windows in file order; live-cut checks; rows after a check's first miss enter as the
map's wrong-draft stand-in, row p + 97 mod N). For each rule and pool: SSD misses per kept token (x 2.1627 MB) and TILES per
kept token (distinct experts COMPUTED per layer, resident ones included). Rules: exact; QOM (resident keep: sure rows keep
ranks 0-1 + any rank 2-9 that is resident after the block's loads; unsure rows all 10); strict forms (no resident keep);
BANDED-K tables; TOP1. The bonus row is gated by its next draft's cf under 'bq'. The sidecar (LSB) pool is not replayed."""
import numpy as np, sys, json, os
from collections import OrderedDict
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
MID = set(range(8, 36)); MB = 2162688 / 1e6
W = load(); doms = ["code", "ifeval", "multi"]
SZ = None
p = os.path.expanduser("~/350-tps/MoonShots/data/laysz45_256.json")
if os.path.exists(p): SZ = json.load(open(p))["C_m"]
POOLS = {"45GB": (SZ if SZ else [266] * 48), "24GB": [147] * 48, "12GB": [51] * 48}

def rows_of(ch, N, standin=True):
    rows = ch["rows"].copy()
    if standin:
        for i in range(ch["kept"], ch["R"]): rows[i] = (ch["rows"][i] + 97) % N
    return rows

def row_k(ch, rule):
    """per-row base ranks kept before residency: returns (k_sure [R] array of ranks per layer, sure mask)"""
    R = ch["R"]; cf = ch["cf"]; has = cf == cf; c = np.nan_to_num(cf, nan=-1)
    thr = rule.get("thr", 0.5); sure = has & (c >= thr)
    if not rule.get("bq", True): sure[ch["L"]] = False
    kk = np.full((R, 48), 10, np.int64)
    ks = rule.get("ks", 2); kk[sure, :] = ks
    for (layers, ks_b, ku_b) in rule.get("bands", []):
        L = np.array(sorted(layers)); kk[np.ix_(sure, L)] = ks_b; kk[np.ix_(~sure, L)] = ku_b
    return kk, sure

RULES = {
 "exact": dict(kind="exact"),
 "QOM cf0.5 resident-keep (bonus exact)": dict(kind="qom", thr=0.5, bq=False),
 "QOM cf0.5 resident-keep + BQ": dict(kind="qom", thr=0.5, bq=True),
 "QOM cf0 resident-keep + BQ (code FAST stack)": dict(kind="qom", thr=0.0, bq=True),
 "strict Q1 cf0.5 (bonus exact)": dict(kind="strict", thr=0.5, bq=False),
 "strict Q1 cf0.5 + BQ (T1)": dict(kind="strict", thr=0.5, bq=True),
 "strict T2: + unsure k6 in L8-35": dict(kind="strict", thr=0.5, bq=True, bands=[(MID, 2, 6)]),
 "strict T4: unsure k8 outside, k5 in L8-35": dict(kind="strict", thr=0.5, bq=True, bands=[(MID, 2, 5), (set(range(0, 8)) | set(range(36, 48)), 2, 8)]),
 "strict TOP1 cf0 + BQ": dict(kind="strict", thr=0.0, bq=True, ks=1),
}
out = {}
for pname, C in POOLS.items():
    for rname, rule in RULES.items():
        tot = {d: dict(miss=0, tiles=0, kept=0, loads=0) for d in doms}
        for dom in doms:
            lru = [OrderedDict() for _ in range(48)]
            def touch(l, es):
                od = lru[l]
                for e in es:
                    if e in od: od.move_to_end(e)
                    else: od[e] = True
                while len(od) > C[l]: od.popitem(last=False)
            for w in W:
                if w["dom"] != dom: continue
                ids = w["ids"]; N = ids.shape[0]
                for p_ in range(0, w["gen"] - 1):          # prompt rows warm the pools
                    for l in range(48): touch(l, ids[p_, l].tolist())
                for ch in checks_of(w):
                    rows = rows_of(ch, N); I = ids[rows]; R = ch["R"]
                    if rule["kind"] == "exact": kk, sure = np.full((R, 48), 10), np.zeros(R, bool)
                    else: kk, sure = row_k(ch, rule)
                    for l in range(48):
                        loads = set()
                        for r in range(R):
                            k = int(kk[r, l])
                            if k > 0: loads.update(I[r, l, :k].tolist())
                        res = lru[l]
                        miss = sum(1 for e in loads if e not in res)
                        comp = set(loads)
                        if rule["kind"] == "qom":
                            for r in range(R):
                                if sure[r]:
                                    for e in I[r, l, 2:].tolist():
                                        if e in res or e in loads: comp.add(e)
                        tot[dom]["miss"] += miss; tot[dom]["tiles"] += len(comp); tot[dom]["loads"] += len(loads)
                        touch(l, comp)
                    for d in [dom]: tot[d]["kept"] += ch["kept"]
        out[(pname, rname)] = {d: dict(mb=tot[d]["miss"] * MB / tot[d]["kept"], tiles=tot[d]["tiles"] / tot[d]["kept"], loads=tot[d]["loads"] / tot[d]["kept"]) for d in doms}
        print(f"{pname:5s} {rname:46s} " + " ".join(f"{d}: {out[(pname, rname)][d]['mb']:6.1f}MB {out[(pname, rname)][d]['tiles']:6.1f}t" for d in doms), flush=True)
json.dump({f"{k[0]}|{k[1]}": v for k, v in out.items()}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_lru_out.json"), "w"), indent=1)
