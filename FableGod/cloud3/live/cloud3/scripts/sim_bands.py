"""Confidence-band k tables, priced for BOTH the distinct count (live-cut, strict, BQ) and an expected flip rate:
expected flips = sum over bands of (share of realized rows in the band) x (the map's measured flip rate of that band under
the uniform cut with the same k: topk2 or topk5 from floor_rows.json, every row perturbed, no union members = an upper bound),
with k >= 8 taken as 0 flips. Rates are computed from floor_rows at the table's own band edges on the realized rows
(A..A+kept-1, joined through the MoonShots checks). Also: per-band reuse flips on reached unsure rows (layer sensitivity)."""
import json, numpy as np, sys, collections
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import union_counts, MID
W = load(); doms = ["code", "ifeval", "multi"]
fr = json.load(open("/root/350-tps/FableGod/floor_rows.json")); F = {(r["nm"], r["j"]): r for r in fr}
EDGES = [0.0, 0.5, 0.9, 0.99, 0.999, 1.01]; BN = ["<0.5", "0.5-0.9", "0.9-0.99", "0.99-0.999", ">=0.999"]
def band(c): return int(np.searchsorted(EDGES, c, side="right") - 1) if c == c and c >= 0 else -1
# 1) per-band flip rates on realized rows (reached), per domain, for topk2 / topk5 and the reuse arms
rates = {d: {a: np.zeros((5, 2)) for a in ["topk2", "topk5", "moe_reuse_prev_g0", "moe_reuse_prev_g1", "moe_reuse_prev_g2", "moe_reuse_prev_g3", "moe_reuse_prev_L08_11", "moe_reuse_prev_L12_35", "moe_reuse_prev_L36_39"]} for d in doms}
shares = {d: np.zeros(6) for d in doms}   # realized rows per band (index 5 = no draft / bonus without draft)
for w in W:
    d = w["dom"]; gen = w["gen"]
    for ch in checks_of(w):
        for i in range(ch["kept"]):
            row = int(ch["rows"][i]); c = ch["cf"][i]; b = band(c); shares[d][b if b >= 0 else 5] += 1
            rec = F.get((w["nm"], row - gen))
            if rec is None or b < 0: continue
            for a in rates[d]:
                rates[d][a][b] += (int(rec["flip"][a]), 1)
for d in doms:
    sh = shares[d] / shares[d].sum()
    print(f"\n== {d}: realized rows by cf band {BN} + no-draft: " + " ".join(f"{x:.3f}" for x in sh))
    for a in rates[d]:
        r = rates[d][a]; print(f"   {a:24s} flips % by band: " + " ".join(f"{100*r[b,0]/max(1,r[b,1]):6.2f}" for b in range(5)))
# 2) tables: k per band (bands <0.5, 0.5-0.9, 0.9-0.99, 0.99-0.999, >=0.999, nodraft), optional band-k for the lowest class in 8-35
TABLES = {
 "Q1 0.9 strict + BQ (ref: map PASS code)": ([10, 10, 2, 2, 2, 10], None),
 "T1 = Q1 0.5 strict + BQ": ([10, 2, 2, 2, 2, 10], None),
 "B1: >=0.9: 2, 0.5-0.9: 5, <0.5: 10": ([10, 5, 2, 2, 2, 10], None),
 "B2: >=0.99: 2, 0.9-0.99: 5, <0.9: 10": ([10, 10, 5, 2, 2, 10], None),
 "NEW-1 live: >=0.999: 2, 0.99-0.999: 4, 0.9-0.99: 8, <0.9: 10": ([10, 10, 8, 4, 2, 10], None),
 "1%-bar live: >=0.999: 2, 0.99-0.999: 5, rest 10": ([10, 10, 10, 5, 2, 10], None),
 "B1 + <0.5 rows k 5 in 8-35": ([10, 5, 2, 2, 2, 10], 5),
 "T1 + <0.5 rows k 5 in 8-35 (T3)": ([10, 2, 2, 2, 2, 10], 5),
 "B2 + <0.9 rows k 5 in 8-35": ([10, 10, 5, 2, 2, 10], ("lt09", 5)),
}
def rate_for(d, b, k):
    if k >= 8: return 0.0
    a = "topk2" if k <= 2 else "topk5"   # k 3-5: the top-5 rate (a floor for k 4 / 5; k 3 unmeasured)
    r = rates[d][a][b]; return r[0] / max(1, r[1])
print("\n== tables: distinct ratio vs exact | expected all-row flips % (upper bound) vs lines 1.34 / 5.42 / 3.69")
for name, (ks, extra) in TABLES.items():
    line = f"{name:58s}"
    for d in doms:
        tot = 0; ex = 0; K = 0; ef = 0.0; n = 0
        for w in W:
            if w["dom"] != d: continue
            for ch in checks_of(w):
                R = ch["R"]; cf = ch["cf"]; I = w["ids"][ch["rows"]]
                kk = np.full((R, 48), 10, np.int64)
                for i in range(R):
                    b = band(cf[i]); k = ks[b] if b >= 0 else ks[5]; kk[i, :] = k
                    if extra is not None:
                        if isinstance(extra, tuple):
                            if b in (0, 1): kk[i, list(MID)] = np.minimum(kk[i, list(MID)], extra[1])
                        elif b == 0: kk[i, list(MID)] = np.minimum(kk[i, list(MID)], extra)
                    if i < ch["kept"] and b >= 0:
                        base = rate_for(d, b, k); n += 1
                        # band cut on top: add half of the topk5 rate for the band-cut rows (28 of 48 layers)
                        if extra is not None and k == 10 and kk[i, list(MID)[0]] < 10: base += 0.5 * rate_for(d, b, 5)
                        ef += base
                    elif i < ch["kept"]: n += 1
                tot += union_counts(I, kk).sum(); ex += union_counts(I, np.full((R, 48), 10)).sum(); K += ch["kept"]
        line += f" | {d}: x{tot/ex:.3f} flips~{100*ef/max(1,n):.2f}%"
    print(line)
