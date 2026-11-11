"""Threshold grid for the three-band table (top-2 at cf >= X, top-5 at [Y, X), top-8 at [Z, Y), exact below) with the
direct-flip bound computed at the grid's own edges from floor_rows (reached rows, uniform topk2 / topk5 arms)."""
import json, numpy as np, sys, itertools
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import union_counts
W = load(); doms = ["code", "ifeval", "multi"]
fr = json.load(open("/root/350-tps/FableGod/floor_rows.json")); F = {(r["nm"], r["j"]): r for r in fr}
# collect per realized row: dom, cf, flips topk2/topk5, and per check the row structure
rows = {d: [] for d in doms}
for w in W:
    d = w["dom"]; gen = w["gen"]
    for ch in checks_of(w):
        for i in range(ch["kept"]):
            row = int(ch["rows"][i]); c = ch["cf"][i]; rec = F.get((w["nm"], row - gen))
            if rec is None or not (c == c): continue
            rows[d].append((c, int(rec["flip"]["topk2"]), int(rec["flip"]["topk5"])))
R = {d: np.array(rows[d]) for d in doms}
def rate(d, lo, hi, arm):
    a = R[d]; m = (a[:, 0] >= lo) & (a[:, 0] < hi)
    return a[m, 1 + arm].mean() if m.sum() >= 20 else np.nan, int(m.sum())
EX = {}
def tiles(d, X, Y, Z):
    tot = 0; ex = 0
    for w in W:
        if w["dom"] != d: continue
        for ch in checks_of(w):
            Rn = ch["R"]; cf = ch["cf"]; I = w["ids"][ch["rows"]]
            kk = np.full((Rn, 48), 10, np.int64)
            for i in range(Rn):
                c = cf[i]
                if not (c == c): continue
                kk[i, :] = 2 if c >= X else 5 if c >= Y else 8 if c >= Z else 10
            tot += union_counts(I, kk).sum()
            key = (d, w["nm"], int(ch["rows"][0]))
            if key not in EX: EX[key] = union_counts(I, np.full((Rn, 48), 10)).sum()
            ex += EX[key]
    return tot / ex
print("X (top-2 above), Y (top-5 above), Z (top-8 above) -> tiles ratio | direct flips %  [code | ifeval | multi]  (n rows in the top-2 band / top-5 band on code)")
res = []
for X, Y, Z in itertools.product([0.98, 0.99, 0.995, 0.999], [0.8, 0.9, 0.95], [0.3, 0.5, 0.7]):
    if not (Z < Y < X): continue
    line = f"X {X:<6} Y {Y:<5} Z {Z:<4}"
    out = []
    for d in doms:
        a = R[d]; n = len(a)
        f2, n2 = rate(d, X, 1.01, 0); f5, n5 = rate(d, Y, X, 1)
        fl = (np.nan_to_num(f2) * n2 + np.nan_to_num(f5) * n5) / n * 100
        t = tiles(d, X, Y, Z)
        out.append((t, fl)); line += f" | {d}: x{t:.3f} {fl:.2f}%"
    res.append(((X, Y, Z), out)); print(line, flush=True)
json.dump([(k, v) for k, v in res], open("/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s/frontier.json", "w"))
