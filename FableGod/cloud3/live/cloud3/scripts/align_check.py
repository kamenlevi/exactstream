import json, numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load(); fr = json.load(open("/root/350-tps/FableGod/floor_rows.json")); F = {(r["nm"], r["j"]): r for r in fr}
for shift in [-2, -1, 0, 1, 2]:
    match = 0; tot = 0; p1m = 0
    for w in W:
        gen = w["gen"]
        for ch in checks_of(w):
            for i in range(ch["R"]):
                row = int(ch["rows"][i]); c = ch["cf"][i]
                rec = F.get((w["nm"], row - (gen - 1) + shift))
                if rec is None or rec["cf"] is None or not (c == c): continue
                tot += 1; match += abs(rec["cf"] - c) < 1e-4
                p1 = w["p1"][row]
                if p1 == p1: p1m += abs(rec["p1"] - p1) < 1e-3
    print(f"shift {shift:+d}: cf matches {match}/{tot} = {match/max(1,tot):.3f}; p1 matches {p1m/max(1,tot):.3f}")
# also: floor_rows j range vs gen
w = W[0]; js = sorted(j for (nm, j) in F if nm == w["nm"]); print(w["nm"], "gen", w["gen"], "ids rows", w["ids"].shape[0], "floor j range", js[0], js[-1], "n", len(js))
