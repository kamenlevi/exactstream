"""How pessimistic is the all-row flip metric for cuts on unsure rows? Join floor_rows.json (per-position flips under the map's
uniform arms, teacher-forced) with the live-cut checks: a position is REACHED if it is one of the check's kept rows
(rows A..A+kept-1), WASTED if after the first miss (A+kept..A+L). Flip rates by (reached / wasted) x (sure / unsure by cf)."""
import json, numpy as np, sys, collections
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
W = load(); fr = json.load(open("/root/350-tps/FableGod/floor_rows.json"))
F = {(r["nm"], r["j"]): r for r in fr}
# map rows: floor_rows 'j' = position in the window (answer rows); our rows index the ids array [T-1]; the walk's row j predicts token j+1.
# floor_rows' j counts answer rows from 0 at the first answer position; our check rows are absolute row indices with gen = first answer token index.
arms = ["topk2", "topk5", "moe_reuse_prev_L12_35", "moe_reuse_prev_L08_11", "shared_zero"]
for dom in ["code", "ifeval", "multi"]:
    cnt = collections.defaultdict(lambda: [0, 0])   # key (arm, reached, sureclass) -> [flips, rows]
    matched = 0; unmatched = 0
    for w in W:
        if w["dom"] != dom: continue
        nm = w["nm"]; gen = w["gen"]
        for ch in checks_of(w):
            for i, row in enumerate(ch["rows"]):
                j = int(row) - (gen - 1)              # answer-row index: row gen-1 predicts the first answer token
                rec = F.get((nm, j))
                if rec is None: unmatched += 1; continue
                matched += 1
                reached = i < ch["kept"]
                cf = ch["cf"][i]; sure = (cf == cf) and cf >= 0.5
                for a in arms:
                    c = cnt[(a, reached, sure)]; c[0] += int(rec["flip"][a]); c[1] += 1
    print(f"\n== {dom}: matched rows {matched}, unmatched {unmatched}")
    print(f"{'arm':24s} {'reached sure':>14s} {'reached unsure':>15s} {'wasted sure':>13s} {'wasted unsure':>14s} | {'all rows':>9s} {'reached only':>13s}")
    for a in arms:
        vals = {}
        for reached in [True, False]:
            for sure in [True, False]:
                f, n = cnt[(a, reached, sure)]; vals[(reached, sure)] = (f, n)
        allf = sum(v[0] for v in vals.values()); alln = sum(v[1] for v in vals.values())
        rf = vals[(True, True)][0] + vals[(True, False)][0]; rn = vals[(True, True)][1] + vals[(True, False)][1]
        fmt = lambda k: f"{100*vals[k][0]/max(1,vals[k][1]):6.2f}% ({vals[k][1]:4d})"
        print(f"{a:24s} {fmt((True,True)):>14s} {fmt((True,False)):>15s} {fmt((False,True)):>13s} {fmt((False,False)):>14s} | {100*allf/alln:8.2f}% {100*rf/rn:12.2f}%")
