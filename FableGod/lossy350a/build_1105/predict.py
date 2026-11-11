"""Pre-registered predictions for T1's arms from the map's own measured uniform neighbours (Fable A, Thu 5 Nov 2026).
CPU, seconds, read-only: decision_map/out/{t4_all,t3_all,top_k8,topk5,qom_0_live_ball+code}.json by row class.
Law used [K on three uniform arms]: non-tie flips are about linear in the routed block's relative error ENERGY E:
  t3_all (every expert at MSB-3, E = 1.00 x MSB-3 = 0.066) minus t4_all (the same path at 4 bits, its numerics floor);
  top_k8 (ranks 8-9 dropped, E = 0.032 from the measured rank energies); topk5 (ranks 5-9 dropped, E = 0.141).
An xs arm's E is its measured whole-expert error in MSB-3 units (lossy350a/arms_plain.out, held-out routed pairs):
  x48 0.17, x40 0.56, x32 1.38 (inputs only, no deflation: what the patch runs).
Low = the quantisation-noise slope (t3_all - t4_all); high = the dropped-rank slope (top_k8)."""
import json, os
D = os.path.expanduser("~/FableGod/decision_map/out")
fr = json.load(open(os.path.expanduser("~/FableGod/floor_rows.json")))
CLS = {(r["nm"], int(r["i"])): r["cls"] for r in fr}; DOM = {r["nm"]: r["dom"] for r in fr}
base = json.load(open(os.path.join(D, "baseline.json"))); DOMS = ("code", "ifeval", "multi")
def read(arm):
    A = json.load(open(os.path.join(D, arm + ".json"))); acc = {}
    for w, v in A.items():
        if w not in base or w not in DOM or "tk" not in v or len(v["tk"]) != len(base[w]["tk"]): continue
        a0 = int(base[w]["a"])
        for j, (ta, tb) in enumerate(zip(v["tk"], base[w]["tk"])):
            c = CLS.get((w, a0 + j))
            if c is None: continue
            for key in ("all", "tie" if c == "tie" else "nontie"):
                x = acc.setdefault((DOM[w], key), [0, 0]); x[0] += int(ta[0] != tb[0]); x[1] += 1
    return {k: (100.0 * v[0] / v[1], v[1]) for k, v in acc.items()}
t4, t3, k8, k5, tw = (read(a) for a in ("t4_all", "t3_all", "top_k8", "topk5", "qom_0_live_ball+code"))
MSB3, E8, E5 = 0.0657, 0.032, 0.141
E = {48: 0.17, 40: 0.56, 32: 1.38}
print("measured neighbours, non-tie flips % (code / IFEval / multi) and their slope per unit of relative error energy:")
for nm, r, e in (("t3_all - t4_all", None, MSB3), ("top_k8", k8, E8), ("topk5", k5, E5)):
    v = [(t3[(d, "nontie")][0] - t4[(d, "nontie")][0]) if r is None else r[(d, "nontie")][0] for d in DOMS]
    print(f"  {nm:16s} E {e:.3f}: " + " / ".join(f"{x:.2f}" for x in v) + "   slope " + " / ".join(f"{x / e:.1f}" for x in v))
lo = {d: (t3[(d, "nontie")][0] - t4[(d, "nontie")][0]) for d in DOMS}                 # per 1.0 x MSB-3
hi = {d: k8[(d, "nontie")][0] / E8 * MSB3 for d in DOMS}
tie = {d: t3[(d, "tie")][1] / t3[(d, "all")][1] for d in DOMS}
print(f"tie rows' share of rows: " + " / ".join(f"{100 * tie[d]:.1f} %" for d in DOMS) +
      "; they flip " + " / ".join(f"{t4[(d, 'tie')][0]:.0f}-{t3[(d, 'tie')][0]:.0f} %" for d in DOMS) + " under t4_all .. t3_all")
print("\n| arm | error x MSB-3 | predicted NON-TIE flips %, code / IFEval / multi (low - high) | standalone all-row incl. the tie floor |")
print("|---|---|---|---|")
for c in (48, 40, 32):
    nt = {d: (E[c] * lo[d], E[c] * hi[d]) for d in DOMS}
    ar = {d: ((1 - tie[d]) * nt[d][0] + tie[d] * t4[(d, "tie")][0], (1 - tie[d]) * nt[d][1] + tie[d] * t3[(d, "tie")][0]) for d in DOMS}
    print(f"| xs{c} | {E[c]:.2f} | " + " / ".join(f"{nt[d][0]:.2f}-{nt[d][1]:.2f}" for d in DOMS) + " | " + " / ".join(f"{ar[d][0]:.1f}-{ar[d][1]:.1f}" for d in DOMS) + " |")
print(f"\nbars (non-tie): 0.87 / 2.22 / 1.63; today's twin on code {tw[('code', 'nontie')][0]:.2f} (all-row {tw[('code', 'all')][0]:.2f}); QOM-BQ on chat 1.19 / 0.91 [K fablec/WALL_1105.md]")
print(f"room for an added non-tie cost: {0.87 - tw[('code', 'nontie')][0]:.2f} / {2.22 - 1.19:.2f} / {1.63 - 0.91:.2f}")
