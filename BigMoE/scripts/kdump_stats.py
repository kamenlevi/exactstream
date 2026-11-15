#!/usr/bin/env python3
"""K-next dump summary (MoonShots' LINE in ~/FableGod/deepdraft/LEVERS_A_MOONSHOTS.md "K-next"): from kdump_<set>.jsonl,
- M1 = P(argmax == y | t30b's depth 0 wrong at the next check), the headroom; M0 = P(argmax == y | t30b right);
- y's containment in the branch row's top-1 / 2 / 3 / 5 / 8;
- by the margin bin of the target row (m_target) and of the branch row itself.
usage: kdump_stats.py [set ...] -> data/deepdraft/ceiling/kdump_stats.json"""
import json, os, sys
import numpy as np

H = os.path.expanduser
out = {}
for set_ in (sys.argv[1:] or ["heplus24", "mbppbank"]):
    p = H(f"~/BigMoE/data/deepdraft/ceiling/kdump_{set_}.jsonl")
    if not os.path.exists(p):
        continue
    R = [json.loads(l) for l in open(p) if l.strip()]
    eq = np.array([r["top8_ids"][0][0] == r["y"] for r in R])
    pos = np.array([next((i for i, x in enumerate(r["top8_ids"][0]) if x == r["y"]), 99) for r in R])
    nxt = np.array([r["t30b_next_d0_right"] for r in R], dtype=object)
    mt = np.array([r["m_target"] if r["m_target"] is not None else np.nan for r in R], dtype=float)
    mb = np.array([r["margin_row"] for r in R])
    wrong = np.array([x is False for x in nxt]); right = np.array([x is True for x in nxt])
    S = {"branches": len(R), "P_argmax_eq_y": float(eq.mean()),
         "M1_given_t30b_next_wrong": [float(eq[wrong].mean()) if wrong.any() else None, int(wrong.sum())],
         "M0_given_t30b_next_right": [float(eq[right].mean()) if right.any() else None, int(right.sum())],
         "y_in_topk": {k: float((pos < k).mean()) for k in (1, 2, 3, 5, 8)},
         "by_target_margin": {}, "by_branch_margin": {}}
    for nm, arr in (("by_target_margin", mt), ("by_branch_margin", mb)):
        for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, np.inf)):
            sel = (arr >= lo) & (arr < hi)
            if sel.any():
                S[nm][f"{lo}-{hi}"] = [float(eq[sel].mean()), int(sel.sum())]
    out[set_] = S
    print(set_, json.dumps(S))
json.dump(out, open(H("~/BigMoE/data/deepdraft/ceiling/kdump_stats.json"), "w"), indent=1)
