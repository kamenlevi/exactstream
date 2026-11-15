#!/usr/bin/env python3
"""Post-miss depth-0 (Sun 15 Nov 03:2x; the orchestrator's GOAL A wall: t30b's depth 0 at the walk's check starts is 91.6 %
on HE+ vs 97.9 % elsewhere). From dd_ceiling's fresh-state cells (true trunk state at every anchor), depth-0 accuracy
conditioned on the previous position's depth-0 guess, and the 125B's own margin m at those positions.
usage: postmiss_d0.py [cells.npz ...] -> data/deepdraft/ceiling/postmiss_d0.json"""
import json, os, sys
import numpy as np
fs = sys.argv[1:] or [os.path.expanduser("~/BigMoE/data/deepdraft/ceiling/cells_t30b_heplus24.npz")]
out = {}
for f in fs:
    z = np.load(f)
    d0 = z["depth"] == 0
    key = {(int(a), int(r)): (bool(t), float(m)) for a, r, t, m in zip(z["fi"][d0], z["row"][d0], z["right"][d0], z["m"][d0])}
    am, ah = [], []
    for (a, r), v in key.items():
        p = key.get((a, r - 1))
        if p is not None:
            (ah if p[0] else am).append(v)
    R = {}
    for nm, v in (("after_miss", am), ("after_hit", ah)):
        acc = np.array([x[0] for x in v]); mm = np.array([x[1] for x in v])
        R[nm] = {"n": len(v), "acc": float(acc.mean()), "m_median": float(np.median(mm)), "near_tie_share_m_lt_0.5": float((mm < 0.5).mean())}
    R["overall_depth0_acc"] = float(z["right"][d0].mean())
    out[os.path.basename(f)] = R
    print(os.path.basename(f), json.dumps(R))
json.dump(out, open(os.path.expanduser("~/BigMoE/data/deepdraft/ceiling/postmiss_d0.json"), "w"), indent=1)
