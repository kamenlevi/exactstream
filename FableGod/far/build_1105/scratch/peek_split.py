import json, os, sys
import numpy as np
DMO = os.path.expanduser("~/FableGod/decision_map/out")
base = json.load(open(os.path.join(DMO, "baseline.json")))
TD = os.path.expanduser("~/TrainLab/data")
Z = np.load(os.path.join(TD, "d02b_walk_hpd0_code.npz"))
names = [str(x) for x in Z["names"]]
for arm in ("ni/qom_0_split0.99_live_ball", "ni/qom_0_live_ball"):
    n_bad = 0
    for nm in sorted(base):
        kp = np.load(os.path.join(DMO, arm, nm))["keep"]; a = base[nm]["a"]
        rows = np.flatnonzero(~kp[:a].all(axis=(1, 2)))
        if len(rows):
            n_bad += 1
            if n_bad <= 3:
                i = names.index(nm) if nm in names else None
                ck = Z[f"{i}_checks"][:3].tolist() if i is not None else None
                print(arm, nm, "a", a, "N", kp.shape[0], "cut prompt rows", rows.tolist()[:8], "kept per row-layer there", kp[rows].sum() / (len(rows) * 48), "first checks (k, m, A)", ck)
    print(arm, "windows with a cut row < a:", n_bad)
