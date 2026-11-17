"""BREAK98 (MoonShots, Tue 17 Nov 2026 07:4x): the ORACLE bound for any within-session state retrieval. For every generated
position t and depth d (anchor a = t - d), the best earlier verified state chosen with hindsight: max over s <= a of
cos(wd[s], wd[t]) (prompt rows included; wd = the head's true input state, 10,240 wide, BigMoE's bank captures). Also
the direction error (angle, radians: R2 0.83 ~ 0.34 on the bank proxy's scale) of that best state, and the mean of the
anchor's own state for scale. If even the hindsight-best earlier state stays far from the true one, no retrieval key
(n-gram, learned, kNN) can deliver the drafted position's state from the session.
usage: ~/BigMoE/.venv/bin/python -I -B svp_oracle.py   (CPU, seconds)"""
import glob
import json
import os

import numpy as np

BANDS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 30)]
SETS = {"heplus24": sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/tf_code_HumanEval_*.npz"))),
        "mbppbank": sorted(glob.glob(os.path.expanduser("~/datasets/deepdraft/bank/dlog__mbpp*.npz")))}
RES, L_ = {}, []
for sname, files in SETS.items():
    acc = {b: [] for b in BANDS}
    accA = {b: [] for b in BANDS}
    for f in files:
        z = np.load(f)
        wd = np.asarray(z["wd"], np.float32)
        g = int(z["gen"])
        u = wd / (np.linalg.norm(wd, axis=1, keepdims=True) + 1e-9)
        C = u @ u.T                                       # [L, L] cosines
        M = np.maximum.accumulate(C, axis=1)             # M[t, a] = max_{s <= a} C[t, s]
        L = wd.shape[0]
        for t in range(g, L):
            for b in BANDS:
                for d in range(b[0], b[1] + 1):
                    a = t - d
                    if a < 0:
                        continue
                    acc[b].append(M[t, a])
                    accA[b].append(C[t, a])
    RES[sname] = {}
    line = f"=== {sname}: hindsight-best earlier state (max cos over s <= anchor) vs the true state, per depth band"
    print(line); L_.append(line)
    for b in BANDS:
        x, xa = np.array(acc[b]), np.array(accA[b])
        ang, anga = np.arccos(np.clip(x, -1, 1)), np.arccos(np.clip(xa, -1, 1))
        row = dict(n=int(len(x)), cos_mean=float(x.mean()), cos_p10=float(np.percentile(x, 10)), cos_p50=float(np.median(x)),
                   cos_p90=float(np.percentile(x, 90)), dir_err_mean=float(ang.mean()), share_dir_le_040=float((ang <= 0.40).mean()),
                   share_dir_le_055=float((ang <= 0.55).mean()), anchor_cos_mean=float(xa.mean()), anchor_dir_mean=float(anga.mean()))
        RES[sname][f"{b[0]}-{b[1]}"] = row
        line = (f"  d {b[0]:2d}-{b[1]:2d}: n {row['n']:6d} | best cos mean {row['cos_mean']:.3f} (p10 {row['cos_p10']:.3f}, "
                f"p50 {row['cos_p50']:.3f}, p90 {row['cos_p90']:.3f}) | direction error mean {row['dir_err_mean']:.3f}; share "
                f"<= 0.40 {row['share_dir_le_040']:.3f}, <= 0.55 {row['share_dir_le_055']:.3f} | anchor's own state: cos "
                f"{row['anchor_cos_mean']:.3f}, dir {row['anchor_dir_mean']:.3f}")
        print(line); L_.append(line)
out = os.path.expanduser("~/MoonShots/week98/break98/svp_oracle")
open(out + ".txt", "w").write("\n".join(L_) + "\n")
json.dump(RES, open(out + ".json", "w"), indent=1)
