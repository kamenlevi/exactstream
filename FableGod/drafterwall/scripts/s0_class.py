#!/usr/bin/env python3
"""s0_class.py (drafterwall, exploratory, CPU): per-class, per-depth teacher-forced accuracy of DFlash2 L1-20 and the MTP
head b5 on the tree3_map tables (every answer position an anchor; depth j predicts the token at A + 1 + j).
Class of the target row by the 125B's own top-3 probabilities there: sure p1 >= 0.99, tie p1 - p2 <= 0.1, mid otherwise.
Teacher-forced = conditional on a right prefix, i.e. the rate a chain sees at that depth."""
import numpy as np, glob, os
TD = os.path.expanduser("~/TrainLab/data/tree3_map")
BANDS = [(0, 0), (1, 2), (3, 6), (7, 10), (11, 14)]
def cls_of(p3):
    p1, p2 = p3[..., 0], p3[..., 1]
    c = np.full(p1.shape, 1, dtype=np.int8)           # 1 = mid
    c[p1 >= 0.99] = 0                                  # 0 = sure
    c[(p1 - p2) <= 0.1] = 2                            # 2 = tie
    return c
for dom in ("code", "ifeval", "multi"):
    T = {k: [] for k in ("true", "d1", "d3", "m1", "m3", "cls", "depth", "valid", "p1", "dp", "mp")}
    for f in sorted(glob.glob(f"{TD}/{dom}/*.npz")):
        z = np.load(f)
        true = z["true"]; n = true.shape[0]
        valid = (true >= 0) & (z["t125_t3"][..., 0] >= 0) & (z["mtp_rank"] >= 0)
        valid &= (z["t125_t3"][..., 0] == true)        # rows where the map's own top-1 is the window token
        T["true"].append(true[valid]); T["valid"].append(valid.sum())
        T["d1"].append((z["dfl_t3"][..., 0] == true)[valid]); T["d3"].append((z["dfl_t3"] == true[..., None]).any(-1)[valid])
        T["m1"].append((z["mtp_t3"][..., 0] == true)[valid]); T["m3"].append((z["mtp_t3"] == true[..., None]).any(-1)[valid])
        T["cls"].append(cls_of(z["t125_p3"])[valid]); T["depth"].append(np.broadcast_to(np.arange(15), true.shape)[valid])
        T["p1"].append(z["t125_p3"][..., 0][valid]); T["dp"].append(z["dfl_p3"][..., 0][valid]); T["mp"].append(z["mtp_p3"][..., 0][valid])
    A = {k: np.concatenate(v) for k, v in T.items() if k != "valid"}
    N = len(A["true"])
    print(f"\n== {dom}: {N} (anchor, depth) cells; class shares sure / mid / tie: " + " / ".join(f"{100 * (A['cls'] == c).mean():.1f}" for c in (0, 1, 2)))
    print("| class | depth | n | DFlash2 top-1 | top-3 | MTP top-1 | top-3 | either top-1 | both wrong |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c, nm in ((0, "sure"), (1, "mid"), (2, "tie"), (-1, "all")):
        for lo, hi in BANDS + [(0, 14)]:
            s = (A["depth"] >= lo) & (A["depth"] <= hi) & ((A["cls"] == c) if c >= 0 else True)
            if s.sum() == 0: continue
            print(f"| {nm} | {lo}-{hi} | {s.sum()} | {100 * A['d1'][s].mean():.1f} | {100 * A['d3'][s].mean():.1f} | {100 * A['m1'][s].mean():.1f} | {100 * A['m3'][s].mean():.1f} | {100 * (A['d1'][s] | A['m1'][s]).mean():.1f} | {100 * (~A['d1'][s] & ~A['m1'][s]).mean():.1f} |")
