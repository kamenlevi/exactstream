"""D-13 DIVERGE read-out (MTP Head, Thu 10-29; REPORT 'D-13'): kept per 16-row check of each arm's D-02b walk on code (24)
and math (60), and the ratios S / U and V / U with a 90 % bootstrap CI over samples (2,000 resamples, seed 13; the same
resampled samples for both arms). Inputs data/d02b_walk_d13{U,S,V}_{code,math}.npz. Output data/d13_ratio.txt.
Usage: taskpolicy -b nice -n 10 [BigMoE venv] python sticky/d13_ratio.py"""
import os
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
rng = np.random.default_rng(13)
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def per_sample(Z):
    return np.array([(sum(int(k) + 1 for k, m, a in Z[f"{i}_checks"]), len(Z[f"{i}_checks"])) for i in range(len(Z["names"]))],
                    np.float64)

say("D-13 DIVERGE: kept per 16-row check (D-02b walk, 1 step); ratio vs U with a 90 % bootstrap CI over samples")
for dom in ("code", "math"):
    W = {a: np.load(os.path.join(TD, f"d02b_walk_d13{a}_{dom}.npz")) for a in "USV"
         if os.path.exists(os.path.join(TD, f"d02b_walk_d13{a}_{dom}.npz"))}
    if "U" not in W:
        say(f"  {dom}: no U walk"); continue
    U = per_sample(W["U"])
    say(f"  {dom}: U {U[:, 0].sum() / U[:, 1].sum():.2f}" + "".join(
        f", {a} {per_sample(W[a])[:, 0].sum() / per_sample(W[a])[:, 1].sum():.2f}" for a in "SV" if a in W))
    for a in "SV":
        if a not in W:
            continue
        X = per_sample(W[a])
        assert [str(x) for x in W[a]["names"]] == [str(x) for x in W["U"]["names"]]
        r = (X[:, 0].sum() / X[:, 1].sum()) / (U[:, 0].sum() / U[:, 1].sum())
        b = []
        for _ in range(2000):
            i = rng.integers(0, len(U), len(U))
            b.append((X[i, 0].sum() / X[i, 1].sum()) / (U[i, 0].sum() / U[i, 1].sum()))
        lo, hi = np.percentile(b, [5, 95])
        say(f"    {a} / U = {r:.3f} (90 % CI {lo:.3f}-{hi:.3f})")
say("Lines: PASS S / U or V / U >= 1.03 on code or math with the CI's low end > 1.00; KILL both <= 1.01 on both; else BETWEEN.")
open(os.path.join(TD, "d13_ratio.txt"), "w").write("\n".join(lines) + "\n")
