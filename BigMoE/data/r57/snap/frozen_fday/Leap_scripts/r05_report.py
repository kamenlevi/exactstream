"""R-05 report (Leap; CPU only): W / U tokens per pass from the per-sample replay JSONs (refine19.py R19_JSON) of the two
arms, block 32 1-step (primary, the '32-row eval') and block 16 1-step, with 90 % paired bootstrap CIs over the 24
samples. Lines (QUEUE R-05): PASS W / U >= 1.05 (then the 50-min run), KILL < 1.02. Writes ~/Leap/data/r05_report.txt"""
import json, os
import numpy as np

D = os.path.expanduser("~/Leap/data")
per = {a: {m: np.array(v, float) for m, v in json.load(open(f"{D}/r05_{a}.json"))["per"].items()} for a in ("U", "W")}
rng = np.random.default_rng(0)
lines = []
for m in ("32:1step", "16:1step"):
    u, w = per["U"][m], per["W"][m]
    T = lambda x, i=slice(None): x[i, 0].sum() / x[i, 1].sum()
    r = T(w) / T(u)
    bs = [T(w, i) / T(u, i) for i in (rng.integers(0, len(u), len(u)) for _ in range(4000))]
    lo, hi = np.percentile(bs, [5, 95])
    lines.append(f"  {m}: U {T(u):.3f}, W {T(w):.3f} tokens per pass; W / U {r:.4f} (90 % CI {lo:.3f}-{hi:.3f})")
    if m == "32:1step":
        v = "PASS" if r >= 1.05 else "KILL" if r < 1.02 else "BETWEEN"
lines.append(f"R-05 verdict (PASS W / U >= 1.05 at block 32, KILL < 1.02): {v}")
print("\n".join(lines))
open(f"{D}/r05_report.txt", "w").write("\n".join(lines) + "\n")
