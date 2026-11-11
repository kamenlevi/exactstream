#!/usr/bin/env python3
"""a5_refresh.py (drafterwall, DESCRIPTIVE; no line). How fresh must the trunk state be? The MTP head b5's chain restarted
from an exact trunk state every k positions inside a 32-row check (unseen d23h tables): the draft at offset i of a check
uses the chain anchored at offset k * floor(i / k), depth i mod k. k = 1 is 'lag 2 everywhere', k = 31 is today's chain.
Also the byte-floor arithmetic [P] with BREAK_THE_WALL's measured union bytes."""
import numpy as np, os
TD = os.path.expanduser("~/TrainLab/data"); out = []
def say(s=""):
    print(s); out.append(s)
for st in ("code", "mbpp"):
    z = np.load(f"{TD}/d23h/{st}_b5.npz"); res = {}
    for k in (1, 2, 3, 4, 6, 8, 12, 16, 31):
        kept = []
        for i in range(len(z["keys"])):
            rt = z[f"rt_{i}"]; r0 = int(z["gen"][i]) - 1 - int(z["a0"][i]); sub = rt[r0:]; m = sub.shape[0]
            p = 0
            while p < m:
                j = 0
                while j < 31 and p + j < m:
                    a = p + (j // k) * k; d = j % k
                    if a >= m or sub[a, d] != 1: break
                    j += 1
                kept.append(j + 1); p += j + 1
        res[k] = np.mean(kept)
    say(f"{st}: kept per 32-row check when the chain restarts from an exact trunk state every k positions: " + ", ".join(f"k={k}: {v:.2f}" for k, v in res.items()))
say("\n[P] byte floor: tok/s = kept x 284 GB/s / (2.7 GB trunk + union bytes of the check); union bytes at 16 / 32 / 64 rows from BREAK_THE_WALL section 3:")
U = {"exact": (8.366, 12.556, 18.119), "banded gate (lossy)": (4.894, 7.777, 11.850), "oracle gate (lossy)": (3.359, 5.367, 8.285)}
K = {"today, HumanEval (DFlash2 16 rows 10.5; MTP 32 rows 12.6)": (10.5, 12.6, None),
     "lag-2 ceiling, MTP alone, HumanEval+": (12.62, 20.05, 27.36), "lag-2 ceiling, perfect pick of two readers, HumanEval+": (13.98, 24.43, 37.28),
     "lag-2 ceiling, MTP alone, MBPP": (9.40, 12.12, 13.52), "lag-2 ceiling, perfect pick of two readers, MBPP": (11.46, 16.78, 20.88)}
for kn, kv in K.items():
    for un, uv in U.items():
        say(f"  {kn} | {un}: " + " / ".join("-" if k is None else f"{k * 284 / (2.7 + u):.0f}" for k, u in zip(kv, uv)) + " tok/s at 16 / 32 / 64 rows")
open(os.path.join(os.path.dirname(__file__), "..", "out", "a5_refresh.txt"), "w").write("\n".join(out) + "\n")
