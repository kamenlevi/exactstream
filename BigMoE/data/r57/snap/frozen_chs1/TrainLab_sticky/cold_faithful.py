"""CPU: how optimistic is the cold-only replay of turn18c128 / turn19 (budget125_tf.cold_observe, arms cC_rR)? There a
cold expert read at 2 bits enters the LRU as if fully resident: its later uses are free full-precision hits. A faithful
engine keeps it as a 2-bit copy in its slot: later uses at ranks >= R hit at 2 bits; a use at a rank < R upgrades it
(an MSB-3 read, + the LSB part if rank < 2, as the v3 reference). On Breakthrough's traces (first half warm, 4-token
passes over the second half): per token, the bytes and the 2-bit USES (row x layer uses computed at 2 bits) of both
models at C slots. "planar": the same faithful model with an upgrade reading only the missing third plane of a planar
records4 store (720,896 B, + the LSB part if rank < 2) instead of a full MSB-3 re-read. usage: cold_faithful.py [C] [R]"""
import glob, os, sys
from collections import OrderedDict
import numpy as np

C = int(sys.argv[1]) if len(sys.argv) > 1 else 128
R2 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
BASE, SIDE, Q2, P3 = 2162688, 770048, 1540096, 720896
T = os.path.expanduser("~/Breakthrough/data/expA/traces125/")
out = [f"cold_faithful C={C} r{R2}: optimistic (turn18c128's replay) vs faithful (2-bit copies stay 2-bit), per token"]
for f in sorted(glob.glob(T + "*_800_g.npz")):
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(48)], 1)[:, :, :10].astype(np.int64)
    n, h = ids.shape[0], ids.shape[0] // 2
    res = {}
    for model in ("optimistic", "faithful", "planar"):
        by, v3b, uses2, tok = 0.0, 0.0, 0, 0
        for L in range(48):
            lru = OrderedDict()                                  # expert -> bits of its resident copy (2 or 3)
            for row in ids[:h, L]:                               # warm: full reads in order
                for e in row:
                    e = int(e)
                    if e in lru:
                        lru.move_to_end(e)
                    else:
                        lru[e] = 3
                        if len(lru) > C:
                            lru.popitem(last=False)
            for t in range(h, n - 3, 4):
                rows = ids[t:t + 4, L]
                best = {}
                for row in rows:
                    for j, e in enumerate(row):
                        best[int(e)] = min(best.get(int(e), 99), j)
                bits = {}
                for e, k in best.items():
                    full = BASE + (SIDE if k < 2 else 0)
                    if e not in lru:
                        v3b += full
                        if k >= R2:
                            by += Q2; bits[e] = 2
                        else:
                            by += full; bits[e] = 3
                    elif model != "optimistic" and lru[e] == 2:
                        if k >= R2:
                            bits[e] = 2                          # hit on the 2-bit copy
                        else:                                    # upgrade read: all of MSB-3, or only the third plane
                            by += (full if model == "faithful" else P3 + (SIDE if k < 2 else 0)); bits[e] = 3
                    else:
                        bits[e] = 3
                for row in rows:                                 # 2-bit uses (row x layer)
                    for e in row:
                        uses2 += bits[int(e)] == 2
                for e in sorted(best, key=lambda e: (e not in lru, best[e])):
                    if e in lru:
                        lru.move_to_end(e)
                        if bits[e] == 3:
                            lru[e] = 3
                    else:
                        lru[e] = bits[e] if model != "optimistic" else 3
                        if len(lru) > C:
                            lru.popitem(last=False)
                if L == 0:
                    tok += 4
        res[model] = (v3b / tok / 1e6, by / tok / 1e6, uses2 / tok)
    name = os.path.basename(f).split("_")[0]
    v3 = res["optimistic"][0]
    out.append(f"  {name:9s} v3 {v3:6.1f} MB | " + " | ".join(
        f"{m} {res[m][1]:6.1f} MB ({1 - res[m][1] / v3:.1%}), 2-bit uses {res[m][2] / 288:.0%} of rank 4-9" for m in res))
print("\n".join(out))
open(os.path.expanduser(f"~/TrainLab/data/cold_faithful_C{C}_r{R2}.txt"), "w").write("\n".join(out) + "\n")
