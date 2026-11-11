"""Wait-top-r streaming on the 125B traces (Breakthrough, 2026-10-19). MoEPrefetch's 'wait top-2' (35B: GSM8K-150
identical at C=64 and C=48): ranks < r are always exact (loaded on a miss = SSD bytes), ranks >= r use the expert if it
is resident, else a substitute: the best-ranked resident expert among the router's ranks 10-19 not already chosen,
else the rank is dropped. Used experts become MRU (LRU per layer, strict capacity C). Ids only (traces have no gates).
Prints per token: SSD loads (x 2.77 MB at 4-bit), substituted ranks, dropped ranks.
"""
import numpy as np
from collections import OrderedDict
T = "/Users/kamen/MoEPrefetch/data/traces125/"; SETS = ["code", "chat", "bulgarian", "math"]; NL = 48
trs = {s: np.stack([np.load(T + f"{s}_800.npz")[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for s in SETS}
def run(tr, C, r):
    loads = subs = drops = 0
    for l in range(NL):
        c = OrderedDict()
        for t in range(tr.shape[0]):
            row = [int(x) for x in tr[t, l]]
            alt = [e for e in row[10:20] if e in c]
            used = []
            for rank, e in enumerate(row[:10]):
                if rank < r or e in c:
                    if e not in c: loads += 1
                    used.append(e)
                else:
                    a = next((x for x in alt if x not in used), None)
                    if a is None: drops += 1
                    else: subs += 1; used.append(a); alt.remove(a)
            for e in used:
                if e in c: c.move_to_end(e)
                else: c[e] = 1
            while len(c) > C: c.popitem(last=False)
    n = tr.shape[0]; return loads / n, subs / n, drops / n
print("per token, mean of 4 sets: SSD loads (MB at 2.77 MB/expert) / substituted ranks / dropped ranks (of 480 uses)")
print("C (res%)        " + "  ".join(f"r={r:<22d}" for r in (1, 2, 3, 4, 10)))
for C in (8, 13, 26, 51, 128):
    cells = []
    for r in (1, 2, 3, 4, 10):
        v = np.mean([run(trs[s], C, r) for s in SETS], 0)
        cells.append(f"{v[0]:5.1f} ({v[0]*2.77:4.0f}MB) {v[1]:5.1f} {v[2]:4.1f}")
    print(f"C={C:<3d} ({100*C/512:4.1f}%) " + "  ".join(cells))
