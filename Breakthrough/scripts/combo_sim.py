"""How the levers multiply (Breakthrough, 2026-10-19). 125B routing traces (MoEPrefetch traces125, top-20 ids per
token/layer). A pass covers K consecutive tokens (K=1 eager, K=4 ~ a depth-3 MTP verify pass when all drafts are
accepted). Per pass and layer: ranks < r of every row are exact (fetched if not resident); a row's rank >= r uses its
expert if resident, else the best-ranked resident expert among that row's ranks 10-19 not already used by the row, else
the rank is dropped. Strict LRU capacity C per layer (experts used in the pass become MRU). r=10 = exact routing.
Cache sizes in GB depend on bits per weight: an expert is 4.9152M weights x b/8 bytes (4.5 bits today, 2.5 if the
bits direction works). Prints SSD MB per token = fetches/token x expert MB.
"""
import numpy as np
from collections import OrderedDict
T = "/Users/kamen/MoEPrefetch/data/traces125/"; SETS = ["code", "chat", "bulgarian", "math"]; NL = 48
trs = {s: np.stack([np.load(T + f"{s}_800.npz")[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for s in SETS}
def run(tr, C, r, K):
    fetch = subs = drops = 0
    for l in range(NL):
        c = OrderedDict()
        for a in range(0, tr.shape[0], K):
            used_all = []
            for t in range(a, min(a + K, tr.shape[0])):
                row = [int(x) for x in tr[t, l]]
                alt = [e for e in row[10:20] if e in c]
                used = []
                for rank, e in enumerate(row[:10]):
                    if rank < r or e in c or e in used_all:
                        if e not in c and e not in used_all: fetch += 1
                        used.append(e)
                    else:
                        x = next((y for y in alt if y not in used), None)
                        if x is None: drops += 1
                        else: subs += 1; used.append(x); alt.remove(x)
                used_all += [e for e in used if e not in used_all]
            for e in used_all:
                if e in c: c.move_to_end(e)
                else: c[e] = 1
            while len(c) > C: c.popitem(last=False)
    n = tr.shape[0]; return fetch / n, subs / n, drops / n
import sys
cases = [(c.split(":")[0], float(c.split(":")[1]), float(c.split(":")[2])) for c in sys.argv[1:]] or [  # (label, GB of expert cache, bits/weight)
    ("12 GB total today (5.4 GB, 4.5 b)", 5.4, 4.5), ("6.9 GB cache, 4.5 b", 6.9, 4.5),
    ("12 GB total + bits/dense diet (7.6 GB, 2.5 b)", 7.6, 2.5), ("6 GB total + diet (1.6 GB, 2.5 b)", 1.6, 2.5),
    ("17 GB cache, 4.5 b (today's 24 GB)", 17.0, 4.5)]
print("SSD MB per token (fetches/token) [substituted/dropped ranks per token], mean of 4 prompts")
for label, gb, b in cases:
    mb = 4.9152 * b / 8; C = int(gb * 1000 / mb / NL)
    print(f"\n{label}: expert {mb:.2f} MB, C={C} per layer ({100*C/512:.1f} %)")
    for K in (1, 4, 16):
        cells = []
        for r in (10, 3, 2, 1):
            f, s_, d = np.mean([run(trs[s], C, r, K) for s in SETS], 0)
            cells.append(f"r={r:<2d} {f*mb:5.0f} MB ({f:5.1f}) [{s_:4.0f}/{d:4.0f}]")
        print(f"  K={K:<3d} " + "   ".join(cells))
