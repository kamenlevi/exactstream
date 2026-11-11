"""Is the 2+1+1 record format (records4: P2 + B3 + B4 parts) worth it over records3 ([MSB-3 | LSB-1]) for plan 4433332222?
Same model as planes3_sim.py (MoEPrefetch's 125B traces *_800.npz, verify windows of 4 tokens with stride 3, warm-up 200
tokens, per layer one LRU over cache units within a byte budget, demand MB/pass = bytes of the units missing when a window
starts; deterministic, no prefetch), with the unit sizes of the concrete formats:
  whole     4-bit expert 2.765 MB (records/)
  records3  M3 = MSB-3 part 2.150 MB (w3 + s3 + b3), L1 = LSB part 0.768 MB (l1 + b)
  records4  P2 = 1.536 MB (2-bit codes + s2 + b2), B3 = 0.768 MB (bit + b3), B4 = 0.768 MB (bit + b)
usage: planes4_sim.py   (CPU, 1 core, a few minutes)"""
import glob, os
from collections import OrderedDict
import numpy as np

E, K, WARM = 512, 10, 200
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
TR = []
for f in files:
    z = np.load(f)
    TR.append([z[f"L{L}"][:, :K].astype(np.int64) for L in range(len(z.files))])
NL = len(TR[0])
FMT = {
    "whole": ({"W": 2.765}, {4: ("W",), 3: ("W",), 2: ("W",)}),
    "records3": ({"M3": 2.150, "L1": 0.768}, {4: ("M3", "L1"), 3: ("M3",), 2: ("M3",)}),
    "records4": ({"P2": 1.536, "B3": 0.768, "B4": 0.768}, {4: ("P2", "B3", "B4"), 3: ("P2", "B3"), 2: ("P2",)}),
}


def run(fmt, plan, budget_mb):
    size, need_of = FMT[fmt]
    dem = miss = P = 0
    for tr in TR:
        T = tr[0].shape[0]
        starts = list(range(0, T - 4, 3))
        P += sum(1 for s in starts if s >= WARM)
        for L in range(NL):
            cache, used = OrderedDict(), 0.0
            for s in starts:
                w = tr[L][s:s + 4]
                need = set()
                for r in range(K):
                    for e in np.unique(w[:, r]):
                        for u in need_of[int(plan[r])]:
                            need.add((int(e), u))
                for u in sorted(need):
                    if u in cache:
                        cache.move_to_end(u)
                for u in sorted(need):
                    if u not in cache:
                        if s >= WARM:
                            dem += size[u[1]]; miss += 1
                        cache[u] = size[u[1]]; used += size[u[1]]
                        while used > budget_mb and cache:
                            _, sz = cache.popitem(last=False); used -= sz
    return dem / P, miss / P


import sys
EXTRA = len(sys.argv) > 1 and sys.argv[1] == "extra"
MENU = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[1] == "menu" else None
print(f"125B traces: {len(TR)} prompts x {TR[0][0].shape[0]} tokens, {NL} layers; demand MB/pass (misses/pass)", flush=True)
for gb in (10, 13, 17, 24):
    b = gb * 1024 / NL
    if MENU:
        cells = [("records3 4433333333", run("records3", "4433333333", b))] + [(f"records4 {pl}", run("records4", pl, b)) for pl in MENU]
        print(f"expert budget {gb:2d} GB: " + " | ".join(f"{n} {d:6.1f} ({m:5.1f})" for n, (d, m) in cells), flush=True)
        continue
    cells = [("whole 4444444444", run("whole", "4444444444", b)),
             ("records3 4433333333", run("records3", "4433333333", b)),
             ("records3 4433332222", run("records3", "4433332222", b)),
             ("records4 4433333333", run("records4", "4433333333", b)),
             ("records4 4433332222", run("records4", "4433332222", b)),
             ("records4 4444222222", run("records4", "4444222222", b))] if not EXTRA else \
            [("records4 4422222222", run("records4", "4422222222", b)), ("records4 4222222222", run("records4", "4222222222", b)),
             ("records4 2222222222", run("records4", "2222222222", b))]
    print(f"expert budget {gb:2d} GB: " + " | ".join(f"{n} {d:6.1f} ({m:5.1f})" for n, (d, m) in cells), flush=True)
