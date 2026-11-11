"""Two caching ideas on the real 125B traces (verify windows of 4 tokens, stride 3; per-layer LRU slot pools; CPU only).
1. Stall-time-aware slot allocation (cost-aware caching, Cao & Irani 1997, with the measured concave SSD curve: a layer with m
   misses stalls t(m) = 0.45/0.71/0.99/1.17 ms for m = 1..4, +0.25 ms per extra). The same total slots, split per layer by greedy
   marginal allocation, either for fewest misses (MoEPrefetch's DP objective) or for least stall time.
2. Draft-informed eviction (Belady's MIN 1966 x speculative decoding): LRU, but a victim that the NEXT verify window needs at
   this layer is skipped. Oracle next window = upper bound for predictors (MTP drafts, prev-pass routing).
Reports misses and stall ms per pass.   usage: stall_alloc_sim.py [FRAC]"""
import glob, os, sys
from collections import OrderedDict
import numpy as np
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
E, K, WARM = 512, 10, 200
tcost = lambda m: 0.0 if m == 0 else [0.45, 0.71, 0.99, 1.17][m - 1] if m <= 4 else 1.17 + 0.25 * (m - 4)
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
TR = []
for f in files:
    z = np.load(f); NL = len(z.files)
    tr = [z[f"L{L}"][:, :K].astype(np.int64) for L in range(NL)]
    wins = [[sorted(set(int(e) for e in tr[L][s:s + 4].ravel())) for s in range(0, tr[0].shape[0] - 4, 3)] for L in range(NL)]
    TR.append(wins)
NL = len(TR[0])


def layer_run(wins_by_prompt, C, belady=False):
    """one layer's LRU over all prompts: (misses list per window, stall ms list) after warm-up."""
    ms, st = [], []
    for wins in wins_by_prompt:
        cache = OrderedDict()
        for i, u in enumerate(wins):
            nxt = set(wins[i + 1]) if (belady and i + 1 < len(wins)) else set()
            miss = [e for e in u if e not in cache]
            for e in u:
                if e in cache: cache.move_to_end(e)
            for e in miss:
                cache[e] = 1
                while len(cache) > C:
                    victim = next((v for v in cache if v not in nxt and v not in u), None) if belady else None
                    if victim is None: cache.popitem(last=False)
                    else: del cache[victim]
            if i * 3 >= WARM:
                ms.append(len(miss)); st.append(tcost(len(miss)))
    return np.mean(ms), np.mean(st)


C0 = int(FRAC * E)
grid = sorted({max(16, int(C0 * f)) for f in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5)} | {C0})
curves = {}
for L in range(NL):
    for C in grid:
        curves[(L, C)] = layer_run([TR[p][L] for p in range(len(TR))], C)
uni_m = sum(curves[(L, C0)][0] for L in range(NL)); uni_s = sum(curves[(L, C0)][1] for L in range(NL))


def allocate(obj):
    """greedy: start every layer at the smallest grid budget, then repeatedly buy the step with the best gain per slot."""
    cur = {L: grid[0] for L in range(NL)}; left = C0 * NL - grid[0] * NL
    while True:
        best = None
        for L in range(NL):
            i = grid.index(cur[L])
            if i + 1 < len(grid):
                d = grid[i + 1] - cur[L]
                if d <= left:
                    gain = (curves[(L, cur[L])][obj] - curves[(L, grid[i + 1])][obj]) / d
                    if best is None or gain > best[0]: best = (gain, L, grid[i + 1], d)
        if best is None: break
        _, L, c, d = best; cur[L] = c; left -= d
    return cur


print(f"125B traces: {len(TR)} prompts, {NL} layers, C={C0}/512 per layer ({FRAC:.0%}), budget grid {grid}")
print(f"  uniform LRU           : misses/pass {uni_m:6.1f}  stall {uni_s:6.2f} ms/pass")
for name, obj in (("alloc for misses", 0), ("alloc for stall ms", 1)):
    a = allocate(obj)
    m = sum(curves[(L, a[L])][0] for L in range(NL)); s = sum(curves[(L, a[L])][1] for L in range(NL))
    print(f"  {name:22s}: misses/pass {m:6.1f}  stall {s:6.2f} ms/pass  (slots {sum(a.values())} vs {C0 * NL})", flush=True)
bm = bs = 0
for L in range(NL):
    m, s = layer_run([TR[p][L] for p in range(len(TR))], C0, belady=True); bm += m; bs += s
print(f"  LRU + next-window oracle eviction (Belady-informed): misses/pass {bm:6.1f}  stall {bs:6.2f} ms/pass", flush=True)
