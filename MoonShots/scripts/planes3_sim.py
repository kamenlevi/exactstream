"""Round 19 PoC A (CPU): three nested bit-planes per expert as separate cache units, on the real 125B traces.
Units: P2 = MSB-2 plane (2-bit codes + scale + bias, 2.5 bits/weight = 1.54 MB), B3 = third bit (1 bit + uint8 offset per
group, 0.69 MB), B4 = last bit (0.69 MB); whole 4-bit expert = 2.77 MB. A plan gives the bits each gate rank needs, e.g.
4433322222 = ranks 0-1 exact (P2+B3+B4), 2-4 MSB-3 (P2+B3), 5-9 MSB-2 (P2). Verify windows of 4 tokens, stride 3; per layer
an LRU over units within a byte budget; demand MB/pass = bytes of units missing when a window starts (no prefetch, no
fallback: deterministic). Quality of plans: scripts/expert_tiers35b.py (35B).   usage: planes3_sim.py"""
import glob, os
from collections import OrderedDict
import numpy as np
E, K, WARM = 512, 10, 200
SIZE = {"W": 2.77, "P2": 1.54, "B3": 0.69, "B4": 0.69}
NEED = {4: ("P2", "B3", "B4"), 3: ("P2", "B3"), 2: ("P2",)}
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
TR = []
for f in files:
    z = np.load(f)
    TR.append([z[f"L{L}"][:, :K].astype(np.int64) for L in range(len(z.files))])
NL = len(TR[0])


def run(plan, budget_mb):
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
                        if plan == "whole":
                            need.add((int(e), "W"))
                        else:
                            for u in NEED[int(plan[r])]:
                                need.add((int(e), u))
                late = [u for u in sorted(need) if u not in cache]
                for u in sorted(need):
                    if u in cache:
                        cache.move_to_end(u)
                for u in late:
                    if s >= WARM:
                        dem += SIZE[u[1]]; miss += 1
                    cache[u] = SIZE[u[1]]; used += SIZE[u[1]]
                    while used > budget_mb and cache:
                        _, sz = cache.popitem(last=False); used -= sz
    return dem / P, miss / P


print(f"125B traces: {len(TR)} prompts, {NL} layers, verify windows of 4 (stride 3); demand MB/pass (all layers)")
BYTES = {4: 4.75, 3: 3.625, 2: 2.5}
for gb in (20, 24, 28, 34):
    b = gb * 1024 / NL
    d, m = run("whole", b)
    print(f"budget {gb} GB: whole-expert LRU {d:6.1f} MB/pass ({m:5.1f} misses)", flush=True)
    for plan in ("4444222222", "4433322222", "4433333333", "4443322222", "4333322222", "4433222222"):
        d, m = run(plan, b)
        rel = sum(BYTES[int(c)] for c in plan) / (4.5 * K)
        print(f"   plan {plan} (read {100 * (rel - 1):+.0f} % bytes per use set): demand {d:6.1f} MB/pass ({m:5.1f} unit misses)",
              flush=True)
