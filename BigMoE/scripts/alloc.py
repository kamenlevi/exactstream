"""Handshake elision by uneven residency. A layer holding all E experts never misses, so it
needs no GPU->CPU->GPU handshake (~0.13-0.16 ms). Per layer, measure exact-LRU stall stats vs
cache size C on the proxy traces, then choose per-layer C (incl. C=E: no handshake) to minimise
modelled stall+handshake time under a total budget. Proxy: 40 layers x 256; scaled to target
in the summary. -> data/alloc.json, data/alloc.txt"""
import json, os, sys
from collections import OrderedDict
import numpy as np
TR = os.path.expanduser("~/MoEPrefetch/traces"); NL, K, E = 40, 8, 256
SETS = ["chat", "code", "bulgarian", "longctx", "realcode"]
GRID = [64, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
HS, LAT1, PER = 0.145e-3, 0.37e-3, 2.765e-3 / 13.4      # s: handshake, first-miss latency, extra miss
S = int(os.environ.get("S", 1)); B = int(os.environ.get("B", 12))
zs = [dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in SETS]
def layer_stats(L, C):
    st = sl = n = 0
    for z in zs:
        act = z[f"L{L}_act"][:, :K]; P = z[f"L{L-1}_pmid1"][:, :B] if (B and L >= 1) else None
        cache = OrderedDict(); T = len(act)
        def touch(e):
            if e in cache: cache.move_to_end(e); return True
            cache[e] = None
            if len(cache) > C: cache.popitem(last=False)
            return False
        for t in range(200): [touch(e) for e in act[t].tolist()]
        for t0 in range(200, T - S + 1, S):
            if P is not None: [touch(e) for e in dict.fromkeys(P[t0:t0+S].reshape(-1).tolist())]
            m = sum(0 if touch(e) else 1 for e in dict.fromkeys(act[t0:t0+S].reshape(-1).tolist()))
            st += m; sl += m > 0; n += 1
    return st / n, sl / n
tab = {}
for L in range(NL):
    tab[L] = {C: layer_stats(L, C) for C in GRID if C < E}
    tab[L][E] = (0.0, 0.0)
def cost(L, C):   # seconds per pass for this layer
    if C == E: return 0.0
    st, sl = tab[L][C]
    return HS + sl * LAT1 + max(0, st - sl) * PER
lines = []
for frac in (0.5, 0.6, 0.7):
    budget = round(frac * E * NL)
    uni = [min(GRID, key=lambda c: abs(c - frac * E))] * NL
    # greedy marginal allocation: start every layer at the smallest C, spend budget on best cost drop per expert slot
    alloc = [GRID[0]] * NL; used = sum(alloc)
    while True:
        best = None
        for L in range(NL):
            i = GRID.index(alloc[L])
            for j in range(i + 1, len(GRID)):
                dc = cost(L, alloc[L]) - cost(L, GRID[j]); ds = GRID[j] - alloc[L]
                if used + ds <= budget and dc > 0 and (best is None or dc / ds > best[0]):
                    best = (dc / ds, L, GRID[j], ds)
        if not best: break
        _, L, c, ds = best; alloc[L] = c; used += ds
    cu = sum(cost(L, uni[L]) for L in range(NL)); ca = sum(cost(L, alloc[L]) for L in range(NL))
    full = [L for L in range(NL) if alloc[L] == E]
    lines.append(f"S={S} B={B} resident {frac:.0%}: uniform C={uni[0]} -> {cu*1e3:5.2f} ms/pass of handshake+stall (proxy 40 layers); "
                 f"optimised -> {ca*1e3:5.2f} ms/pass ({sum(alloc)} slots of {budget}); {len(full)} layers fully resident: {full}")
    lines.append(f"   alloc: {alloc}")
    print(lines[-2], flush=True)
open(os.path.expanduser(f"~/BigMoE/data/alloc_S{S}_B{B}.txt"), "w").write("\n".join(lines) + "\n")
