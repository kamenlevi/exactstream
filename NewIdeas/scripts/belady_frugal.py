"""Cache-policy headroom for frugal's waited ranks (NewIdeas 2026-10-22, CPU, 125B expA traces, 4 prompts, decode).
Per layer, a cache of C experts; per token the ranks 0-1 must be present (a miss = one waited read, then inserted);
ranks 2-9 are computed only if resident. Policies: LRU (hits at any rank refresh), LRU01 (only rank 0-1 uses refresh),
LFU-decay (score x0.97 per token, +1 per rank 0-1 use, +0.3 per lower-rank use), OPT (Belady on the rank 0-1 request
stream: evict the expert whose next rank 0-1 use is farthest; the bound for any demand-admission policy)."""
import glob
import numpy as np
from collections import OrderedDict

NL = 48
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
TR = []
for f in fs:
    d = np.load(f)
    L = np.stack([d[f"L{l}"][:, :10] for l in range(NL)], 1).astype(np.int64)
    G = np.stack([d[f"G{l}"][:, :10] for l in range(NL)], 1).astype(np.float64)
    TR.append((L, G / G.sum(-1, keepdims=True)))
ntok = sum(L.shape[0] for L, W in TR)


def run(C, pol):
    waited = 0; me = 0.0
    for L, W in TR:
        T = L.shape[0]
        for l in range(NL):
            ids = L[:, l, :]
            if pol == "opt":
                nxt = {}; NEXT = np.full((T, 2), 10 ** 9, np.int64)       # next rank 0-1 use time of each request
                for t in range(T - 1, -1, -1):
                    for r in range(2):
                        e = int(ids[t, r]); NEXT[t, r] = nxt.get(e, 10 ** 9)
                    for r in range(2): nxt[int(ids[t, r])] = t
                cache = {}                                                 # expert -> next rank 0-1 use
            elif pol == "lfu":
                cache = {}; sc = {}
            else:
                cache = OrderedDict()
            for t in range(T):
                if pol == "lfu":
                    for e in sc: sc[e] *= 0.97
                for r in range(10):
                    e = int(ids[t, r])
                    if e in cache:
                        if pol == "lru" or (pol == "lru01" and r < 2): cache.move_to_end(e)
                        if pol == "opt" and r < 2: cache[e] = NEXT[t, r]
                        if pol == "lfu": sc[e] = sc.get(e, 0) + (1.0 if r < 2 else 0.3)
                        continue
                    if r >= 2:
                        me += W[t, l, r] ** 2; continue
                    waited += 1
                    if pol == "opt":
                        if len(cache) >= C:
                            v = max(cache, key=cache.get)
                            if cache[v] < NEXT[t, r]:                    # bypass: the new one is needed last
                                continue
                            del cache[v]
                        cache[e] = NEXT[t, r]
                    elif pol == "lfu":
                        if len(cache) >= C:
                            v = min(cache, key=lambda x: sc.get(x, 0)); del cache[v]
                        cache[e] = 1; sc[e] = sc.get(e, 0) + 1.0
                    else:
                        cache[e] = 1
                        if len(cache) > C: cache.popitem(last=False)
    return waited / ntok, me / ntok / NL


print(f"{len(TR)} prompts, {ntok} decode tokens")
for C in (51, 26):
    for pol in ("lru", "lru01", "lfu", "opt"):
        w, e = run(C, pol)
        print(f"  C={C} {pol:5s}: waited {w:5.2f}/token  missed E(ranks 2-9) {e:.4f}/layer", flush=True)
