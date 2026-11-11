"""How much of Belady's OPT gain needs only short-horizon knowledge? (125B, data/verify_trace125.npz, per layer C slots)
Policy LA(W): on eviction, protect every expert the next W passes will use (known exactly: the bound of a perfect
W-pass route predictor), evict the least recently used of the rest (LRU fallback; if all are protected, evict the one
used farthest in the future). W = 0 is plain LRU; W = inf is OPT. Misses per pass in 4-bit MB.
usage: lookahead_bound125.py   (CPU, ~1 min, 1 core)"""
import numpy as np

z = np.load("/Users/kamen/MoonShots/data/verify_trace125.npz")
ids = z["ids"].astype(np.int32)
P, L = ids.shape[:2]
seqs = [[set(np.unique(ids[p, l]).tolist()) for p in range(P)] for l in range(L)]
MB = 2.765


def next_use(seq):
    """nu[t][e]: the next pass index > t that uses e (for e used or cached at t), as a dict per t, built backwards."""
    nxt_after = [None] * len(seq)
    last = {}
    for t in range(len(seq) - 1, -1, -1):
        nxt_after[t] = dict(last)                       # next use strictly after t
        for e in seq[t]:
            last[e] = t
    return nxt_after


def la(seq, C, W, nu):
    cache, miss = {}, 0                                 # e -> last use
    for t, need in enumerate(seq):
        for e in need:
            miss += e not in cache
            cache[e] = t
        if len(cache) > C:
            after = nu[t]
            def key(e):
                n = after.get(e, 10 ** 9)
                prot = (n - t) <= W                     # used within the next W passes
                # unprotected first (evict LRU among them); protected last (evict farthest next use among them)
                return (prot, cache[e] if not prot else -n)
            cand = sorted((e for e in cache if e not in need), key=key)
            for e in cand[:len(cache) - C]:
                del cache[e]
    return miss


NU = [next_use(s) for s in seqs]
for C in (64, 96, 128, 192):
    row = []
    for W in (0, 1, 2, 4, 8, 16, 10 ** 6):
        m = sum(la(s, C, W, nu) for s, nu in zip(seqs, NU)) / P
        row.append(f"W={'inf' if W > 1000 else W}: {m * MB:6.1f}")
    print(f"C={C:3d} ({48 * C * MB / 1e3:4.1f} GB) MB/pass  " + " | ".join(row), flush=True)
