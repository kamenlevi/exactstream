"""First-principles bound on SSD traffic for the 125B (data/verify_trace125.npz: 401 committed MTP verify passes x 48
layers x 4 rows x top-10). Per layer, a cache of C expert slots; compare LRU (today's policy) with Belady's OPT (evict
the expert whose next use is farthest: the floor for ANY replacement / perfect-prediction prefetch policy) at the same C.
Also: distinct experts per pass vs rows (how verify width amortizes expert loads).
usage: opt_bound125.py   (CPU, seconds)"""
import numpy as np

z = np.load("/Users/kamen/MoonShots/data/verify_trace125.npz")
ids = z["ids"].astype(np.int32)
P, L = ids.shape[:2]
print("distinct experts per layer per pass by rows used:",
      {r: round(float(np.mean([len(np.unique(ids[p, l, :r])) for p in range(P) for l in range(0, L, 4)])), 2) for r in (1, 2, 3, 4)})
seqs = [[np.unique(ids[p, l]) for p in range(P)] for l in range(L)]   # per layer: the set each pass needs


def lru(seq, C):
    cache, miss = {}, 0
    for t, need in enumerate(seq):
        for e in need:
            miss += e not in cache
            cache[e] = t
        if len(cache) > C:
            for e, _ in sorted(cache.items(), key=lambda kv: kv[1])[:len(cache) - C]:
                del cache[e]
    return miss


def opt(seq, C):
    nxt, last = [], {}
    for t in range(len(seq) - 1, -1, -1):
        d = {}
        for e in seq[t]:
            d[e] = last.get(e, 10 ** 9)
            last[e] = t
        nxt.append(d)
    nxt = nxt[::-1]
    cache, miss = {}, 0                                  # e -> its next use
    for t, need in enumerate(seq):
        for e in need:
            miss += e not in cache
            cache[e] = nxt[t][e]
        if len(cache) > C:
            prot = set(need.tolist())
            cand = sorted(((v, e) for e, v in cache.items() if e not in prot), reverse=True)
            for v, e in cand[:len(cache) - C]:
                del cache[e]
    return miss


MB = 2.765                                               # one 4-bit expert
for C in (16, 32, 64, 96, 128, 192, 256):
    ml = sum(lru(s, C) for s in seqs) / P
    mo = sum(opt(s, C) for s in seqs) / P
    print(f"C={C:3d} ({48 * C * MB / 1e3:4.1f} GB): misses/pass LRU {ml:6.1f} ({ml * MB:6.1f} MB) | OPT {mo:6.1f} "
          f"({mo * MB:6.1f} MB) | OPT/LRU {mo / ml:.2f}", flush=True)
