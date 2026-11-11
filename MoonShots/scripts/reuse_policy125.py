"""Learned-reuse replacement for the expert cache (CPU-cache idea: Mockingjay, Shah et al. HPCA 2022, predicts each line's
reuse distance and evicts the one whose predicted next use is farthest; Hawkeye, Jain & Lin ISCA 2016, learns OPT's
decisions from the past). Exactness-preserving (a cache policy never changes outputs). Trace: data/verify_trace125.npz,
per layer C slots, misses per pass in 4-bit MB, vs LRU and Belady OPT (scripts/opt_bound125.py).
Policies (all online, using only the past):
  rd-ema    predicted next use = last use + EMA of the expert's observed reuse distances (passes); evict the farthest
  rd-last   the same with the last observed reuse distance
  rd-mix    max(predicted next use, LRU age guard): rd-ema but an expert unseen for > 2x its predicted distance is stale
usage: reuse_policy125.py   (CPU, ~1 min, 1 core)"""
import numpy as np

z = np.load("/Users/kamen/MoonShots/data/verify_trace125.npz")
ids = z["ids"].astype(np.int32)
P, L = ids.shape[:2]
seqs = [[set(np.unique(ids[p, l]).tolist()) for p in range(P)] for l in range(L)]
MB = 2.765


def run(seq, C, mode, a=0.5):
    cache, last, rd, miss = set(), {}, {}, 0
    gm = 8.0                                             # prior reuse distance for never-reused experts (passes)
    for t, need in enumerate(seq):
        for e in need:
            if e in last:
                d = t - last[e]
                rd[e] = d if (mode == "rd-last" or e not in rd) else a * d + (1 - a) * rd[e]
            miss += e not in cache
            cache.add(e)
            last[e] = t
        if len(cache) > C:
            def pred(e):
                if mode == "lru":
                    return -last[e]
                nxt = last[e] + rd.get(e, gm)
                if mode == "rd-mix" and t - last[e] > 2 * rd.get(e, gm):
                    nxt = 10 ** 9                        # stale: overdue by 2x its usual gap
                return nxt
            cand = sorted((e for e in cache if e not in need), key=pred, reverse=True)
            for e in cand[:len(cache) - C]:
                cache.discard(e)
    return miss


for C in (64, 96, 128, 192):
    out = []
    for mode in ("lru", "rd-ema", "rd-last", "rd-mix"):
        m = sum(run(s, C, mode) for s in seqs) / P
        out.append(f"{mode} {m * MB:6.1f}")
    print(f"C={C:3d} ({48 * C * MB / 1e3:4.1f} GB) MB/pass  " + " | ".join(out), flush=True)
