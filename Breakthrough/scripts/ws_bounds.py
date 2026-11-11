"""Bounds for the budget (Breakthrough, 2026-10-19). Same 125B traces as ws_sim.py (top-10 of 20 ids per token/layer).
(A) Belady (MIN) misses per token at C slots per layer: the best any exact cache policy can do (no prefetch limits).
(B) Chunked processing: tokens handled K at a time (one pass per chunk, union of the chunk's experts per layer),
    LRU over chunk unions with C slots per layer (MRU = experts of the latest chunk). Misses per token = loads / tokens.
    K=1 is eager decode; K=4 ~ a depth-3 MTP verify pass; K>=16 needs a drafter that is accepted that far.
"""
import numpy as np
from collections import OrderedDict
T = "/Users/kamen/MoEPrefetch/data/traces125/"
SETS = ["code", "chat", "bulgarian", "math"]; NL = 48
def load(s):
    d = np.load(T + f"{s}_800.npz")
    return np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32)[:, :, :10]
trs = {s: load(s) for s in SETS}

def belady(seq_steps, C):
    # seq_steps: list of sets (experts needed at each step). MIN over steps: on a miss with a full cache,
    # evict the cached expert (not needed now) whose next use is farthest.
    n = len(seq_steps); nxt = {}; next_use = [None] * n
    for i in range(n - 1, -1, -1):
        next_use[i] = {e: nxt.get(e, 10**9) for e in seq_steps[i]}
        for e in seq_steps[i]: nxt[e] = i
    # next use AFTER step i for expert e: computed lazily with per-expert position lists
    pos = {}
    for i, st in enumerate(seq_steps):
        for e in st: pos.setdefault(e, []).append(i)
    ptr = {e: 0 for e in pos}
    cache = set(); miss = 0
    for i, st in enumerate(seq_steps):
        for e in st: ptr[e] += 1  # advance past current use
        for e in st:
            if e in cache: continue
            miss += 1
            if len(cache) >= C:
                cand = [x for x in cache if x not in st]
                def nu(x):
                    p = pos[x]; k = ptr[x]
                    return p[k] if k < len(p) else 10**9
                v = max(cand, key=nu); cache.remove(v)
            cache.add(e)
    return miss

def chunk_lru(tr, C, K):
    # strict capacity C per layer: a chunk's union streams through; afterwards the cache keeps the C most recently
    # used experts, where a chunk's experts count as used in order of their frequency within the chunk (most frequent last).
    loads = 0
    for l in range(NL):
        c = OrderedDict()
        for a in range(0, tr.shape[0], K):
            u, cnt = np.unique(tr[a:a + K, l], return_counts=True)
            for e in u[np.argsort(cnt, kind="stable")]:
                e = int(e)
                if e in c: c.move_to_end(e)
                else:
                    loads += 1; c[e] = 1
            while len(c) > C: c.popitem(last=False)
    return loads / tr.shape[0]

print("(A) misses per token (of 480 uses), mean over 4 sets: LRU vs Belady")
for C in (13, 26, 51, 128):
    lr = np.mean([chunk_lru(trs[s], C, 1) for s in SETS])
    bl = np.mean([sum(belady([set(map(int, trs[s][t, l])) for t in range(trs[s].shape[0])], C) for l in range(NL)) / trs[s].shape[0] for s in SETS])
    print(f"C={C:3d} ({100*C/512:4.1f}%): LRU {lr:6.1f}  Belady {bl:6.1f}  ({lr/bl:.2f}x)")
print("\n(B) chunked passes: expert loads per token (mean over sets); rows = chunk K, cols = C slots per layer")
Cs = (13, 26, 51, 128)
print("K      " + "  ".join(f"C={C:<5d}" for C in Cs))
for K in (1, 4, 16, 64, 256):
    print(f"{K:<6d} " + "  ".join(f"{np.mean([chunk_lru(trs[s], C, K) for s in SETS]):7.1f}" for C in Cs))
