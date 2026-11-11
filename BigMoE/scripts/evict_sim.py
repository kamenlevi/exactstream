"""Eviction-rule replay on recorded v3 traces (NewIdeas I9): per gated (pass, layer) the unique expert ids
(BIGMOE_ORACLE_REC, key = seq*1024 + L), replayed per layer through a C-slot cache under LRU (the engine's rule),
LFU with exponential decay per pass, and Belady OPT (the bound). Misses = demand reads the gate waits on (prefetch is not
modelled: the same for every rule to first order). Base layers (L < side0) use C_m slots, sidecar layers C_l.
usage: evict_sim.py TRACE.bin [C_m] [C_l] [side0]   (CPU only)"""
import sys, heapq
from collections import OrderedDict, defaultdict
import numpy as np

path = sys.argv[1]
C_m = int(sys.argv[2]) if len(sys.argv) > 2 else 147
C_l = int(sys.argv[3]) if len(sys.argv) > 3 else 46
SIDE0 = int(sys.argv[4]) if len(sys.argv) > 4 else 49
raw = open(path, "rb").read()
rows, o = [], 0
while o < len(raw):
    k, n = np.frombuffer(raw, np.int64, 2, o); o += 16
    rows.append((int(k), np.frombuffer(raw, np.int32, int(n), o).tolist())); o += 4 * int(n)
per = defaultdict(list)                                   # layer -> [(seq, ids)] in pass order
for k, ids in sorted(rows):
    per[k % 1024].append((k // 1024, ids))


def lru(seq, C):
    c, miss = OrderedDict(), 0
    for _, ids in seq:
        for e in ids:
            if e in c:
                c.move_to_end(e)
            else:
                miss += 1
                if len(c) >= C:
                    for v in c:                           # evict the LRU expert not needed by this pass
                        if v not in ids:
                            del c[v]; break
                c[e] = 1
        for e in ids:
            c.move_to_end(e)
    return miss


def lfu(seq, C, decay):
    score, res, miss = defaultdict(float), set(), 0
    for t, (_, ids) in enumerate(seq):
        w = decay ** -t                                   # decayed counts without touching every entry
        for e in ids:
            score[e] += w
        for e in ids:
            if e not in res:
                miss += 1
                if len(res) >= C:
                    v = min((x for x in res if x not in ids), key=lambda x: score[x])
                    res.discard(v)
                res.add(e)
    return miss


def opt(seq, C):
    nxt = []                                              # for each (pass, id): the next pass that uses id
    last = {}
    for t in range(len(seq) - 1, -1, -1):
        d = {e: last.get(e, 1 << 30) for e in seq[t][1]}
        nxt.append(d)
        for e in seq[t][1]:
            last[e] = t
    nxt.reverse()
    res, nu, miss = set(), {}, 0
    for t, (_, ids) in enumerate(seq):
        for e in ids:
            if e not in res:
                miss += 1
                if len(res) >= C:
                    v = max((x for x in res if x not in ids), key=lambda x: nu.get(x, 1 << 30))
                    res.discard(v)
                res.add(e)
        for e in ids:
            nu[e] = nxt[t][e]
    return miss


tot = defaultdict(int)
for L, seq in per.items():
    C = C_l if L >= SIDE0 else C_m
    tot["uses"] += sum(len(ids) for _, ids in seq)
    tot["LRU"] += lru(seq, C)
    for d in (0.99, 0.97, 0.9):
        tot[f"LFU{d}"] += lfu(seq, C, d)
    tot["OPT"] += opt(seq, C)
passes = len({k // 1024 for k, _ in rows})
print(f"{path}: {passes} passes, {len(per)} layers, {tot['uses']} uses (C_m {C_m}, C_l {C_l})")
for k in ("LRU", "LFU0.99", "LFU0.97", "LFU0.9", "OPT"):
    print(f"  {k:8s} misses {tot[k]:7d}  {tot[k] / passes:6.1f}/pass  {100 * (tot[k] / tot['LRU'] - 1):+6.1f} % vs LRU")
