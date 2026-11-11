"""Scan-resistant insertion policies for the expert cache (CPU caches: LIP/BIP, Qureshi et al. ISCA 2007; SRRIP, Jaleel
et al. ISCA 2010; 2Q, Johnson & Shasha 1994). OPT's advantage over LRU here is keeping frequently reused experts and
dropping one-off experts at once; these policies try to get that online. Exact (a policy never changes outputs).
Trace data/verify_trace125.npz, per layer C slots, 4-bit MB per pass.
  lru        baseline
  lip-k      a newly inserted expert starts k passes "old" (evicted first unless reused soon); a hit makes it MRU
  srrip-n    n-bit re-reference prediction values: insert at 2^n - 2, hit -> 0, evict the max (aging all until one is max)
  2q-f       admit to a probation FIFO of f*C slots; promote to the protected LRU on a reuse
usage: insert_policy125.py   (CPU, ~1 min, 1 core)"""
import numpy as np

z = np.load("/Users/kamen/MoonShots/data/verify_trace125.npz")
ids = z["ids"].astype(np.int32)
P, L = ids.shape[:2]
seqs = [[set(np.unique(ids[p, l]).tolist()) for p in range(P)] for l in range(L)]
MB = 2.765


def lip(seq, C, k):
    age, miss = {}, 0                                    # e -> effective last-use time
    for t, need in enumerate(seq):
        for e in need:
            if e in age:
                age[e] = t
            else:
                miss += 1
                age[e] = t - k
        if len(age) > C:
            cand = sorted((e for e in age if e not in need), key=lambda e: age[e])
            # newly inserted experts of this pass are protected for this pass only if C allows
            for e in cand[:len(age) - C]:
                del age[e]
            if len(age) > C:                             # all candidates gone: drop the oldest of this pass's new ones
                for e in sorted(age, key=lambda e: age[e])[:len(age) - C]:
                    del age[e]
    return miss


def srrip(seq, C, n):
    M = (1 << n) - 1
    rr, miss = {}, 0
    for t, need in enumerate(seq):
        for e in need:
            if e in rr:
                rr[e] = 0
            else:
                miss += 1
                rr[e] = M - 1
        while len(rr) > C:
            cand = [e for e in rr if e not in need]
            if not cand:
                break
            mx_ = max(rr[e] for e in cand)
            if mx_ < M:
                for e in rr:
                    if e not in need:
                        rr[e] += M - mx_
            victim = next(e for e in cand if rr[e] >= M)
            del rr[victim]
    return miss


def twoq(seq, C, f):
    A = max(1, int(f * C))                               # probation FIFO size
    prob, prot, miss = {}, {}, 0                         # e -> insertion time / last use
    for t, need in enumerate(seq):
        for e in need:
            if e in prot:
                prot[e] = t
            elif e in prob:
                del prob[e]
                prot[e] = t
            else:
                miss += 1
                prob[e] = t
        while len(prob) + len(prot) > C:
            pc = [e for e in prob if e not in need]
            if len(prob) > A and pc:
                del prob[min(pc, key=lambda e: prob[e])]
                continue
            qc = [e for e in prot if e not in need]
            if qc:
                del prot[min(qc, key=lambda e: prot[e])]
            elif pc:
                del prob[min(pc, key=lambda e: prob[e])]
            else:
                break
    return miss


for C in (64, 96, 128, 192):
    out = [f"lru {sum(lip(s, C, 0) for s in seqs) / P * MB:6.1f}"]
    for k in (1, 2, 4):
        out.append(f"lip-{k} {sum(lip(s, C, k) for s in seqs) / P * MB:6.1f}")
    for n in (2, 3):
        out.append(f"srrip-{n} {sum(srrip(s, C, n) for s in seqs) / P * MB:6.1f}")
    for f in (0.25, 0.5):
        out.append(f"2q-{f} {sum(twoq(s, C, f) for s in seqs) / P * MB:6.1f}")
    print(f"C={C:3d} ({48 * C * MB / 1e3:4.1f} GB) MB/pass  " + " | ".join(out), flush=True)
