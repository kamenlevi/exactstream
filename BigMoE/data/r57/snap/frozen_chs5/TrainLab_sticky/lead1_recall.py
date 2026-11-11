"""NEXTCHECK companion (MTP Head, Tue 11-03, CPU): the within-check look-ahead on the same capture, for comparison with the
next-check predictor. Lead 1 = layer L + 1's router on layer L's MoE input (R-57's forecast, which BigMoE's pre-route on
the partial sum tries to beat tonight): per-row recall@10 / @16 of L + 1's true top-10, and the share of check p's misses at
L + 1 (LRU replay at C 256, as nextcheck_pred.py) that the union of the rows' top-k forecasts names, with how many reads
that union costs. Output: data/nextcheck_pred.txt (appended)."""
import os, sys
sys.argv = sys.argv[:1]
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nextcheck_pred as N                                    # its loaders (it re-runs its own screen on import)
lines = []
def say(s=""):
    print(s, flush=True); lines.append(s)
C = 256
r10 = r16 = nrow = 0
cov = {k: [0, 0, 0] for k in (10, 16)}                       # misses named, misses, reads
for name, P in N.prompts:
    M = N.lru_misses(P, C)
    for t, (s, rec) in enumerate(P):
        if t < 3:
            continue
        for L in range(1, N.NL - 1):
            x, _ = rec[L]
            ids1 = rec[L + 1][1]
            sc = x @ N.RT[L + 1].T
            o = np.argsort(-sc, axis=-1)
            for a, b in zip(o, ids1):
                tb = set(b.tolist()); r10 += len(tb & set(a[:10].tolist())); r16 += len(tb & set(a[:16].tolist())); nrow += 1
            miss = set(M[t][L + 1].tolist())
            served = set(N.SERV[(s, L + 1)].tolist())
            for k in cov:
                u = set(o[:, :k].ravel().tolist())
                cov[k][0] += len(u & miss); cov[k][1] += len(miss); cov[k][2] += len(u - (served - miss))   # reads = forecast ids not resident
say(f"  lead 1 (L+1's router on L's MoE input, C {C} LRU replay): per-row recall@10 {r10 / (10 * nrow):.3f}, @16 {r16 / (10 * nrow):.3f} "
    f"of the true top-10 ({nrow} rows)")
for k, (a, b, c) in cov.items():
    say(f"    union of the rows' top-{k}: names {a / max(b, 1):.3f} of the check's misses at L+1, {c / max(b, 1):.2f} reads per miss")
open(os.path.join(N.TD, "nextcheck_pred.txt"), "a").write("\n".join(lines) + "\n\n")
