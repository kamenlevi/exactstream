"""Zero-shot switch caps on the 125B expA traces (NewIdeas 21:55, CPU): per layer, a token may take at most M experts that
were not in the previous token's selected set; extra new ones (lowest-ranked first) are swapped for the best-ranked
experts of the previous set among the router's top-20. Reports, at C=51 (10 %) and C=26 (5 %): LRU hit rate of the
selected experts, cold reads/token for frugal coverage (ranks 0-1 waited, others resident-only) and for full coverage,
and the swap gate energy (displaced w^2 + added w^2, top-10 renormalised) per layer."""
import glob, numpy as np
from collections import OrderedDict
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
D = [np.load(f) for f in fs]
NL, K = 48, 10
def run(C, M):
    hits = uses = ntok = 0; fr_reads = full_reads = 0; en = 0.0; nl = 0
    for d in D:
        ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)
        gp = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
        T = ids.shape[0]; ntok += T
        for l in range(NL):
            cf, cx = OrderedDict(), OrderedDict()          # frugal cache, full-coverage cache
            prev = set()
            for t in range(T):
                e20, p20 = ids[t, l], gp[t, l]
                w = p20 / p20[:K].sum()
                sel = list(range(K))                        # positions in the top-20 list
                if prev and M < K:
                    new = [j for j in sel if int(e20[j]) not in prev]
                    if len(new) > M:
                        drop = new[M:]                      # keep the M best-ranked new ones
                        cand = [j for j in range(20) if int(e20[j]) in prev and j not in sel]
                        add = cand[:len(drop)]
                        sel = [j for j in sel if j not in drop[:len(add)]] + add
                        dd = drop[:len(add)]
                        en += (w[dd] ** 2).sum() + (w[add] ** 2).sum()
                nl += 1
                prev = set(int(e20[j]) for j in sel)
                order = sorted(sel, key=lambda j: -p20[j])
                for r, j in enumerate(order):
                    e = int(e20[j]); uses += 1
                    if e in cx: cx.move_to_end(e); hits += 1
                    else:
                        full_reads += 1; cx[e] = 1
                        if len(cx) > C: cx.popitem(last=False)
                    if e in cf: cf.move_to_end(e)
                    elif r < 2:
                        fr_reads += 1; cf[e] = 1
                        if len(cf) > C: cf.popitem(last=False)
    return hits / uses, fr_reads / ntok, full_reads / ntok, en / nl
for C in (51, 26):
    for M in (10, 6, 4, 2):
        h, fr, fu, e = run(C, M)
        print(f"C={C} cap M={M:2d}: LRU hit {h:.3f}  frugal reads/token {fr:5.1f}  full-coverage reads {fu:6.1f}  swap energy/layer {e:.4f}", flush=True)
