"""Intra-pass sharing bias (NewIdeas, CPU): 4 consecutive decode tokens of the 125B expA traces stand in for a 4-row verify
pass. Per layer, rows choose their top-10 in order; with bias delta, experts that are resident OR already chosen by an
earlier row of the same pass get +delta on their log-prob (ranks 0-1 pinned). Reads per pass = distinct cold experts among
all rows' choices (one read serves every row that uses it); reported per token, plus swapped true mass per row-layer."""
import glob, numpy as np
from collections import OrderedDict
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
D = [np.load(f) for f in fs]
NL, R = 48, 4
def run(C, delta, share):
    reads = ntok = 0; swapped = 0.0; nrl = 0
    for d in D:
        ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)
        gp = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
        T = (ids.shape[0] // R) * R; ntok += T
        for l in range(NL):
            c = OrderedDict()
            for p0 in range(0, T, R):
                chosen = set()
                for t in range(p0, p0 + R):
                    e20, lg = ids[t, l], np.log(np.maximum(gp[t, l], 1e-12))
                    b = lg.copy()
                    if delta:
                        favour = np.array([(int(e) in c) or (share and int(e) in chosen) for e in e20])
                        b += delta * favour
                        b[:2] = 1e9
                    top = np.argsort(-b)[:10]
                    swapped += (gp[t, l][:10].sum() - gp[t, l][top].sum()) / gp[t, l][:10].sum(); nrl += 1
                    chosen |= set(int(e20[j]) for j in top)
                cold = [e for e in chosen if e not in c]
                reads += len(cold)
                for e in chosen:
                    if e in c: c.move_to_end(e)
                    else:
                        c[e] = 1
                        if len(c) > C: c.popitem(last=False)
    return reads / ntok, swapped / nrl
for C in (51, 97, 128):
    out = []
    for delta, share in ((0.0, False), (0.25, False), (0.25, True), (0.5, False), (0.5, True)):
        r, sw = run(C, delta, share)
        out.append(f"d{delta}{'+share' if share else ''}: {r:6.1f} ({sw:.3f})")
    print(f"C={C:3d}: reads/token in 4-row passes (swapped true mass per row-layer)  " + "  ".join(out), flush=True)
