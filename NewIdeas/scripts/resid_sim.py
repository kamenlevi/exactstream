"""Reads per token vs residency x near-tie re-routing on the 125B expA traces (top-20 ids + probs, 2,926 decode tokens),
exact coverage of the (possibly re-routed) top-10 or of the top-8, ranks 0-1 never displaced (NewIdeas, CPU)."""
import glob, numpy as np
from collections import OrderedDict
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
D = [np.load(f) for f in fs]
NL = 48
def run(C, delta, KC=10):
    reads = ntok = 0
    for d in D:
        ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)
        gp = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
        T = ids.shape[0]; ntok += T
        for l in range(NL):
            c = OrderedDict()
            for t in range(T):
                e20, lg = ids[t, l], np.log(np.maximum(gp[t, l], 1e-12))
                b = lg.copy()
                if delta:
                    b += delta * np.array([int(e) in c for e in e20])
                    b[:2] = 1e9                                   # ranks 0-1 never displaced (trace is score-ordered)
                top = np.argsort(-b)[:10]
                top = top[np.argsort(-lg[top])][:KC]
                for j in top:
                    e = int(e20[j])
                    if e in c: c.move_to_end(e)
                    else:
                        reads += 1; c[e] = 1
                        if len(c) > C: c.popitem(last=False)
    return reads / ntok
for C in (51, 77, 97, 128):
    row = [f"{run(C, dl, KC):6.1f}" for KC in (10, 8) for dl in (0.0, 0.25, 0.5)]
    print(f"C={C:3d} ({C / 512:4.1%}): reads/token  top-10: d0 {row[0]} d.25 {row[1]} d.5 {row[2]} | top-8: d0 {row[3]} d.25 {row[4]} d.5 {row[5]}", flush=True)
