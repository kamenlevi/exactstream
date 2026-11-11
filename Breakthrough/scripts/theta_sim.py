"""Per-expert wait rule by gate share, with optional half-byte MSB-2 reads (Breakthrough, 2026-10-20 22:10).
125B traces with router probabilities; frugal LRU cache of whole 4-bit experts (C per layer); no substitution.
Rule per token-layer: ranks 0-1 always computed at 4 bits (read if missing). A missing rank r >= 2 with renormalised
top-10 share w_r >= theta is read (waited): 'full' = whole expert (2.77 MB, enters the cache), 'msb2' = MSB-2 planes
only (1.54 MB, used once at 2 bits, not cached). Below theta it is skipped.
Prints loads/token, MB/token, skipped mass, mass computed at 2 bits.  usage: theta_sim.py [C ...]"""
import glob, sys
import numpy as np
from collections import OrderedDict

NL = 48; FULL, MSB2 = 2.765, 1.536
sets = {f.split("/")[-1].split("_")[0]: np.load(f)
        for f in sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))}
ids = {k: np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int32) for k, d in sets.items()}
gp = {k: np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float32) for k, d in sets.items()}
w = {k: g[:, :, :10] / g[:, :, :10].sum(-1, keepdims=True) for k, g in gp.items()}


def run(C, theta, mode):
    mb = loads = ntok = 0.0
    skip, low = [], []
    for k in ids:
        tr, wt = ids[k], w[k]
        ntok += tr.shape[0]
        for l in range(NL):
            c = OrderedDict()
            for t in range(tr.shape[0]):
                s = lo = 0.0
                used = []
                for r in range(10):
                    e, p = int(tr[t, l, r]), float(wt[t, l, r])
                    if e in c:
                        used.append(e); continue
                    if r < 2 or (p >= theta and mode == "full"):
                        loads += 1; mb += FULL; used.append(e)
                    elif p >= theta:
                        loads += 1; mb += MSB2; lo += p
                    else:
                        s += p
                for e in used:
                    if e in c: c.move_to_end(e)
                    else: c[e] = 1
                while len(c) > C:
                    c.popitem(last=False)
                skip.append(s); low.append(lo)
    return loads / ntok, mb / ntok, np.mean(skip), np.mean(low)


for C in [int(a) for a in sys.argv[1:]] or [51, 26]:
    print(f"C={C} ({100 * C / 512:.0f} %): loads/token, MB/token | skipped mass | mass at 2 bits", flush=True)
    for mode in ("full", "msb2"):
        for theta in (1.0, 0.15, 0.10, 0.07, 0.05, 0.03):
            ld, mb, s, lo = run(C, theta, mode)
            print(f"  {mode:4s} theta {theta:4.2f}: {ld:6.1f} loads {mb:6.0f} MB | skipped {s:.3f} | 2-bit {lo:.3f}", flush=True)
