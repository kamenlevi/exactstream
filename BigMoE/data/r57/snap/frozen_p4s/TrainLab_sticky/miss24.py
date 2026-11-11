"""TrainLab (10-22): predicted misses per 4-row verify pass at 24 GB (C = 128 expert slots per layer, LRU) on the 125B
traces, natural routing vs the per-pass expert budget (U experts by summed w^2). Teacher-forced 800-token texts, no
drafts and no JIT prefetch: a trend for turn14's misses / pass, not its number. CPU only."""
import glob, os
from collections import OrderedDict
import numpy as np
fs = sorted(glob.glob("/Users/kamen/Breakthrough/data/expA/traces125/*_800_g.npz"))
import sys
NL, K, C = 48, 10, 128
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
US = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [18, 16]
for f in fs:
    name = os.path.basename(f).split("_")[0]
    d = np.load(f)
    ids = np.stack([d[f"L{l}"] for l in range(NL)], 1).astype(np.int64)
    g = np.stack([d[f"G{l}"] for l in range(NL)], 1).astype(np.float64)
    w = g / g[:, :, :K].sum(-1, keepdims=True)
    T = ids.shape[0] // B * B
    res = []
    for U in [None] + US:
        miss = 0
        for l in range(NL):
            lru = OrderedDict()
            for b in range(0, T, B):
                blk, wb = ids[b:b + B, l, :K], w[b:b + B, l, :K]
                if U is None:
                    used = set(int(e) for e in blk.ravel())
                else:
                    c = {}
                    for r in range(B):
                        for e, x in zip(blk[r], wb[r]):
                            c[int(e)] = c.get(int(e), 0.0) + x * x
                    used = set(sorted(c, key=c.get, reverse=True)[:U])
                for e in used:
                    if e in lru:
                        lru.move_to_end(e)
                    else:
                        miss += 1 if b > 8 * B else 0                 # after a short warm-up
                        lru[e] = None
                        if len(lru) > C:
                            lru.popitem(last=False)
        passes = (T // B - 9)
        res.append((U, miss / passes))
    nat = res[0][1]
    print(f"{name:9s} " + " | ".join(f"{'natural' if U is None else f'b{U}'}: misses/pass({B}) {m:.1f}" + ("" if U is None else f" ({m / nat:.2f}x)")
                                    for U, m in res), flush=True)
