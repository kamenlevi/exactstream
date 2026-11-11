"""SlotStream's idea, kill test (CPU, 125B traces): ONE slot pool shared by all 48 layers (global LRU) vs per-layer LRU pools
at the same total slots (exact: a cache policy only). Teacher-forced 800-token texts, B-row passes (verify width), top-10,
no drafts / JIT (a trend, as TrainLab's miss24.py). Misses counted after a 64-token warm-up.
Pre-registered: global/per-layer misses >= 0.95 at 24 GB (C=128) -> not worth building; <= 0.85 -> worth an engine A/B."""
import glob, os, sys
from collections import OrderedDict
import numpy as np
NL, K = 48, 10
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
for C in (128, 51):
    tot = {"layer": 0, "global": 0, "rows": 0}
    for f in sorted(glob.glob(os.path.expanduser("~/Breakthrough/data/expA/traces125/*_800_g.npz"))):
        d = np.load(f)
        ids = np.stack([d[f"L{l}"][:, :K] for l in range(NL)], 1).astype(np.int64)
        T = ids.shape[0] // B * B
        per = [OrderedDict() for _ in range(NL)]
        glob_ = OrderedDict()
        ml = mg = 0
        for b in range(0, T, B):
            warm = b >= 64
            for l in range(NL):
                used = set(int(e) for e in ids[b:b + B, l].ravel())
                lru = per[l]
                for e in used:
                    if e in lru:
                        lru.move_to_end(e)
                    else:
                        ml += warm
                        lru[e] = None
                        if len(lru) > C:
                            lru.popitem(last=False)
                for e in used:
                    k = (l, e)
                    if k in glob_:
                        glob_.move_to_end(k)
                    else:
                        mg += warm
                        glob_[k] = None
                        if len(glob_) > C * NL:
                            glob_.popitem(last=False)
        rows = T - 64
        tot["layer"] += ml; tot["global"] += mg; tot["rows"] += rows
        print(f"C={C:3d} B={B} {os.path.basename(f).split('_')[0]:9s} misses/token per-layer {ml / rows:6.1f}  global {mg / rows:6.1f}"
              f"  ratio {mg / max(ml, 1):.3f}")
    print(f"C={C:3d} B={B} ALL       misses/token per-layer {tot['layer'] / tot['rows']:6.1f}  global {tot['global'] / tot['rows']:6.1f}"
          f"  ratio {tot['global'] / max(tot['layer'], 1):.3f}")
