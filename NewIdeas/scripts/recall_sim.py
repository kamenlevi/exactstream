"""Value of look-ahead recall (NewIdeas, 2026-10-21; CPU). On the 125B pf traces (actual top-10 per layer), an oracle-mixed
predictor prefetches each true cold expert of layer l (issued at layer l-1) with probability RHO, plus WASTE wrong reads
per layer; the rest of the cold experts are waited (exact coverage of the top-KC ranks; ranks >= KC dropped). Speed with
the calibrated eager model (30.4 ms + 0.31 ms per waited read; bandwidth floor reads x MB / 14.6 GB/s)."""
import glob
import numpy as np
from collections import OrderedDict

NL = 48
TR = [np.load(f) for f in sorted(glob.glob("/Users/kamen/MoEPrefetch/data/traces125/*_400_pf.npz"))]
rng = np.random.default_rng(0)


def sim(C, rho, waste, KC=10):
    reads = waited = ntok = 0
    for d in TR:
        T = d["L0"].shape[0]; ntok += T
        caches = [OrderedDict() for _ in range(NL)]
        for t in range(T):
            for l in range(NL):
                c = caches[l]
                act = [int(e) for e in d[f"L{l}"][t, :KC]]
                if l > 0:
                    for e in act:
                        if e not in c and rng.random() < rho:
                            c[e] = 1; reads += 1
                            if len(c) > C: c.popitem(last=False)
                    reads += waste
                for e in act:
                    if e in c:
                        c.move_to_end(e)
                    else:
                        waited += 1; reads += 1; c[e] = 1
                        if len(c) > C: c.popitem(last=False)
    return reads / ntok, waited / ntok


for C, label in ((51, "10 %"), (128, "25 % (24 GB setting)")):
    for KC in (10, 8):
        print(f"--- C={C} ({label}), coverage top-{KC}; eager tok/s at 4-bit / MSB-3 bytes")
        for rho, waste in ((0.0, 0), (0.6, 2), (0.7, 2), (0.8, 2), (0.9, 2), (0.95, 2), (1.0, 0)):
            r, w = sim(C, rho, waste, KC)
            sp = [1000 / max(30.4 + 0.31 * w, r * mb / 14.6) for mb in (2.77, 2.15)]
            print(f"recall {rho:4.2f} (+{waste} wasted/layer): reads {r:6.1f} waited {w:6.1f} -> {sp[0]:5.1f} / {sp[1]:5.1f} tok/s", flush=True)
