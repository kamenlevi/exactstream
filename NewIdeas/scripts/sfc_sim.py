"""Stall-free coverage on real 125B decode traces (NewIdeas, 2026-10-21; CPU). MoEPrefetch's *_400_pf traces: actual top-20
per layer (L<l>, by score) and the look-ahead router's top-32 prediction for layer l+1 made at layer l (P1_<l>); 4 domains
x 400 decode tokens. Per-layer LRU of C slots. Policies (per layer l, actual top-10):
  pf=k   at layer l-1 the predicted top-k for layer l that are not resident are read (arrive by l's router; inserted)
  cold rank < W  -> waited (critical path);  cold rank >= W -> 'late' (half a layer late) or 'drop' or 'wait'
Outputs per token: reads (incl. prefetch), waited reads, late reads per layer histogram, prefetch waste; then the speed
model (scripts/spdmodel.py calibration: eager t = 30.4 ms + 0.31 ms per waited read + late-burst excess over SLACK;
bandwidth floor reads x MB / 14.6 GB/s)."""
import glob, sys
import numpy as np
from collections import OrderedDict

NL, K = 48, 10
fs = sorted(glob.glob("/Users/kamen/MoEPrefetch/data/traces125/*_400_pf.npz"))
TR = [np.load(f) for f in fs]
BURST = ([0, 1, 2, 4, 8], [0, 0.336, 0.561, 0.975, 1.79])


def sim(C=51, pf=0, W=2, rest="late", maxpf=99, drop_from=10):
    reads = waited = late_tot = waste = ntok = 0
    hist = np.zeros(5)
    for d in TR:
        T = d["L0"].shape[0]; ntok += T
        caches = [OrderedDict() for _ in range(NL)]
        for t in range(T):
            pend = set()
            for l in range(NL):
                c = caches[l]
                act = [int(e) for e in d[f"L{l}"][t, :K]]
                if pf and l > 0:                                  # prefetch issued at layer l-1 for layer l
                    n = 0
                    for e in d[f"P1_{l - 1}"][t, :pf]:
                        e = int(e)
                        if e not in c and n < maxpf:
                            c[e] = 1; reads += 1; n += 1
                            if e not in act: waste += 1
                            if len(c) > C: c.popitem(last=False)
                nl = 0
                for r, e in enumerate(act):
                    if e in c:
                        c.move_to_end(e); continue
                    if r < W or rest == "wait":
                        waited += 1; reads += 1
                    elif rest == "late" and r < drop_from:
                        late_tot += 1; reads += 1; nl += 1
                    else:
                        continue                                  # dropped: not read, not inserted
                    c[e] = 1
                    if len(c) > C: c.popitem(last=False)
                hist[min(nl, 4)] += 1
    return reads / ntok, waited / ntok, late_tot / ntok, hist / hist.sum(), waste / ntok


def speed(reads, waited, hist, slack=0.5, mb=2.77):
    late_cost = NL * sum(hist[n] * max(0.0, float(np.interp(n if n < 4 else 4.5, *BURST)) - slack) for n in range(1, 5))
    t_lat = 30.4 + 0.31 * waited + late_cost
    t_bw = reads * mb / 14.6
    return 1000 / max(t_lat, t_bw), t_lat, t_bw


if __name__ == "__main__":
    rows = [("exact, no prefetch", dict(pf=0, rest="wait")),
            ("exact + prefetch top-10", dict(pf=10, rest="wait")),
            ("exact + prefetch top-16", dict(pf=16, rest="wait")),
            ("frugal (drop cold ranks >= 2)", dict(pf=0, rest="drop")),
            ("frugal + prefetch top-10", dict(pf=10, rest="drop")),
            ("half-late cold ranks >= 2", dict(pf=0, rest="late")),
            ("prefetch top-10 + half-late", dict(pf=10, rest="late")),
            ("prefetch top-16 + half-late", dict(pf=16, rest="late")),
            ("prefetch top-10 + half-late ranks 2-7, drop 8-9", dict(pf=10, rest="late", drop_from=8)),
            ("prefetch top-10 + half-late, W=1", dict(pf=10, rest="late", W=1))]
    for C in (51, 26):
        print(f"--- C={C} ({C / 512:.0%}), 1600 decode tokens; speeds = eager model (4-bit / MSB-3 bytes)")
        for name, kw in rows:
            r, w, lt, h, ws = sim(C=C, **kw)
            s4, tl, tb = speed(r, w, h); s3, _, _ = speed(r, w, h, mb=2.15)
            print(f"{name:48s} reads {r:6.1f} (waste {ws:5.1f})  waited {w:5.1f}  late {lt:5.1f}  late/layer 0/1/2/3/4+ "
                  + "/".join(f"{x:.2f}" for x in h) + f"  -> {s4:5.1f} / {s3:5.1f} tok/s (lat {tl:.0f} ms, bw {tb:.0f} ms)", flush=True)
