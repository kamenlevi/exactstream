"""Timing model with REAL forecasts (NewIdeas 2026-10-22): the 125B capture tf_samelayer125.py (exact decode calls, Eval
suite tf items; T true top-10 + P probs, S same-layer pre-attention top-16, D / DC d=1 tap plain / MoEPrefetch-corrected
top-16 for the NEXT layer). SSD model from stage_time.py (SSDV=1 or 2). Frugal (ranks 0-1 waited, lower ranks only if
resident) in 4-row MTP passes on the fast engine, and the EXACT track (all top-10 waited) on an eager-v3-like engine.
Forecasts: d1 = DC top-k issued at layer l-1's router point ('after' its demand reads); same = S top-k issued at the
layer's start (lead = the attention half); both = union."""
import glob, os, sys
import numpy as np
from collections import OrderedDict

sys.argv = sys.argv[:1] + sys.argv[1:]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_time.py")).read()
ns = {"__name__": "lib"}
_argv = sys.argv; sys.argv = ["lib"]; exec(compile(src, "stage_time", "exec"), ns); sys.argv = _argv
SSD, REC, NL = ns["SSD"], ns["REC"], ns["NL"]

SEQ = []
for f in sorted(glob.glob(os.path.expanduser("~/NewIdeas/data/samelayer/S*_e16.npz"))):
    z = np.load(f)
    T = np.stack([z[f"T{l}"] for l in range(NL)], 1).astype(np.int64)            # [n, NL, 10]
    S = np.stack([z[f"S{l}"] for l in range(NL)], 1).astype(np.int64)            # [n, NL, 16]
    DC = np.stack([z[f"DC{l}"] for l in range(NL - 1)] + [z[f"DC{NL - 2}"]], 1).astype(np.int64)   # DC[:, l] -> layer l+1
    SEQ.append((os.path.basename(f), T, S, DC))


def run(mode, C, kd, ks, A, B, rows=4, adv=3, over=3.0, conf=0):
    """mode 'frugal' (ranks 0-1 waited) or 'exact' (top-10 waited). kd = d1 corrected top-k ('after'), ks = same-layer top-k."""
    tot = 0.0; ntok = 0; npass = 0; waited = 0; reads = 0
    nw = 2 if mode == "frugal" else 10
    for name, T, S, DC in SEQ:
        caches = [OrderedDict() for _ in range(NL)]; ssd = SSD(); t = 0.0
        for p0 in range(0, T.shape[0] - rows, adv):
            t0 = t; npass += 1; ntok += adv; staged = {}
            R = range(p0, p0 + rows)
            for l in range(NL):
                c = caches[l]; cur = staged; staged = {}; need = {}
                if ks:
                    for tt in R:
                        for e in S[tt, l, :ks]:
                            e = int(e)
                            if e not in c and e not in cur: cur[e] = ssd.issue(t); reads += 1
                if conf and l > 0:                         # same-layer CONFIRMED: in both S[:conf] and the d=1 forecast DC[:conf]
                    for tt in R:
                        dset = set(int(x) for x in DC[tt, l - 1, :conf])
                        for e in S[tt, l, :conf]:
                            e = int(e)
                            if e in dset and e not in c and e not in cur: cur[e] = ssd.issue(t); reads += 1
                t += A
                for tt in R:
                    for r in range(10):
                        e = int(T[tt, l, r])
                        if e in c: c.move_to_end(e); continue
                        if e in need or r >= nw: continue
                        if e in cur: need[e] = cur[e]
                        else: need[e] = ssd.issue(t); waited += 1; reads += 1
                        c[e] = 1
                        if len(c) > C: c.popitem(last=False)
                ready = max([t] + [ssd.when(i) for i in need.values()])
                if kd and l + 1 < NL:
                    for tt in R:
                        for e in DC[tt, l, :kd]:
                            e = int(e)
                            if e not in caches[l + 1] and e not in staged: staged[e] = ssd.issue(ready); reads += 1
                t = ready + B
            t += over; tot += t - t0
    return 1000 * ntok / tot, waited / npass, reads / npass, tot / npass


if __name__ == "__main__" and "more" in sys.argv:
    print(f"SSD model v{os.environ.get('SSDV', '2')}: precision-first variants")
    print("FRUGAL fast engine C=51:")
    base = None
    for kd, ks, cf in ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (2, 1, 0), (0, 0, 2), (0, 0, 3), (0, 0, 4), (2, 0, 3)):
        tps, w, rd, ms = run("frugal", 51, kd, ks, 0.30, 0.33, conf=cf)
        base = base or tps
        print(f"  d1c k={kd} same k={ks} confirmed@{cf}: {tps:5.1f} tok/s ({100 * (tps / base - 1):+.0f} %)  waited {w:5.1f}  reads {rd:6.1f}/pass", flush=True)
    print("EXACT eager-v3-like C=128:")
    base = None
    for kd, ks, cf in ((0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 2, 0), (0, 4, 0), (0, 0, 8), (0, 0, 12), (2, 0, 10)):
        tps, w, rd, ms = run("exact", 128, kd, ks, 0.45, 0.50, over=5.0, conf=cf)
        base = base or tps
        print(f"  d1c k={kd} same k={ks} confirmed@{cf}: {tps:5.1f} tok/s ({100 * (tps / base - 1):+.0f} %)  waited {w:5.1f}  reads {rd:6.1f}/pass", flush=True)


if __name__ == "__main__" and "more" not in sys.argv:
    print(f"SSD model v{os.environ.get('SSDV', '2')}; {len(SEQ)} capture files, "
          f"{sum(x[1].shape[0] for x in SEQ)} decode calls", flush=True)
    print("FRUGAL, fast engine, 4-row MTP passes (0.30+0.33 ms/layer + 3 ms), C=51:")
    base = None
    for kd, ks in ((0, 0), (2, 0), (0, 2), (0, 3), (0, 4), (2, 2), (2, 3)):
        tps, w, rd, ms = run("frugal", 51, kd, ks, 0.30, 0.33)
        base = base or tps
        print(f"  d1c k={kd} same k={ks}: {tps:5.1f} tok/s ({100 * (tps / base - 1):+.0f} %)  {ms:5.1f} ms/pass  waited {w:5.1f}"
              f"  reads {rd:6.1f}/pass", flush=True)
    print("EXACT track, eager-v3-like (0.45+0.50 ms/layer + 5 ms), C=128, 4-row passes:")
    base = None
    for kd, ks in ((0, 0), (12, 0), (0, 10), (0, 12), (0, 16), (12, 12)):
        tps, w, rd, ms = run("exact", 128, kd, ks, 0.45, 0.50, over=5.0)
        base = base or tps
        print(f"  d1c k={kd:2d} same k={ks:2d}: {tps:5.1f} tok/s ({100 * (tps / base - 1):+.0f} %)  {ms:5.1f} ms/pass  waited {w:5.1f}"
              f"  reads {rd:6.1f}/pass", flush=True)
