"""Plane-granular expert caching on the real 125B traces (round 18; CPU only). An expert's 4-bit codes split into an MSB
plane (2 bits + scale + both biases, 2.625 bits/weight, 1.61 MB) and an LSB plane (2 bits/weight, 1.23 MB); whole 4-bit
expert 2.77 MB. Verify windows of 4 tokens, stride 3 (as admission_sim / stall_alloc_sim); traces are ranked router top-20
(MoEPrefetch traces125, from waittop125 trace mode). Per layer, an LRU over cache units within a byte budget.
  whole      : the exact model, LRU of whole experts; every use needs all of it (BigMoE today)
  planes-R   : mix-R (GSM8K-equal on the 35B at R=4 of 8): uses at gate rank < R need MSB + LSB, the others MSB only;
               MSB and LSB planes are separate LRU units, so cold LSB planes leave first (a compression cache, Douglis 1993)
  pinMSB-R   : every MSB plane pinned (when they fit), LSB planes LRU in the remaining bytes
  fallback-R : planes-R, but a missing LSB is never waited for (game-engine mip fallback): that use runs MSB-only and the
               LSB streams in the background; demand bytes = MSB misses only; cost = the fallback rate of top-R uses
Reports demand (stall) MB/pass, background MB/pass, misses/pass and the fallback rate.   usage: planes_cache_sim.py"""
import glob, os
from collections import OrderedDict
import numpy as np
E, K, WARM, NL = 512, 10, 200, 48
WHOLE, MSB, LSB = 2.77, 1.61, 1.23                       # MB per unit
LSBFIRST = False                                         # SliceMoE-style: a fetched LSB plane goes to the LRU end
files = sorted(glob.glob(os.path.expanduser("~/MoEPrefetch/data/traces125/*_800.npz")))
TR = []
for f in files:
    z = np.load(f)
    TR.append([z[f"L{L}"][:, :K].astype(np.int64) for L in range(len(z.files))])
NL = len(TR[0])


def windows(tr):
    T = tr.shape[0]
    for s in range(0, T - 4, 3):
        yield s, tr[s:s + 4]                              # [4 tokens, K ranks]


def run(policy, budget_mb, R):
    dem = bg = miss = fb = top = P = 0
    for tr_layers in TR:
        for L in range(NL):
            cache, used = OrderedDict(), 0.0
            pin = policy == "pinMSB" and E * MSB <= budget_mb
            cap = budget_mb - (E * MSB if pin else 0.0)
            for s, w in windows(tr_layers[L]):
                if policy == "whole":
                    need = {(int(e), "W") for e in np.unique(w)}
                elif policy == "msbfull":               # MSB plane files + the original 4-bit files (no LSB files):
                    topw = {int(e) for e in np.unique(w[:, :R])}         # top-rank uses need the whole expert
                    need = {(e, "W") for e in topw}
                    for e in {int(e) for e in np.unique(w)} - topw:      # low-rank uses take the whole one if resident
                        need.add((e, "W") if (e, "W") in cache else (e, "M"))
                    for e in topw:                                         # a whole expert makes its MSB unit redundant
                        if (e, "M") in cache:
                            used -= cache.pop((e, "M"))
                else:
                    need = {(int(e), "M") for e in np.unique(w)} if not pin else set()
                    need |= {(int(e), "L") for e in np.unique(w[:, :R])}
                warm = s >= WARM
                late = []
                for u in sorted(need):
                    if u in cache:
                        cache.move_to_end(u)
                    else:
                        late.append(u)
                for u in late:
                    size = WHOLE if u[1] == "W" else MSB if u[1] == "M" else LSB
                    if LSBFIRST and u[1] == "L":           # SliceMoE: LSB slices lowest priority -> insert at the LRU end
                        if warm:
                            miss += 1
                            if policy == "fallback":
                                bg += size; fb += int((w[:, :R] == u[0]).sum())
                            else:
                                dem += size
                        cache[u] = size; cache.move_to_end(u, last=False); used += size
                        while used > cap and cache:
                            _, sz = cache.popitem(last=False); used -= sz
                        continue
                    if warm:
                        miss += 1
                        if policy == "fallback" and u[1] == "L":
                            bg += size
                            fb += int((w[:, :R] == u[0]).sum())    # top-R uses of this expert in the window fall back
                        else:
                            dem += size
                    cache[u] = size; used += size
                    while used > cap and cache:
                        _, sz = cache.popitem(last=False); used -= sz
                if warm:
                    top += w[:, :R].size if policy != "whole" else 0
            P += sum(1 for s, _ in windows(tr_layers[0]) if s >= WARM) if L == 0 else 0
    return dem / P, bg / P, miss / P, (fb / top if top else 0.0)


print(f"125B traces: {len(TR)} prompts x 800 tokens, {NL} layers, verify windows of 4 (stride 3); MB per pass, all layers")
import sys
if sys.argv[1:] == ["small"]:                    # Kamen's target: today's speed in 24 GB total -> ~17 GB expert cache
    for gb in (17, 20, 34):
        b = gb * 1024 / NL
        d, _, m, _ = run("whole", b, K)
        print(f"budget {gb} GB: whole-expert LRU: demand {d:6.1f} MB/pass ({m:5.1f} misses)", flush=True)
        for lf in (False, True):
            LSBFIRST = lf
            for pol in ("planes", "fallback"):
                d, g, m, f = run(pol, b, 4)
                print(f"   {pol:8s} R=4{' LSB-first' if lf else '          '}: demand {d:6.1f} MB/pass, background {g:6.1f}"
                      + (f", fallback {100 * f:.1f} % of top-4 uses" if pol == "fallback" else ""), flush=True)
        LSBFIRST = False
    sys.exit()
if sys.argv[1:] == ["msbfull"]:                  # disk-light variant: only MSB plane files are written (~40 GB for the 125B)
    for gb in (24, 34, 45):
        b = gb * 1024 / NL
        d, _, m, _ = run("whole", b, K)
        print(f"budget {gb} GB: whole-expert LRU: demand {d:6.1f} MB/pass, misses {m:5.1f}/pass", flush=True)
        for R in (4, 5):
            d, _, m, _ = run("msbfull", b, R)
            print(f"   msbfull R={R}: demand {d:6.1f} MB/pass, unit misses {m:5.1f}/pass", flush=True)
    sys.exit()
for gb in (24, 34, 45):
    b = gb * 1024 / NL
    d, _, m, _ = run("whole", b, K)
    print(f"budget {gb} GB ({gb / 68 * 100:.0f} % of 4-bit experts): whole-expert LRU: demand {d:6.1f} MB/pass, "
          f"misses {m:5.1f}/pass", flush=True)
    for R in (4, 5):
        for pol in ("planes", "pinMSB", "fallback"):
            d, g, m, f = run(pol, b, R)
            print(f"   {pol:8s} R={R}: demand {d:6.1f} MB/pass, background {g:6.1f}, unit misses {m:5.1f}/pass"
                  + (f", fallback {100 * f:.1f} % of top-{R} uses" if pol == "fallback" else ""), flush=True)
