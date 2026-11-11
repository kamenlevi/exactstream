"""Predict the gain of spare-slot mid-token prefetch on the proxy traces (35B: 40 layers x 256).
Per layer: main LRU of C_main experts + SPARE prefetch slots. Prefetch for layer L uses layer L-1's
residual through layer L's router (pmid1, top-B), issued one layer ahead. A miss whose expert was
prefetched waits only max(0, read - window) instead of the full read.
Model: stall time per pass = sum over stalled layers of (full read if any miss was not prefetched,
else max(0, read - window)). -> data/sim_pf.txt"""
import os
from collections import OrderedDict
import numpy as np
TR = os.path.expanduser("~/MoEPrefetch/traces"); NL, K, E = 40, 8, 256
SETS = ["chat", "code", "bulgarian", "longctx", "realcode"]
zs = [dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in SETS]
READ = 0.50e-3                    # measured: cold stalled layer ~0.5 ms (35B)


def run(C, spare, B, S, window, start=200):
    t_stall = n = full = part = 0
    for z in zs:
        T = len(z["doc"])
        acts = [z[f"L{L}_act"][:, :K] for L in range(NL)]
        preds = [z[f"L{L-1}_pmid1"][:, :B] if L >= 1 else None for L in range(NL)]
        main = [OrderedDict() for _ in range(NL)]
        pf = [OrderedDict() for _ in range(NL)]
        for t0 in range(0, T - S + 1, S):
            for L in range(NL):
                need = list(dict.fromkeys(acts[L][t0:t0 + S].reshape(-1).tolist()))
                if spare and preds[L] is not None:            # prefetch issued while layer L-1 runs
                    for e in dict.fromkeys(preds[L][t0:t0 + S].reshape(-1).tolist()):
                        if e in main[L] or e in pf[L]: continue
                        pf[L][e] = None
                        if len(pf[L]) > spare: pf[L].popitem(last=False)
                miss = [e for e in need if e not in main[L]]
                covered = [e for e in miss if e in pf[L]]
                for e in need:
                    if e in main[L]: main[L].move_to_end(e)
                for e in miss:
                    pf[L].pop(e, None)
                    while len(main[L]) >= C - spare:
                        for v in main[L]:
                            if v not in need: del main[L][v]; break
                    main[L][e] = None
                if t0 >= start and miss:
                    if len(covered) < len(miss): t_stall += READ; full += 1
                    else: t_stall += max(0.0, READ - window); part += 1
            if t0 >= start: n += 1
    return t_stall / n * 1e3, full / n, part / n


lines = []
C = 154
for S in (1, 3):
    base = run(C, 0, 12, S, 0.0)
    lines.append(f"S={S} no prefetch (C={C}): stall {base[0]:.2f} ms/pass, stalled layers {base[1]:.1f}")
    for spare, B, win in [(12, 12, 0.20e-3), (12, 12, 0.35e-3), (16, 16, 0.20e-3), (8, 8, 0.20e-3)]:
        r = run(C, spare, B, S, win)
        lines.append(f"S={S} spare={spare:2d} B={B:2d} window={win*1e6:.0f}us: stall {r[0]:.2f} ms/pass "
                     f"({(1 - r[0]/base[0])*100:+.0f}% less), full-read layers {r[1]:.1f}, partly hidden {r[2]:.1f}")
    print("\n".join(lines[-5:]), flush=True)
open(os.path.expanduser("~/BigMoE/data/sim_pf.txt"), "w").write("\n".join(lines) + "\n")
