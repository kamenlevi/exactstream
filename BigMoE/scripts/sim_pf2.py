"""Depth-2 look-ahead prefetch on the proxy traces (35B: 40 layers x 256): besides layer L-1's
prediction for layer L (pmid1, head start ~200 us), layer L-2's prediction (pmid2, head start
~450 us) is prefetched too. Same model as sim_pf.py: a miss whose expert was prefetched waits
max(0, READ - head start). -> data/sim_pf2.txt"""
import os
from collections import OrderedDict
import numpy as np
TR = os.path.expanduser("~/MoEPrefetch/traces"); NL, K, E = 40, 8, 256
SETS = ["chat", "code", "bulgarian", "longctx", "realcode"]
zs = [dict(np.load(os.path.join(TR, f"{s}.npz"))) for s in SETS]
READ, W1, W2 = 0.50e-3, 0.20e-3, 0.45e-3


def run(C, spare, B1, B2, S, start=200):
    t_stall = n = 0
    for z in zs:
        T = len(z["doc"])
        acts = [z[f"L{L}_act"][:, :K] for L in range(NL)]
        p1 = [z[f"L{L-1}_pmid1"][:, :B1] if (B1 and L >= 1) else None for L in range(NL)]
        p2 = [z[f"L{L-2}_pmid2"][:, :B2] if (B2 and L >= 2) else None for L in range(NL)]
        main = [OrderedDict() for _ in range(NL)]
        pf = [OrderedDict() for _ in range(NL)]          # expert -> head start
        for t0 in range(0, T - S + 1, S):
            for L in range(NL):
                need = list(dict.fromkeys(acts[L][t0:t0 + S].reshape(-1).tolist()))
                for preds, w in ((p2[L], W2), (p1[L], W1)):
                    if not spare or preds is None: continue
                    for e in dict.fromkeys(preds[t0:t0 + S].reshape(-1).tolist()):
                        if e in main[L] or e in pf[L]: continue
                        pf[L][e] = w
                        if len(pf[L]) > spare: pf[L].popitem(last=False)
                miss = [e for e in need if e not in main[L]]
                for e in need:
                    if e in main[L]: main[L].move_to_end(e)
                if t0 >= start and miss:
                    if all(e in pf[L] for e in miss):
                        t_stall += max(0.0, READ - min(pf[L][e] for e in miss))
                    else:
                        t_stall += READ
                for e in miss:
                    pf[L].pop(e, None)
                    while len(main[L]) >= C - spare:
                        for v in main[L]:
                            if v not in need: del main[L][v]; break
                    main[L][e] = None
            if t0 >= start: n += 1
    return t_stall / n * 1e3


lines = []
for S in (1, 3):
    base = run(154, 0, 0, 0, S)
    for spare, B1, B2 in [(12, 12, 0), (16, 8, 8), (20, 10, 10), (24, 12, 12)]:
        r = run(154, spare, B1, B2, S)
        lines.append(f"S={S} spare={spare:2d} B1={B1:2d} B2={B2:2d}: stall {r:.2f} ms/pass "
                     f"({(1 - r / base) * 100:+.0f}% less than no prefetch {base:.2f})")
        print(lines[-1], flush=True)
open(os.path.expanduser("~/BigMoE/data/sim_pf2.txt"), "w").write("\n".join(lines) + "\n")
