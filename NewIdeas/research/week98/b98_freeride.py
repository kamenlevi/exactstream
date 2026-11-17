#!/usr/bin/env python3
"""BREAK98: the FREE RIDE behind a cheap repair (NewIdeas, Tue 17 Nov; time = the commit's). When the head's draft goes
wrong at depth e, how many of its following tokens (drafted conditioned on the WRONG token at e) still equal the true
tokens? If many, a repair can swap just the token at e and keep the old tail (no re-draft levels; only the catch-up of
layers 1..L). From BigMoE's 127-node k8 dumps: y = the true token at a node's depth (checked position-based, 100 %).
At each anchor: the last on-true node; its best wrong child (max path_p); then that child's greedy chain (max path_p child
per level, inside the 127 pops). Free ride = consecutive matches with y after e.
usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B b98_freeride.py"""
import os
import numpy as np
CE = os.path.expanduser("~/BigMoE/data/deepdraft/ceiling")
for tag in ("heplus24", "mbppbank"):
    z = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz", allow_pickle=True)
    fi, an, nd, par, dp, tok, y, ot, pp = (z[k] for k in ("fi", "an", "node", "parent", "depth", "token", "y", "on_true", "path_p"))
    key = fi.astype(np.int64) * 10_000_000 + an.astype(np.int64)
    o = np.lexsort((nd, key)); key, nd, par, dp, tok, y, ot, pp = (a[o] for a in (key, nd, par, dp, tok, y, ot, pp))
    b = np.flatnonzero(np.append(True, key[1:] != key[:-1])); b = np.append(b, len(key))
    rides, depth_avail = [], []
    for b0, b1 in zip(b[:-1], b[1:]):
        N = nd[b0:b1]; P = par[b0:b1]; D = dp[b0:b1]; T = tok[b0:b1]; Y = y[b0:b1]; O = ot[b0:b1]; Pp = pp[b0:b1]
        node_ix = {int(n): k for k, n in enumerate(N)}
        kids = {}
        for k, pa in enumerate(P):
            kids.setdefault(int(pa), []).append(k)
        last = -1                                             # the deepest on-true node (or the root, -1)
        ont = [k for k in range(len(N)) if O[k]]
        if ont:
            last = int(N[max(ont, key=lambda k: D[k])])
        ch = [k for k in kids.get(last, []) if not O[k]]
        if not ch:
            continue                                          # the true child is the only child present, or none
        w = max(ch, key=lambda k: Pp[k])                      # the head's best wrong token at depth e
        r = 0; cur = w; avail = 0
        while True:
            cc = kids.get(int(N[cur]), [])
            if not cc:
                break
            cur = max(cc, key=lambda k: Pp[k]); avail += 1
            if T[cur] == Y[cur]:
                r += 1
            else:
                break
        rides.append(r); depth_avail.append(avail)
    R = np.array(rides); Av = np.array(depth_avail)
    has = Av >= 1
    print(f"== {tag}: {len(R)} anchors with a wrong best child; tail present below it in {100 * has.mean():.0f} %")
    print(f"   free ride after the wrong token (consecutive true tokens in the old tail): mean {R[has].mean():.2f}, "
          f"P(>=1) {100 * (R[has] >= 1).mean():.0f} %, P(>=4) {100 * (R[has] >= 4).mean():.0f} %, P(>=8) {100 * (R[has] >= 8).mean():.0f} %; "
          f"median tail length present {np.median(Av[has]):.0f}")
