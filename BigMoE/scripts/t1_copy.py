#!/usr/bin/env python3
"""T1 COPYABILITY (the orchestrator 22:4x; theory: a fresh state gives the head (a) retrieval from the far prefix and
(b) composition of the new tokens). On MBPP 100, per eval cell (anchor s, band j: the target is ids[s + j + 2], the
context ids[:s + j + 2], true tokens), from per-cell dumps of t30b's own chain and of the true-wd oracle (dd_taps.py
evalrep --dump), the share of target tokens that:
  - seen: appear anywhere earlier in the context;
  - far: last appeared > 64 tokens back;
  - lookup: are what a prompt lookup predicts (the longest earlier suffix match, n = 4..1, the token after its latest
    earlier occurrence), and the n it matched at.
Categories:
  - FIXED: the chain misses, the oracle hits (what freshness buys), at 7-22 and 23-30;
  - HIT0: the chain's depth-0 hits (reference);
  - HIT23: the chain's hits at 23-30;
  - MISS23: both miss at 23-30.
-> data/deepdraft/bank/t1_copy.json"""
import json, os
import numpy as np

H = os.path.expanduser
ch = np.load(H("~/datasets/deepdraft/t1_chain.npz")); orc = np.load(H("~/datasets/deepdraft/t1_oracle.npz"))
MB = H("~/TrainLab/data/dlog/d1102mbpp")
cat = {k: {"n": 0, "seen": 0, "far": 0, "lookup": 0, "lk_n": [0, 0, 0, 0, 0]} for k in
       ("FIXED_7-22", "FIXED_23-30", "HIT0", "HIT23", "MISS23", "ALL23")}
for fi, nm in enumerate(ch["files"]):
    ids = np.asarray(np.load(os.path.join(MB, str(nm)))["ids"]).tolist(); N = len(ids)
    seen, far, lk, lkn = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, np.int8)
    last = {}; grams = [dict() for _ in range(5)]                  # n-gram (n = 1..4) -> the position after its latest occurrence
    for p in range(N):
        t = ids[p]
        if t in last:
            seen[p] = True; far[p] = p - last[t] > 64
        for n in (4, 3, 2, 1):                                    # the prediction for p from ids[:p]
            if p - n >= 0:
                q = grams[n].get(tuple(ids[p - n:p]))
                if q is not None:
                    lk[p] = ids[q] == t; lkn[p] = n
                    break
        for n in (1, 2, 3, 4):                                    # register the n-grams that END at p - 1 ... after predicting p
            if p - n >= 0:
                grams[n][tuple(ids[p - n:p])] = p
        last[t] = p
    rc, ro, a0 = ch[f"rt_{fi}"], orc[f"rt_{fi}"], int(ch[f"a0_{fi}"])
    for i in range(rc.shape[0]):
        for j in range(31):
            c, o = int(rc[i, j]), int(ro[i, j]); pos = a0 + i + j + 2
            if c < 0 or o < 0 or pos >= N:
                continue
            keys = []
            if c == 0 and o == 1 and 7 <= j <= 22: keys.append("FIXED_7-22")
            if c == 0 and o == 1 and j >= 23: keys.append("FIXED_23-30")
            if c == 1 and j == 0: keys.append("HIT0")
            if c == 1 and j >= 23: keys.append("HIT23")
            if c == 0 and o == 0 and j >= 23: keys.append("MISS23")
            if j >= 23: keys.append("ALL23")
            for k in keys:
                d = cat[k]; d["n"] += 1; d["seen"] += int(seen[pos]); d["far"] += int(far[pos]); d["lookup"] += int(lk[pos])
                if lk[pos]: d["lk_n"][int(lkn[pos])] += 1
out = {}
for k, d in cat.items():
    if d["n"]:
        out[k] = {"cells": d["n"], "seen": d["seen"] / d["n"], "far": d["far"] / d["n"], "lookup": d["lookup"] / d["n"],
                  "lookup_by_n": {n: d["lk_n"][n] / d["n"] for n in (1, 2, 3, 4)}}
        print(f"T1 {k:12s} cells {d['n']:6d}: seen earlier {100 * out[k]['seen']:.1f} %, far (> 64 back) {100 * out[k]['far']:.1f} %, "
              f"prompt lookup right {100 * out[k]['lookup']:.1f} % (n=1/2/3/4: " + "/".join(f"{100 * out[k]['lookup_by_n'][n]:.1f}" for n in (1, 2, 3, 4)) + ")")
json.dump(out, open(H("~/BigMoE/data/deepdraft/bank/t1_copy.json"), "w"), indent=1)
