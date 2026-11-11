"""Resident-only picker masks (Breakthrough, Tue 2026-11-03; the staged-verification lead of the BREAKTHROUGH PUSH, the
orchestrator 17:2x; pre-registered SCOREBOARD 4229db8). The cheap tree pass of a staged check reads nothing from the SSD:
every realized row of an HPD-0 check (A .. first miss) keeps only the experts resident in the base pool at its check's
start (all ranks; keep [rows, 48, 10] rank-sorted as DAY9's routing capture); prompt rows and rows outside the walk keep
all 10. The pool moves by the exact pass alone: each check then loads its union (all ranks of its rows; the rows after
the first miss enter by the wrong-draft stand-in p + 97, as dm_q1u_masks.py).
The LRU replay is NewIdeas' (~/NewIdeas/scripts/dm_oea_masks.py, the qom arms, Tue 17:0x), copied: per layer, pool
sizes = ~/MoonShots/data/laysz45_256.json C_m (the 45 GB default's base pool, mean 295 of 512), carried across a
domain's windows in walk order, the prompt rows' top-10 warm it, plain LRU (the engine adds a popularity prior).
Self-check (selfcheck): the same replay in NewIdeas' qom mode at theta 0 on live checks equals its qom_0_live masks bit
for bit (~/NewIdeas/data/dm_masks/qom_0_live).
Usage (BigMoE venv): python dm_res_masks.py [res | selfcheck] -> out/res_c256/<window>.npz"""
import json, os, sys
from collections import OrderedDict
import numpy as np
import dm_gate
from dm_q1u_masks import live_segments, OUT, base, G
CL = json.load(open(os.path.expanduser("~/MoonShots/data/laysz45_256.json")))["C_m"]

def touch(c, e, C):
    if e in c: c.move_to_end(e)
    else:
        c[e] = 1
        if len(c) > C: c.popitem(last=False)

def build(mode, out_dir, live=False, thr=None):
    """mode 'res': realized rows keep the experts resident at their check's start; 'qom' (thr): NewIdeas' Q1-ON-MISS rule
    (sure rows keep ranks 0-1 + the check's union members + what is resident after the check's loads)"""
    os.makedirs(out_dir, exist_ok=True); st = {}
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        cache = [OrderedDict() for _ in range(48)]               # the base pool per layer, carried across windows
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids)
            g = np.nan_to_num(G[nm], nan=-1.0); keep = np.ones((N, 48, 10), bool)
            s = st.setdefault(dom, np.zeros(6))                 # kept / all realized (row, layer, rank); rank-0 kept / all;
            for p in range(base[nm]["a"]):                      # loads, load misses
                for l in range(48):
                    for e in ids[p, l].tolist(): touch(cache[l], e, CL[l])
            for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
                for (k, m, A, dc) in (live_segments(k0_, m0_, A0_, dc0_) if live else [(k0_, m0_, A0_, dc0_)]):
                    last, fm = min(A + m, N - 1), min(A + k, N - 1)
                    rows, cfs = [], []
                    for p in range(A, last + 1):
                        if p <= fm: rows.append(p); cfs.append(-1.0 if live and k == m and p == A + m else g[p])
                        else: rows.append((p + 97) % N); cfs.append(dc[p - A] if p - A < m else -1.0)
                    if not rows: continue                        # a live segment past the window's end
                    k0 = np.full(len(rows), 10) if mode == "res" else np.where(np.array(cfs) >= thr, 2, 10)
                    for l in range(48):
                        c = cache[l]
                        if mode == "res":                        # resident at the check's start = what the cheap pass has
                            for p in range(A, fm + 1):
                                kp = np.fromiter((e in c for e in ids[p, l].tolist()), bool, 10)
                                keep[p, l] = kp; s[0] += kp.sum(); s[1] += 10; s[2] += kp[0]; s[3] += 1
                        U = np.unique(np.concatenate([ids[r, l, :k0[j]] for j, r in enumerate(rows)]))
                        for e in U.tolist(): s[4] += 1; s[5] += e not in c; touch(c, e, CL[l])
                        if mode == "qom":                        # members, then what is resident after the loads
                            res = np.fromiter(c.keys(), np.int64, len(c))
                            for j, p in enumerate(range(A, fm + 1)):
                                if k0[j] < 10:
                                    keep[p, l, k0[j]:] = np.isin(ids[p, l, k0[j]:], U) | np.isin(ids[p, l, k0[j]:], res)
                            for j, r in enumerate(rows):         # kept resident experts are used: touch them
                                if k0[j] < 10:
                                    for e in ids[r, l, k0[j]:].tolist():
                                        if e in c: c.move_to_end(e)
            np.savez_compressed(os.path.join(out_dir, nm), keep=keep)
    return st

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "res"
    if what == "selfcheck":
        od = os.path.join(OUT, "_selfcheck_qom_0_live"); build("qom", od, live=True, thr=0.0)
        ref = os.path.expanduser("~/NewIdeas/data/dm_masks/qom_0_live")
        bad = [f for f in sorted(os.listdir(od)) if not np.array_equal(np.load(os.path.join(od, f))["keep"],
                                                                         np.load(os.path.join(ref, f))["keep"])]
        print(f"self-check qom_0_live vs NewIdeas': {'ALL EQUAL (36)' if not bad else f'{len(bad)} windows DIFFER: {bad[:3]}'}")
    else:
        st = build("res", os.path.join(OUT, "res_c256"))
        for dom, s in st.items():
            print(f"res_c256 {dom}: realized row-layer experts kept {100 * s[0] / s[1]:.1f} %, rank 0 kept "
                  f"{100 * s[2] / s[3]:.1f} %; the exact passes' load miss rate {100 * s[5] / s[4]:.1f} % of {int(s[4])} loads")
