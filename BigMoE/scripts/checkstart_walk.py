#!/usr/bin/env python3
"""Check starts in t30b's decode walk (Sun 15 Nov 2026 03:3x; the orchestrator's GOAL A wall + MoonShots' K-next spec).
From dd_ceiling's cells (fresh-state t30b chain: depth / right / rank / fi / row) and the bank's windows (ids / gen / tk / lp):
- the walk: kept_walk from row 1 (anchor s0 = gen - 1); a check at anchor s keeps its leading right drafts (k <= 31) + the
  bonus, the next check starts at s + k + 1;
- per check start: t30b's depth-0 right there, split by what ended the previous check (a miss at depth k / a full 31-draft
  accept), and the 125B's margin m (top-1 - top-2, nats) at the row it predicts from;
- the miss list for the K-next verify dump: per check that ended at a miss (depth k): t = s + k + 2 (the missed position),
  d_k = tk[t - 1][rank] (rank < 32; else skipped), b = ids[t] (the 125B's token there = the bonus), y = ids[t + 1] (the
  next check's depth-0 target), d_(k+1) / d_(k+2) where recoverable (for a 3-row branch), m at rows t - 1 and t, and
  whether t30b's depth 0 at the next check (anchor t - 1) is right.
usage: checkstart_walk.py heplus24|mbppbank -> data/deepdraft/ceiling/checkstart_<set>.json + kmiss_<set>.json"""
import json, os, sys
import numpy as np

H = os.path.expanduser
SET = sys.argv[1]
z = np.load(H(f"~/BigMoE/data/deepdraft/ceiling/cells_t30b_{SET}.npz"))
files = [str(f) for f in z["files"]]
D, R, RT, RK, FI = z["depth"], z["row"], z["right"], z["rank"], z["fi"]
starts = {"after_miss": [], "after_full": [], "first": []}   # (right, m) at each check start's depth 0
miss = []
for fi, fn in enumerate(files):
    b = np.load(H(f"~/datasets/deepdraft/bank/{fn}"))
    ids, gen = [int(x) for x in b["ids"]], int(b["gen"])
    lp, tk = np.asarray(b["lp"], np.float32), np.asarray(b["tk"])
    sel = FI == fi
    cell = {}                                            # (anchor s, depth j) -> (right, rank)
    for d, r, rt, rk in zip(D[sel], R[sel], RT[sel], RK[sel]):
        cell[(int(r) - int(d) - 1, int(d))] = (bool(rt), int(rk))
    s, prev = gen - 1, "first"
    while (s, 0) in cell:
        r0_, _ = cell[(s, 0)]
        mrow = s + 1                                     # depth 0 predicts ids[s + 2] from row s + 1
        starts[prev].append((r0_, float(lp[mrow, 0] - lp[mrow, 1]) if mrow < len(lp) else float("nan")))
        k = 0
        while k < 31 and cell.get((s, k), (False, 99))[0]:
            k += 1
        if k < 31 and (s, k) in cell:                    # the check ended at a miss at depth k
            t = s + k + 2
            rk = cell[(s, k)][1]
            if t + 1 < len(ids) and rk < 32:
                ex = []
                for e in (1, 2):                         # d_(k+1), d_(k+2) from the same anchor's chain, where recoverable
                    c = cell.get((s, k + e))
                    ex.append(int(tk[t - 1 + e][c[1]]) if c is not None and c[1] < 32 and t - 1 + e < len(tk) else None)
                nx = cell.get((t - 1, 0))
                miss.append(dict(file=fn, s=s, k=k, t=t, d=int(tk[t - 1][rk]), rank=rk, b=ids[t], y=ids[t + 1],
                                 d1=ex[0], d2=ex[1], m_miss=float(lp[t - 1, 0] - lp[t - 1, 1]),
                                 m_target=float(lp[t, 0] - lp[t, 1]) if t < len(lp) else None,
                                 t30b_next_d0_right=None if nx is None else bool(nx[0])))
            prev = "after_miss"
        else:
            prev = "after_full"
        s = s + k + 1
out = {}
for nm, v in starts.items():
    if v:
        a = np.array([x[0] for x in v]); m = np.array([x[1] for x in v])
        out[nm] = {"n": len(v), "depth0_acc": float(a.mean()), "m_median": float(np.nanmedian(m)),
                   "near_tie_share_m_lt_0.5": float(np.nanmean(m < 0.5))}
out["all_anchors_depth0_acc"] = float(RT[D == 0].mean())
out["misses_for_dump"] = len(miss)
out["misses_skipped_rank_outside_top32_note"] = "drafts outside the 125B's top-32 at the miss row are not recoverable (skipped)"
json.dump(out, open(H(f"~/BigMoE/data/deepdraft/ceiling/checkstart_{SET}.json"), "w"), indent=1)
json.dump(miss, open(H(f"~/BigMoE/data/deepdraft/ceiling/kmiss_{SET}.json"), "w"))
print(SET, json.dumps(out))
