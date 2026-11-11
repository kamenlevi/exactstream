"""LOSSY350-FAR, where FAST's remaining demand misses come from (Fable FAR, Thu 2026-11-05 16:1x; CPU, numpy, one core, < 1.5 GB).
Frame: far_frame.py (cap45, prompts 2-6, per-layer LRU of K slots). Ranks: cap45's ids are re-sorted per row by the layer's own
router logits (routers.npz w<l>, recomputed from the captured x; layers without a router keep the captured order), after a
check of whether the captured order is already rank order.
QOM replay (today's FAST preset, simplified: ranks 0..KS-1 of every row are demand loads; other ranks computed only if
resident or co-loaded). For KS 2 and KS 1, rows = live (min(16, kept + 2)):
  demand misses per check split by the best rank at which any row needs the expert (0 / 1), and by who needs it at a demand
  rank: a KEPT row (j < kept), only the FIRST REJECTED row (j = kept: the bonus row's successor, its draft was wrong), only
  rows AFTER it; the stalled layers each class causes alone; cold demand terms per kept row (rank 0 / rank 1).
REACH variants (no cf in cap45: the oracle "kept" split bounds what a reach gate can reach):
  NOWAIT-REJ   misses needed only by rejected rows are not waited for (dropped for those rows, read in the background)
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_defer2.py -> far_defer2.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("far_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
P, NL, E = F.P, F.NL, F.E
OUT = os.path.expanduser("~/FableGod/far/far_defer2.txt")
A0, B = 0.27, 0.155
t0 = time.time()
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


# rank order: re-sort every check's ids by the router's logits
R = np.load(os.path.join(F.CAP, "routers.npz"))
RK = {int(k[1:]): k for k in R.keys() if k[0] == "w" and k[1:].isdigit()}
ok = tot = 0
for p, (name, cks, _) in enumerate(P):
    z = np.load(os.path.join(F.CAP, name + "_gate_0.npz"))
    seq, L, rows, xoff = z["seq"], z["L"], z["rows"], z["x_off"]
    first = {}
    for n in range(len(seq)):
        key = (int(seq[n]), int(L[n]))
        if key not in first:
            first[key] = n
    X = z["x"]
    for c in cks:
        for l in range(NL):
            if l not in RK:
                continue
            n = first[(c["seq"], l)]
            x = (X[xoff[n]:xoff[n] + 16].astype(np.uint32) << 16).view(np.float32)
            W = R[RK[l]].astype(np.float32)
            lg = x @ (W.T if W.shape[1] == x.shape[1] else W)
            g = np.take_along_axis(lg, c["rows"][l], 1)
            ok += int((np.diff(g, axis=1) <= 1e-3).all(1).sum()); tot += 16
            o = np.argsort(-g, axis=1, kind="stable")
            c["rows"][l] = np.take_along_axis(c["rows"][l], o, 1)
    del X, z
say(f"FAR DEFER2 ({time.strftime('%a %Y-%m-%d %H:%M')}): FAST's demand misses by rank and row class, cap45 frame (prompts 2-6)")
say(f"  rank check: the captured ids were already in router-logit order in {ok} / {tot} rows ({100 * ok / max(tot, 1):.1f} %); routers for "
    f"{len(RK)} layers (missing: {sorted(set(range(NL)) - set(RK))}); ids re-sorted by logits ({time.time() - t0:.0f} s)")


def run(K, KS, mode):
    res = np.zeros((NL, E), bool); last = np.zeros((NL, E))
    A = dict(checks=0, kept=0, rows=0, miss=0, m_r0=0, m_r1=0, m_kept=0, m_first=0, m_after=0, stl=0, stl_kept=0, block=0.0, block_kept=0.0,
             cold0=0, cold1=0, keptrows=0, drop=0.0, dropk=0.0)
    Wt = np.array([0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010]); Wt = Wt / Wt.sum()
    i = 0
    for p, (_, cks, res0) in enumerate(P):
        for t in range(-1, len(cks)):
            i += 1
            cnt = p >= 1 and t >= 0
            if t < 0:
                for l in range(NL):
                    need = res0[l]
                    res[l, need] = True; last[l, need] = i
                    over = int(res[l].sum()) - K
                    if over > 0:
                        cand = np.nonzero(res[l])[0]; cand = cand[~np.isin(cand, need)]
                        res[l, cand[np.argsort(last[l, cand], kind="stable")[:over]]] = False
                continue
            c = cks[t]; k = c["kept"]
            nr = min(16, k + 2)
            if cnt:
                A["checks"] += 1; A["kept"] += k; A["rows"] += nr; A["keptrows"] += min(k, nr)
            for l in range(NL):
                rw = c["rows"][l][:nr]
                dem = rw[:, :KS]
                d_all = np.unique(dem)
                cold = d_all[~res[l, d_all]]
                dk = np.unique(dem[:k]) if k > 0 else np.array([], int)
                cold_k = cold[np.isin(cold, dk)]                     # needed at a demand rank by a kept row
                cold_f = cold[~np.isin(cold, dk) & np.isin(cold, dem[k:k + 1])]
                cold_a = cold[~np.isin(cold, dk) & ~np.isin(cold, dem[k:k + 1])]
                waited = cold_k if mode == "NOWAIT-REJ" else cold
                avail = res[l].copy(); avail[waited] = True
                okm = avail[rw]
                if cnt:
                    A["miss"] += len(cold); A["m_kept"] += len(cold_k); A["m_first"] += len(cold_f); A["m_after"] += len(cold_a)
                    A["m_r0"] += int(np.isin(cold, rw[:, 0]).sum()); A["m_r1"] += int((~np.isin(cold, rw[:, 0])).sum())
                    A["stl"] += len(cold) > 0; A["stl_kept"] += len(cold_k) > 0
                    if len(cold):
                        A["block"] += A0 + B * len(cold)
                    if len(cold_k):
                        A["block_kept"] += A0 + B * len(cold_k)
                    cm = np.zeros(E, bool); cm[cold] = True
                    A["cold0"] += int(cm[rw[:k, 0]].sum())
                    if KS > 1:
                        A["cold1"] += int(cm[rw[:k, 1]].sum())
                    A["drop"] += float((Wt[None, :] * ~okm).sum()); A["dropk"] += float((Wt[None, :] * ~okm[:k]).sum())
                used = np.unique(rw[okm])
                last[l, used] = i
                res[l, cold] = True                                  # every cold demand expert is read (waited for or in the background)
                over = int(res[l].sum()) - K
                if over > 0:
                    nm = np.zeros(E, bool); nm[used] = True; nm[cold] = True
                    cand = np.nonzero(res[l] & ~nm)[0]
                    if cand.size < over:
                        cand = np.nonzero(res[l])[0]
                    res[l, cand[np.argsort(last[l, cand], kind="stable")[:over]]] = False
    return A


say("per K, KS (demand ranks), mode: demand misses per check (per kept token) | by best rank: rank 0 / rank 1 | by who needs it: a kept row / "
    "only the first rejected row / only later rows | stalled layers: all / from kept rows alone | blocking wait model ms per check: all / kept "
    "rows' misses alone | cold demand terms per kept row over 48 layers: rank 0 / rank 1 | dropped mass per row-layer: all rows / kept rows")
for K in F.KS:
    for KS in (2, 1):
        for mode in ("WAIT", "NOWAIT-REJ"):
            A = run(K, KS, mode)
            n = A["checks"]; m = max(A["miss"], 1)
            say(f"  K {K} KS {KS} {mode:10s}: miss {A['miss'] / n:5.1f} ({A['miss'] / A['kept']:4.1f}) | r0 {100 * A['m_r0'] / m:4.1f} % / r1 {100 * A['m_r1'] / m:4.1f} % | "
                f"kept {100 * A['m_kept'] / m:4.1f} % / first rej {100 * A['m_first'] / m:4.1f} % / later {100 * A['m_after'] / m:4.1f} % | stl {A['stl'] / n:4.1f} / "
                f"{A['stl_kept'] / n:4.1f} | block {A['block'] / n:5.1f} / {A['block_kept'] / n:5.1f} ms | cold per kept row r0 {A['cold0'] / max(A['keptrows'], 1):.2f} / r1 "
                f"{A['cold1'] / max(A['keptrows'], 1):.2f} | drop {A['drop'] / (A['rows'] * NL):.4f} / {A['dropk'] / max(A['keptrows'] * NL, 1):.4f} ({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
