"""LOSSY350-FAR, the late top-2 terms on the DECISION MAP's own windows, by domain and drafter confidence (Fable FAR, Thu
2026-11-05 16:2x; CPU, numpy, one core, < 1 GB, seconds).
Data (read-only): Breakthrough's DAY9 routing capture ~/FableGod/decision_map/out/moonshots/*.npz, one per map window (36):
ids [N, 48, 10] teacher-forced rank-sorted expert ids of every row, conf [N] = DFlash2's cf of the draft that row verifies
(NaN = no draft), checks [[a, n, k]] = the walk's checks after the engine's running-cf cut at 0.3 (start row, rows, kept),
gen = the first generated row (loader as MoonShots' band45.py load_capture). These are the windows QOM's map flips
(qom_0_live_bq 0.39 / 4.09 / 3.26 %) were measured on.
Replay: per domain, the windows chained; per-layer LRU of K slots; the prompt rows as prefill chunks (all 10 ranks loaded,
uncounted); each check under QOM's rule (ranks 0-1 of every row are demand loads; ranks 2-9 computed only if resident or
co-loaded). Rows j < kept are on the true path (kept rows); rows j >= kept carry a wrong draft, stand-in = row (p + 97) mod N
of the same window (dm_q1u_masks' convention).
Per domain and K: rows and kept per check; demand misses per check / per kept token; stalled layers; the blocking wait
model (0.27 ms per stalled layer + 0.155 ms per miss); cold rank-0 / rank-1 terms per kept row; the split of the kept rows'
cold top-2 terms by the row's cf band (no draft / < 0.5 / 0.5-0.9 / 0.9-0.99 / >= 0.99) beside the band's share of kept rows;
and the wait that stays if only the kept rows with cf < 0.9 or no draft are waited for (SURE-NW), for ranks 0-1 and for
rank 0 only.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_defer3.py [K ...] -> far_defer3.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
D = os.path.expanduser("~/FableGod/decision_map/out/moonshots")
OUT = os.path.expanduser("~/FableGod/far/far_defer3.txt")
PAUSE = os.path.expanduser("~/.local/share/quiet/PAUSE")
NL, E = 48, 512
A0, B = 0.27, 0.155
KS_ = tuple(int(a) for a in sys.argv[1:]) or (266, 147)
BANDS = ("no draft", "< 0.5", "0.5-0.9", "0.9-0.99", ">= 0.99")
t0 = time.time()
lines = []
if os.path.exists(PAUSE):
    sys.exit("PAUSE set: not running")


def say(s):
    print(s, flush=True); lines.append(s)


def band(c):
    if not np.isfinite(c):
        return 0
    return 1 if c < 0.5 else 2 if c < 0.9 else 3 if c < 0.99 else 4


W = {}
for f in sorted(x for x in os.listdir(D) if x.endswith(".npz")):
    z = np.load(os.path.join(D, f), allow_pickle=False)
    W.setdefault(str(z["domain"]), []).append((f[:-4], z["ids"].astype(np.int64), z["conf"].astype(float), int(z["gen"]),
                                               [tuple(int(v) for v in c) for c in z["checks"].tolist()]))


def evict(res, last, keep, K):
    over = int(res.sum()) - K
    if over > 0:
        cand = np.nonzero(res & ~keep)[0]
        if cand.size < over:
            cand = np.nonzero(res)[0]
        res[cand[np.argsort(last[cand], kind="stable")[:over]]] = False


def run(dom, K):
    res = np.zeros((NL, E), bool); last = np.zeros((NL, E))
    A = dict(checks=0, rows=0, kept=0, miss=0, stl=0, block=0.0, c0=0, c1=0, keptrows=0, m_sure=0, stl_un=0, block_un=0.0,
             block_un0=0.0, stl_un0=0, drop=0.0)
    cb = np.zeros((2, 5)); rb = np.zeros(5)
    Wt = np.array([0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010]); Wt = Wt / Wt.sum()
    i = 0
    for name, ids, conf, gen, checks in W[dom]:
        N = len(ids)
        for s in range(0, gen - 1, 128):                             # the prompt: prefill chunks, every rank loaded
            i += 1
            ch = ids[s:min(s + 128, gen - 1)]
            for l in range(NL):
                need = np.unique(ch[:, l])
                keep = np.zeros(E, bool); keep[need] = True
                res[l, need] = True; last[l, need] = i
                evict(res[l], last[l], keep, K)
        for a, n, k in checks:
            n = min(n, N - a); k = min(k, n)                         # a window's last check may run past its rows (band45 truncates too)
            if n <= 0:
                continue
            i += 1
            rows = np.array([a + j if j < k else (a + j + 97) % N for j in range(n)])
            bj = [band(conf[a + j]) for j in range(n)]
            unsure = np.array([j < k and bj[j] <= 2 for j in range(n)])   # kept rows with cf < 0.9 or no draft
            A["checks"] += 1; A["rows"] += n; A["kept"] += k; A["keptrows"] += min(k, n)
            for j in range(min(k, n)):
                rb[bj[j]] += 1
            for l in range(NL):
                rw = ids[rows, l]
                dem = rw[:, :2]
                d_all = np.unique(dem)
                cold = d_all[~res[l, d_all]]
                cm = np.zeros(E, bool); cm[cold] = True
                avail = res[l] | cm
                okm = avail[rw]
                A["miss"] += len(cold); A["stl"] += len(cold) > 0
                A["drop"] += float((Wt[None, :] * ~okm).sum())
                if len(cold):
                    A["block"] += A0 + B * len(cold)
                    cu = cold[np.isin(cold, dem[unsure])] if unsure.any() else cold[:0]
                    cu0 = cold[np.isin(cold, dem[unsure, 0])] if unsure.any() else cold[:0]
                    if len(cu):
                        A["block_un"] += A0 + B * len(cu); A["stl_un"] += 1
                    if len(cu0):
                        A["block_un0"] += A0 + B * len(cu0); A["stl_un0"] += 1
                    for j in range(min(k, n)):
                        if cm[rw[j, 0]]:
                            cb[0, bj[j]] += 1; A["c0"] += 1
                        if cm[rw[j, 1]]:
                            cb[1, bj[j]] += 1; A["c1"] += 1
                used = np.unique(rw[okm])
                keep = np.zeros(E, bool); keep[used] = True; keep[cold] = True
                last[l, used] = i
                res[l, cold] = True
                evict(res[l], last[l], keep, K)
    return A, cb, rb


say(f"FAR DEFER3 ({time.strftime('%a %Y-%m-%d %H:%M')}): QOM's cold top-2 terms on the decision map's 36 windows (Breakthrough's DAY9 capture), "
    f"per domain; windows: " + ", ".join(f"{d} {len(v)}" for d, v in sorted(W.items())))
for K in KS_:
    for dom in sorted(W):
        A, cb, rb = run(dom, K)
        n = A["checks"]; kr = max(A["keptrows"], 1)
        say(f"K {K} {dom:7s}: checks {n}, rows {A['rows'] / n:4.1f}, kept {A['kept'] / n:4.1f} | demand misses {A['miss'] / n:5.1f} per check "
            f"({A['miss'] / max(A['kept'], 1):4.1f} per kept token), stalled layers {A['stl'] / n:4.1f}, wait model {A['block'] / n:5.1f} ms | cold terms per "
            f"kept row: rank 0 {A['c0'] / kr:.2f}, rank 1 {A['c1'] / kr:.2f} | dropped mass per row-layer {A['drop'] / (A['rows'] * NL):.4f}")
        tot = max(cb.sum(), 1)
        say("     kept rows' cold top-2 terms by the row's cf band (share of the terms | share of kept rows | cold top-2 terms per row of the band): "
            + "; ".join(f"{b} {100 * cb[:, q].sum() / tot:4.1f} % | {100 * rb[q] / max(rb.sum(), 1):4.1f} % | {cb[:, q].sum() / max(rb[q], 1):.2f}"
                        for q, b in enumerate(BANDS)))
        say(f"     SURE-NW (wait only for kept rows with cf < 0.9 or no draft): wait model {A['block_un'] / n:5.1f} ms ({100 * A['block_un'] / max(A['block'], 1e-9):.0f} % "
            f"of today's), stalled layers {A['stl_un'] / n:4.1f}; and rank 0 only: {A['block_un0'] / n:5.1f} ms ({100 * A['block_un0'] / max(A['block'], 1e-9):.0f} %), "
            f"{A['stl_un0'] / n:4.1f} layers ({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
