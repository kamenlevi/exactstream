"""FAR hybrid (Fable FAR, Wed 2026-11-04 22:2x; Kamen's idea via the orchestrator): a NEAR and a FAR forecaster together, like
a tournament branch predictor. M-38's frame (chained cap45 replay, K 266 / 147), FH look-ahead H 8.
NEAR (next check only; a decision at item i protects what check i + 1 needs):
  NEAR-V1   the next check's true experts with per-use recall r 0.75 (V1's layer look-ahead level, 0.72-0.79 [K TRIED]),
            the dropped uses replaced by popularity false positives (M-38's noise model, drawn once per use)
  NEAR-P1   the next check's true experts (= M-38's perfect H 1; the bound of any next-check forecaster)
  NEAR-P2   the next two checks' true experts (perfect H 2)
FAR: RIDGE-1 top-16 on all rows of checks i + 1 .. i + 8 (far_ridge.py's stand-in: recall 0.70 / precision 0.64, -14.6 %).
Combinations: (a) TIER: near = protected, far = kept if room (farthest forecast first), the rest LRU or M-14's learned key;
(b) CHOOSER: per layer an online EMA of each forecaster's precision (near resolved one item later, far resolved at the
horizon), the better one takes the protected tier; (c) SCORE: s_e = beta * [near] + gamma^(d_far - 1) * [far within 8],
lowest score leaves first (LRU / learned key tiebreak).
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_hybrid.py (one core, < 3 GB, ~3 min) -> far_hybrid.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("far_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
spec2 = importlib.util.spec_from_file_location("far_combo", os.path.expanduser("~/FableGod/far/far_combo.py"))
P, NL, E, NI, H, BIG = F.P, F.NL, F.E, F.NI, F.H, F.BIG
ITEMS, PRE, PRM, PCHK, PEND, TRUE, NXW = F.ITEMS, F.PRE, F.PRM, F.PCHK, F.PEND, F.TRUE, F.NXW
OUT = os.path.expanduser("~/FableGod/far/far_hybrid.txt")
t0 = time.time()
lines = []
ITEM_OF = {(int(PRM[i]), int(PCHK[i])): i for i in range(NI) if PCHK[i] >= 0}


def say(s):
    print(s, flush=True); lines.append(s)


# ---- M-14's learned key (far_combo.py's copy) -----------------------------------------------------------------------
POP = []
for i in range(len(P)):
    pop, n = np.zeros((NL, E)), 0
    for j, (_, cks, _) in enumerate(P):
        if j != i:
            for c in cks:
                for l in range(NL):
                    pop[l, np.unique(c["rows"][l])] += 1
            n += len(cks)
    POP.append(pop / n)
POPI = np.stack([np.stack([POP[PRM[i]][l] for i in range(NI)]) for l in range(NL)]) + 1e-6
ROWS = [[] for _ in range(NL)]
for l in range(NL):
    for p, (_, cks, res0) in enumerate(P):
        ROWS[l].append((np.zeros(len(res0[l])), np.zeros(len(res0[l]), bool)))
        for c in cks:
            rw = c["rows"][l]
            need, cnt = np.unique(rw, return_counts=True)
            ROWS[l].append((cnt / 16, np.isin(need, rw[:c["kept"]])))


def feats(st, cand, i, l, p, pre_set):
    rec = np.log1p(i - st["last"][cand])
    dc = np.log1p(st["score"][cand] / st["w"])
    return np.stack([rec, dc, POP[p][l, cand], st["nrows"][cand], st["kept"][cand].astype(float),
                     st["pre"][cand].astype(float), pre_set[cand].astype(float), np.full(len(cand), l / NL),
                     np.ones(len(cand))], 1)


def fp_draw(T, w, drop, rng):
    g = np.log(w) - np.log(-np.log(rng.random(T.shape)))
    g[T] = -np.inf
    order = np.argsort(-g, axis=-1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(E), order.shape), axis=-1)
    return rank < drop[..., None]


def noisy_fc(r, seed):
    rng = np.random.default_rng(seed)
    keep = TRUE & (rng.random(TRUE.shape) < r)
    drop = (TRUE & ~keep).sum(-1)
    FC = keep | fp_draw(TRUE, POPI, drop, rng)
    FC[:, PRE] = False
    return FC


def replay(K, pol, near=None, far=None, model=None, collect=None, only_p=None, beta=1.5, gamma=0.85, rest="lru", near_h=2):
    """near / far: [48, NI, 512] bool per-check forecasts; the decision at item i sees near[:, i + 1] (and i + 2 for P2)
    and far[:, i + 1 .. i + 8] (within the prompt). pol: lru, learn, near, far, tier, chooser, score."""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    nxt_far = F.next_from(far) if far is not None else None
    nxt_near = F.next_from(near) if near is not None else None
    for l in range(NL):
        st = {"res": np.zeros(E, bool), "last": np.zeros(E), "score": np.zeros(E), "w": 1.0, "nrows": np.zeros(E),
              "kept": np.zeros(E, bool), "pre": np.zeros(E, bool)}
        pre_set = np.zeros(E, bool)
        ema = {"near": 0.5, "far": 0.5}                                  # (b): per-layer online precision estimates
        pend = []                                                       # (decision i, protected near mask, far mask)
        for i, (p, t, need, pre) in enumerate(ITEMS[l]):
            nr, kr = ROWS[l][i]
            st["w"] /= 0.99
            if pre:
                pre_set = np.zeros(E, bool); pre_set[need] = True
            hit = st["res"][need]
            if t >= 0:
                miss[p][t] += int((~hit).sum())
            if pol == "chooser" and pend:                               # resolve past forecasts against this item's needs
                nm0 = np.zeros(E, bool); nm0[need] = True
                keep_ = []
                for (i0, mn, mf, hits_f) in pend:
                    if i0 + 1 == i and mn.any():
                        ema["near"] = 0.9 * ema["near"] + 0.1 * float((mn & nm0).sum() / mn.sum())
                    hits_f |= (mf & nm0)
                    if i - i0 >= H or PRM[i] != PRM[i0]:
                        if mf.any():
                            ema["far"] = 0.9 * ema["far"] + 0.1 * float(hits_f.sum() / mf.sum())
                    else:
                        keep_.append((i0, mn, mf, hits_f))
                pend = keep_
            st["last"][need] = i
            st["score"][need] += st["w"]
            st["nrows"][need] = nr; st["kept"][need] = kr; st["pre"][need] = pre
            st["res"][need] = True
            over = int(st["res"].sum()) - K
            if collect is not None and t >= 0:
                cand = np.nonzero(st["res"])[0]
                cand = cand[~np.isin(cand, need)]
                if cand.size:
                    sel = cand if cand.size <= 64 else np.random.default_rng(i * 131 + l).choice(cand, 64, replace=False)
                    collect.append((p, feats(st, sel, i, l, p, pre_set), np.log1p([F.next_use(l, e, i) for e in sel.tolist()])))
            if over <= 0:
                continue
            nm = np.zeros(E, bool); nm[need] = True
            cand = np.nonzero(st["res"] & ~nm)[0]
            if cand.size < over:
                cand = np.nonzero(st["res"])[0]
            if pol == "lru":
                key = st["last"][cand]
            elif pol == "learn":
                key = -(feats(st, cand, i, l, p, pre_set) @ model)
            else:
                tail = st["last"][cand] if rest == "lru" else -(feats(st, cand, i, l, p, pre_set) @ model)
                dn = (nxt_near[l, i] - i)[cand] if near is not None else np.full(cand.size, BIG)
                df = (nxt_far[l, i] - i)[cand] if far is not None else np.full(cand.size, BIG)
                fn, ff = dn <= near_h, df <= H                           # near: within its own (1-2 check) horizon
                if pol == "near":
                    key = np.where(fn, 1e9 - 1e3 * dn, 0.0) + tail
                elif pol == "far":
                    key = np.where(ff, 1e9 - 1e3 * df, 0.0) + tail
                elif pol == "tier":
                    key = np.where(fn, 3e9, 0.0) + np.where(ff, 1e9 - 1e3 * df, 0.0) + tail
                elif pol == "chooser":
                    hi, lo = (fn, ff) if ema["near"] >= ema["far"] else (ff, fn)
                    dlo = df if lo is ff else dn
                    key = np.where(hi, 3e9, 0.0) + np.where(lo, 1e9 - 1e3 * np.where(lo, dlo, 0), 0.0) + tail
                    mn = np.zeros(E, bool); mn[cand[fn]] = True
                    mf = np.zeros(E, bool); mf[cand[ff]] = True
                    pend.append((i, mn, mf, np.zeros(E, bool)))
                else:                                                   # score
                    s = beta * fn + np.where(ff, gamma ** (np.where(ff, df, 1) - 1), 0.0)
                    key = 1e6 * s + tail / 1e6
            vict = cand[np.argsort(key, kind="stable")[:over]]
            st["res"][vict] = False
        while os.path.exists(F.PAUSE):
            time.sleep(20)
    return miss


def fit_models(K):
    smp = []
    replay(K, "lru", collect=smp)
    X = np.concatenate([x for _, x, _ in smp]); Y = np.concatenate([y for _, _, y in smp])
    PID = np.concatenate([np.full(len(y), q) for q, _, y in smp])
    return {p: np.linalg.lstsq(X[PID != p], Y[PID != p], rcond=None)[0] for p in range(1, len(P))}


def lopo(K, pol, models, **kw):
    out = [np.zeros(len(c)) for _, c, _ in P]
    for p in range(1, len(P)):
        out[p] = replay(K, pol, model=models[p], only_p=p, **kw)[p]
    return out


say(f"FAR HYBRID ({time.strftime('%a %Y-%m-%d %H:%M')}): NEAR + FAR forecasters on M-38's frame; H {H}; K {F.KS}")
# forecasts
mod = importlib.util.module_from_spec(spec2)
sys_argv = sys.argv; sys.argv = [sys.argv[0]]
src = open(os.path.expanduser("~/FableGod/far/far_combo.py")).read().split("say(f\"FAR COMBO")[0]
ns = {"__name__": "far_combo_part", "__file__": os.path.expanduser("~/FableGod/far/far_combo.py")}
exec(src, ns)
FAR_R = ns["ridge_fc"]()                                            # RIDGE-1 a0.3 top-16 all rows, per check [48, NI, 512]
say(f"  far ridge forecast built ({time.time() - t0:.0f} s)")
NEAR_P = TRUE.copy(); NEAR_P[:, PRE] = False                         # perfect next checks (P1 uses d <= 1, P2 d <= 2)
NEAR_V1 = noisy_fc(0.75, 4242)                                       # V1-level next-check forecast
res = {}
for K in F.KS:
    lru = F.pooled(replay(K, "lru"))
    models = fit_models(K)
    lrn = F.pooled(lopo(K, "learn", models))
    pct = lambda v: f"{100 * (lru - v) / lru:5.1f} %"
    say(f"K {K}: LRU {lru:.1f}; learned key {lrn:.1f} ({pct(lrn)}) ({time.time() - t0:.0f} s)")
    for tag, near, far, pol, rst, kw in (
            ("FAR ridge (8 checks) + LRU", None, FAR_R, "far", "lru", {}),
            ("FAR ridge + learned key", None, FAR_R, "far", "learn", {}),
            ("NEAR-V1 next 1 check (r 0.75) + LRU", NEAR_V1, None, "near", "lru", {"near_h": 1}),
            ("NEAR-V1 next 2 checks (r 0.75) + LRU", NEAR_V1, None, "near", "lru", {"near_h": 2}),
            ("NEAR-P1 perfect next check + LRU", NEAR_P, None, "near", "lru", {"near_h": 1}),
            ("NEAR-P2 perfect next 2 checks + LRU", NEAR_P, None, "near", "lru", {"near_h": 2}),
            ("(a) TIER V1(2) > FAR > LRU", NEAR_V1, FAR_R, "tier", "lru", {"near_h": 2}),
            ("(a) TIER V1(1) > FAR > LRU", NEAR_V1, FAR_R, "tier", "lru", {"near_h": 1}),
            ("(a) TIER V1(2) > FAR > learned key", NEAR_V1, FAR_R, "tier", "learn", {"near_h": 2}),
            ("(a) TIER P2 > FAR > learned key", NEAR_P, FAR_R, "tier", "learn", {"near_h": 2}),
            ("(b) CHOOSER per layer (V1(2), FAR) + LRU", NEAR_V1, FAR_R, "chooser", "lru", {"near_h": 2}),
            ("(c) SCORE b1.5 g0.85 (V1(2), FAR) + LRU", NEAR_V1, FAR_R, "score", "lru", {"beta": 1.5, "gamma": 0.85}),
            ("(c) SCORE b3 g0.7 (V1(2), FAR) + LRU", NEAR_V1, FAR_R, "score", "lru", {"beta": 3.0, "gamma": 0.7}),
            ("(c) SCORE b1.5 g0.85 (V1(2), FAR) + learned key", NEAR_V1, FAR_R, "score", "learn", {"beta": 1.5, "gamma": 0.85})):
        if rst == "learn":
            m = F.pooled(lopo(K, pol, models, near=near, far=far, rest="learn", **kw))
        else:
            m = F.pooled(replay(K, pol, near=near, far=far, **kw))
        res[(K, tag)] = m
        say(f"  K {K} {tag:46s}: {m:6.1f} misses / kept token, {pct(m)} fewer than LRU ({time.time() - t0:.0f} s)")
say(f"({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
