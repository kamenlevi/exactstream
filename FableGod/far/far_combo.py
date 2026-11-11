"""FAR combo (Fable FAR, Wed 2026-11-04): FH foresight + M-14's learned eviction key among the unforecast residents (instead
of LRU), on M-38's frame. M-14's features / fit copied from foresight45.py (leave-one-prompt-out, Belady labels on LRU
states). Forecasts: PERFECT-ALL (M-38's perfect H 8), PERFECT-KEPT (the kept rows' true routes), RIDGE-1 (far_ridge.py's
s 1, alpha 0.3, top-16, all 16 rows, oracle future x_1). Also r_m: recall of the forecast on the positives whose last use is
> 2 checks back (the residents LRU is about to lose) vs on the recent ones.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_combo.py (one core, < 3 GB, ~3 min) -> far_combo.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("far_frame", os.path.expanduser("~/FableGod/far/far_frame.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
P, NL, E, NI, H, BIG = F.P, F.NL, F.E, F.NI, F.H, F.BIG
ITEMS, PRE, PRM, PCHK, NXW = F.ITEMS, F.PRE, F.PRM, F.PCHK, F.NXW
OUT = os.path.expanduser("~/FableGod/far/far_combo.txt")
t0 = time.time()
lines = []
ITEM_OF = {(int(PRM[i]), int(PCHK[i])): i for i in range(NI) if PCHK[i] >= 0}
NF = 9


def say(s):
    print(s, flush=True); lines.append(s)


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
ROWS = [[] for _ in range(NL)]                                       # per layer, per item: (rows using each need / 16, kept-row use)
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


def replay(K, pol, nxt=None, model=None, collect=None, stats=None, only_p=None):
    """pol: lru, learn, fh (LRU among the unforecast), fhlearn (the learned key among the unforecast)"""
    miss = [np.zeros(len(c)) for _, c, _ in P]
    for l in range(NL):
        st = {"res": np.zeros(E, bool), "last": np.zeros(E), "score": np.zeros(E), "w": 1.0, "nrows": np.zeros(E),
              "kept": np.zeros(E, bool), "pre": np.zeros(E, bool)}
        pre_set = np.zeros(E, bool)
        for i, (p, t, need, pre) in enumerate(ITEMS[l]):
            nr, kr = ROWS[l][i]
            st["w"] /= 0.99
            if pre:
                pre_set = np.zeros(E, bool); pre_set[need] = True
            hit = st["res"][need]
            if t >= 0:
                miss[p][t] += int((~hit).sum())
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
                vict = cand[np.argsort(st["last"][cand], kind="stable")[:over]]
            elif pol == "learn":
                pred = feats(st, cand, i, l, p, pre_set) @ model
                vict = cand[np.argsort(-pred, kind="stable")[:over]]
            else:
                d = (nxt[l, i] - i)[cand].astype(float)
                fc = d <= H
                if stats is not None and (only_p is None or p == only_p):
                    y = NXW[l, i, cand] - i <= H
                    old = i - st["last"][cand] > 2
                    for key_, m_ in (("all", np.ones_like(y)), ("old", old), ("recent", ~old)):
                        stats[key_] = stats.get(key_, np.zeros(3, int)) + np.array([int((y & m_).sum()), int((y & fc & m_).sum()), int((fc & m_).sum())])
                if pol == "fhlearn":
                    rest = feats(st, cand, i, l, p, pre_set) @ model      # larger = farther next use = leaves first
                    key = np.where(fc, 1e9 - 1e3 * d, 0.0) - rest
                else:
                    key = np.where(fc, 1e9 - 1e3 * d, 0.0) + st["last"][cand]
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


def lopo_run(K, pol, models, nxt=None, stats=None):
    out = [np.zeros(len(c)) for _, c, _ in P]
    for p in range(1, len(P)):
        out[p] = replay(K, pol, nxt=nxt, model=models[p], stats=stats, only_p=p)[p]
    return out


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def ridge_fc(s=1, alpha=0.3, k=16):
    RW = []
    for p, (name, cks, _) in enumerate(P):
        z = np.load(os.path.join(F.CAP, name + "_gate_0.npz"))
        seq, L, rows, xoff = z["seq"], z["L"], z["rows"], z["x_off"]
        xs = z["x"]
        want = {}
        for n, (q, l, r) in enumerate(zip(seq, L, rows)):
            if int(l) == s and int(r) == 16 and int(q) not in want:
                want[int(q)] = int(xoff[n])
        X, Y = [], []
        for t, c in enumerate(cks):
            o = want[c["seq"]]
            X.append(bf16(xs[o:o + 16]))
            y = np.zeros((16, NL, E), bool)
            for l in range(NL):
                np.put_along_axis(y[:, l], c["rows"][l], True, axis=1)
            Y.append(y)
        X = np.concatenate(X)
        RW.append((np.concatenate([X, np.ones((len(X), 1), np.float32)], 1), np.concatenate(Y)))
    lam = alpha * float(np.mean([np.mean(np.sum(X * X, 1)) for X, _ in RW]))
    FC = np.zeros((NL, NI, E), bool)
    for p in range(len(P)):
        Xtr = np.concatenate([RW[q][0] for q in range(len(P)) if q != p])
        Ytr = np.concatenate([RW[q][1] for q in range(len(P)) if q != p]).reshape(len(Xtr), -1).astype(np.float32)
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32), Xtr.T @ Ytr)
        pred = (RW[p][0] @ W).reshape(len(RW[p][0]), NL, E)
        top = np.argsort(-pred, axis=2)[:, :, :k]
        for t in range(len(P[p][1])):
            i = ITEM_OF[(p, t)]
            for l in range(NL):
                FC[l, i, top[t * 16:(t + 1) * 16, l].ravel()] = True
        del Xtr, Ytr, W, pred
    return FC


say(f"FAR COMBO ({time.strftime('%a %Y-%m-%d %H:%M')}): FH foresight + the learned key among the unforecast residents; M-38's frame; H {H}")
FC_KEPT = np.zeros((NL, NI, E), bool)
for p, (_, cks, _) in enumerate(P):
    for t, c in enumerate(cks):
        i = ITEM_OF[(p, t)]
        for l in range(NL):
            FC_KEPT[l, i, c["rows"][l][:c["kept"]]] = True
NX_KEPT = F.next_from(FC_KEPT)
NX_RIDGE = F.next_from(ridge_fc())
say(f"  ridge forecast built ({time.time() - t0:.0f} s)")
for K in F.KS:
    lru = F.pooled(replay(K, "lru"))
    models = fit_models(K)
    lrn = F.pooled(lopo_run(K, "learn", models))
    say(f"K {K}: LRU {lru:.1f}, learned key {lrn:.1f} ({100 * (lru - lrn) / lru:.1f} % fewer) ({time.time() - t0:.0f} s)")
    for tag, nxt in (("PERFECT-ALL", NXW), ("PERFECT-KEPT", NX_KEPT), ("RIDGE-1 top16 all16", NX_RIDGE)):
        S = {}
        fh = F.pooled(replay(K, "fh", nxt=nxt, stats=S))
        fl = F.pooled(lopo_run(K, "fhlearn", models, nxt=nxt))
        rm = {k_: (v[1] / max(v[0], 1), v[1] / max(v[2], 1), v[0]) for k_, v in S.items()}
        say(f"  {tag:20s}: FH + LRU {fh:6.1f} ({100 * (lru - fh) / lru:5.1f} % fewer than LRU) | FH + learned key {fl:6.1f} "
            f"({100 * (lru - fl) / lru:5.1f} %) | recall / precision on positives: all {rm['all'][0]:.2f} / {rm['all'][1]:.2f}, "
            f"last use > 2 checks back {rm['old'][0]:.2f} / {rm['old'][1]:.2f} ({100 * rm['old'][2] / rm['all'][2]:.0f} % of positives), "
            f"recent {rm['recent'][0]:.2f} / {rm['recent'][1]:.2f} ({time.time() - t0:.0f} s)")
say(f"({time.time() - t0:.0f} s)")
open(OUT, "w").write("\n".join(lines) + "\n")
