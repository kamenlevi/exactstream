#!/usr/bin/env python3
"""EARLY DEATH stage 2: the TREE probe pre-registered in research/week98/EARLYDEATH_PREREG.md (NewIdeas; written Mon 16 Nov
21:2x, before any capture data; run Wed on BigMoE's Tue-night capture).

Question: at layer L of a tree check, can the 125B's partial state rank each spine node's children well enough to drop
the wrong ones from layers L+1..48?

Units (prereg): BigMoE's real 127-node k8 dumps (tree_t30b_{heplus24,mbppbank}_n127k8.npz). A unit = the root (row an+1)
or an ON-TRUE node at depth d (row p = an+2+d; its token = ids[p], checked), with >= 2 children. Children's tokens sit at
p+1 and are scored from the state at row p. A unit whose true next token is not among its children has no false drop.
Probes per child (scores comparable across units):
  P-draft   log of the drafter's edge probability (path_p(child) / path_p(parent); root: path_p), as a gap to the
            unit's best child (the same within-unit form as the lens probes).
  P-lens    the 125B's own head (ed_lens.Lens: final hyper-connection mixer + lm_head) on the layer-L widened state; score =
            the child's logit minus the best child's logit (<= 0).
  P-tuned   a ridge map layer-L state (10,240) -> the head's 2,560-wide input computed from the FINAL state (alpha by
            inner CV), cross-fitted by file parity, then lm_head; the same gap score.
  Note: a gap score never drops a unit's best child, so units whose true token is absent keep one child (conservative).
  P-comb    logistic on [P-tuned gap, P-draft, P-lens gap], cross-fitted by file parity.
Metric: the threshold at which <= 2 % of units with an on-true child drop it; at it, the share of OFF-spine children
dropped. Reported pooled (threshold from all units) and CROSS-FIT (threshold from the other parity's files), with a
95 % file-bootstrap CI (1,000 draws).
LINES (Fable B's, layer 16, best held-out non-oracle probe): PASS >= 90 % droppable at <= 2 % false drops, and P-tuned or
P-comb beats P-draft by >= 10 pt; KILL < 60 %; BETWEEN otherwise. The chain arm is information only (amendment 2).

Modes:
  --standin final   the bank's FINAL widened stream stands in for the layer (no capture needed). P-lens is then the
                    125B itself, so it must drop ~all off-spine children at ~0 false drops: the pipeline self-check.
  --capture PATH    BigMoE's mid_<tag>.npz (fields fi / row / layer + one [N, 10240] state array), layers 8 / 16 / 24.
Usage: taskpolicy -b ~/BigMoE/.venv/bin/python -I -B ed_stage2.py --set he --standin final
CPU only, one process, <= 4 GB.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ed_lens import Lens  # noqa: E402

H = os.path.expanduser("~")
CE = f"{H}/BigMoE/data/deepdraft/ceiling"
BANK = f"{H}/datasets/deepdraft/bank"
TAGS = {"he": ("heplus24", "tf_code_HumanEval_*.npz"), "mbpp": ("mbppbank", "dlog__mbpp*.npz")}
FD, RNG = 0.02, np.random.default_rng(16)


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


LINES = []


def load_units(tag, pat):
    z = np.load(f"{CE}/tree_t30b_{tag}_n127k8.npz", allow_pickle=True)
    files = [str(f) for f in z["files"]]
    bank = {os.path.basename(f): f for f in glob.glob(f"{BANK}/{pat}")}
    ids = [np.load(bank[f])["ids"] for f in files]
    tk0 = [np.load(bank[f])["tk"][:, 0] for f in files]
    N = int(z["nodes"])
    fi, an, dp, tok, ot, par, pp, nd = (z[k] for k in ("fi", "an", "depth", "token", "on_true", "parent", "path_p", "node"))
    o = np.lexsort((nd, an, fi))                           # group by tree (fi, an), node index within
    T = len(fi) // N
    fi, an, dp, tok, ot, par, pp, nd = (a[o].reshape(T, N) for a in (fi, an, dp, tok, ot.astype(bool), par, pp, nd))
    assert (fi == fi[:, :1]).all() and (an == an[:, :1]).all() and (nd == np.arange(N)).all(), "tree grouping failed"
    U = {k: [] for k in ("fi", "row", "unit", "tok", "on", "edge", "arg")}   # one entry per CHILD; 'unit' groups children
    u = 0; chk = [0, 0]
    for t in range(T):
        f_, a_ = int(fi[t, 0]), int(an[t, 0])
        kids = {}
        for j in range(N):
            kids.setdefault(int(par[t, j]), []).append(j)
        parents = [(-1, a_ + 1, 1.0)] + [(j, a_ + 2 + int(dp[t, j]), float(pp[t, j])) for j in range(N) if ot[t, j]]
        for pj, row, ppar in parents:
            if pj >= 0:
                chk[0] += int(ids[f_][row] == tok[t, pj]) if row < len(ids[f_]) else 0; chk[1] += 1
            ch = kids.get(pj, [])
            if len(ch) < 2 or row >= len(ids[f_]) - 1:
                continue
            for c in ch:
                U["fi"].append(f_); U["row"].append(row); U["unit"].append(u); U["tok"].append(int(tok[t, c]))
                U["on"].append(bool(ot[t, c])); U["edge"].append(float(np.log(max(pp[t, c] / max(ppar, 1e-30), 1e-30))))
                U["arg"].append(bool(row < len(tk0[f_]) and int(tok[t, c]) == int(tk0[f_][row])))
            u += 1
    U = {k: np.asarray(v) for k, v in U.items()}
    say(f"[{tag}] {T} trees, {u} units (root + on-true parents with >= 2 children), {len(U['tok'])} children, "
        f"{int(U['on'].sum())} on-true; alignment ids[an+2+d] == token on {chk[0]}/{chk[1]} on-true parents")
    assert chk[0] >= 0.999 * chk[1], "alignment check failed"
    on_ne_arg = int((U["on"] & ~U["arg"]).sum()); arg_ne_on = int((U["arg"] & ~U["on"]).sum())
    say(f"  truth labels: on-true children that are NOT the 125B's stored argmax {on_ne_arg} of {int(U['on'].sum())}; "
        f"argmax children that are not on-true {arg_ne_on} (bank text vs tk[row, 0])")
    return files, ids, U


def states_for(mode, files, rows_by_file, layer=None, cap=None):
    """{fi: {row: state[10240] float32}} for the requested rows"""
    out = {}
    if mode == "final":
        bank = {os.path.basename(f): f for f in glob.glob(f"{BANK}/*.npz")}
        for f_, rows in rows_by_file.items():
            wd = np.load(bank[files[f_]])["wd"]
            out[f_] = {r: wd[r].astype(np.float32) for r in rows}
        return out
    sel = cap["layer"] == layer
    for f_, rows in rows_by_file.items():
        m = sel & (cap["fi"] == f_)
        idx = dict(zip(cap["row"][m].tolist(), np.nonzero(m)[0].tolist()))
        out[f_] = {r: cap["X"][idx[r]].astype(np.float32) for r in rows if r in idx}
    return out


def child_gaps(L, M, ru, U):
    """M [n_rows, 2560] = the head's input per unique row; ru[unit] = the unit's row index. Score per child = its logit
    minus the best child's logit in its unit (<= 0)."""
    lg = np.empty(len(U["tok"]))
    for a in range(0, len(lg), 8192):
        sl = slice(a, a + 8192)
        lg[sl] = np.einsum("nd,nd->n", M[ru[U["unit"][sl]]], L.rows(U["tok"][sl]))
    best = np.full(U["unit"].max() + 1, -np.inf)
    np.maximum.at(best, U["unit"], lg)
    return lg - best[U["unit"]]


def ridge_dual(Xtr, Ytr, Xte, alphas=(1e-3, 1e-2, 1e-1, 1.0), val=None):
    """Ridge in DUAL form (rows << 10,240 features): pred = Xte Xtr^T (Xtr Xtr^T + s I)^-1 Ytr, s = alpha * mean diag.
    alpha by inner CV on `val` (a boolean mask over Xtr's rows: fit on ~val, score MSE on val), then refit on all."""
    def pred(Xa, Ya, Xb, a):
        K = Xa @ Xa.T; s_ = a * np.trace(K) / len(K)
        K[np.diag_indices_from(K)] += s_
        return (Xb @ Xa.T) @ np.linalg.solve(K, Ya)
    best = None
    if val is not None and val.any() and (~val).any():
        for a in alphas:
            e = float(((pred(Xtr[~val], Ytr[~val], Xtr[val], a) - Ytr[val]) ** 2).mean())
            if best is None or e < best[1]:
                best = (a, e)
    a = best[0] if best else alphas[1]
    return pred(Xtr, Ytr, Xte, a).astype(np.float32), a


def logistic(F, y, iters=30, l2=1e-3):
    w = np.zeros(F.shape[1] + 1); Xb = np.c_[F, np.ones(len(F))]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + l2 * w; Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        w -= np.linalg.solve(Hm, g)
    return w


TRUTH = "arg"                                              # amendment 3: the 125B's stored argmax at the unit's row


def metric(score, U, files_mask=None, tau=None):
    m = np.ones(len(score), bool) if files_mask is None else files_mask[U["fi"]]
    on = m & U[TRUTH]; off = m & ~U[TRUTH]
    if tau is None:
        tau = np.quantile(score[on], FD) if on.any() else -np.inf   # <= 2 % of on-true children fall below tau
    return float((score[off] < tau).mean()) if off.any() else float("nan"), float((score[on] < tau).mean()), tau


def report(name, score, U, nf):
    global TRUTH
    par = (np.arange(nf) % 2 == 0)
    TRUTH = "on"; info = metric(score, U); TRUTH = "arg"
    share, fdr, tau = metric(score, U)
    ca = metric(score, U, ~par, metric(score, U, par)[2])          # threshold from even files, scored on odd
    cb = metric(score, U, par, metric(score, U, ~par)[2])
    boots = []
    for _ in range(1000):
        w = np.bincount(RNG.integers(0, nf, nf), minlength=nf)
        rep = np.repeat(np.arange(len(score)), w[U["fi"]])
        if not len(rep):
            continue
        Ub = {TRUTH: U[TRUTH][rep], "fi": U["fi"][rep]}
        boots.append(metric(score[rep], Ub)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    xs = (ca[0] + cb[0]) / 2
    say(f"  {name:22s} droppable {100 * share:5.1f} % [{100 * lo:5.1f}, {100 * hi:5.1f}] at false drops {100 * fdr:4.2f} % "
        f"(tau {tau:.3f}); cross-fit threshold: {100 * xs:5.1f} % at false drops {100 * (ca[1] + cb[1]) / 2:4.2f} % | "
        f"bank-text truth (info): {100 * info[0]:5.1f} % at {100 * info[1]:4.2f} %")
    return {"share": share, "ci": [lo, hi], "fdr": fdr, "tau": float(tau), "xfit_share": xs, "xfit_fdr": (ca[1] + cb[1]) / 2,
            "banktext_share": info[0], "banktext_fdr": info[1]}


def run(a):
    tag, pat = TAGS[a.set]
    files, ids, U = load_units(tag, pat)
    nf = len(files)
    urow = np.zeros(U["unit"].max() + 1, np.int64); ufi = np.zeros_like(urow)
    urow[U["unit"]] = U["row"]; ufi[U["unit"]] = U["fi"]
    rows_by_file = {}
    for f_, r in zip(ufi.tolist(), urow.tolist()):
        rows_by_file.setdefault(f_, set()).add(r)
    L = Lens()
    res = {"set": a.set, "mode": a.standin or a.capture, "units": int(U["unit"].max() + 1), "children": int(len(U["tok"]))}
    eb = np.full(U["unit"].max() + 1, -np.inf); np.maximum.at(eb, U["unit"], U["edge"])
    U["edge_gap"] = U["edge"] - eb[U["unit"]]              # the same within-unit gap form as the lens probes
    res["P-draft"] = report("P-draft (edge gap)", U["edge_gap"], U, nf)
    cap = None
    if a.capture:
        z = np.load(a.capture)
        # data access only (Tue 17 Nov 08:0x, BigMoE's note): each z[k] access re-reads the member, so name the state
        # array directly and read it ONCE (MBPP's X is 2.56 GB); no analysis change.
        key = "X" if "X" in z.files else [k for k in z.files if k not in ("fi", "row", "layer")][0]
        cap = {"fi": z["fi"], "row": z["row"], "layer": z["layer"], "X": z[key]}
        assert cap["X"].ndim == 2 and cap["X"].shape[1] == 10240, f"capture state array {key}: {cap['X'].shape}"
    layers = ["final"] if a.standin == "final" else [int(x) for x in a.layers.split(",")]
    key = ufi * 1_000_000 + urow                           # one state per unique (file, row); units share rows
    uk, ru = np.unique(key, return_inverse=True)
    rfi, rrow = uk // 1_000_000, uk % 1_000_000
    fin = states_for("final", files, rows_by_file)
    Yf = np.stack([fin[int(f_)][int(r)] for f_, r in zip(rfi, rrow)])   # the WIDENED final state per row (P-tuned's target)
    del fin
    res["rows"] = int(len(uk))
    for ly in layers:
        if ly == "final":
            X = Yf                                          # the stand-in: the layer IS the final state (no copy)
        else:
            S = states_for("cap", files, rows_by_file, ly, cap)
            have = np.array([int(r) in S.get(int(f_), {}) for f_, r in zip(rfi, rrow)])
            if not have.all():
                say(f"  layer {ly}: {int((~have).sum())} rows lack a state (their units score as the zero state)")
            X = np.stack([S[int(f_)][int(r)] if h else np.zeros(10240, np.float32) for f_, r, h in zip(rfi, rrow, have)])
            del S
        say(f"[{a.set}] layer {ly}: {len(X)} rows")
        g_lens = child_gaps(L, L.mixed(X), ru, U)
        res[f"P-lens L{ly}"] = report(f"P-lens L{ly}", g_lens, U, nf)
        par_r = (rfi % 2 == 0)
        Mt = np.empty((len(X), 2560), np.float32); alphas = {}
        for half in (True, False):                          # fit on one parity's rows, predict the other's (widened)
            tr = par_r == half
            Rhat, al = ridge_dual(X[tr], Yf[tr] - X[tr], X[~tr], val=(rfi[tr] % 4 < 2))
            Yhat = X[~tr] + Rhat                            # residual form: identity + correction (amendment 3)
            alphas["even" if half else "odd"] = al
            Mt[~tr] = L.mixed(Yhat)                         # then the 125B's real mixer, then lm_head (prereg form)
        g_tuned = child_gaps(L, Mt, ru, U)
        res[f"P-tuned L{ly}"] = report(f"P-tuned L{ly}", g_tuned, U, nf)
        res[f"P-tuned L{ly}"]["alpha"] = alphas
        F = np.c_[g_tuned, U["edge_gap"], g_lens]
        comb = np.empty(len(F)); par_c = (U["fi"] % 2 == 0)
        for half in (True, False):
            w = logistic(F[par_c == half], U[TRUTH][par_c == half].astype(float))
            comb[par_c != half] = np.c_[F[par_c != half], np.ones(int((par_c != half).sum()))] @ w
        res[f"P-comb L{ly}"] = report(f"P-comb L{ly}", comb, U, nf)
    out = os.path.join(HERE, f"ed_stage2_{a.set}_{'standin' if a.standin else 'capture'}")
    json.dump(res, open(out + ".json", "w"), indent=1, default=float)
    open(out + ".txt", "w").write("\n".join(LINES) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="he", choices=list(TAGS))
    ap.add_argument("--standin", default="", choices=["", "final"])
    ap.add_argument("--capture", default="")
    ap.add_argument("--layers", default="8,16,24")
    a = ap.parse_args()
    t0 = time.time()
    run(a)
    say(f"wall {time.time() - t0:.0f} s")
