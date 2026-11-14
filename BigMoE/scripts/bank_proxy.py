#!/usr/bin/env python3
"""DEEPDRAFT PROXY screen (Sat 14 Nov 2026; the orchestrator 18:0x). A CPU filter, seconds to a minute, for candidate
predictors of the trunk's widened stream (wd) at drafted rows. It is never a verdict: the real check is the REPLACE eval
(dd_taps.py evalrep, ~3 min of GPU). README: data/deepdraft/bank/PROXY.md.

  bank_proxy.py cache                       build the memory-mapped proxy set from the bank (once, after the build)
  bank_proxy.py ridge --inputs tap45        a ridge over named inputs (tap5 tap15 tap25 tap35 tap45; comma-separated)
  bank_proxy.py custom --py my_map.py       a python file with fit(Xtr: dict, Ytr) -> model and predict(model, X: dict)
                                            -> Y; X maps every input name to a float32 array
Inputs are the raw taps of a row (the mean of the 4 HC streams after layer L), the row being predicted. The target is
that row's wd. Rows are answer rows (gen - 1 .. n - 1): the rows the head reads at drafted positions.
- Train rows: a fixed subsample (every 3rd answer row) of the bank's 360 train files.
- Val rows: every answer row of its 40 val files.
Output: held-out R2, the RMS-normalized relative (direction) error, and the predicted MBPP 23-30 from data/deepdraft/bank/
calibration.json, monotone piecewise-linear in the DIRECTION ERROR (the primary measure; Sandbox98's proxy_check). >= 88
predicted -> run the REPLACE eval.
Scope: only predictors that output a wd estimate. A learned stand-in lives in the head's own space (t30b's carried
stand-in has R2 -2.32 against wd yet drafts 72 % at depth), so it must be judged by the drafting eval itself.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np

H = os.path.expanduser
BANK = H("~/datasets/deepdraft/bank")
PC = os.path.join(BANK, "proxy")
SPLIT = H("~/BigMoE/data/deepdraft/bank/split.json")
CAL = H("~/BigMoE/data/deepdraft/bank/calibration.json")
TAPS = {"tap5": 0, "tap15": 1, "tap25": 2, "tap35": 3, "tap45": 4}
LINE = 88.0


def cmd_cache(a):
    s = json.load(open(SPLIT)); os.makedirs(PC, exist_ok=True)
    for part, names, step in (("tr", s["train"], 3), ("va", s["val"], 1)):
        R, W, meta = [], [], []
        for nm in names:
            z = np.load(os.path.join(BANK, nm))
            n, gen = z["raw"].shape[0], int(z["gen"])
            rows = np.arange(gen - 1, n)[::step]
            R.append(np.asarray(z["raw"])[rows]); W.append(np.asarray(z["wd"])[rows]); meta += [(nm, int(r)) for r in rows]
        R, W = np.concatenate(R), np.concatenate(W)
        np.save(os.path.join(PC, f"raw_{part}.npy"), R); np.save(os.path.join(PC, f"wd_{part}.npy"), W)
        json.dump(meta, open(os.path.join(PC, f"rows_{part}.json"), "w"))
        print(f"[proxy] {part}: {len(R)} rows from {len(names)} files -> {PC}", flush=True)


def load(part, names):
    raw = np.load(os.path.join(PC, f"raw_{part}.npy"), mmap_mode="r")
    return {nm: np.asarray(raw[:, TAPS[nm] * 2560:(TAPS[nm] + 1) * 2560], np.float32) for nm in names}, \
        np.load(os.path.join(PC, f"wd_{part}.npy"), mmap_mode="r")


def score(P, Y):
    Y = np.asarray(Y, np.float32); ym = Y.mean(0)
    r2 = 1 - float(((P - Y) ** 2).sum()) / float(((Y - ym) ** 2).sum())
    pn = P / np.sqrt((P ** 2).mean(-1, keepdims=True) + 1e-12); yn = Y / np.sqrt((Y ** 2).mean(-1, keepdims=True) + 1e-12)
    return r2, float(np.sqrt(((pn - yn) ** 2).sum() / (yn ** 2).sum()))


def structure(P, Y, k=64):
    """Sandbox98 (11-14 19:51): the head tolerates unbiased noise around the true state far better than map error at the
    same R2 (true W + noise at R2 0.51 -> 82.7 at MBPP 23-30, a ridge map at 0.44 -> 59.2). So beside R2 the proxy reports:
    shrink, the slope of (P - ym) on (Y - ym) (1 = unbiased; ridge maps shrink); r2_unshrunk, R2 after dividing that slope
    out; bias, the share of the squared error in the mean offset; r2_pc, R2 on the top-k principal directions of Y (most of
    the variance) and r2_rest on the remainder. With ~/datasets/deepdraft/ckpt/head_in.npz (the head's input norm weight +
    fc_hidden) it also reports r2_head, the R2 in the head's own fused input space."""
    Y = np.asarray(Y, np.float32); ym = Y.mean(0); Yc = Y - ym; Pc = P - ym; E = P - Y
    shrink = float((Pc * Yc).sum() / (Yc * Yc).sum())
    out = {"shrink": shrink, "r2_unshrunk": 1 - float(((Pc / max(shrink, 1e-6) - Yc) ** 2).sum()) / float((Yc ** 2).sum()),
           "bias": float((E.mean(0) ** 2).sum() * len(E) / max((E ** 2).sum(), 1e-12))}
    sub = Yc[:: max(1, len(Yc) // 4000)]
    _, _, Vt = np.linalg.svd(sub, full_matrices=False); V = Vt[:k].T                     # [10240, k]
    yp, ep = Yc @ V, E @ V
    out["r2_pc"] = 1 - float((ep ** 2).sum()) / float((yp ** 2).sum())
    out["r2_rest"] = 1 - float(((E - ep @ V.T) ** 2).sum()) / max(float(((Yc - yp @ V.T) ** 2).sum()), 1e-12)
    hf = os.path.join(H("~/datasets/deepdraft/ckpt"), "head_in.npz")
    if os.path.exists(hf):
        h = np.load(hf); g, F = h["norm_w"].astype(np.float32), h["fc_hidden"].astype(np.float32)   # [10240], [2560, 2560]
        hc, grouped = int(h["hc"]), bool(h["grouped"])
        def head_in(X):
            Xg = X.reshape(len(X), hc, -1) if grouped else X
            Xn = (Xg / np.sqrt((Xg ** 2).mean(-1, keepdims=True) + float(h["eps"]))).reshape(len(X), -1)
            Xn = Xn * (np.tile(g, hc) if g.shape[0] * hc == X.shape[1] else g)
            return (Xn.reshape(len(X), hc, -1) @ F.T).reshape(len(X), -1)
        HP, HY = head_in(P), head_in(Y); hm = HY.mean(0)
        out["r2_head"] = 1 - float(((HP - HY) ** 2).sum()) / float(((HY - hm) ** 2).sum())
    return out


def predict_acc(ne, family="ridge"):
    """the primary measure is the RMS-normalized direction error (Sandbox98's proxy_check, 11-14 20:1x: it orders all 7
    predictors monotonically against the read-outs, R2 doesn't): accuracy, made non-increasing in the error, interpolated"""
    if not os.path.exists(CAL):
        return None, "no calibration yet"
    pts = sorted([p for p in json.load(open(CAL))["points"] if p.get("family", "ridge") in (family, "oracle") and "ne" in p],
                 key=lambda p: p["ne"])
    if len(pts) < 2:
        return None, f"no direction-error calibration for the family {family}"
    xs = np.array([p["ne"] for p in pts]); ys = np.minimum.accumulate(np.array([p["mbpp_23_30"] for p in pts]))   # monotone
    return float(np.interp(ne, xs, ys)), f"{family}: {len(pts)} points on the direction error {xs.min():.2f}-{xs.max():.2f}"


def report(tag, r2, ne, t0, st=None, family="ridge"):
    acc, note = predict_acc(ne, family)
    verdict = "no calibration" if acc is None else ("REPLACE-worthy" if acc >= LINE else "below the line")
    print(f"[proxy] {tag}: held-out R2 {r2:.4f}, normalized error {ne:.4f}; predicted MBPP 23-30 "
          f"{'-' if acc is None else f'{acc:.1f}'} ({note}); {verdict} (line {LINE}); {time.time() - t0:.0f} s", flush=True)
    if st:
        print("[proxy]   structure: " + ", ".join(f"{k} {v:.4f}" for k, v in st.items()), flush=True)


def cmd_ridge(a):
    t0 = time.time(); names = a.inputs.split(",")
    Xtr, Ytr = load("tr", names); Xva, Yva = load("va", names)
    X = np.concatenate([Xtr[k] for k in names], 1); Xv = np.concatenate([Xva[k] for k in names], 1)
    Y = np.asarray(Ytr, np.float32)
    xm, ym = X.mean(0), Y.mean(0); Xc = X - xm
    XtX = Xc.T.astype(np.float64) @ Xc; XtY = Xc.T.astype(np.float64) @ (Y - ym)
    dg = float(np.mean(np.diag(XtX))); best = None
    for lf in ([a.lam] if a.lam else (1e-3, 1e-2, 1e-1)):
        W = np.linalg.solve(XtX + lf * dg * np.eye(X.shape[1]), XtY).astype(np.float32)
        r2, ne = score((Xv - xm) @ W + ym, Yva)
        if best is None or r2 > best[0]:
            best = (r2, ne, lf, W)
    r2, ne, lf, W = best
    st = structure((Xv - xm) @ W + ym, Yva)
    if a.save:
        np.savez(a.save, W=W, xm=xm, ym=ym, cols=np.concatenate([np.arange(TAPS[k] * 2560, (TAPS[k] + 1) * 2560) for k in names]))
    report(f"ridge {a.inputs} (lambda {lf:g} x mean diag)", r2, ne, t0, st, "ridge")


def cmd_custom(a):
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("cand", a.py); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    names = getattr(m, "INPUTS", list(TAPS))
    Xtr, Ytr = load("tr", names); Xva, Yva = load("va", names)
    model = m.fit(Xtr, np.asarray(Ytr, np.float32))
    Pv = np.asarray(m.predict(model, Xva), np.float32)
    r2, ne = score(Pv, Yva)
    report(f"custom {os.path.basename(a.py)} ({','.join(names)})", r2, ne, t0, structure(Pv, Yva), getattr(m, "FAMILY", "custom"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "ridge", "custom"])
    ap.add_argument("--inputs", default="tap45"); ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--save", default=None); ap.add_argument("--py", default=None)
    a = ap.parse_args()
    {"cache": cmd_cache, "ridge": cmd_ridge, "custom": cmd_custom}[a.cmd](a)
