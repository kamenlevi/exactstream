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
Output: held-out R2, the RMS-normalized relative error, and the predicted MBPP 23-30 from data/deepdraft/bank/
calibration.json (monotone piecewise-linear in R2, flat beyond the ends). >= 88 predicted -> run the REPLACE eval.
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


def predict_acc(r2):
    if not os.path.exists(CAL):
        return None, "no calibration yet"
    pts = sorted(json.load(open(CAL))["points"], key=lambda p: p["r2"])
    xs = np.array([p["r2"] for p in pts]); ys = np.maximum.accumulate(np.array([p["mbpp_23_30"] for p in pts]))   # monotone
    return float(np.interp(r2, xs, ys)), f"{len(pts)} calibration points, R2 {xs.min():.2f}-{xs.max():.2f}"


def report(tag, r2, ne, t0):
    acc, note = predict_acc(r2)
    verdict = "no calibration" if acc is None else ("REPLACE-worthy" if acc >= LINE else "below the line")
    print(f"[proxy] {tag}: held-out R2 {r2:.4f}, normalized error {ne:.4f}; predicted MBPP 23-30 "
          f"{'-' if acc is None else f'{acc:.1f}'} ({note}); {verdict} (line {LINE}); {time.time() - t0:.0f} s", flush=True)


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
    if a.save:
        np.savez(a.save, W=W, xm=xm, ym=ym, cols=np.concatenate([np.arange(TAPS[k] * 2560, (TAPS[k] + 1) * 2560) for k in names]))
    report(f"ridge {a.inputs} (lambda {lf:g} x mean diag)", r2, ne, t0)


def cmd_custom(a):
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("cand", a.py); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    names = getattr(m, "INPUTS", list(TAPS))
    Xtr, Ytr = load("tr", names); Xva, Yva = load("va", names)
    model = m.fit(Xtr, np.asarray(Ytr, np.float32))
    r2, ne = score(np.asarray(m.predict(model, Xva), np.float32), Yva)
    report(f"custom {os.path.basename(a.py)} ({','.join(names)})", r2, ne, t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "ridge", "custom"])
    ap.add_argument("--inputs", default="tap45"); ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--save", default=None); ap.add_argument("--py", default=None)
    a = ap.parse_args()
    {"cache": cmd_cache, "ridge": cmd_ridge, "custom": cmd_custom}[a.cmd](a)
