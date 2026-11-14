#!/usr/bin/env python3
"""LAYERS (docs/PREREG_DEEPDRAFT_LAYERS_2026-11-14.md): ridge maps from the raw taps of a layer set to the trunk's widened
stream. Both come from the same re-prefill (dd_reprefill.py -> ~/datasets/deepdraft/rawtaps/<key>.npz, raw [n, 5 x 2560],
taps 5/15/25/35/45 in that order, and wd [n, 10240]).
- Train: the first 90 of the 100 re-prefilled training files; lambda picked on the last 10. Every row is used.
- Sets: {5}, {5,15}, {5,15,25}, all 5, {45}. numpy, streaming sums, float64 normal equations.
-> ~/datasets/deepdraft/ckpt/ridge_L<set>.npz (W, xm, ym, cols) + data/deepdraft/tapsq/layers_fit.json
usage: dd_layers.py"""
import json, os, time
import numpy as np

H = os.path.expanduser
RT = H("~/datasets/deepdraft/rawtaps")
names = json.load(open(H("~/BigMoE/data/deepdraft/tapsq/split.json")))["train"][:100]
tr, va = names[:90], names[90:100]
SETS = {"5": [0], "5_15": [0, 1], "5_15_25": [0, 1, 2], "all": [0, 1, 2, 3, 4], "45": [4]}


def load(nm):
    z = np.load(os.path.join(RT, nm))
    return np.asarray(z["raw"]).astype(np.float32), np.asarray(z["wd"]).astype(np.float32)


TR = [load(n) for n in tr]; VA = [load(n) for n in va]
print(f"[layers] {len(TR)} train files ({sum(len(x) for x, _ in TR)} rows), {len(VA)} validation files", flush=True)
out = {}
for name, li in SETS.items():
    t0 = time.time()
    cols = np.concatenate([np.arange(i * 2560, (i + 1) * 2560) for i in li])
    d = len(cols)
    Sxx = np.zeros((d, d)); Sxy = np.zeros((d, 10240)); sx = np.zeros(d); sy = np.zeros(10240); n = 0
    for x, y in TR:
        x = x[:, cols]
        Sxx += (x.T @ x).astype(np.float64); Sxy += (x.T @ y).astype(np.float64); sx += x.sum(0); sy += y.sum(0); n += len(x)
    xm, ym = sx / n, sy / n
    XtX = Sxx - n * np.outer(xm, xm); XtY = Sxy - n * np.outer(xm, ym)
    dg = float(np.mean(np.diag(XtX)))
    best = None; res = {}
    for lf in (1e-4, 1e-3, 1e-2, 1e-1):
        W = np.linalg.solve(XtX + lf * dg * np.eye(d), XtY).astype(np.float32)
        num = den = num_n = den_n = 0.0
        for x, y in VA:
            p = (x[:, cols] - xm) @ W + ym
            num += float(((p - y) ** 2).sum()); den += float(((y - ym) ** 2).sum())
            pn = p / np.sqrt((p ** 2).mean(-1, keepdims=True) + 1e-12); yn = y / np.sqrt((y ** 2).mean(-1, keepdims=True) + 1e-12)
            num_n += float(((pn - yn) ** 2).sum()); den_n += float((yn ** 2).sum())
        res[str(lf)] = {"r2": 1 - num / den, "rel_err_normalized": (num_n / den_n) ** 0.5}
        if best is None or res[str(lf)]["r2"] > best[0]:
            best = (res[str(lf)]["r2"], lf, W)
    r2, lf, W = best
    np.savez(H(f"~/datasets/deepdraft/ckpt/ridge_L{name}.npz"), W=W, xm=xm.astype(np.float32), ym=ym.astype(np.float32), cols=cols)
    out[name] = {"layers": [(5, 15, 25, 35, 45)[i] for i in li], "inputs": d, "lambda_factor": lf, "val": res,
                 "r2": r2, "rel_err_normalized": res[str(lf)]["rel_err_normalized"], "secs": time.time() - t0}
    print(f"[layers] {name}: {d} inputs, lambda {lf:g}: R2 {r2:.4f}, normalized error {res[str(lf)]['rel_err_normalized']:.4f} ({time.time() - t0:.0f} s)", flush=True)
json.dump(out, open(H("~/BigMoE/data/deepdraft/tapsq/layers_fit.json"), "w"), indent=1)
