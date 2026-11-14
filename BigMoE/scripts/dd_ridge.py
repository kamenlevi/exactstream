#!/usr/bin/env python3
"""REPLACE-H step 2 (Sat 14 Nov 2026; docs/PREREG_DEEPDRAFT_REPLACE_2026-11-14.md): a ridge map from h (DFlash2's fused
multi-layer input, 2560) to wd (the trunk's widened stream, 10240, the MTP head's native input), fit on captured rows.
numpy only, streaming:
- per file, the sums S_xx, S_xy, s_x and s_y, then centred;
- train files: the first NTR of the 795 n1112 files t30b never saw (tapsq/split.json); validation: the next NVA;
- every row where h is nonzero (the last 2048 prompt rows + the answer);
- lambda picked on the validation files from {1e-4, 1e-3, 1e-2, 1e-1} x mean(diag(XtX));
- reports the relative error, raw and after each row's RMS normalization (the head's pre_fc_norm sees directions).
-> ~/datasets/deepdraft/ckpt/ridge_h2wd.npz (W [2560, 10240], xm, ym) + data/deepdraft/tapsq/ridge_h2wd.json
usage: dd_ridge.py [NTR=400] [NVA=40]"""
import json, os, sys, time
import numpy as np

H = os.path.expanduser
CAP = H("~/FableGod/deepdraft/capture/n1112")
NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NVA = int(sys.argv[2]) if len(sys.argv) > 2 else 40
names = json.load(open(H("~/BigMoE/data/deepdraft/tapsq/split.json")))["train"]
tr, va = names[:NTR], names[NTR:NTR + NVA]


def rows(name):
    z = np.load(os.path.join(CAP, name))
    h = (np.asarray(z["h"]).astype(np.uint32) << 16).view(np.float32)
    nz = np.abs(h).sum(-1) > 0
    return h[nz], np.asarray(z["wd"])[nz].astype(np.float32)


t0 = time.time()
Sxx = np.zeros((2560, 2560)); Sxy = np.zeros((2560, 10240)); sx = np.zeros(2560); sy = np.zeros(10240); n = 0
for nm in tr:
    x, y = rows(nm)
    Sxx += (x.T @ x).astype(np.float64); Sxy += (x.T @ y).astype(np.float64)
    sx += x.sum(0); sy += y.sum(0); n += len(x)
xm, ym = sx / n, sy / n
XtX = Sxx - n * np.outer(xm, xm); XtY = Sxy - n * np.outer(xm, ym)
print(f"[ridge] {len(tr)} train files, {n} rows; sums in {time.time() - t0:.0f} s", flush=True)
V = [rows(nm) for nm in va]


def err(W):
    num = den = num_n = den_n = 0.0
    for x, y in V:
        p = (x - xm) @ W + ym
        num += float(((p - y) ** 2).sum()); den += float(((y - ym) ** 2).sum())
        pn = p / np.sqrt((p ** 2).mean(-1, keepdims=True) + 1e-12); yn = y / np.sqrt((y ** 2).mean(-1, keepdims=True) + 1e-12)
        num_n += float(((pn - yn) ** 2).sum()); den_n += float((yn ** 2).sum())
    return num / den, (num_n / den_n) ** 0.5


best, res = None, {}
d = float(np.mean(np.diag(XtX)))
for lf in (1e-4, 1e-3, 1e-2, 1e-1):
    W = np.linalg.solve(XtX + lf * d * np.eye(2560), XtY).astype(np.float32)
    fvu, rn = err(W)
    res[str(lf)] = {"fvu": fvu, "rel_err_normalized": rn}
    print(f"[ridge] lambda {lf:g} x mean diag: unexplained variance {fvu:.4f} (R2 {1 - fvu:.4f}); RMS-normalized relative error {rn:.4f}", flush=True)
    if best is None or fvu < best[0]:
        best = (fvu, lf, W)
fvu, lf, W = best
np.savez(H("~/datasets/deepdraft/ckpt/ridge_h2wd.npz"), W=W, xm=xm.astype(np.float32), ym=ym.astype(np.float32))
out = {"train_files": len(tr), "val_files": len(va), "rows": n, "lambda_factor": lf, "val": res,
       "best": {"fvu": fvu, "r2": 1 - fvu, "rel_err_normalized": res[str(lf)]["rel_err_normalized"]}, "secs": time.time() - t0}
json.dump(out, open(H("~/BigMoE/data/deepdraft/tapsq/ridge_h2wd.json"), "w"), indent=1)
print(f"[ridge] best lambda {lf:g}: R2 {1 - fvu:.4f}, normalized relative error {res[str(lf)]['rel_err_normalized']:.4f}; "
      f"saved; {time.time() - t0:.0f} s", flush=True)
