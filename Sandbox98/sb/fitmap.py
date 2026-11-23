"""Ridge maps from captured features to W (the head's input: the target's post-norm final hidden), for REPLACE-style
evals (BigMoE's LAYERS / REPLACE-H analog) and the R2 proxy. numpy on the CPU (<= 4 BLAS threads), streaming X'X.
- features: taps by layer index (core.TAPS = 2, 7, 12, 17, 22), concatenated; rows = every answer row (from the first
  anchor on) of the first --ntr training files; lambda picked on the next --nva files (x mean diag of X'X).
- R2 = 1 - SSE / SST on the validation rows (dd_ridge.py's fvu), plus the RMS-normalized relative error.
-> ckpt/maps/NAME.npz (Wm [in, d], xm, ym, taps) + results/maps.json[NAME]
usage: fitmap.py --name L22 --taps 22 [--ntr 200 --nva 30]"""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4"); os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, json, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mlx.core as mx
from sb import core as C

MAPS = os.path.join(C.CKPT, "maps")


def feats(f, taps, r0=0):
    tz = np.load(f[:-4] + ".taps.npz")["taps"]
    idx = [C.TAPS.index(t) for t in taps]
    return np.concatenate([tz[i, r0:].astype(np.float32) for i in idx], -1)


def target(f, r0=0):
    z = np.load(f)
    return np.array(C.load_W(z)[0].astype(mx.float32))[r0:]


def rows(f, taps):
    z = np.load(f)
    r0 = C.first_anchor(z["asst"])
    return feats(f, taps, r0), target(f, r0)


_M = {}
def apply_map(name, f):
    """[S, d] mx fp32: the map's prediction of W for every row of a capture file"""
    if name not in _M:
        m = np.load(os.path.join(MAPS, name + ".npz"))
        _M[name] = (mx.array(m["Wm"]), mx.array(m["xm"]), mx.array(m["ym"]), [int(t) for t in m["taps"]])
    Wm, xm, ym, taps = _M[name]
    X = mx.array(feats(f, taps))
    return (X - xm) @ Wm + ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--taps", required=True)
    ap.add_argument("--ntr", type=int, default=200); ap.add_argument("--nva", type=int, default=30)
    a = ap.parse_args()
    C.limit_memory(4)
    C.load_target(); C.drop_trunk()
    taps = [int(t) for t in a.taps.split(",")]
    s = json.load(open(os.path.join(C.DATA, "split.json")))
    fs = [os.path.join(C.DATA, "mc", k + ".npz") for k in s["train"]]
    tr, va = fs[:a.ntr], fs[a.ntr:a.ntr + a.nva]
    t0 = time.time()
    din, d = 2048 * len(taps), 2048
    Sxx = np.zeros((din, din)); Sxy = np.zeros((din, d)); sx = np.zeros(din); sy = np.zeros(d); n = 0
    for f in tr:
        x, y = rows(f, taps)
        Sxx += (x.T @ x).astype(np.float64); Sxy += (x.T @ y).astype(np.float64)
        sx += x.sum(0); sy += y.sum(0); n += len(x)
    xm, ym = sx / n, sy / n
    XtX = Sxx - n * np.outer(xm, xm); XtY = Sxy - n * np.outer(xm, ym)
    V = [rows(f, taps) for f in va]
    print(f"[map] {a.name} taps {taps}: {len(tr)} files, {n} rows; sums in {time.time() - t0:.0f}s", flush=True)

    def err(W):
        num = den = num_n = den_n = 0.0
        for x, y in V:
            p = (x - xm) @ W + ym
            num += float(((p - y) ** 2).sum()); den += float(((y - ym) ** 2).sum())
            pn = p / np.sqrt((p ** 2).mean(-1, keepdims=True) + 1e-12); yn = y / np.sqrt((y ** 2).mean(-1, keepdims=True) + 1e-12)
            num_n += float(((pn - yn) ** 2).sum()); den_n += float((yn ** 2).sum())
        return num / den, (num_n / den_n) ** 0.5

    best, res = None, {}
    dg = float(np.mean(np.diag(XtX)))
    for lf in (1e-4, 1e-3, 1e-2, 1e-1):
        W = np.linalg.solve(XtX + lf * dg * np.eye(din), XtY).astype(np.float32)
        fvu, rn = err(W)
        res[str(lf)] = {"r2": 1 - fvu, "rel_err_normalized": rn}
        print(f"[map]   lambda {lf:g}: R2 {1 - fvu:.4f}, normalized rel err {rn:.4f}", flush=True)
        if best is None or fvu < best[0]:
            best = (fvu, lf, W, rn)
    os.makedirs(MAPS, exist_ok=True)
    np.savez(os.path.join(MAPS, a.name + ".npz"), Wm=best[2], xm=xm.astype(np.float32), ym=ym.astype(np.float32),
             taps=np.array(taps))
    jf = os.path.join(C.RES, "maps.json")
    allm = json.load(open(jf)) if os.path.exists(jf) else {}
    allm[a.name] = {"taps": taps, "r2": 1 - best[0], "lambda": best[1], "rel_err_normalized": best[3], "rows": n,
                    "grid": res, "ntr": a.ntr, "nva": a.nva}
    json.dump(allm, open(jf, "w"), indent=1)
    print(f"[map] {a.name}: R2 {1 - best[0]:.4f} (lambda {best[1]:g}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
