#!/usr/bin/env python3
"""DEEPDRAFT pre-proof check (Thu 12 Nov 00:xx; the orchestrator, PLAN step 2's question): how fast does the model's state
drift over 30 tokens? This is CPU-only, on features already recorded.
- dlog captures (TrainLab data/dlog, read-only):
  - h: DFlash2's context input, hidden_norm(fc(taps)), the drafter's input feature; 2560, bf16 bits.
  - wd: the trunk's pre-mixer widened stream after the last layer (the MTP head's input); 10240, fp16.
  - Code = d1102mbpp (MBPP prompts); chat = n1101 (multilingual MGSM dialogue / story / explain answers).
- d02a (TrainLab data/d02a, read-only): f = the 5 drafter taps (the mean of the 4 HC streams after layers 5 / 15 / 25 /
  35 / 45), on IFEval (tf_ifeval) and multilingual (tf_multi) answers; the first and last tap are reported.
- Metric: the mean cosine between the feature at answer row t and at t + d, d = 1..30, over every t with both rows inside
  the 125B's answer. "raw" is the plain cosine. "ctr" subtracts each window's mean answer feature first, which removes
  the large shared direction raw cosines are dominated by.
- Up to MAXF files per source, in name order (deterministic).
usage: ~/BigMoE/.venv/bin/python -I scripts/dd_drift.py [out.txt]"""
import glob, os, sys

import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
MAXF, D = 40, 30


def bf16(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def curve(X):
    """X [n, dim] float32 answer rows -> (raw [D], ctr [D]) mean cosine at lag d, with the pair counts"""
    out = []
    for Y in (X, X - X.mean(axis=0, keepdims=True)):
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        c = np.full(D, np.nan); n = np.zeros(D, int)
        for d in range(1, D + 1):
            if len(Yn) > d:
                v = (Yn[:-d] * Yn[d:]).sum(axis=1)
                c[d - 1], n[d - 1] = v.sum(), len(v)
        out.append((c, n))
    return out


def pool(curves):
    res = []
    for k in range(2):
        s = sum(np.nan_to_num(c[k][0]) for c in curves); n = sum(c[k][1] for c in curves)
        res.append(s / np.maximum(n, 1))
    return res


def dlog(night, key):
    cs = []
    for f in sorted(glob.glob(os.path.join(TD, "dlog", night, "*.npz")))[:MAXF]:
        z = np.load(f)
        a = np.asarray(z["asst"]).astype(bool)[:-1] if "asst" in z else None
        X = z[key]
        X = bf16(np.asarray(X)) if key == "h" else np.asarray(X).astype(np.float32)
        if a is not None and len(a) == len(X):
            X = X[a]
        X = X[np.abs(X).sum(axis=1) > 0]                  # logged rows only (unlogged rows are zero)
        if len(X) > D + 1:
            cs.append(curve(X))
    return pool(cs), len(cs)


def d02a(prefix, tap):
    cs = []
    for f in sorted(glob.glob(os.path.join(TD, "d02a", prefix + "_*.npz")))[:MAXF]:
        z = np.load(f)
        F = np.asarray(z["f"])[:, tap, :].astype(np.float32)
        g = int(z["gen"]) - int(z["off"]) if "off" in z else int(z["gen"])
        X = F[max(g, 0):]
        if len(X) > D + 1:
            cs.append(curve(X))
    return pool(cs), len(cs)


def main():
    dst = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/FableGod/deepdraft/out/drift.txt")
    rows = [("code h (dlog d1102mbpp)", dlog("d1102mbpp", "h")), ("chat h (dlog n1101)", dlog("n1101", "h")),
            ("code wd trunk (d1102mbpp)", dlog("d1102mbpp", "wd")), ("chat wd trunk (n1101)", dlog("n1101", "wd")),
            ("ifeval tap L5 (d02a)", d02a("tf_ifeval", 0)), ("ifeval tap L45 (d02a)", d02a("tf_ifeval", 4)),
            ("multi tap L45 (d02a)", d02a("tf_multi", 4))]
    L = ["DEEPDRAFT drift (BigMoE scripts/dd_drift.py, Thu 12 Nov): mean cosine of a feature at answer row t vs t + d, over the "
         "125B's own answers; raw / ctr (each window's mean answer feature removed). Files per source <= 40, name order."]
    sel = [1, 2, 3, 5, 10, 15, 20, 25, 30]
    L.append(f"{'source':30s} {'files':>5s}  " + "  ".join(f"d{d:<2d} raw/ctr" for d in sel))
    for name, ((raw, ctr), nf) in rows:
        L.append(f"{name:30s} {nf:>5d}  " + "  ".join(f"{raw[d - 1]:.3f}/{ctr[d - 1]:+.3f}" for d in sel))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
