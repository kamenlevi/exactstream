"""landmark_screen.py (Breakthrough, Tue 17 Nov 2026 07:4x; BREAK98): screen LANDMARK-1, pre-registered in
docs/IDEAS_1117_landmark.md (e23697a) before computing.
At a drafted position p of depth d (true 125B states exist only up to p - d), the landmark is the most recent q <= p - d
whose last k tokens equal p's (longest k in 8, 6, 4, 3, 2). How often is there one, and how close is the 125B's state
at q to its state at p (taps after layers 25 / 35 / 45), against the anchor state f[p - d] and the neighbour f[p - 1]?
Plus the 125B's next-token agreement top1[q] == top1[p]. Data: ~/TrainLab/data/d17f (the 125B's greedy text + states).
Usage: ~/BigMoE/.venv/bin/python -I -B scripts/break98/landmark_screen.py   -> data/break98/landmark_screen.out"""
import bisect
import glob
import os
import resource
import time

import numpy as np

SRC = os.path.expanduser("~/TrainLab/data/d17f")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "break98", "landmark_screen.out")
KS = (8, 6, 4, 3, 2)
DS = (1, 4, 8, 16)
TAPS = {25: 2, 35: 3, 45: 4}
SETS = {"code": "tf_code_HumanEval_*.npz", "gsm": "tf_gsm_*.npz", "ifeval": "tf_ifeval_*.npz", "multi": "tf_multi_*.npz"}
lines = []


def say(s=""):
    lines.append(s)
    print(s, flush=True)


def unit(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def window(path):
    z = np.load(path)
    ids = z["ids"].astype(np.int64)
    gen = int(z["gen"]) - int(z["off"])
    F = {t: unit(z["f"][:, i, :]) for t, i in TAPS.items()}
    idx = {k: {} for k in KS}
    for k in KS:
        for q in range(k - 1, len(ids)):
            idx[k].setdefault(tuple(ids[q - k + 1:q + 1]), []).append(q)
    return ids, gen, F, z["top1"].astype(np.int64), idx


def landmark(ids, idx, p, d):
    for k in KS:
        if p - k + 1 < 0:
            continue
        L = idx[k].get(tuple(ids[p - k + 1:p + 1]))
        if not L:
            continue
        j = bisect.bisect_right(L, p - d) - 1
        if j >= 0:
            return L[j], k
    return None, 0


def q3(x):
    return (f"{np.median(x):.3f} [{np.percentile(x, 25):.3f}, {np.percentile(x, 75):.3f}]" if len(x) else "-")


t0 = time.time()
verdict_inputs = None
say("LANDMARK-1 (pre-registered, docs/IDEAS_1117_landmark.md): landmark = the most recent q <= p - d with the same last "
    "k tokens (longest k in 8/6/4/3/2); answer positions; the 125B's own states (d17f)")
for sname, pat in SETS.items():
    W = [window(p) for p in sorted(glob.glob(os.path.join(SRC, pat)))]
    say("")
    say(f"== {sname}: {len(W)} windows, {sum(len(w[0]) - w[1] - 1 for w in W)} answer positions")
    say("| d | coverage | by longest k (8/6/4/3/2) | cos landmark t45 | cos anchor f[p-d] t45 | cos neighbour f[p-1] t45 "
        "| landmark - anchor (paired median) | |u-v| landmark / anchor t45 | top1 agree | cos landmark t25 / t35 |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for d in DS:
        n = 0
        byk = {k: 0 for k in KS}
        cl, ca, cn, dl, agree, c25, c35 = [], [], [], [], [], [], []
        for ids, gen, F, top1, idx in W:
            for p in range(gen + 1, len(ids)):
                if p - d < 0:
                    continue
                n += 1
                q, k = landmark(ids, idx, p, d)
                if q is None:
                    continue
                byk[k] += 1
                a, b = float(F[45][q] @ F[45][p]), float(F[45][p - d] @ F[45][p])
                cl.append(a)
                ca.append(b)
                cn.append(float(F[45][p - 1] @ F[45][p]))
                dl.append(a - b)
                agree.append(top1[q] == top1[p])
                c25.append(float(F[25][q] @ F[25][p]))
                c35.append(float(F[35][q] @ F[35][p]))
        m = len(cl)
        cov = m / max(n, 1)
        ul = np.sqrt(np.maximum(2 - 2 * np.array(cl), 0)) if m else np.array([])
        ua = np.sqrt(np.maximum(2 - 2 * np.array(ca), 0)) if m else np.array([])
        say(f"| {d} | {100 * cov:.1f} % ({m}/{n}) | {'/'.join(str(byk[k]) for k in KS)} | {q3(cl)} | {q3(ca)} | {q3(cn)} | "
            f"{np.median(dl) if m else float('nan'):+.3f} | {np.median(ul) if m else float('nan'):.3f} / "
            f"{np.median(ua) if m else float('nan'):.3f} | {100 * np.mean(agree) if m else float('nan'):.1f} % | "
            f"{np.median(c25) if m else float('nan'):.3f} / {np.median(c35) if m else float('nan'):.3f} |")
        if sname == "code" and d == 8:
            verdict_inputs = dict(cov=cov, ml=float(np.median(cl)) if m else -1, ma=float(np.median(ca)) if m else 9,
                                  ag=float(np.mean(agree)) if m else 0)
v = verdict_inputs
prom = v["cov"] >= 0.25 and v["ml"] >= v["ma"] + 0.15 and v["ag"] >= 0.80
kill = v["cov"] < 0.10 or v["ml"] <= v["ma"]
say("")
say(f"VERDICT (code, d = 8): coverage {100 * v['cov']:.1f} % (PROMISING >= 25, KILL < 10); median cos landmark "
    f"{v['ml']:.3f} vs anchor {v['ma']:.3f} (PROMISING needs +0.15, KILL if <=); top1 agreement {100 * v['ag']:.1f} % "
    f"(PROMISING >= 80) -> {'PROMISING' if prom else ('KILL' if kill else 'BETWEEN')}")
say(f"wall {time.time() - t0:.1f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
