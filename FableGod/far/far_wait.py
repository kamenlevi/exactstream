"""FAR read (4) tonight, CPU only (Fable FAR, Wed 2026-11-04 22:1x; the orchestrator's follow-up for BigMoE): is the SSD wait
latency-bound (per-layer round trips: the number of calls with misses) or bandwidth-bound (bytes: the number of misses)?
Data: BigMoE's r57 per-check logs ~/BigMoE/data/r57/<tag>_checks.json (read-only): per check miss (expert misses), io (demand
SSD busy ms: the load batches' wall time), svc (gate service ms of calls with misses, post seen -> release = the GPU's spin
bound), wt (ms), bld (ms), ck (check + commit ms), rows, ws (warm-split flag), paused; <tag>_runs.json adds per run 'slow' =
the median calls with misses per check (the stalled layers; not logged per check) and fast (FAST mode) / C (from the tag).
Fits, per tag (checks >= 3, ws 0, not paused, pooled over its prompts): io ~ a + b * miss and svc ~ a + b * miss (b = marginal
ms per miss; 2.163 MB per miss -> implied GB/s; a = the zero-miss intercept); per-check io / miss vs misses per check (a
latency-bound SSD: falling with misses); across runs: io_med ~ alpha * slow_med + beta * miss_med.
usage: ~/BigMoE/.venv/bin/python -I far_wait.py -> stdout + far_wait.txt (one core, seconds)"""
import os, glob, json, re, time
import numpy as np
R = os.path.expanduser("~/BigMoE/data/r57")
OUT = os.path.expanduser("~/FableGod/far/far_wait.txt")
MSB_MB = 2.162688
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def fit(x, y):
    X = np.stack([np.ones(len(x)), x], 1)
    b, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yh = X @ b
    r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    se = np.sqrt(((y - yh) ** 2).sum() / max(len(y) - 2, 1) / max(((x - x.mean()) ** 2).sum(), 1e-9))
    return b[0], b[1], se, r2


say(f"FAR WAIT ({time.strftime('%a %Y-%m-%d %H:%M')}): SSD wait vs misses on BigMoE's r57 per-check logs (read-only)")
say("per tag: n checks | misses/check (median) | io ms (median) | svc ms | ck ms | io ~ a + b*miss: b ms/miss [GB/s implied], a ms, R2 | "
    "svc ~ a + b*miss | io/miss ms at the low / high miss tercile | slow (calls with misses, run medians)")
ALLRUN = []
for p in sorted(glob.glob(os.path.join(R, "*_checks.json")), key=os.path.getmtime):
    tag = os.path.basename(p)[:-len("_checks.json")]
    m = re.search(r"_(\d+)r?$", tag)
    C = int(m.group(1)) if m else -1
    try:
        d = json.load(open(p))
        runs = {(r["key"], r["arm"]): r for r in json.load(open(os.path.join(R, tag + "_runs.json")))}
    except Exception as e:
        continue
    X, modes = [], set()
    for rec in d:
        if rec.get("paused"):
            continue
        r = runs.get((rec["key"], rec["arm"]))
        fast = bool(r and r.get("fast"))
        modes.add("FAST" if fast else "EXACT")
        ws = rec.get("ws") or [0] * len(rec["ck"])
        n = len(rec["ck"])
        for i in range(2, n):
            if ws[i]:
                continue
            X.append((rec["miss"][i], rec["io"][i], rec["svc"][i], rec["wt"][i], rec["bld"][i], rec["ck"][i], rec["rows"][i], fast))
        if r:
            ALLRUN.append((tag, C, fast, r["miss"], r["io"], r.get("svc", float("nan")), r.get("slow", float("nan")), r["ck"]))
    if len(X) < 12:
        continue
    A = np.array([x[:7] for x in X], float)
    miss, io, svc, wt, bld, ck = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]
    a1, b1, se1, r21 = fit(miss, io)
    a2, b2, se2, r22 = fit(miss, svc)
    ratio = io / np.maximum(miss, 1)
    q = np.quantile(miss, [1 / 3, 2 / 3])
    lo, hi = ratio[miss <= q[0]], ratio[miss >= q[1]]
    slows = [x[6] for x in ALLRUN if x[0] == tag]
    say(f"  {tag:13s} C {C:3d} {'/'.join(sorted(modes)):10s} n {len(X):4d} | miss {np.median(miss):6.0f} | io {np.median(io):6.1f} | svc "
        f"{np.median(svc):6.1f} | ck {np.median(ck):6.1f} | io: b {b1:.3f} +- {se1:.3f} ms/miss [{MSB_MB / max(b1, 1e-6):5.1f} GB/s], "
        f"a {a1:6.1f}, R2 {r21:.2f} | svc: b {b2:.3f}, a {a2:6.1f}, R2 {r22:.2f} | io/miss {np.median(lo):.3f} / {np.median(hi):.3f} | "
        f"slow {np.median(slows) if slows else float('nan'):.0f}")
say("across runs (one point per prompt run, medians): io_med ~ alpha * slow_med + beta * miss_med, by mode and C")
for mode in (False, True):
    for C in sorted({x[1] for x in ALLRUN}):
        pts = [x for x in ALLRUN if x[2] == mode and x[1] == C and np.isfinite(x[6])]
        if len(pts) < 8:
            continue
        M = np.array([[1.0, x[6], x[3]] for x in pts]); y = np.array([x[4] for x in pts])
        b = np.linalg.lstsq(M, y, rcond=None)[0]
        yh = M @ b; r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        bs = fit(np.array([x[6] for x in pts]), y); bm = fit(np.array([x[3] for x in pts]), y)
        say(f"  {'FAST' if mode else 'EXACT'} C {C:3d}: {len(pts):3d} runs; slow {np.median([x[6] for x in pts]):.0f} calls with misses "
            f"(range {min(x[6] for x in pts):.0f}-{max(x[6] for x in pts):.0f}), miss {np.median([x[3] for x in pts]):.0f}, io {np.median(y):.1f} ms | "
            f"joint: {b[1]:.2f} ms per stalled call + {b[2]:.3f} ms per miss, R2 {r2:.2f} | io~slow alone R2 {bs[3]:.2f} | io~miss alone R2 {bm[3]:.2f} "
            f"(b {bm[1]:.3f} ms/miss)")
open(OUT, "w").write("\n".join(lines) + "\n")
