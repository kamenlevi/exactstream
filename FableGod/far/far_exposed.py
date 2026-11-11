"""NEVERWAIT, the SHAPE of check time against service time (Fable FAR, Thu 2026-11-05 16:4x; CPU, numpy, seconds; logs only).
The engine's cold-vs-warm samples (far_warmsplit.py) say exposed = -26.6 + 1.06 x service on heavy 16-row checks: a hidden
allowance per check (the resident experts computed under each stalled layer's read), then 1:1. If that holds on live checks,
the wait a never-wait check removes is service minus the allowance, not b x service. Two reads of BigMoE's per-check logs:
1. per tag: check ms ~ prompt + c x rows + d x position + f(service), f piecewise linear with a knee:
   f = b1 x min(service, h) + b2 x max(service - h, 0), h scanned; the removable wait per check = mean f(service).
2. partial residuals by service bin (check minus the fitted prompt, rows and position terms).
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_exposed.py > far_exposed.txt"""
import json, os, time
import numpy as np
R = os.path.expanduser("~/BigMoE/data/r57")
print(f"FAR EXPOSED ({time.strftime('%a %Y-%m-%d %H:%M')}): how much of the gate service time a live check pays for")
for tag in ("qallt_208", "qallft_208", "u3ct_208", "qalq_160", "u3cq_160", "dkvt_224"):
    try:
        d = json.load(open(os.path.join(R, tag + "_checks.json")))
    except Exception:
        continue
    X = []
    for pi, rec in enumerate(d):
        if rec.get("paused"):
            continue
        cum = np.cumsum(rec["kept"])
        for i in range(2, len(rec["ck"])):
            X.append((pi, rec["rows"][i], rec["kept"][i], rec["ck"][i], rec["miss"][i], rec["svc"][i], rec["drf"][i], cum[i - 1]))
    A = np.array(X, float); pi, rows, kept, ck, miss, svc, drf, pos = A.T
    P = np.eye(int(pi.max()) + 1)[pi.astype(int)]
    ok = np.ones(len(ck), bool)
    for _ in range(2):                                               # drop hiccup checks (> 3 sd residual) once
        M = np.concatenate([P, rows[:, None], pos[:, None], svc[:, None]], 1)
        b = np.linalg.lstsq(M[ok], ck[ok], rcond=None)[0]
        res = ck - M @ b
        ok = np.abs(res) < 3 * res[ok].std()
    lin = b[-1]
    best = None
    for h in (0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30):
        f1, f2 = np.minimum(svc, h), np.maximum(svc - h, 0)
        cols = [P, rows[:, None], pos[:, None], f2[:, None]] + ([f1[:, None]] if h > 0 else [])
        M2 = np.concatenate(cols, 1)
        b2 = np.linalg.lstsq(M2[ok], ck[ok], rcond=None)[0]
        sse = float(((ck[ok] - M2[ok] @ b2) ** 2).sum())
        hi = b2[len(P[0]) + 2]; lo = b2[-1] if h > 0 else float("nan")
        rem = float((hi * f2 + (lo * f1 if h > 0 else 0)).mean())
        if best is None or sse < best[0]:
            best = (sse, h, lo, hi, rem)
    M0 = np.concatenate([P, rows[:, None], pos[:, None]], 1)
    part = ck - M0 @ b[:-1]                                         # check minus prompt, rows, position (linear model's terms)
    q = np.quantile(svc[ok], [0, .1, .25, .5, .75, .9, 1.0])
    bins = "; ".join(f"service {svc[ok & (svc >= lo_) & (svc <= hi_)].mean():5.1f} -> {part[ok & (svc >= lo_) & (svc <= hi_)].mean():5.1f}"
                     for lo_, hi_ in zip(q[:-1], q[1:]))
    sse, h, lo, hi, rem = best
    tps = lambda cut: 1000 * kept.sum() / (ck + drf - cut).sum()
    print(f"  {tag}: n {int(ok.sum())} | service mean {svc.mean():.1f} ms, check {ck.mean():.1f} | linear slope {lin:.2f} -> removable {lin * svc.mean():.1f} ms | best knee h {h} ms: "
          f"slope below {lo:.2f}, above {hi:.2f} -> removable {rem:.1f} ms per check | tok/s now {tps(0):.1f}, wait removed: linear {tps(lin * svc):.1f}, knee model "
          f"{tps(hi * np.maximum(svc - h, 0) + (lo * np.minimum(svc, h) if h > 0 else 0)):.1f}, fully serial {tps(svc):.1f}")
    print(f"     partial residual (check minus prompt, rows, position terms) by service bin: {bins}")
