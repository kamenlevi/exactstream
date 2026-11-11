"""LOSSY350-FAR: how much of the SSD service time is on the check's critical path (Fable FAR, Thu 2026-11-05 16:2x; CPU, numpy,
seconds). BigMoE's per-check logs ~/BigMoE/data/r57/<tag>_checks.json (read-only; checks >= 3, not paused). Fits per tag:
check ms ~ prompt effects + b * service ms + c * rows [+ d * position (tokens generated before the check)]; the wait removed
at the fitted slope b: check -> check - b * service, tok/s = kept / (check + draft). Position matters: early checks have more
misses AND a shorter context (corr(service, position) -0.2 .. -0.4), which biases the slope down without it.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_ckfit.py > far_ckfit.txt"""
import json, os
import numpy as np
R = os.path.expanduser("~/BigMoE/data/r57")
for tag in ("qallt_208", "qallft_208", "u3ct_208", "qalq_160", "u3cq_160"):
    try:
        d = json.load(open(os.path.join(R, tag + "_checks.json")))
    except Exception as e:
        print(tag, "missing"); continue
    X = []
    for pi, rec in enumerate(d):
        if rec.get("paused"):
            continue
        cum = np.cumsum(rec["kept"])
        for i in range(2, len(rec["ck"])):
            X.append((pi, rec["rows"][i], rec["kept"][i], rec["ck"][i], rec["miss"][i], rec["io"][i], rec["svc"][i], rec["drf"][i], i, cum[i - 1]))
    A = np.array(X, float); pi, rows, kept, ck, miss, io, svc, drf, idx, pos = A.T
    P = np.eye(int(pi.max()) + 1)[pi.astype(int)]
    out = []
    for name, cols in (("svc+rows", [svc, rows]), ("svc+rows+pos", [svc, rows, pos]), ("svc+rows+pos+pos2", [svc, rows, pos, pos ** 2 / 100])):
        M = np.concatenate([P] + [c[:, None] for c in cols], 1)
        b, *_ = np.linalg.lstsq(M, ck, rcond=None)
        res = ck - M @ b
        nb = len(cols)
        # robust: drop residual outliers > 3 sd and refit once
        keep = np.abs(res) < 3 * res.std()
        b2, *_ = np.linalg.lstsq(M[keep], ck[keep], rcond=None)
        out.append(f"{name}: b_svc {b[-nb]:.2f} (robust {b2[-nb]:.2f}), c_rows {b[-nb + 1]:.2f}" + (f", d_pos {b[-nb + 2]:.3f} ms per token" if nb >= 3 else "") + f", resid sd {res.std():.1f}")
    cc = np.corrcoef(svc, pos)[0, 1]
    print(f"{tag}: n {len(ck)}, corr(svc, position) {cc:+.2f}, corr(miss, position) {np.corrcoef(miss, pos)[0, 1]:+.2f}\n   " + "\n   ".join(out))
    M = np.concatenate([P, svc[:, None], rows[:, None], pos[:, None]], 1)
    b, *_ = np.linalg.lstsq(M, ck, rcond=None)
    ck0 = ck - b[-3] * svc
    print(f"   wait removed at this slope: check {ck.mean():.1f} -> {ck0.mean():.1f} ms; tok/s {1000 * kept.sum() / (ck + drf).sum():.1f} -> {1000 * kept.sum() / (ck0 + drf).sum():.1f}")
