"""Fit12 T2 (Sun 8 Nov 07:3x): CPU audit of the 12 GB chat loop's exact checks, from chat12's own logs
(/Users/kamen/Fit12/data/scr/chat12: arm ex, 12 IFEval prompts, CAP 256; 4-row verify calls stopping at the first miss).
Question: how many of the rows each check reads are wasted on chat (kept ~2.3 of 4 rows read), does the first draft's
selector cf predict a short check, and what does one exact call cost (fixed ms + ms per GB)? This prices a cf-adaptive
first-call width (exact-preserving by WV1 / WV2: the verify is width-invariant) before anything is built.
Per check (check 0 dropped: cold pools): kept = len(new), cf0, exact-phase ms / MB / misses, calls = (kept - 1) // 4 + 1
(4-row calls; the cap / EOS check can be short). Fit: exact ms = a * calls + b * GB (least squares).
Usage: python -I -B chatw_audit.py [DIR] [TAG]   (CPU only, seconds)"""
import glob, json, os, sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/chat12"
TAG = sys.argv[2] if len(sys.argv) > 2 else "chat12"
rows = []
for f in sorted(glob.glob(os.path.join(D, f"{TAG}_*_ex.json"))):
    key = os.path.basename(f)[len(TAG) + 1:-len("_ex.json")]
    lg = json.load(open(f))
    ph = json.load(open(f[:-5] + "_ph.json"))
    ex = {p[1]: p for p in ph if p[0] == "exact"}
    dr = {p[1]: p for p in ph if p[0] == "draft"}
    for k, (S, path, kept, cf0) in enumerate(lg):
        if k == 0 or k not in ex or path != 0:
            continue
        p = ex[k]
        rows.append(dict(key=key, k=k, kept=kept, cf0=cf0, ms=(p[6] - p[2]) * 1e3, mb=(p[8] - p[4]) / 1e6,
                         miss=p[9] - p[5], dms=((dr[k][6] - dr[k][2]) * 1e3) if k in dr else 0.0,
                         calls=(kept - 1) // 4 + 1))
n = len(rows)
kept = np.array([r["kept"] for r in rows]); cf = np.array([r["cf0"] if r["cf0"] is not None else 1.0 for r in rows])
ms = np.array([r["ms"] for r in rows]); mb = np.array([r["mb"] for r in rows]); calls = np.array([r["calls"] for r in rows])
miss = np.array([r["miss"] for r in rows]); dms = np.array([r["dms"] for r in rows])
print(f"{n} checks (check 0 dropped) on {len(set(r['key'] for r in rows))} prompts from {D}")
print(f"kept per check {kept.mean():.3f}; exact ms {ms.mean():.1f}; MB {mb.mean():.0f}; misses {miss.mean():.0f} "
      f"({mb.sum() / max(1, miss.sum()):.2f} MB per miss); draft ms {dms.mean():.1f}; calls {calls.mean():.3f}")
print("kept distribution: " + ", ".join(f"{v}: {np.mean(kept == v) * 100:.1f} %" for v in range(1, 9)) +
      f", >8: {np.mean(kept > 8) * 100:.1f} %")
rr = 4 * calls
print(f"rows read per check {rr.mean():.2f}, rows needed (= kept) {kept.mean():.2f}: wasted {100 * (1 - kept.sum() / rr.sum()):.1f} %")
A = np.stack([calls, mb / 1e3], 1)
(a, b), *_ = np.linalg.lstsq(A, ms, rcond=None)
res = ms - A @ np.array([a, b])
print(f"fit: exact ms = {a:.1f} * calls + {b:.1f} * GB (rms residual {np.sqrt(np.mean(res ** 2)):.1f} ms; "
      f"{1e3 / b:.1f} GB/s marginal)")
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .97), (.97, 1.01)):
    m = (cf >= lo) & (cf < hi)
    if m.sum():
        print(f"  cf0 [{lo:.2f}, {hi:.2f}): {m.sum():4d} checks ({100 * m.mean():4.1f} %), kept {kept[m].mean():.2f}, "
              f"P(kept 1) {np.mean(kept[m] == 1):.2f}, P(kept <= 2) {np.mean(kept[m] <= 2):.2f}, MB {mb[m].mean():.0f}")
json.dump(dict(n=n, a_ms_per_call=a, b_ms_per_gb=b, rows=rows), open(os.path.join(D, "chatw_audit.json"), "w"))
