"""NEVERWAIT, the cleanest CPU estimate of how serial the SSD wait is (Fable FAR, Thu 2026-11-05 16:5x; numpy, seconds).
BigMoE's per-check logs ~/BigMoE/data/r57/<tag>_checks.json (read-only). Two runs of the SAME prompt that produced the SAME
checks (identical rows and kept sequences: the exact loop, or a FAST arm whose tokens did not move) differ only in what the
cache held and in timing noise. Per check i (>= 3rd), paired: d_check = ck_B - ck_A, d_service = svc_B - svc_A. Same rows,
same context length, same position: the slope of d_check on d_service is the share of the service time on the check's
critical path, without the cross-sectional confounds of far_ckfit.py.
Pairs are discovered automatically: any two tags sharing >= 3 prompts with identical (rows, kept) sequences; a pair is kept
when the two runs differ in service time by >= 2 ms per check on average (different cache size or state) or is asked for.
usage: OMP_NUM_THREADS=1 ~/BigMoE/.venv/bin/python -I far_pairs.py [tagA tagB ...] > far_pairs.txt"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob, json, sys, time
import numpy as np
R = os.path.expanduser("~/BigMoE/data/r57")
T0 = time.time() - 4 * 86400                                         # tags written in the last four days
tags = {}
for p in sorted(glob.glob(os.path.join(R, "*_checks.json")), key=os.path.getmtime):
    if os.path.getmtime(p) < T0:
        continue
    tag = os.path.basename(p)[:-len("_checks.json")]
    try:
        d = json.load(open(p))
        runs = {(r["key"], r["arm"]): r for r in json.load(open(os.path.join(R, tag + "_runs.json")))}
    except Exception:
        continue
    recs = {}
    for rec in d:
        if rec.get("paused") or len(rec["ck"]) < 6:
            continue
        r = runs.get((rec["key"], rec["arm"]))
        recs[(rec["key"], rec["arm"])] = (rec, bool(r and r.get("fast")))
    if recs:
        tags[tag] = (recs, time.strftime("%a %H:%M", time.localtime(os.path.getmtime(p))))
print(f"FAR PAIRS ({time.strftime('%a %Y-%m-%d %H:%M')}): paired per-check differences between runs with identical checks; {len(tags)} tags")
print("pair (A -> B) | mode | prompts, checks | misses A -> B | service ms A -> B | check ms A -> B | mean d_check / mean d_service | "
      "OLS slope of d_check on d_service (through 0) | R2")
want = [(sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv) - 1, 2)]
names = sorted(tags)
out = []
for ia, a in enumerate(names):
    for b in names[ia + 1:]:
        A, B = tags[a][0], tags[b][0]
        D = []; np_ = 0; fast = set()
        for key in A:
            ks = [k for k in B if k[0] == key[0]]
            for kb in ks:
                ra, fa = A[key]; rb, fb = B[kb]
                n = min(len(ra["ck"]), len(rb["ck"]))
                if n < 6 or ra["rows"][:n] != rb["rows"][:n] or ra["kept"][:n] != rb["kept"][:n]:
                    continue
                np_ += 1; fast.add(("FAST" if fa else "EXACT") + "/" + ("FAST" if fb else "EXACT"))
                for i in range(2, n):
                    D.append((ra["ck"][i], rb["ck"][i], ra["svc"][i], rb["svc"][i], ra["miss"][i], rb["miss"][i]))
                break
        if np_ < 3 or len(D) < 60:
            continue
        D = np.array(D); dck = D[:, 1] - D[:, 0]; dsv = D[:, 3] - D[:, 2]
        if abs(dsv.mean()) < 2.0 and (a, b) not in want and (b, a) not in want:
            continue
        keep = np.abs(dck - np.median(dck)) < 6 * (np.median(np.abs(dck - np.median(dck))) + 1e-9) + 20      # drop paused / hiccup checks
        x, y = dsv[keep], dck[keep]
        X = np.stack([np.ones(len(x)), x], 1)
        co = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - ((y - X @ co) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        s0 = float((x * y).sum() / max((x * x).sum(), 1e-9))
        out.append((abs(dsv.mean()), f"  {a} ({tags[a][1]}) -> {b} ({tags[b][1]}) | {'+'.join(sorted(fast))} | {np_}, {len(x)} | "
                    f"{D[keep, 4].mean():.0f} -> {D[keep, 5].mean():.0f} | {D[keep, 2].mean():.1f} -> {D[keep, 3].mean():.1f} | {D[keep, 0].mean():.1f} -> "
                    f"{D[keep, 1].mean():.1f} | {y.mean() / x.mean() if abs(x.mean()) > 1e-9 else float('nan'):.2f} | {co[1]:.2f} ({s0:.2f}) | {r2:.2f}"))
for _, line in sorted(out, reverse=True):
    print(line)
print(f"({len(out)} pairs; {time.time() - (T0 + 4 * 86400):.0f} s)")
