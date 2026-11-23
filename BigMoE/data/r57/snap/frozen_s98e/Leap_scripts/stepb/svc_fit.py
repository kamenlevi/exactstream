"""SVC-FIT (Leap, Sat 2026-11-07 00:3x; the orchestrator 23:5x: price NW v3 (S1) against the cycle floor, not GPU time
alone). Per real 16-row check of tonight's C 224 runs (steady passes: real 2, real 3, the commit pass; checks 2..12):
how does check + commit grow with the gate's service time (svc: the ext's reads + releases for the check, ms) and the
misses? A slope ~1 on svc above its floor means the expert kernels wait (spin) on the release for that long: the time
NW v3 takes off the critical path (late demand pairs, dropped optional uses, 3-bit cold sidecars).
usage: svc_fit.py <flat2.json> [more] -> stdout (and ~/Leap/data/stepb/svc_fit.txt)"""
import json
import os
import sys

import numpy as np

OUT = os.path.expanduser("~/Leap/data/stepb/svc_fit.txt")


def say(s):
    print(s, flush=True)
    open(OUT, "a").write(s + "\n")


def rows_of(paths):
    out = []
    for p in paths:
        d = json.load(open(p))
        real = [r for r in d["runs"] if r.get("real")]
        for r in real[1:]:                               # the first real pass loads its rows (discarded)
            pc = r["pc"]
            for i in range(1, len(pc["ck"])):
                out.append(dict(run=os.path.basename(p).replace("_flat2.json", ""), label=r["label"], ck=pc["ck"][i],
                                svc=pc["svc"][i], io=pc["io"][i], miss=pc["miss"][i], rows=pc["rows"][i],
                                uses=pc["uses"][i], stl=pc["stl"][i], drf=pc["drf"][i], bld=pc["bld"][i], wt=pc["wt"][i]))
    return out


def fit(xs, ys):
    A = np.vstack([xs, np.ones_like(xs)]).T
    (b, a), res, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    r = np.corrcoef(xs, ys)[0, 1]
    return a, b, r


def main(paths):
    R = rows_of(paths)
    say(f"\nSVC-FIT {len(R)} checks from {len(paths)} runs ({', '.join(sorted({x['run'] for x in R}))})")
    for key in ("svc", "io", "miss", "stl"):
        xs = np.array([x[key] for x in R], float)
        ys = np.array([x["ck"] for x in R], float)
        a, b, r = fit(xs, ys)
        say(f"  check + commit = {a:.2f} + {b:.3f} x {key}  (r {r:.2f}); {key} median {np.median(xs):.2f}, p10-p90 "
            f"{np.percentile(xs, 10):.2f}-{np.percentile(xs, 90):.2f}")
    for run in sorted({x["run"] for x in R}):
        S = [x for x in R if x["run"] == run]
        xs = np.array([x["svc"] for x in S], float)
        ys = np.array([x["ck"] for x in S], float)
        a, b, r = fit(xs, ys)
        say(f"  {run}: {len(S)} checks, check + commit median {np.median(ys):.2f}, svc median {np.median(xs):.2f}; "
            f"fit {a:.2f} + {b:.3f} x svc (r {r:.2f})")
    xs = np.array([x["svc"] for x in R], float)
    ys = np.array([x["ck"] for x in R], float)
    a, b, r = fit(xs, ys)
    med_svc = float(np.median(xs))
    say(f"  -> if the release wait left the critical path entirely (NW v3's ideal), check + commit falls by ~b x svc = "
        f"{b * med_svc:.2f} ms at the median svc ({med_svc:.2f}); vs the median check + commit {np.median(ys):.2f} "
        f"and a ~68 ms cycle: x{68 / max(1.0, 68 - b * med_svc):.3f} tok/s (upper bound; the cycle floor ~40 ms CPU is "
        f"not reached)")


if __name__ == "__main__":
    main(sys.argv[1:])
