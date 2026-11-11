"""Fit12 T2 FIXED COST PROFILE (Sun 8 Nov 11:4x; pre-registration: /Users/kamen/Fit12/PLAN_12GB_90_fit12.md 'T2 FIXED COST
PROFILE'): the reader of /Users/kamen/Fit12/scripts/chw15.sh. Per exact call (t3_twopass.TS, F12_GSTAT=1): stamps t0..t5 =
start, after forward(), after mx.eval, after tolist + compare, after commit, after the lsha hash; the loader's (proc_s, io_s,
cpu_wait_s, misses, stalled, uses) at t0, t1, t2. Disk bytes per call from the CALLS log (same order); draft ms per check
from the phase log. Prints the per-stage table by (first / later call, width), the intercept + slope of each stage against
the call's GB read, and the per-check split. Usage: python -I -B chw15_read.py [DIR] [TAG]"""
import glob, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "/Users/kamen/Fit12/data/scr/chw15"
TAG = sys.argv[2] if len(sys.argv) > 2 else "chw15p"
rows, ph_all = [], []
for f in sorted(glob.glob(os.path.join(D, f"{TAG}_*_ex_ts.json"))):
    ts = json.load(open(f))
    cl = json.load(open(f.replace("_ts.json", "_calls.json")))
    ph = json.load(open(f.replace("_ts.json", "_ph.json")))
    assert len(ts) == len(cl), (f, len(ts), len(cl))
    for a, c in zip(ts, cl):
        assert a[:4] == c[:4], (f, a[:4], c[:4])
        t = a[4:10]
        g0, g1, g2 = a[10:16], a[16:22], a[22:28]
        rows.append(dict(first=a[1] == 0, w=a[2], kept=a[3],
                         build=(t[1] - t[0]) * 1e3, ev=(t[2] - t[1]) * 1e3, host=(t[3] - t[2]) * 1e3,
                         commit=(t[4] - t[3]) * 1e3, hsh=(t[5] - t[4]) * 1e3, tot=(t[5] - t[0]) * 1e3,
                         proc=(g2[0] - g1[0]) * 1e3, io=(g2[1] - g1[1]) * 1e3, proc_b=(g1[0] - g0[0]) * 1e3, io_b=(g1[1] - g0[1]) * 1e3,
                         miss=g2[3] - g1[3], stalled=g2[4] - g1[4], gb=(c[10] - c[6]) / 1e9,
                         cpu_f=(a[29] - a[28]) * 1e3 if len(a) >= 31 else float("nan"),   # (chw16) main-thread CPU
                         cpu_e=(a[30] - a[29]) * 1e3 if len(a) >= 31 else float("nan")))  # over t0..t1, t1..t2
    ph_all += [p for p in ph]
if not rows:
    sys.exit(f"no {TAG} TS rows in {D}")
for r in rows:
    r["ovh"] = r["proc"] - r["io"]
    r["idle"] = r["ev"] - r["proc"]
    r["busy"] = r["proc_b"] + r["proc"]                 # (v2) the loader busy over t0..t2 (forward() syncs inside: the reads
    r["rd"] = r["io_b"] + r["io"]                       # mostly land in the build window); rd = its reads
    r["nl"] = r["build"] + r["ev"] - r["busy"]          # (v2) t0..t2 with the loader idle: Python + GPU critical path + syncs
K = ["build", "ev", "proc", "io", "ovh", "idle", "host", "commit", "hsh", "tot", "busy", "rd", "nl"]
if all(r["cpu_f"] == r["cpu_f"] for r in rows):      # (chw16) the CPU stamps exist
    K += ["cpu_f", "cpu_e"]
print(f"T2 FIXED COST PROFILE ({D}, {TAG}): {len(rows)} exact calls; ms per call (mean). ev = mx.eval window = loader proc"
      f" (io + ovh) + loader idle (the GPU side's path); proc_b = loader busy while forward() built the graph")
print("  group            n   GB   miss | " + " ".join(f"{k:>6}" for k in K) + " | proc_b")
grp = {}
for r in rows:
    grp.setdefault(("first" if r["first"] else "later", r["w"]), []).append(r)
for g in sorted(grp):
    v = grp[g]
    m = {k: np.mean([r[k] for r in v]) for k in K + ["gb", "miss", "proc_b"]}
    print(f"  {g[0]:5} w{g[1]}    {len(v):5} {m['gb']:5.2f} {m['miss']:6.0f} | " + " ".join(f"{m[k]:6.1f}" for k in K)
          + f" | {m['proc_b']:5.2f}")
m = {k: np.mean([r[k] for r in rows]) for k in K + ["gb"]}
print(f"  all          {len(rows):5} {m['gb']:5.2f}        | " + " ".join(f"{m[k]:6.1f}" for k in K))
ws = sorted({r["w"] for r in rows})
X = np.array([[1.0 * (r["w"] == w) for w in ws] + [r["gb"]] for r in rows])
print("  fit per stage: ms = a[w] + b * GB (least squares over all calls; a[w] = the stage's fixed part at width w)")
for k in K:
    y = np.array([r[k] for r in rows])
    cf, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = float(np.sqrt(np.mean((X @ cf - y) ** 2)))
    print(f"    {k:6}: " + ", ".join(f"a[{w}] {cf[i]:6.1f}" for i, w in enumerate(ws)) + f"; b {cf[-1]:6.1f} ms per GB; rms {rms:.1f}")
dr = [(p[6] - p[2]) * 1e3 for p in ph_all if p[0] == "draft"]
ex = [(p[6] - p[2]) * 1e3 for p in ph_all if p[0] == "exact"]
tot_calls = sum(r["tot"] for r in rows)
print(f"  per check: draft {np.mean(dr):.1f} ms (n {len(dr)}), exact phase {np.mean(ex):.1f} ms (n {len(ex)}); exact calls sum"
      f" {tot_calls / 1e3:.1f} s vs exact phases {sum(ex) / 1e3:.1f} s, drafts {sum(dr) / 1e3:.1f} s")
sh = {k: sum(r[k] for r in rows) / max(1e-9, sum(ex) + sum(dr)) for k in ["build", "proc", "io", "ovh", "idle", "host", "commit", "hsh", "busy", "rd", "nl"]}
print("  share of draft + exact time: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in sh.items()))
json.dump(dict(n=len(rows), groups={f"{g[0]}_w{g[1]}": {k: float(np.mean([r[k] for r in v])) for k in K + ["gb", "miss"]}
                                     | {"n": len(v)} for g, v in grp.items()}, share=sh,
               draft_ms=float(np.mean(dr)), exact_ms=float(np.mean(ex))),
          open(os.path.join(D, "chw15_prof.json"), "w"), indent=1)
