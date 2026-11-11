"""Fit12 S1u profile reader (Mon 9 Nov 23:1x; NewIdeas' number check 22:55: the 108.2 ms 'outside reads' is a fit split [P]).
READ-ONLY. Reads <tag>_<key>_<arm>_p1ts.json (t3_onecall.TS1, F12_GSTAT=1): per wide pass-1 call, stamps t0 (start), t1
(graph built), t2 (eval done), t3 (commit done); the loader's (proc_s, io_s, cpu_wait_s, misses, stalled, uses) at t0, t1,
t2; the main thread's CPU at t0, t1, t2; the process's disk bytes at t0, t2. Per call [K]:
- wall = t2 - t0 (the call; commit apart); busy = the loader's proc time over t0..t2; io = its read time over t0..t2;
- idle = wall - busy: the call's time with the loader idle (Python + the GPU's critical path + syncs). This is the measured
  "outside reads"; busy may still overlap GPU work, so idle is a lower bound of the non-read work (chw16's 'nl' convention);
- cpu_build = the main thread's CPU over t0..t1 (Python graph build); GB = disk bytes over t0..t2.
Prints means and medians, the read rate (GB / io), a least-squares busy-vs-GB fit, and the pre-registered line (outside
reads <= 90 ms) on the MEASURED idle time. Usage: python -I -B s1u_prof.py RUN_DIR TAG ARM"""
import glob, json, os, statistics as st, sys
import numpy as np

D, TAG, ARM = sys.argv[1], sys.argv[2], sys.argv[3]
R = []
for f in sorted(glob.glob(os.path.join(D, f"{TAG}_*_{ARM}_p1ts.json"))):
    for a in json.load(open(f)):
        t0, t1, t2, t3 = a[4:8]
        g0, g1, g2 = a[8:14], a[14:20], a[20:26]
        c0, c1, c2 = a[26:29]
        b0, b1 = a[29:31]
        wall = (t2 - t0) * 1e3
        busy = (g2[0] - g0[0]) * 1e3
        R.append(dict(rows=a[2], kept=a[3], wall=wall, build=(t1 - t0) * 1e3, ev=(t2 - t1) * 1e3, commit=(t3 - t2) * 1e3,
                      busy=busy, io=(g2[1] - g0[1]) * 1e3, idle=wall - busy, cpu_build=(c1 - c0) * 1e3, cpu_ev=(c2 - c1) * 1e3,
                      gb=(b1 - b0) / 1e9, miss=g2[3] - g0[3], stalled=g2[4] - g0[4]))
if not R:
    sys.exit(f"no {TAG} {ARM} p1ts rows in {D}")
K = ["wall", "build", "ev", "commit", "busy", "io", "idle", "cpu_build", "cpu_ev", "gb", "miss", "stalled"]
L = [f"S1u PROFILE ({D}, {TAG}:{ARM}): {len(R)} wide calls, {st.mean(r['rows'] for r in R):.1f} rows, kept {st.mean(r['kept'] for r in R):.2f} [K]"]
L.append("  " + " | ".join(f"{k} {st.mean(r[k] for r in R):.{3 if k == 'gb' else 1}f} (med {st.median(r[k] for r in R):.{3 if k == 'gb' else 1}f})" for k in K))
gb = np.array([r["gb"] for r in R]); bu = np.array([r["busy"] for r in R]); io = np.array([r["io"] for r in R])
A = np.vstack([np.ones_like(gb), gb]).T
(a_, b_), *_ = np.linalg.lstsq(A, bu, rcond=None)
L.append(f"  read rate: {gb.sum() / (io.sum() / 1e3):.2f} GB/s over the loader's io time; busy = {a_:.1f} + {b_:.1f} ms per GB (fit "
         f"over {len(R)} calls; the chw16 exact-call fit was 83.7 ms per GB)")
idle = st.mean(r["idle"] for r in R)
L.append(f"LINE (pre-registered, now on the measured split): outside reads = the loader-idle time {idle:.1f} ms (<= 90) -> "
         f"{'PASS' if idle <= 90 else 'FAIL'}")
print("\n".join(L))
