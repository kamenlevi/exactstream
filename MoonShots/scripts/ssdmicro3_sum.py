"""M-30 QUEUEMIX summary (MoonShots, Tue 2026-11-03; pre-registered in ~/NewIdeas/QUEUE.md M-30 before any timing):
prices scripts/ssdmicro3.c's raw calls (medians per point) against the lines.
A = the demand call (2 base records, GCD) alone after 0.8 ms idle; A0 = alone back to back.
Lines:
  L-busy (lever 2's premise): over the busy arms (F = 2 / 4 forecast reads issued Delta = 100-800 us before the demand
     call, default policy) and the mixed arms (issued with it): PASS = some arm's demand call is >= 100 us faster than A;
     KILL = every busy and mixed arm's demand call is slower than A; else BETWEEN.
  L-priority: PASS = a throttle / utility arm's demand call is >= 50 us faster than the same arm at the default policy
     AND its forecast reads end <= 1 ms after the demand call's end; KILL = the forecasts end > 2 ms late or the demand
     is < 20 us faster; else BETWEEN.
usage: python ssdmicro3_sum.py raw.tsv [out.txt] -> data/ssdmicro3.txt"""
import os, sys, time
import numpy as np

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/MoonShots/data/ssdmicro3.txt")
rows, meta, cols = {}, [], None
for ln in open(sys.argv[1]):
    f = ln.rstrip("\n").split("\t")
    if ln.startswith("#"):
        meta.append(ln[2:].strip())
    elif f[0] == "round":
        cols = f[6:]
    else:
        rows.setdefault((f[2], int(f[3]), int(f[4]), f[5]), []).append([float(x) for x in f[6:]])
R = {k: np.array(v) for k, v in rows.items()}
C = {c: i for i, c in enumerate(cols)}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def med(k, c="T_us"):
    return float(np.median(R[k][:, C[c]]))


A, A0 = ("A", 0, 0, "default"), ("A0", 0, 0, "default")
n = len(R[A])
say(f"== M-30 QUEUEMIX ({time.strftime('%a %H:%M')}): {len(R)} points x {n} rounds; {'; '.join(meta)}")
say(f"  demand call = 2 base records by GCD: alone after 0.8 ms idle {med(A):.1f} us, back to back {med(A0):.1f}")
say("  set F Delta policy: demand call us (vs alone after idle) | forecast reads: first start, last end vs the demand "
    "call's start, mean read us")
for k in sorted(R, key=lambda k: (k[0], k[1], k[2], k[3])):
    if k[0] in ("A", "A0"):
        continue
    say(f"    {k[0]} {k[1]} {k[2]:4d} {k[3]:8s}: {med(k):7.1f} ({med(k) - med(A):+7.1f}) | {med(k, 'fc_start_us'):7.1f} "
        f"{med(k, 'fc_end_us'):8.1f} {med(k, 'fc_read_us'):7.1f}")
busy = [k for k in R if k[0] in ("B", "M") and k[3] == "default"]
best = min(busy, key=med)
gain = med(A) - med(best)
say(f"  L-busy: best busy / mixed arm {best}: demand {med(best):.1f} us, {gain:+.1f} us vs alone after idle -> "
    + ("PASS" if gain >= 100 else "KILL" if all(med(k) > med(A) for k in busy) else "BETWEEN") + " (line 100 us)")
res = []
for k in R:
    if k[3] in ("throttle", "utility"):
        d = (k[0], k[1], k[2], "default")
        res.append((k, med(d) - med(k), med(k, "fc_end_us") - med(k)))
for k, dg, late in res:
    say(f"    {k}: demand {dg:+.1f} us faster than at the default policy; forecasts end {late:+.0f} us after the demand call")
ok = [r for r in res if r[1] >= 50 and r[2] <= 1000]
say("  L-priority: " + ("PASS" if ok else "KILL" if all(r[2] > 2000 or r[1] < 20 for r in res) else "BETWEEN")
    + " (line: demand >= 50 us faster with the forecasts done <= 1 ms after it)")
for f in (2, 4):
    k = ("C", f, -1, "default")
    say(f"  chained F={f}: forecast reads start {med(k, 'fc_start_us'):.1f} us into the demand call (right after it), "
        f"read {med(k, 'fc_read_us'):.1f} us each; all done {med(k, 'fc_end_us') - med(k):.1f} us after the demand call")
open(OUT, "a").write("\n".join(lines) + "\n")
