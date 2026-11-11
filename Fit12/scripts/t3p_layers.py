"""Fit12 T3P (Tue 3 Nov 2026): per-layer split of T3's phases from the loader's call log (BIGMOE_CALLLOG: one row per
gate call = layer L (base pools 0..47, LSB sidecars side0 + L), pass seq, misses, t_seen = the loader saw the routing row,
t_rel = it released the GPU) and fit_t3.py's phase marks (F12_PHLOG=1: name, check, then at start and at end the
CLOCK_UPTIME_RAW time, gate seq, disk bytes, misses). Both clocks are CLOCK_UPTIME_RAW, so each call is matched to its
phase by time. Per phase kind (p1 = pass 1, exact = pass 2 / the exact check), per gated pass:
  wait  = sum over calls of t_rel - t_seen   (the loader holds the GPU: reads + table work; GPU idle at the gate)
  gap   = sum over consecutive trunk layers of t_seen(L + 1) - t_rel(L)  (GPU side: tier_glu of L, attention + router of
          L + 1, the route copy and the loader's poll latency)
  edge  = the phase's time outside the gated span (embedding / first attention before layer 0, the head after layer 47)
Usage: t3p_layers.py CALLS.csv 'PHASE_GLOB' [ARM_FILTER]   (CPU only)"""
import collections, csv, glob, json, os, sys

calls = [dict(L=int(r["L"]), seq=int(r["seq"]), m=int(r["misses"]), ts=float(r["t_seen"]), tr=float(r["t_rel"]))
         for r in csv.DictReader(open(sys.argv[1]))]
calls.sort(key=lambda c: c["ts"])
TS = [c["ts"] for c in calls]
NL = 48
side_Ls = sorted({c["L"] for c in calls if c["L"] >= NL})
side0 = side_Ls[0] if side_Ls else None
if side_Ls and len(side_Ls) != NL:   # an MTP layer at 48 would shift side0: take the 48 highest
    side0 = side_Ls[-NL]
trunk = lambda L: L - side0 if side0 is not None and L >= side0 else (L if L < NL else None)
filt = sys.argv[3] if len(sys.argv) > 3 else None

import bisect
agg = collections.defaultdict(lambda: dict(phases=0, passes=0, t=0.0, wait=0.0, gap=0.0, edge=0.0, misses=0, calls=0,
                                           bytes=0, wait0=0.0, n0=0, lw=[0.0] * NL, lg=[0.0] * NL, lm=[0] * NL))
for f in sorted(glob.glob(sys.argv[2])):
    if filt and filt not in os.path.basename(f):
        continue
    kind_tag = os.path.basename(f)[:-len("_ph.json")]
    for name, chk, t0, s0, d0, m0, t1, s1, d1, m1 in json.load(open(f)):
        a = agg[name]
        a["phases"] += 1; a["t"] += t1 - t0; a["bytes"] += d1 - d0
        i0, i1 = bisect.bisect_left(TS, t0), bisect.bisect_right(TS, t1)
        win = calls[i0:i1]
        by = collections.defaultdict(list)
        for c in win:
            by[c["seq"]].append(c)
        span_all = 0.0
        for seq, E in by.items():
            a["passes"] += 1
            E.sort(key=lambda c: c["ts"])
            span_all += E[-1]["tr"] - E[0]["ts"]
            per_L = collections.defaultdict(list)
            for c in E:
                w = c["tr"] - c["ts"]
                a["wait"] += w; a["misses"] += c["m"]; a["calls"] += 1
                if c["m"] == 0:
                    a["wait0"] += w; a["n0"] += 1
                tL = trunk(c["L"])
                if tL is not None:
                    per_L[tL].append(c)
                    a["lw"][tL] += w; a["lm"][tL] += c["m"]
            Ls = sorted(per_L)
            for L_, L2 in zip(Ls, Ls[1:]):
                g = min(c["ts"] for c in per_L[L2]) - max(c["tr"] for c in per_L[L_])
                a["gap"] += g; a["lg"][L2] += g
        a["edge"] += (t1 - t0) - span_all

for name, a in agg.items():
    P = max(1, a["passes"]); ph = max(1, a["phases"])
    print(f"{name}: {a['phases']} phases, {a['passes']} gated passes ({a['passes'] / ph:.2f} per phase), "
          f"{1e3 * a['t'] / ph:.1f} ms per phase | per pass: wait {1e3 * a['wait'] / P:.1f} ms, gap {1e3 * a['gap'] / P:.1f} ms, "
          f"misses {a['misses'] / P:.0f}, calls {a['calls'] / P:.0f} | edge {1e3 * a['edge'] / ph:.1f} ms per phase | "
          f"bytes {a['bytes'] / P / 1e6:.0f} MB per pass, {a['bytes'] / max(1, a['misses']) / 1e6:.2f} MB per miss, "
          f"{a['bytes'] / max(1e-9, a['wait']) / 1e9:.2f} GB/s over the wait | zero-miss calls: {a['n0']} "
          f"({1e6 * a['wait0'] / max(1, a['n0']):.0f} us each)")
    band = lambda v: " ".join(f"{1e3 * sum(v[b:b + 8]) / P:5.1f}" for b in range(0, NL, 8))
    print(f"  per 8 layers (0-7 .. 40-47), ms per pass: wait {band(a['lw'])} | gap {band(a['lg'])} | misses "
          + " ".join(f"{sum(a['lm'][b:b + 8]) / P:5.0f}" for b in range(0, NL, 8)))
