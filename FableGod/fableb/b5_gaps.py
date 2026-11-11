"""b5_gaps.py (Fable B, mission B2): the GPU's idle gaps inside one eager 16-row check, from Leap's B5 Metal System Trace
(~/Leap/data/b5_gpu_intervals.xml: xctrace's metal-gpu-intervals = command-buffer-level GPU activity; the Oct-25 engine,
12-layer resident slice, variant A eager). Per measured check (b5_trace_wall.txt t0 / t2, aligned as Leap's b5_gpu.py):
busy intervals, their durations, and the gaps between consecutive intervals (histogram). CPU only; reads Leap's files."""
import os
import xml.etree.ElementTree as ET

D = os.path.expanduser("~/Leap/data")
PID = "8882"
ids, rows = {}, []
for row in ET.parse(f"{D}/b5_gpu_intervals.xml").getroot().iter("row"):
    vals = []
    for el in row:
        for sub in el.iter():
            if "id" in sub.attrib:
                ids[sub.attrib["id"]] = sub
        vals.append(ids[el.attrib["ref"]] if "ref" in el.attrib else el)
    start, dur, chan, depth, proc = int(vals[0].text), int(vals[1].text), vals[2].text, int(vals[5].text), vals[10]
    pid = proc.find("pid")
    if pid is None and "ref" in proc.attrib:
        pid = ids[proc.attrib["ref"]].find("pid")
    pid = ids[pid.attrib["ref"]] if pid is not None and "ref" in pid.attrib else pid
    rows.append((start, start + dur, chan, depth, pid.text if pid is not None else "?"))
mine = sorted((a, z, d) for a, z, c, d, p in rows if p == PID)
depths = sorted({d for a, z, d in mine})
print("intervals of the traced process: %d; depths %s; per depth %s" % (len(mine), depths, {d: sum(1 for a, z, dd in mine if dd == d) for d in depths}))
# merged busy union (as Leap) and the measured block after the last >= 0.8 s gap
union = []
for a, z, d in mine:
    if union and a <= union[-1][1]:
        union[-1][1] = max(union[-1][1], z)
    else:
        union.append([a, z])
gaps = [(union[i][1], union[i + 1][0]) for i in range(len(union) - 1) if union[i + 1][0] - union[i][1] > 0.8e9]
b0, b1 = gaps[-1][1], union[-1][1]
st = [tuple(map(float, l.split()[:2])) for l in open(f"{D}/b5_trace_wall.txt")]
off = b1 / 1e9 - st[-1][1]
hist_all = [0, 0, 0, 0, 0]
print("per check: wall, busy, intervals (merged), gaps between consecutive busy intervals: count, sum, histogram (< 5 us, 5-20, 20-100, 100-1000, > 1 ms)")
for k, (t0, t2) in enumerate(st):
    w0, w1 = (t0 + off) * 1e9, (t2 + off) * 1e9
    blk = [(max(a, w0), min(z, w1)) for a, z in union if z > w0 and a < w1]
    busy = sum(z - a for a, z in blk) / 1e6
    gs = [(blk[i + 1][0] - blk[i][1]) / 1e3 for i in range(len(blk) - 1)]   # us
    h = [sum(1 for g in gs if g < 5), sum(1 for g in gs if 5 <= g < 20), sum(1 for g in gs if 20 <= g < 100),
         sum(1 for g in gs if 100 <= g < 1000), sum(1 for g in gs if g >= 1000)]
    if k >= 2:
        hist_all = [x + y for x, y in zip(hist_all, h)]
    first = (blk[0][0] - w0) / 1e6 if blk else float("nan")
    print("  check %d: wall %6.2f ms, busy %6.2f, first GPU work after t0 %5.2f ms; %3d busy intervals, median %6.1f us (max %7.1f); %3d gaps, sum %5.2f ms, %s"
          % (k, (w1 - w0) / 1e6, busy, first, len(blk), sorted((z - a) / 1e3 for a, z in blk)[len(blk) // 2] if blk else 0,
             max(((z - a) / 1e3 for a, z in blk), default=0), len(gs), sum(gs) / 1e3, h))
print("checks 2-9 gap histogram (< 5 us, 5-20, 20-100, 100-1000, > 1 ms): %s" % hist_all)
