"""B5: GPU busy vs wall per 16-row check from the Metal System Trace of b5_slice.py trace (variant A, eager).
Reads data/b5_gpu_intervals.xml (xctrace export of the metal-gpu-intervals table; id/ref compressed) and
data/b5_trace_wall.txt (perf_counter t0 / t2 per measured check). The measured block runs from the last >= 0.8 s gap in
python's GPU work to its end (the closing 1 s sleep has no GPU work after it); clocks are aligned on the block's last GPU end = last check's t2.
Usage: python3 b5_gpu.py [pid] -> ~/Leap/data/b5_gpu.txt"""
import os, sys
import xml.etree.ElementTree as ET

D = os.path.expanduser("~/Leap/data")
PID = sys.argv[1] if len(sys.argv) > 1 else "8882"

ids, rows = {}, []
for row in ET.parse(f"{D}/b5_gpu_intervals.xml").getroot().iter("row"):
    vals = []
    for el in row:
        for sub in el.iter():                                   # register every element with an id (refs point to any)
            if "id" in sub.attrib:
                ids[sub.attrib["id"]] = sub
        vals.append(ids[el.attrib["ref"]] if "ref" in el.attrib else el)
    start, dur, chan, depth, proc = int(vals[0].text), int(vals[1].text), vals[2].text, int(vals[5].text), vals[10]
    pid = proc.find("pid")
    if pid is None and "ref" in proc.attrib:
        pid = ids[proc.attrib["ref"]].find("pid")
    pid = ids[pid.attrib["ref"]] if pid is not None and "ref" in pid.attrib else pid
    rows.append((start, start + dur, chan, depth, pid.text if pid is not None else "?"))

mine = sorted((a, z) for a, z, c, d, p in rows if p == PID)
chans = sorted({(c, d) for a, z, c, d, p in rows if p == PID})
union = []                                                      # merged busy intervals, ns
for a, z in mine:
    if union and a <= union[-1][1]:
        union[-1][1] = max(union[-1][1], z)
    else:
        union.append([a, z])
gaps = [(union[i][1], union[i + 1][0]) for i in range(len(union) - 1) if union[i + 1][0] - union[i][1] > 0.8e9]
assert gaps, "no 1 s gap"
b0, b1 = gaps[-1][1], union[-1][1]                              # measured block: after the last long gap
blk = [(a, z) for a, z in union if a >= b0 and z <= b1]

st = [tuple(map(float, l.split()[:2])) for l in open(f"{D}/b5_trace_wall.txt")]
off = blk[-1][1] / 1e9 - st[-1][1]                              # trace s = perf_counter s + off
out = [f"B5 trace (pid {PID}): {len(mine)} GPU intervals, channels {chans}; block {len(blk)} busy runs, "
       f"{(b1 - b0) / 1e6:.1f} ms long; first GPU start - first t0 = {(blk[0][0] / 1e9 - (st[0][0] + off)) * 1e3:.2f} ms"]
tot_w = tot_b = 0.0
for k, (t0, t2) in enumerate(st):
    w0, w1 = (t0 + off) * 1e9, (t2 + off) * 1e9
    busy = sum(max(0.0, min(z, w1) - max(a, w0)) for a, z in blk) / 1e6
    first = min((a for a, z in blk if a >= w0), default=w1)
    wall = (t2 - t0) * 1e3
    out.append(f"  check {k}: wall {wall:6.2f} ms, GPU busy {busy:6.2f} ms, idle {wall - busy:5.2f} ms "
               f"(first GPU work {(first - w0) / 1e6:5.2f} ms after t0)")
    if k >= 2:
        tot_w += wall; tot_b += busy
n = len(st) - 2
out.append(f"checks 2-{len(st) - 1} (after the GPU clock ramp): wall {tot_w / n:.2f}, busy {tot_b / n:.2f}, idle "
           f"{(tot_w - tot_b) / n:.2f} ms per 12-layer check -> x4 idle {4 * (tot_w - tot_b) / n:.1f} ms per 48-layer check")
print("\n".join(out))
open(f"{D}/b5_gpu.txt", "w").write("\n".join(out) + "\n")
