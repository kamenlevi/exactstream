"""c25_report.py (Leap, Mon 16 Nov 2026 13:2x; CHECK25): the c25 engine A/B report from c25.json (which c25_engine dumps
after every window, so a run cut by its cap still reports its finished windows). Per arm: median whole check (verify
build + submit, the wait, the commit), expert GB per check (an upper bound: tier-0 experts counted at full 4-bit),
misses, overflowed checks; then the paired per-round diffs / ratios, and the efficiency vs the 280 GB/s floor for the
arms whose bytes are known (C5: 2.13 GB dense + the measured expert GB; C6 likewise).
usage: python3 c25_report.py [c25.json] [--md]"""
import json
import os
import statistics as st
import sys

P = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.expanduser(
    "~/Leap/data/week98/c25/c25.json")
d = json.load(open(P))
REC = [r for r in d["rec"] if r["phase"] == "same"]
arms = d["arms"]
W = {}
for r in REC:
    W.setdefault((r["key"], r["round"]), {})[r["arm"]] = r
med = lambda xs: st.median(xs) if xs else float("nan")
q = lambda xs: (sorted(xs)[len(xs) // 4], sorted(xs)[(3 * len(xs)) // 4]) if len(xs) >= 4 else (float("nan"),) * 2
DENSE_GB = {"F32c5": 2.133, "F32c6i": 2.133, "F32c6g": 2.133}           # C5's dense bytes (Fit12 lossy32_price.json dense l45_v1v3)
out = []
p = out.append
p(f"C25 REPORT from {P}: {len(W)} (window, round) samples, windows {sorted({k for k, _ in W})}")
p("| arm | checks | median ms | build + submit | wait | commit | expert GB (upper) | misses | overflowed |")
p("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
summ = {}
for a in arms:
    rs = [v[a] for v in W.values() if a in v]
    if not rs:
        continue
    tot = [1e3 * sum(r["w"]) for r in rs]
    summ[a] = dict(ms=med(tot), gb=med([r["exp_gb"] for r in rs]))
    p(f"| {a} | {len(rs)} | {med(tot):.2f} | {med([1e3 * r['w'][0] for r in rs]):.2f} | {med([1e3 * r['w'][1] for r in rs]):.2f} | "
      f"{med([1e3 * r['w'][2] for r in rs]):.2f} | {med([r['exp_gb'] for r in rs]):.2f} | {med([r['miss'] for r in rs]):.1f} | "
      f"{sum(1 for r in rs if r.get('ovf'))} |")
p("")
PAIRS = [("F32s", "F16", "ratio"), ("F32o", "F32s", "diff"), ("F32d", "F32o", "diff"), ("F32df", "F32d", "diff"),
         ("F32if", "F32df", "diff"), ("F32if6", "F32if", "diff"), ("F32c5", "F32df", "diff"), ("F32c6i", "F32c5", "diff"),
         ("F32df", "F32s", "diff"), ("F32c5", "F32s", "diff"), ("F32c6i", "F32s", "diff"), ("F32c5", "F16", "ratio"),
         ("F32c6i", "F16", "ratio"), ("F32g", "F32o", "diff"), ("F32c6g", "F32c6i", "diff"), ("F32c6g", "F16", "ratio")]
p("| pair | median | IQR | rounds |")
p("|---|---:|---|---:|")
for x, y, kind in PAIRS:
    xs = [(sum(v[x]["w"]) / sum(v[y]["w"])) if kind == "ratio" else 1e3 * (sum(v[x]["w"]) - sum(v[y]["w"]))
          for v in W.values() if x in v and y in v]
    if not xs:
        continue
    lo, hi = q(xs)
    unit = "" if kind == "ratio" else " ms"
    p(f"| {x} {'/' if kind == 'ratio' else '-'} {y} | {med(xs):+.3f}{unit} | [{lo:+.3f}, {hi:+.3f}] | {len(xs)} |")
p("")
for a, dg in DENSE_GB.items():
    if a in summ:
        gb = dg + summ[a]["gb"]
        p(f"{a}: {gb:.2f} GB per check (dense {dg} + experts {summ[a]['gb']:.2f}, upper) -> floor {gb / 0.28:.1f} ms at 280 GB/s; "
          f"measured {summ[a]['ms']:.1f} ms = {summ[a]['ms'] / (gb / 0.28):.2f}x the floor (whole check incl. the commit)")
txt = "\n".join(out)
print(txt)
if "--md" in sys.argv:
    open(P.replace(".json", "_report.md"), "w").write(txt + "\n")
