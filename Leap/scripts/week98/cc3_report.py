"""cc3_report.py (Leap, Sat 14 Nov 2026): the report of cc3_runner.sh (WEEK98 route 3 screen 3). Per run tag: the loop's
warm split (runs.json "wspl": the 16-row check cold, then warm at 1 / 4 / 8 / 16 rows, each rolled back) joined with the
probe's dense-only passes (d1 / d16 / d32: the routed-expert call stubbed) and the real 32-row verify (s32, slot path).
Medians over the probe checks; per layer = / 48. Derived: experts E1 = warm1 - d1, E16 = warm16 - d16; misses M16 = cold16
- warm16; the experts' per-row slope 8 -> 16 rows; a widened-gate 32-row estimate = d32 + E16 + 16 x slope x 0.72 (new
distinct experts per added row fall from ~3.0 at 8-16 rows to ~2.15 at 16-32, D32 / L1-24) + M16 x 1.5 (unmeasured).
Usage: cc3_report.py <data dir>"""
import glob
import json
import os
import re
import statistics as st
import sys

D = sys.argv[1]
TAGS = ("a45", "a45b", "a45c", "b24", "b24b")


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


for tag in TAGS:
    pj = os.path.join(D, f"{tag}_probe.jsonl")
    rr = sorted(glob.glob(os.path.join(D, tag, "cli_*_runs.json")))
    if not os.path.exists(pj) and not rr:
        continue
    probes = [json.loads(x) for x in open(pj)] if os.path.exists(pj) else []
    runs = json.load(open(rr[-1])) if rr else []
    txt = open(os.path.join(D, f"{tag}.txt")).read() if os.path.exists(os.path.join(D, f"{tag}.txt")) else ""
    agree = re.findall(r"tokens == REF2 row (\d+) / (\d+)", txt)
    peak = re.findall(r"lifetime peak ([0-9.]+) GB", txt)
    W = {"cold16": [], "w1": [], "w4": [], "w8": [], "w16": [], "miss_cold": [], "miss_w16": [], "rows": []}
    for r in runs:
        for s in (r.get("wspl") or []):
            c = s.get("cold")
            if c:
                W["cold16"].append(c[1]); W["miss_cold"].append(c[2]); W["rows"].append(c[0])
            for w in s.get("warm", []):
                k = {1: "w1", 4: "w4", 8: "w8"}.get(w[0], "w16" if w[0] >= 13 else None)
                if k:
                    W[k].append(w[1])
                    if k == "w16":
                        W["miss_w16"].append(w[2])
    Pp = {}
    for p in probes:
        for q in p.get("passes", []):
            Pp.setdefault(q["p"], []).append(q)
    pm = {k: med([q["ms"] for q in v]) for k, v in Pp.items()}
    pmiss = {k: med([q["miss"] for q in v]) for k, v in Pp.items()}
    errs = [p["err"] for p in probes if p.get("err")]
    m = {k: med(v) for k, v in W.items()}
    print(f"== {tag}: runs {len(runs)}, probe checks {len(probes)}, errors {len(errs)} {errs[:1]}; tokens == REF2 {agree}; "
          f"peak {peak[-1] if peak else '-'} GB")
    print(f"   warm split (n {len(W['cold16'])}, rows {f(m['rows'], 0)}): cold {f(m['cold16'])} ms (misses {f(m['miss_cold'], 0)}); "
          f"warm 1 / 4 / 8 / 16 rows {f(m['w1'])} / {f(m['w4'])} / {f(m['w8'])} / {f(m['w16'])} ms")
    print(f"   probe: dense-only 1 / 16 / 32 rows {f(pm.get('d1'))} / {f(pm.get('d16'))} / {f(pm.get('d32'))} ms; "
          f"real 32-row verify (slot path) {f(pm.get('s32'))} ms, misses {f(pmiss.get('s32'), 0)} (n {len(Pp.get('s32', []))})")
    E1 = m["w1"] - pm["d1"] if m["w1"] is not None and pm.get("d1") is not None else None
    E16 = m["w16"] - pm["d16"] if m["w16"] is not None and pm.get("d16") is not None else None
    M16 = m["cold16"] - m["w16"] if m["cold16"] is not None and m["w16"] is not None else None
    sl = (m["w16"] - m["w8"]) / 8 if m["w16"] is not None and m["w8"] is not None else None
    print(f"   split at 16 rows: dense {f(pm.get('d16'))} + experts {f(E16)} + misses {f(M16)} = cold {f(m['cold16'])} ms; "
          f"at 1 row: dense {f(pm.get('d1'))} + experts {f(E1)}; expert slope 8->16 {f(sl, 2)} ms per row")
    if pm.get("d32") is not None and E16 is not None and sl is not None and M16 is not None:
        est = pm["d32"] + E16 + 16 * sl * 0.72 + 1.5 * M16
        print(f"   32 rows: today = 2 x 16 gated ~{f(2 * m['cold16'])} ms (exact) or the slot path {f(pm.get('s32'))} ms; "
              f"a gate carrying 320 ids ~{f(est)} ms (estimate); dense 32 / 16 = {f(pm['d32'] / pm['d16'], 2)}")
    print("   per layer (/ 48): " + ", ".join(f"{k} {f(v / 48, 2)}" for k, v in
                                          (("cold16", m["cold16"]), ("w1", m["w1"]), ("w16", m["w16"]), ("d1", pm.get("d1")),
                                           ("d16", pm.get("d16")), ("d32", pm.get("d32")), ("s32", pm.get("s32"))) if v is not None))
