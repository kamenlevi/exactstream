#!/usr/bin/env python3 -I
"""wall_cutpoints.py (FableGod, mission 14). CPU, plain json. On D-23's 1,000 logged checks (dfl arm, C 224, t 0.3):
how predictable is the position where a check's chain breaks from the drafter's own confidence profile? For the
draft-ahead idea: draft the next block from the block end and from the K lowest-confidence positions while the SSD
is busy; the hit rate is the share of checks whose cut point is among those candidates. Also the median SSD-busy and
draft time per check."""
import json, os, statistics as st
R = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
R = [r for r in R if r.get("arm") == "dfl" and not r.get("invalid")]
print("| type | checks | full keeps | breaks | break at the lowest-cf position | in the 2 lowest | in the 3 lowest | hit with candidates = end + 2 lowest | median io ms / draft ms per check |")
print("|---|---|---|---|---|---|---|---|---|")
for typ in ("code", "math", "chat", "agent"):
    checks = full = breaks = k1 = k2 = k3 = 0; ios = []; drs = []
    for r in R:
        if r["type"] != typ: continue
        ios.append(r["io"]); drs.append(r["dr"])
        for (rows, kept, ck, dr), cf in zip(r["pc"], r["cf"]):
            checks += 1
            if kept >= rows: full += 1; continue
            breaks += 1
            j = int(kept)                                    # the rejected row's draft index
            c = [cf[i] if i < len(cf) else 0.0 for i in range(rows)]
            order = sorted(range(rows), key=lambda i: c[i])  # ascending confidence
            if j == order[0]: k1 += 1
            if j in order[:2]: k2 += 1
            if j in order[:3]: k3 += 1
    print(f"| {typ} | {checks} | {100 * full / checks:.0f} % | {breaks} | {100 * k1 / max(breaks, 1):.0f} % | {100 * k2 / max(breaks, 1):.0f} % | "
          f"{100 * k3 / max(breaks, 1):.0f} % | {100 * (full + k2) / checks:.0f} % | {st.median(ios):.0f} / {st.median(drs):.1f} |")
