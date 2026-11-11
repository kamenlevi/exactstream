#!/usr/bin/env python3 -I
"""safari_gpu.py (FableGod, mission 19, H3). CPU, plain json. Per-check timing from BigMoE's A6 runs at C 160
(~/BigMoE/data/r57/a6{c,k,p}_160_checks.json): per prompt the per-check arrays (rows, kept, ck = check ms, drf = draft
ms, and whatever else the file holds). Medians per arm and prompt type, the draft's share of the cycle, the first check's
excess (K-80), and the GPU-idle share where the file records waits / SSD time."""
import json, os, statistics as st
from collections import defaultdict
D = os.path.expanduser("~/BigMoE/data/r57")
def med(x): x = [v for v in x if v is not None]; return st.median(x) if x else float("nan")
for f in sorted(os.listdir(D)):
    if not (f.startswith("a6") and f.endswith("_checks.json")): continue
    runs = json.load(open(f"{D}/{f}"))
    keys = sorted({k for r in runs for k in r.keys()}); print(f"\n### {f}: {len(runs)} prompts; keys: {', '.join(keys)}")
    arr_keys = [k for k in keys if isinstance(runs[0].get(k), list)]
    by = defaultdict(list)
    for r in runs: by[(r.get("arm"), r["key"].split("/")[0])].append(r)
    print("| arm | type | prompts | checks | kept/check | rows/check | check ms (med) | draft ms (med) | draft share of cycle | first check vs median | " + " | ".join(k for k in arr_keys if k not in ("rows", "kept", "ck", "drf")) + " |")
    print("|---|---|---|---|---|---|---|---|---|---|" + "---|" * len([k for k in arr_keys if k not in ("rows", "kept", "ck", "drf")]))
    for (arm, typ), rs in sorted(by.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        ck = [v for r in rs for v in r["ck"][1:]]; drf = [v for r in rs for v in r["drf"][1:]]
        kept = [v for r in rs for v in r["kept"][1:]]; rows = [v for r in rs for v in r["rows"][1:]]
        first = [r["ck"][0] for r in rs if r["ck"]]
        extras = []
        for k in arr_keys:
            if k in ("rows", "kept", "ck", "drf"): continue
            vals = [v for r in rs for v in (r.get(k) or [])[1:] if isinstance(v, (int, float))]
            extras.append(f"{med(vals):.1f}" if vals else "—")
        print(f"| {arm} | {typ} | {len(rs)} | {len(ck)} | {med(kept):.1f} | {med(rows):.1f} | {med(ck):.0f} | {med(drf):.1f} | {100 * med(drf) / (med(drf) + med(ck)):.0f} % | {med(first):.0f} vs {med(ck):.0f} | " + " | ".join(extras) + " |")
