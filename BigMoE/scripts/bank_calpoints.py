#!/usr/bin/env python3
"""Rebuild data/deepdraft/bank/calibration.json with the direction error (the PROXY's primary measure) and the structure
measures of each calibration map. They are parsed from bank_proxy.py's lines in data/deepdraft/bank/calibrate.log; the
real REPLACE accuracies come from bank_calibrate.sh's eval files. Run after the calibration ends."""
import json, os, re
B = os.path.expanduser("~/BigMoE/data/deepdraft/bank")
log = open(os.path.join(B, "calibrate.log")).read()
meas = {}
for m in re.finditer(r"ridge (\S+) \(lambda [^)]*\): held-out R2 ([-0-9.]+), normalized error ([0-9.]+)[^\n]*\n\[proxy\]\s+structure: ([^\n]+)", log):
    st = {k: float(v) for k, v in (kv.split() for kv in m.group(4).split(", "))}
    meas[m.group(1).replace(",", "_")] = dict(r2=float(m.group(2)), ne=float(m.group(3)), **st)
cal = json.load(open(os.path.join(B, "calibration.json")))
for p in cal["points"]:
    if p["set"] in meas:
        p.update(meas[p["set"]]); p["family"] = "ridge"
    elif p["set"].startswith("oracle"):
        p.update(ne=0.0, family="oracle")
cal["primary"] = "ne (the RMS-normalized direction error), per family"
json.dump(cal, open(os.path.join(B, "calibration.json"), "w"), indent=1)
for p in sorted(cal["points"], key=lambda p: p.get("ne", 9)):
    print(f"{p['set']:32s} ne {p.get('ne', float('nan')):.4f} R2 {p['r2']:.4f} -> MBPP 23-30 {p['mbpp_23_30']:.2f}" +
          "".join(f", {k} {p[k]:.3f}" for k in ("shrink", "r2_pc", "r2_head") if k in p))
