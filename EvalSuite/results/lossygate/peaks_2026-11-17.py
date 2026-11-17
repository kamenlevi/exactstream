#!/usr/bin/env python3
# Lifetime peaks per process, read from runner 6x's per-process logs (results/card/es_cd<tag>_<start>_<proc>.log).
# Each answer's "mem:" line ends "lifetime peak after prefill X -> after decode Y GB"; the warm-ups end
# "lifetime peak X -> Y GB". A process's lifetime peak is its largest Y (monotonic within a process).
# Usage: python3 results/lossygate/peaks_2026-11-17.py  (from ~/EvalSuite)
import glob, re

RUNS = [  # (prefix, what)
    ("es_cdlx45cc1_160408", "lx45cc1 forward (Mon night, the close-call base)"),
    ("es_cdlx45cc1_rev_160518", "lx45cc1 reverse (Mon night)"),
    ("es_cdlx45t1c1_161202", "lx45t1c1 forward, B1 (Mon by day; 12 processes after a resume)"),
    ("es_cdt1cx45t1c1_161313", "t1cx45t1c1 forward, V1 (Mon by day)"),
    ("es_cdccx45cc1_170056", "ccx45cc1 forward (Tue night)"),
    ("es_cdccx45cc1_rev_170208", "ccx45cc1 reverse (Tue night)"),
    ("es_cdccx45cc1d2_170322", "ccx45cc1d2 forward (Tue night)"),
]
ANS = re.compile(r"lifetime peak after prefill [0-9.]+ -> after decode ([0-9.]+) GB")
WU = re.compile(r"warm-up.*?lifetime peak [0-9.]+ -> ([0-9.]+) GB")

for prefix, what in RUNS:
    procs = []
    for f in sorted(glob.glob(f"results/card/{prefix}_*.log")):
        t = open(f).read()
        a = [float(x) for x in ANS.findall(t)]
        w = [float(x) for x in WU.findall(t)]
        if a or w:
            procs.append((f.rsplit("_", 1)[1][:-4], max(w) if w else 0.0, max(a + w)))
    if not procs:
        print(f"{prefix}: no per-process logs")
        continue
    wmax = max(p[1] for p in procs)
    lmax = max(p[2] for p in procs)
    lmin = min(p[2] for p in procs)
    over = sum(p[2] > 42.5 for p in procs)
    print(f"{what}: {len(procs)} processes; warm-up peak max {wmax:.1f} GB; "
          f"process lifetime peaks {lmin:.1f}-{lmax:.1f} GB (max {lmax:.1f}); above 42.5 GB: {over}")
    print("   " + " ".join(f"{p[0]}:{p[2]:.1f}" for p in procs))
