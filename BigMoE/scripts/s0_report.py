"""Wall 1: the S0 split and the zero-hit split from r57_screen's R57_STALL dumps ({TAG}_stall.json: laymap + per-run trace
events), beside stall_report.py (docs/PLAN_wall1_lever1.md; Leap's W1-PICKUP; Fable A's D1 spec).
1. Zero-hit split (any trace): per served base call (L, seq), the gap from layer L-1's release to L's routing being seen.
   That gap holds layer L's attention + router on the GPU and the post's trip to the loader. Calls are split by what the
   GPU had to do between the post and the spin:
   - "zero-hit": no resident expert (no phase-1 work);
   - "hits": some resident, some missing;
   - "no-miss": all resident.
   Leap's harness: a device-scope post followed only by the spin can stay unseen for ms. If so, the zero-hit gaps are far
   above the others, and BIGMOE_POST_SYS is the bigger lever.
2. S0 split (dumps from ext/build_w1: 7-field reads, kind 5 / 6 events), per served call with misses (medians, ms):
   - pre: seen -> ensure() starts;
   - plan + wake: ensure() starts -> the first demand read's block starts;
   - pread: one read's preadv;
   - copy: pread end -> bytes in the slot;
   - tail: the last read's bytes -> ensure() returns;
   - release: the release writes.
usage: .venv/bin/python scripts/s0_report.py [DIR=data/r57] [TAGS=c45t_160,c45tj_160] -> DIR/s0_report.txt"""
import json, os, sys
import numpy as np

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/BigMoE/data/r57")
TAGS = (sys.argv[2] if len(sys.argv) > 2 else "c45t_160,c45tj_160").split(",")
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def med(x):
    return f"{1e3 * float(np.median(x)):.3f}" if len(x) else "-"


for tag in TAGS:
    try:
        d = json.load(open(os.path.join(D, f"{tag}_stall.json")))
    except Exception as e:
        say(f"\n{tag}: no stall dump ({e})"); continue
    real = {int(b): i for i, (b, s) in enumerate(d["laymap"])}       # base ext id -> model layer
    bid = [int(b) for b, s in d["laymap"]]
    gaps = {"zero-hit": [], "hits": [], "no-miss": []}
    svc = {"zero-hit": [], "hits": [], "no-miss": []}
    s0 = {k: [] for k in ("pre", "plan_wake", "pread", "copy", "tail", "release", "reads")}
    nrun = 0
    for run in d["runs"]:
        if run.get("paused"):
            continue
        nrun += 1
        ev = {}
        for L, seq, kind, t0, t1 in run["tr"]:
            ev.setdefault((int(L), int(seq)), {})[int(kind)] = (t0, t1)
        cov = {(int(c[0]), int(c[1])): c for c in run["cov"]}
        rds = {}
        for r in run["rd"]:
            rds.setdefault(int(r[0]), []).append(r)
        for v in rds.values():
            v.sort(key=lambda r: r[3])
        for (L, seq), k in ev.items():
            if L not in real or 1 not in k or (L, seq) not in cov:
                continue
            c = cov[(L, seq)]
            need, resident = c[2], c[3] + c[4]                   # hit + prefetched in time = resident at the call
            cls = "no-miss" if resident >= need else ("zero-hit" if resident == 0 else "hits")
            svc[cls].append(k[1][1] - k[1][0])
            mL = real[L]
            prev = ev.get((bid[mL - 1], seq), {}).get(1) if mL > 0 else None
            if prev:
                gaps[cls].append(k[1][0] - prev[1])
            if 5 in k and 6 in k and resident < need:           # S0 (ext/build_w1 traces only)
                te0, te1 = k[5]
                rs = [r for r in rds.get(L, []) if len(r) >= 7 and r[2] in (0, 2) and te0 <= r[3] <= te1]
                if rs:
                    s0["pre"].append(te0 - k[1][0]); s0["plan_wake"].append(rs[0][3] - te0)
                    s0["pread"].append(float(np.median([r[6] - r[5] for r in rs])))
                    s0["copy"].append(float(np.median([r[4] - r[6] for r in rs])))
                    s0["tail"].append(te1 - max(r[4] for r in rs)); s0["release"].append(k[6][1] - k[6][0])
                    s0["reads"].append(len(rs) / 1e3)
    say(f"\n== {tag}: {nrun} runs; served base calls by class (count): " +
        ", ".join(f"{c} {len(svc[c])}" for c in svc))
    say("| class | L-1 release -> L seen, median ms (p90) | seen -> release, median ms |")
    say("|---|---|---|")
    for c in gaps:
        p90 = f"{1e3 * float(np.percentile(gaps[c], 90)):.3f}" if gaps[c] else "-"
        say(f"| {c} | {med(gaps[c])} ({p90}) | {med(svc[c])} |")
    if s0["pre"]:
        say(f"S0 per call with misses ({len(s0['pre'])} calls, medians ms): pre {med(s0['pre'])}, plan + wake "
            f"{med(s0['plan_wake'])}, pread {med(s0['pread'])}, copy {med(s0['copy'])}, tail {med(s0['tail'])}, release "
            f"{med(s0['release'])}; demand reads per call {1e3 * float(np.median(s0['reads'])):.0f}")
    else:
        say("S0: no kind 5 / 6 events (the dump comes from an ext before 5565a56): only the zero-hit split")
open(os.path.join(D, "s0_report.txt"), "w").write("\n".join(OUT) + "\n")
