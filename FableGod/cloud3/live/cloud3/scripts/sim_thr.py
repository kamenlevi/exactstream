"""Threshold sweep for the strict table (distinct per kept token ratio vs exact) and the bonus row's next-draft cf distribution."""
import numpy as np, sys
sys.path.insert(0, "/tmp/claude-0/-home-user-OjoX/d0b96668-0fdb-5652-b283-b9e2b9efeb75/scratchpad/s")
from sim_union import load, checks_of
from sim_p3 import banded, union_counts, MID
W = load(); doms = ["code", "ifeval", "multi"]
print("bonus row's next-draft cf (when it exists): share >= 0 / 0.3 / 0.5 / 0.7 / 0.9, and the share of checks ending at L = 15 (no next draft):")
for dom in doms:
    cfs = []; nodraft = 0; n = 0
    for w in W:
        if w["dom"] != dom: continue
        for ch in checks_of(w):
            n += 1; c = ch["cf"][ch["L"]]
            if c == c: cfs.append(c)
            else: nodraft += 1
    cfs = np.array(cfs)
    print(f"  {dom:6s}: n {n}, no next draft {nodraft/n:.2f}; of the rest: " + " ".join(f">= {t}: {np.mean(cfs >= t):.2f}" for t in [0, 0.3, 0.5, 0.7, 0.9]))
print("\nthreshold sweep (strict, + BQ at the same thr): thr -> T1 / T2(unsure k6 in 8-35) / T4 ratios")
base = None
for thr in [0.0, 0.3, 0.5, 0.7, 0.9, 0.99]:
    line = f"  thr {thr:<5}"
    for dom in doms:
        tot = {"ex": 0, "t1": 0, "t2": 0, "t4": 0}; K = 0
        for w in W:
            if w["dom"] != dom: continue
            for ch in checks_of(w):
                I = w["ids"][ch["rows"]]; K += ch["kept"]
                tot["ex"] += union_counts(I, np.full((ch["R"], 48), 10)).sum()
                tot["t1"] += union_counts(I, banded(ch, thr, 2, 10, {})).sum()
                tot["t2"] += union_counts(I, banded(ch, thr, 2, 10, {"mid": (list(MID), (2, 6))})).sum()
                tot["t4"] += union_counts(I, banded(ch, thr, 2, 10, {"mid": (list(MID), (2, 5)), "early": (list(range(0, 8)), (2, 8)), "late": (list(range(36, 48)), (2, 8))})).sum()
        line += f" | {dom}: T1 x{tot['t1']/tot['ex']:.3f} T2 x{tot['t2']/tot['ex']:.3f} T4 x{tot['t4']/tot['ex']:.3f}"
    print(line)
