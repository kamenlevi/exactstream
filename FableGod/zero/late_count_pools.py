"""LOSSY350_ZERO follow-up 2 (zero session, Thu 5 Nov 2026): the same count as late_count.py at uniform pool sizes, to
reconcile with FAR's replays (266 slots) and the engine's by-day cache (C 208, 6.6 demand misses per kept token in
BigMoE's qallt_208 logs). CPU only, seconds. Writes late_count_pools.out beside this file."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytes10_replay import windows, POOLS, MR  # noqa: E402
from late_count import run, summ  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = windows(); L = []; t0 = time.time()
cm45 = POOLS["45"][0]
L.append(f"laysz45_256 per-layer sizes: mean {np.mean(cm45):.1f}, min {min(cm45)}, max {max(cm45)}")
L.append("| domain | pool | kept / rows per block | demand misses per block (per kept) | rank 0 / rank 1 / tails | stalled layers (all / rank 0 only waits) | late events per row | MoE mass late per row-layer (all / rank 0 waits) |")
L.append("|---|---|---|---|---|---|---|---|")
for name, Cm in [("laysz45_256", cm45), ("266", [266] * 48), ("224", [224] * 48), ("208", [208] * 48), ("160", [160] * 48), ("147", [147] * 48)]:
    for dom in ("code", "ifeval", "multi"):
        s = summ(run(W[dom], Cm))
        L.append(f"| {dom} | {name} | {s['kept_per_block']:.1f} / {s['rows_per_block']:.1f} | {s['miss_mean']:.1f} ({s['miss_mean'] / s['kept_per_block']:.2f}) | "
                 f"{s['r0']:.1f} / {s['r1']:.1f} / {s['tail']:.1f} | {s['stalled']:.1f} / {s['stalled_r0']:.1f} | {s['late_events_per_row']:.2f} | "
                 f"{100 * s['late_mass_per_rowlayer']:.2f} % / {100 * s['late_mass_r1only_per_rowlayer']:.2f} % |")
        print(L[-1], flush=True)
L.append(f"({time.time() - t0:.0f} s)")
open(os.path.join(HERE, "late_count_pools.out"), "w").write("\n".join(L) + "\n")
print(L[0]); print(L[-1])
