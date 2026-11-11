"""LOSSY350_ZERO follow-up 4 (zero session, Thu 5 Nov 2026): late_terms.py at the base-pool sizes that bracket BigMoE's
qallt_208 run (C_m 226 uniform + a 64-slot volatile victim tier = 290 when the tier is intact; its log's 6.6 misses per
kept are ~3.0 base + ~3.6 sidecar reads by the bytes). CPU only, seconds. Writes late_terms_base.out."""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/FableGod/fablec"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bytes10_replay import windows  # noqa: E402
from late_terms import run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = windows(); L = []; t0 = time.time()
L.append("| domain | base slots (uniform), _ball | demand misses per kept | late terms per realized row: nw1 (own top-2 / members) | nw1r1 | late MoE mass per realized row-layer: nw1 / nw1r1 | realized rows with a late term: nw1 / nw1r1 |")
L.append("|---|---|---|---|---|---|---|")
for C in (226, 266, 290):
    for dom in ("code", "ifeval", "multi"):
        s = run(W[dom], [C] * 48, "all"); rl = s["real"] * 48
        L.append(f"| {dom} | {C} | {s['miss'] / s['kept']:.2f} | {s['late'] / s['real']:.2f} ({s['late_top2'] / s['real']:.2f} / {s['late_member'] / s['real']:.2f}) | "
                 f"{s['late_r1'] / s['real']:.2f} | {100 * s['late_mass'] / rl:.2f} % / {100 * s['late_mass_r1'] / rl:.2f} % | {100 * s['rows_hit'] / s['real']:.0f} % / {100 * s['rows_hit_r1'] / s['real']:.0f} % |")
        print(L[-1], flush=True)
L.append(f"({time.time() - t0:.0f} s)")
open(os.path.join(HERE, "late_terms_base.out"), "w").write("\n".join(L) + "\n"); print(L[-1])
