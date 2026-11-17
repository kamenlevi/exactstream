#!/usr/bin/env python3
"""BREAK98: what IN-CHECK REPAIR nets after its cost per round (NewIdeas, Tue 17 Nov; time = the commit's). Desk model on
measured parts; kept from b98_repairtree.txt (tree-32, real dumps, HE+).
Per round (the check paused at layer L): a readout pass over 32 rows (~1 head level) + the head's tail re-draft (~12 levels
for the ~16-row tail on HE+; ADAPTK.md h16 13.56 levels per 15 pops) + the catch-up of layers 1..L for the ~16 tail rows
(dense L/48 + the tail's experts in L layers; per layer ~0.40 ms at the bound = dense 0.158 + experts 0.22 + cores 0.02,
from Lunatic's bound split of a 25 ms 32-row check; x2.26 on today's engine, 56.4 ms).
Cycle = check + draft + overhead (+ rounds). Bound: 25 + 3 + 1.5 ms (head 0.1 ms / level). Today: 56.4 + 19.1 + 4 ms
(head 0.62 ms / level).
usage: /usr/bin/python3 -I b98_repaircost.py"""
KEPT = {0: 16.26, (1, 1.0): 22.06, (1, 0.5): 19.20, (1, 0.3): 17.96, (2, 1.0): 24.06, (2, 0.5): 19.98, (2, 0.3): 18.16}
SC = {"bound": dict(check=25.0, draft=3.0, ov=1.5, lvl=0.10, layer=0.40),
      "today": dict(check=56.4, draft=19.1, ov=4.0, lvl=0.62, layer=0.40 * 2.26)}
for name, s in SC.items():
    base = s["check"] + s["draft"] + s["ov"]
    t0 = 1000 * KEPT[0] / base
    print(f"== {name}: tree-32 {KEPT[0]} kept / {base:.1f} ms = {t0:.0f} tok/s")
    for L in (4, 8, 16):
        per = s["lvl"] * 1 + s["lvl"] * 12 + s["layer"] * L
        out = []
        for key in ((1, 1.0), (1, 0.5), (2, 1.0), (2, 0.5)):
            R, q = key
            cyc = base + R * per
            t = 1000 * KEPT[key] / cyc
            out.append(f"R{R} q{q}: {t:.0f} ({100 * (t / t0 - 1):+.0f} %)")
        print(f"  L {L:2d}: {per:.1f} ms per round -> " + "; ".join(out))
