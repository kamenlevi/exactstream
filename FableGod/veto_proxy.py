#!/usr/bin/env python3 -I
"""veto_proxy.py (FableGod, mission 30, idea #6 lookahead-consistency acceptance). The veto ("accept a near-tie at row r
only if row r+1 is sure given the draft") needs the 125B's distribution at r+1 conditioned on the DRAFT token, which no
recording holds. This proxy uses relax_positions.json (RELAX_350: per tested position dom, depth, cls, right, lp0 =
the 125B's top-1 log-prob, lpd = the draft's log-prob, cf) and the TRUE-prefix class of the next position (the next
entry of the same check, depth + 1): how often R1(0.5)'s accepted-but-wrong tokens (the deviations) sit before a sure
row, and how often its accepted-and-right ones do. If the draft-conditioned next row behaved like the true-prefix one,
the veto would keep that share of each."""
import json
P = json.load(open("/Users/kamen/FableGod/relax_positions.json"))
by = {}
for i, x in enumerate(P):
    nxt = P[i + 1] if i + 1 < len(P) else None
    if nxt is None or nxt["dom"] != x["dom"] or nxt["depth"] != x["depth"] + 1: continue
    r1 = x["lpd"] >= x["lp0"] - 0.5
    if not r1: continue
    key = (x["dom"], "deviation" if not x["right"] else "right"); c = by.setdefault(key, [0, 0, 0])
    c[0] += 1; c[1] += nxt["cls"] == "sure"; c[2] += nxt["cls"] == "mid"
for (dom, kind), (n, s, m) in sorted(by.items()):
    print(f"  {dom} R1-accepted {kind} rows with a next row in the record: {n}; next row sure {100 * s / n:.1f} %, mid {100 * m / n:.1f} %, tie {100 * (n - s - m) / n:.1f} %")
print("  reading: the veto keeps the 'next row sure' share; the deviations' true next row is a proxy only (the draft-conditioned row is unrecorded)")
