"""An alternative to pass 2, counted on the map (Breakthrough, Mon 2026-11-02 19:3x; CPU, no model): the DEFERRED ANCHOR.
Each check runs ONE pass: its anchor row exact (top-10), its other rows top-2. Walking the block, a certified row is
trusted as in the two-pass check; the first UNcertified row ends the check without a bonus (the accepted drafts before it
are kept) and becomes the next check's anchor, so it runs exact there. No second dense pass ever runs; the price is
shorter checks (more passes and drafts per token). Quality: the committed rows are certified rows at top-2 and exact
anchors, the cert_top2 mixed runs' semantics (certified anchors run exact here, a little better).
Model of the drafter: from an anchor inside an exact walk check, the first miss stays the exact walk's (A + accepted);
blocks are 16 rows (15 drafts + the bonus). Per theta and domain: kept per check, dense passes per 100 tokens (two-pass:
checks + pass 2s; deferred: checks), exact rows per 100 tokens (two-pass: uncertified rows up to the first miss;
deferred: anchors), and checks per 100 tokens (drafter runs).
PIPELINED BONUS (the narrow version): when the only uncertified row at or before the first miss IS the miss row, skip
pass 2 and take that row's pass-1 argmax as a provisional bonus; the next check's anchor row (exact in pass 1) verifies
it. Right guess: one dense pass saved. Wrong guess: that next check keeps 1 token, the same dense passes as pass 2 would
have cost. So it never costs more passes; "pass 2 in" drops to the blocks with an uncertified row BEFORE the miss.
Usage (BigMoE venv): python dm_defer.py -> out/DEFER_read.txt"""
import json, os
import numpy as np
import dm_gate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
base = json.load(open(os.path.join(OUT, "baseline.json"))); S = json.load(open(os.path.join(OUT, "topk2.json")))
CHECKS = {}
for dom, wf in dm_gate.MAPW.items():
    M = np.load(os.path.join(dm_gate.TD, wf))
    for s, nm in enumerate([str(x) for x in M["names"]][:12]):
        CHECKS[nm] = [(int(A), int(k)) for (k, m, A) in M[f"{s}_checks"]]
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
say("Deferred anchor vs two-pass on the map's exact walks (per 100 tokens; kept = tokens per check)")
say("| domain | theta | two-pass: kept | pass 2 in | dense passes | exact rows | pipelined bonus: pass 2 in | its guess right | dense passes | deferred anchor: kept | dense passes = checks |")
say("|---|---|---|---|---|---|---|---|---|---|---|")
for dom in ("code", "ifeval", "multi"):
    for thr in (0.99, 0.95, 0.9, 0.8, 0.7):
        T = {k: 0 for k in ("tok", "chk", "p2", "x2", "dchk", "dtok", "mo", "mr")}
        for nm, b in base.items():
            if b["dom"] != dom: continue
            a = b["a"]; p1 = np.exp(np.array(S[nm]["lp"])[:, 0]); n = a + len(p1) + 1
            right = np.r_[np.full(a, True), np.array(S[nm]["tk"])[:, 0] == np.array(b["tk"])[:, 0], False]   # pass-1 argmax = exact
            ok = np.r_[np.full(a, True), p1 >= thr, False]          # trusted rows: prefill and certified
            ck = [(A, min(A + k, n - 2)) for A, k in CHECKS[nm] if A <= n - 2]
            for A, e in ck:
                T["chk"] += 1; T["tok"] += e - A + 1; u = int((~ok[A:e + 1]).sum()); T["p2"] += u > 0; T["x2"] += u
                if u == 1 and not ok[e]: T["mo"] += 1; T["mr"] += bool(right[e])     # only the miss row: a provisional bonus
            ends = np.array([e for _, e in ck]); t, last = ck[0][0], ck[-1][1]
            while t <= last:
                e = min(int(ends[np.searchsorted(ends, t)]), t + 15); kept = e - t + 1; nt = e + 1
                for r in range(t + 1, e + 1):
                    if not ok[r]: kept, nt = r - t, r; break
                T["dchk"] += 1; T["dtok"] += kept; t = nt
            assert T["dtok"] == T["tok"] or t > last, (nm, T)
        h = 100 / T["tok"]
        say(f"| {dom} | {thr} | {T['tok'] / T['chk']:.1f} | {100 * T['p2'] / T['chk']:.0f} % | {(T['chk'] + T['p2']) * h:.1f} | "
            f"{T['x2'] * h:.1f} | {100 * (T['p2'] - T['mo']) / T['chk']:.0f} % | {T['mr']}/{T['mo']} | {(T['chk'] + T['p2'] - T['mr']) * h:.1f} | "
            f"{T['dtok'] / T['dchk']:.1f} | {T['dchk'] * 100 / T['dtok']:.1f} |")
open(os.path.join(OUT, "DEFER_read.txt"), "w").write("\n".join(lines) + "\n")
