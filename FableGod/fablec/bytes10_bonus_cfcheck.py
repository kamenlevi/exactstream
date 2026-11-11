"""Screen 1(c) follow-up (Fable C, Wed 4 Nov 2026, 19:5x): what the gated bonus rows are. CPU only, seconds.
A real bonus row (bytes10_bonus_masks.py: a live segment with k == m) is one of three kinds: (A) the live cut (cumulative cf
product < 0.3) fell inside an accepted run: the cut draft was accepted in the walk, g[b] = its cf; (B) the cut fell exactly on
the check's first miss: the cut draft was rejected, g[b] = the cf of a wrong draft; (C) end of a fully kept 15-draft check:
no draft, g = nan (-1), never gated by a threshold. Prints per domain the kinds, how many clear cf 0.5 / 0.9 per kind, and
the net flips per gated row from the map read-outs (results_wed, 19:38). Output: bytes10_bonus_cfcheck.out."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/FableGod/decision_map"))
from bytes10_replay import windows, live_segments  # noqa: E402

NET = {"code": (4, 1, 5), "ifeval": (7, 41, 48), "multi": (10, 23, 33)}   # 1(c) net flips, only-ungated, only-gated [K 19:38]
NET0 = {"code": None, "ifeval": None, "multi": None}                      # filled from the overnight cf-0 union-only arm below
lines = []
for dom, ws in windows().items():
    kinds = {"A": [], "B": [], "C": []}
    for w in ws:
        N = len(w["ids"]); g = w["g"]
        for (k0, m0, A0), dc0 in zip(w["checks"], w["dcf"]):
            for (k, m, A, dc) in live_segments(k0, m0, A0, dc0):
                if A > N - 1 or k != m or A + m > N - 1: continue
                b = A + m; s = A - A0
                kind = "C" if s + m >= m0 else ("B" if s + m == k0 else "A")
                kinds[kind].append(float(g[b]))
    nb = sum(len(v) for v in kinds.values())
    lines.append(f"{dom}: real bonus rows {nb} = A (cut inside an accepted run) {len(kinds['A'])}, B (cut on the first miss) "
                 f"{len(kinds['B'])}, C (end of a fully kept check, no draft) {len(kinds['C'])}")
    for t in (0.5, 0.9):
        c = {k: int((np.array(v) >= t).sum()) if v else 0 for k, v in kinds.items()}
        lines.append(f"  cf >= {t}: {sum(c.values())} gated = A {c['A']} + B {c['B']} + C {c['C']}")
    for k, v in kinds.items():
        if v and k != "C":
            a = np.array(v); lines.append(f"  kind {k}: median cf {np.median(a):.3f}, share >= 0.5 {100 * (a >= 0.5).mean():.1f} %, "
                                          f">= 0.9 {100 * (a >= 0.9).mean():.1f} %, < 0.3 {100 * (a < 0.3).mean():.1f} %")
    g5 = sum(int((np.array(v) >= 0.5).sum()) for v in kinds.values() if v)
    net, og, ob = NET[dom]
    lines.append(f"  1(c) at cf 0.5: {ob} / {og} discordant pairs, net +{net} flips on {g5} gated rows = {100 * net / max(g5, 1):.1f} % "
                 f"per gated row (sd ~{100 * np.sqrt(ob + og) / max(g5, 1):.1f} points); at cf 0 every bonus row is gated: see the note")
out = "\n".join(lines); print(out); open("bytes10_bonus_cfcheck.out", "w").write(out + "\n")


# ---- part 2 (20:0x): per-kind flips of the 1(c) arm against the exact reference (results_wed, tk[row][0] = the arm's argmax)
import gzip, json  # noqa: E402
RW = os.path.expanduser("~/Breakthrough/data/dm_s7/results_wed")
arm = json.load(gzip.open(os.path.join(RW, "qom_0.5_live_bonus5g.json.gz"))); ref = json.load(gzip.open(os.path.join(RW, "s7_ref.json.gz")))
lines = ["", "part 2: all-row flips of qom_0.5_live_bonus5g vs the exact reference (s7_ref), by row kind [K map 19:38, CPU join]"]
for dom, ws in windows().items():
    acc = {"bonus A gated": [], "bonus B gated": [], "bonus A ungated": [], "bonus B ungated": [], "bonus C (exact)": [],
           "drafted rows, cf >= 0.5 (QOM-gated)": [], "drafted rows, cf < 0.5": [], "all answer rows": []}
    for w in ws:
        nm, N, g, a = w["nm"], len(w["ids"]), w["g"], w["a"]
        ta, tr = arm[nm]["tk"], ref[nm]["tk"]
        assert len(ta) == len(tr) and arm[nm]["a"] == ref[nm]["a"], nm
        off = N - len(ta)                                  # tk rows start at the first answer row
        flip = {r: int(ta[r - off][0] != tr[r - off][0]) for r in range(off, N)}
        bonus = {}
        for (k0, m0, A0), dc0 in zip(w["checks"], w["dcf"]):
            for (k, m, A, dc) in live_segments(k0, m0, A0, dc0):
                if A > N - 1 or k != m or A + m > N - 1: continue
                s = A - A0; bonus[A + m] = "C" if s + m >= m0 else ("B" if s + m == k0 else "A")
        for r in range(off, N):
            if r not in flip: continue
            acc["all answer rows"].append(flip[r])
            if r in bonus:
                kd = bonus[r]
                if kd == "C": acc["bonus C (exact)"].append(flip[r])
                else: acc[f"bonus {kd} {'gated' if g[r] >= 0.5 else 'ungated'}"].append(flip[r])
            elif g[r] >= 0.5: acc["drafted rows, cf >= 0.5 (QOM-gated)"].append(flip[r])
            else: acc["drafted rows, cf < 0.5"].append(flip[r])
    lines.append(f"{dom}: " + "; ".join(f"{k} {sum(v)}/{len(v)} = {100 * sum(v) / max(len(v), 1):.1f} %" for k, v in acc.items() if v))
out2 = "\n".join(lines); print(out2); open("bytes10_bonus_cfcheck.out", "a").write(out2 + "\n")
