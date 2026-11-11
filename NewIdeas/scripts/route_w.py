"""ROUTE-W replay (NewIdeas, Tue 3 Nov 2026 17:5x; CPU only, writes data/ROUTEW_read.txt): the Lunatic's Route W ("one
user is a batch", ~/FableGod/lunatic/LUNACY2.md sec 2) falsification test. S of the user's streams verify in ONE merged
pass; every (expert, layer) group is read once per pass for all rows that route to it. Metric: expert groups per kept
token = sum over passes of the merged union's distinct experts per layer (48 layers) / the kept tokens of all streams in
those passes. The Lunatic's lines (pre-registered in LUNACY2.md before this replay): PASS <= 60 groups per kept token,
KILL > 80 (8 streams, Q1-UNION masks, mixed domains; his shape is 32 rows per stream, which needs the 'horizon fix').
Data: Breakthrough's decision-map windows (36: 12 HumanEval, 12 IFEval, 12 multi; ids = the 125B's own rank-sorted
routes, teacher-forced), the HPD-0 walks' checks with DFlash2 cf. Blocks per stream: (live) the walk's checks with the
default's running-cf 0.3 cut (dm_oea_masks.live_segments), or (walk16) the walk's own checks, all rows. Rows after a
check's first miss enter with the wrong-draft stand-in (row p + 97), as in every union arm. Kept per block = accepted
drafts + 1. Arms: EXACT (all 10 ranks of every row); Q1U <theta> (rows with cf >= theta keep ranks 0-1 + any rank 2-9
already in the MERGED union: across streams, members are free); HYB <theta> u8 (unsure rows top-8 + members). Streams:
S random windows (mixed: any domain), 40 draws per S, passes while every stream still has a block.
Usage (BigMoE venv): python route_w.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dm_oea_masks import live_segments, OUT, dm_gate  # noqa: E402
DST = os.path.expanduser("~/NewIdeas/data/ROUTEW_read.txt")
rng = np.random.default_rng(7)


def blocks_of(live):
    base = json.load(open(os.path.join(OUT, "baseline.json"))); G = dm_gate.load(); W = []
    for dom, wf in dm_gate.WALK.items():
        Z = np.load(os.path.join(dm_gate.TD, wf))
        for i, nm in enumerate(str(x) for x in Z["names"]):
            if nm not in base: continue
            ids = np.load(os.path.join(OUT, "routing", nm))["ids"]; N = len(ids); g = np.nan_to_num(G[nm], nan=-1.0)
            bl = []
            for (k0_, m0_, A0_), dc0_ in zip(Z[f"{i}_checks"].tolist(), Z[f"{i}_dcf"].tolist()):
                for (k, m, A, dc) in (live_segments(k0_, m0_, A0_, dc0_) if live else [(k0_, m0_, A0_, dc0_)]):
                    last, fm = min(A + m, N - 1), min(A + k, N - 1)
                    rows, cfs = [], []
                    for p in range(A, last + 1):
                        if p <= fm: rows.append(p); cfs.append(-1.0 if live and k == m and p == A + m else g[p])
                        else: rows.append((p + 97) % N); cfs.append(dc[p - A] if p - A < m else -1.0)
                    if rows: bl.append((ids[rows], np.array(cfs), min(k, fm - A) + 1))
            W.append((dom, nm, bl))
    return W


def groups(block_list, arm):
    """distinct experts per layer of the merged pass, summed over 48 layers"""
    kind, thr, ku = arm
    k0s = []
    for ids, cfs, kept in block_list:
        if kind == "EXACT": k0s.append(np.full(len(cfs), 10))
        else: k0s.append(np.where(cfs >= thr, 2, ku))
    tot = 0
    for l in range(48):
        U = set()
        for (ids, cfs, kept), k0 in zip(block_list, k0s):
            for j in range(len(k0)): U.update(ids[j, l, :k0[j]].tolist())
        tot += len(U)
    return tot


def tps(a):
    """the Lunatic's pass model (LUNACY2 sec 1a) [P]: trunk 12.7 ms bytes + 3.6 fixed + 0.32 ms per row, experts 4.5 ms +
    10.15 us per (expert, layer) group; whole model in RAM, draft hidden, no SSD: aggregate kept tokens per second"""
    g, k, r, n = a
    ms = n * (12.7 + 3.6 + 4.5) + 0.32 * r + 0.01015 * g
    return 1000.0 * k / ms


ARMS = {"EXACT": ("EXACT", None, 10), "Q1U 0.9": ("Q1U", 0.9, 10), "Q1U 0.5": ("Q1U", 0.5, 10),
        "HYB 0.9 u8": ("HYB", 0.9, 8)}
lines = []
def say(s=""): print(s, flush=True); lines.append(s)
say("Route W replay: expert groups per kept token in one merged pass of S streams (Lunatic's lines: PASS <= 60, KILL > 80);")
say("(in brackets: aggregate tok/s by the Lunatic's pass model, whole model in RAM, draft hidden, no SSD [P])")
for shape in ("live", "walk16"):
    W = blocks_of(shape == "live")
    say(f"\nblocks = {shape}: " + ", ".join(f"{d} rows / kept per block {np.mean([len(b[1]) for w in W if w[0] == d for b in w[2]]):.1f} / "
                                     f"{np.mean([b[2] for w in W if w[0] == d for b in w[2]]):.2f}" for d in ("code", "ifeval", "multi")))
    say("| S streams (mixed) | " + " | ".join(ARMS) + " |")
    say("|---|" + "---|" * len(ARMS))
    for S in (1, 2, 4, 8, 12):
        acc = {a: [0, 0, 0, 0] for a in ARMS}                 # groups, kept, rows, passes
        for draw in range(40 if S > 1 else 36):
            pick = [W[draw]] if S == 1 else [W[i] for i in rng.choice(len(W), S, replace=False)]
            n = min(len(w[2]) for w in pick)
            for t in range(n):
                bl = [w[2][t] for w in pick]; kept = sum(b[2] for b in bl)
                rows = sum(len(b[1]) for b in bl)
                for a, spec in ARMS.items():
                    acc[a][0] += groups(bl, spec); acc[a][1] += kept; acc[a][2] += rows; acc[a][3] += 1
        say(f"| {S} | " + " | ".join(f"{acc[a][0] / max(acc[a][1], 1):.1f} ({tps(acc[a]):.0f})" for a in ARMS) + " |")
open(DST, "w").write("\n".join(lines) + "\n")
