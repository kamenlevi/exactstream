"""PMIN12 (Breakthrough Sun 1 Nov 11:5x; lines in SCOREBOARD 'PMIN12 LINES', written before the run; CPU): would spec125's
p_min (stop the MTP chain where the head's running top-1 probability < p_min) speed up the 12 GB line, whose 3:3:0 passes
always verify 4 rows? Replays MTP Head's D-23 mtp7 checks (read-only ~/TrainLab/data/d23_runs.jsonl; head b3; per check the
chain's per-step top-1 p ('cf') and kept of 8 rows) as a 3-draft chain: rows(p) = 1 + #of the first 3 steps whose running p >=
p_min, kept(p) = min(kept, 4, rows(p)). Cost per pass at 12 GB: c(r) = 14.5 + 58.5 x r^0.7185 ms (Leap's SSD shape through the
12 GB line's warm AR step ~73 ms and its 4-row pass ~173 ms) + draft 2.2 ms per drafted step (early exit saves the rest).
tok/s per type = sum kept / sum cost (D-23's 18 prompts: code 4, math 4, chat 6, agent 4).
Usage: [BigMoE venv] python pmin12.py -> stdout + data/trees/pmin12.txt"""
import collections, json, os
import numpy as np

R = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
P = [r for r in R if r["arm"] == "mtp7" and not (r["paused"] or r["invalid"] or r["cut"])]
c = lambda r: 14.5 + 58.5 * r ** 0.7185
PS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


T = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0]))
for r in P:
    pc = np.array(r["pc"], float)
    cf = np.array([x[:3] + [0.0] * (3 - len(x[:3])) for x in r["cf"]], float)
    k3 = np.minimum(pc[:, 1], 4)
    for p in PS:
        nd = (np.cumprod(cf, axis=1) >= p).sum(1) if p > 0 else np.full(len(pc), 3)
        rows = nd + 1
        k = np.minimum(k3, rows)
        ms = c(rows) + 2.2 * np.minimum(nd + 1, 3)                    # the step that fails the test is drafted too
        for g in (r["type"], "whole set"):
            a = T[g][p]; a[0] += k.sum(); a[1] += ms.sum(); a[2] += rows.sum(); a[3] += len(pc)
say(f"PMIN12: {len(P)} mtp7 prompts, {sum(len(r['pc']) for r in P)} checks replayed as 3-draft chains; 12 GB cost c(r) = "
    f"14.5 + 58.5 r^0.7185 ms + 2.2 ms per drafted step")
say("  p_min    " + "".join(f"{g:>24s}" for g in ("code", "math", "chat", "agent", "whole set")))
for p in PS:
    cells = []
    for g in ("code", "math", "chat", "agent", "whole set"):
        k, ms, rows, n = T[g][p]; k0, ms0 = T[g][0.0][:2]
        cells.append(f"{1000 * k / ms:5.1f} ({100 * (k / ms) / (k0 / ms0) - 100:+5.1f}%) {k / n:4.2f}k {rows / n:3.1f}r")
    say(f"  {p:<8}" + "".join(f"{x:>24s}" for x in cells))
say(f"  AR at 12 GB (1 row, no draft): {1000 / c(1):.1f} tok/s")
open(os.path.expanduser("~/Breakthrough/data/trees/pmin12.txt"), "w").write("\n".join(lines) + "\n")
