"""S45d replay (Breakthrough Sun 1 Nov 11:5x; CPU; follow-up to the TREES side lead, before any GPU run): would the 45 GB
default's DFlash2 truncation at t > 0.3 decode faster? Replays MTP Head's D-23 logged checks of the dfl arm (read-only:
~/TrainLab/data/d23_runs.jsonl; the 45 GB default at C 224, t 0.3, 18 prompts: code 4, math 4, chat 6, agent 4). Per check
the log has rows, kept, check ms, draft ms and all 15 step confidences (DFlash2 drafts every depth at once), so raising t only
drops verified rows: rows(t) = 1 + #steps whose running product >= t (capped at the logged rows), kept(t) = min(kept, rows(t)),
check(t) = the logged check ms - slope x (rows - rows(t)); the slope from D-23b's fit (one intercept per prompt, one shared
slope, the dfl checks after the first two). Draft time unchanged; the overhead per check (decode time - logged draft + check)
as measured. tok/s(t) = sum kept(t) / sum (overhead + draft + check(t)) per prompt; types and the whole set as D-23's read-out
(sum of tokens / sum of times). t 0.3 must reproduce the measured tok/s.
Usage: [BigMoE venv] python s45d_replay.py -> stdout + data/trees/s45d_replay.txt"""
import collections, json, os
import numpy as np

R = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
P = [r for r in R if r["arm"] == "dfl" and not (r["paused"] or r["invalid"] or r["cut"])]
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


# shared slope, per-prompt intercepts (D-23b's fit): demean rows and check ms within each prompt
X, Y = [], []
for r in P:
    pc = np.array(r["pc"], float)[2:]
    if len(pc) >= 2:
        X.append(pc[:, 0] - pc[:, 0].mean()); Y.append(pc[:, 2] - pc[:, 2].mean())
X, Y = np.concatenate(X), np.concatenate(Y)
b = float((X * Y).sum() / (X * X).sum())
TS = (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
say(f"S45d replay of D-23's dfl arm ({len(P)} prompts, {sum(r['checks'] for r in P)} checks; 45 GB default, C 224, t 0.3 live): "
    f"check cost {b:.2f} ms per verified row (D-23b: 7.82)")
T = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))   # type -> t -> tokens, ms, rows, checks
for r in P:
    pc, cf = np.array(r["pc"], float), np.array(r["cf"], float)
    dec_ms = 1000 * r["n"] / r["tok_s"]
    ovh = (dec_ms - pc[:, 2].sum() - pc[:, 3].sum()) / len(pc)
    for t in TS:
        n_t = np.minimum((np.cumprod(cf, axis=1) >= t).sum(1) + 1, pc[:, 0])
        k_t = np.minimum(pc[:, 1], n_t)
        ms = ovh + pc[:, 3] + pc[:, 2] - b * (pc[:, 0] - n_t)
        for g in (r["type"], "whole set"):
            a = T[g][t]; a[0] += k_t.sum(); a[1] += ms.sum(); a[2] += n_t.sum(); a[3] += len(pc)
    T["_check"][r["key"]] = (r["tok_s"], r["n"] / (dec_ms / 1000))
say("  estimator check (t 0.3 per prompt, replayed / measured): "
    + f"{np.median([v[1] / v[0] for v in T['_check'].values()]):.3f} median")
say("  t        " + "".join(f"{g:>22s}" for g in ("code", "math", "chat", "agent", "whole set")))
for t in TS:
    cells = []
    for g in ("code", "math", "chat", "agent", "whole set"):
        k, ms, rows, ch = T[g][t]; k0, ms0 = T[g][0.3][:2]
        cells.append(f"{1000 * k / ms:6.1f} ({100 * (k / ms) / (k0 / ms0) - 100:+5.1f}%) {rows / ch:4.1f}r")
    say(f"  {t:<8}" + "".join(f"{c:>22s}" for c in cells))
open(os.path.expanduser("~/Breakthrough/data/trees/s45d_replay.txt"), "w").write("\n".join(lines) + "\n")
