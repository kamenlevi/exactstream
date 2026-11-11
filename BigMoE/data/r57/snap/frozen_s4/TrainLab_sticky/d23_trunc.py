"""D-23b MTP TRUNC (MTP Head, Sun 11-01, CPU only): would a running-confidence truncation of the MTP K7 chain (L1-14's rule,
the one the 45 GB default's DFlash2 arm uses at t 0.3) make MTP win where D-23 lost? Exact by construction: the cut only
decides how many drafted rows the check verifies; outputs stay the 125B's greedy tokens.
Replay of D-23's logged checks (data/d23_runs.jsonl; per check: rows, kept, check ms, draft ms, the chain's per-step top-1
probabilities): rows(t) = 1 + #steps whose running product of top-1 p is >= t, kept(t) = min(kept, rows(t)).
Check cost per row: a fixed-effects fit (one intercept per prompt, one shared slope) on the dfl arm's checks, whose rows vary
with its own truncation; validated by predicting the mtp7 arm's mean check at 8 rows. Draft: (A) all 7 steps drafted as now,
(B) early exit, 2.5 ms per step not drafted (D-20: K7 17-21 ms, K15 37-43 ms) less 0.3 ms per step for the extra sync.
tok/s(t) = sum kept(t) / sum (overhead + draft(t) + check(t)) per prompt, the overhead per check (decode time minus logged
draft + check) kept as measured; the dfl arm is scored with the same estimator. Type and whole-set sums as D-23's read-out
(each prompt's tokens / tok/s). Fit on the same 18 prompts it scores: a per-type t is in-sample, so a single global t and a
leave-one-prompt-out choice are reported too. Output: data/d23_trunc.txt"""
import json, os, collections
import numpy as np

TD = os.path.expanduser("~/TrainLab/data")
R = [json.loads(l) for l in open(os.path.join(TD, "d23_runs.jsonl"))]
D = collections.defaultdict(dict)
for r in R:
    if not (r["paused"] or r["invalid"] or r["cut"]):
        D[r["key"]][r["arm"]] = r
keys = [k for k in D if {"dfl", "mtp7"} <= set(D[k])]
TYPES = ("code", "math", "chat", "agent")
lines = []
def say(s):
    print(s, flush=True); lines.append(s)

def arrays(r):
    pc = np.array(r["pc"], float)                      # rows, kept, check ms, draft ms
    dec = r["n"] / r["tok_s"]
    ov = (dec - 1e-3 * (pc[:, 2] + pc[:, 3]).sum()) / len(pc)
    return pc, 1e3 * ov

# ---- check cost per row: shared slope, per-prompt intercepts, dfl checks (skip the first two: warm caches)
xs, ys = [], []
for k in keys:
    pc, _ = arrays(D[k]["dfl"])
    x, y = pc[2:, 0], pc[2:, 2]
    xs.append(x - x.mean()); ys.append(y - y.mean())
b = float((np.concatenate(xs) * np.concatenate(ys)).sum() / (np.concatenate(xs) ** 2).sum())
say(f"D-23b MTP TRUNC (CPU replay of D-23's {len(keys)} prompts, 45 GB default at C 224, head b3)")
say(f"  check cost: {b:.2f} ms per verified row (fixed-effects fit on the dfl arm's checks)")
errs = []
for k in keys:
    pd_, _ = arrays(D[k]["dfl"]); pm, _ = arrays(D[k]["mtp7"])
    a_p = pd_[2:, 2].mean() - b * pd_[2:, 0].mean()
    pred, obs = a_p + b * 8, pm[2:, 2].mean()
    errs.append(obs / pred)
say(f"  validation: the mtp7 arm's mean check at 8 rows / the fit's prediction from the dfl arm: median {np.median(errs):.3f}, "
    f"range {min(errs):.3f}-{max(errs):.3f} (1 = the per-row model transfers across arms)")
SLOPE = b

def score_mtp(r, t, early):
    pc, ov = arrays(r)
    cf = r["cf"]
    kept_t, time_t = 0.0, 0.0
    for (rows, kept, ck, dr), c in zip(pc, cf):
        steps = int((np.cumprod(np.array(c, float)) >= t).sum()) if t > 0 else len(c)
        steps = min(steps, int(rows) - 1)
        rt = steps + 1
        kt = min(kept, rt)
        ckt = ck - SLOPE * (rows - rt)
        drt = dr - (2.5 - 0.3) * (rows - rt) if early else dr
        kept_t += kt; time_t += ov + drt + ckt
    return kept_t / time_t * 1e3

def score_dfl(r):
    pc, ov = arrays(r)
    return pc[:, 1].sum() / (len(pc) * ov + pc[:, 2].sum() + pc[:, 3].sum()) * 1e3

TS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
typ = {k: D[k]["dfl"]["type"] for k in keys}
n = {k: D[k]["dfl"]["n"] for k in keys}
S_dfl = {k: score_dfl(D[k]["dfl"]) for k in keys}
obs_dfl = {k: D[k]["dfl"]["tok_s"] for k in keys}
say(f"  estimator check: dfl's replayed tok/s / its measured tok/s, median "
    f"{np.median([S_dfl[k] / obs_dfl[k] for k in keys]):.3f} (the estimator drops the first token's share only)")

def agg(ks, f):
    return sum(n[k] for k in ks) / sum(n[k] / f(k) for k in ks)

for early in (False, True):
    say(f"\n== draft {'(B) early exit, 2.2 ms saved per step not drafted' if early else '(A) all 7 steps drafted (as now)'}")
    say("  t     " + "  ".join(f"{T:>14s}" for T in TYPES) + "   whole set (mtp everywhere / the switch: dfl code+math)")
    say("  dfl   " + "  ".join(f"{agg([k for k in keys if typ[k] == T], lambda k: S_dfl[k]):14.1f}" for T in TYPES)
        + f"   {agg(keys, lambda k: S_dfl[k]):.1f}")
    best = {}
    for t in TS:
        S = {k: score_mtp(D[k]["mtp7"], t, early) for k in keys}
        cells = []
        for T in TYPES:
            ks = [k for k in keys if typ[k] == T]
            m, d = agg(ks, lambda k: S[k]), agg(ks, lambda k: S_dfl[k])
            cells.append(f"{m:6.1f} ({m / d - 1:+5.1%})")
            best.setdefault(T, []).append((m / d - 1, t))
        whole = agg(keys, lambda k: S[k])
        sw = agg(keys, lambda k: S_dfl[k] if typ[k] in ("code", "math") else S[k])
        dw = agg(keys, lambda k: S_dfl[k])
        say(f"  {t:<4}  " + "  ".join(cells) + f"   {whole:.1f} ({whole / dw - 1:+.1%}) / {sw:.1f} ({sw / dw - 1:+.1%})")
    # leave-one-prompt-out: per type, t chosen on the other prompts of that type, scored on the held-out one
    gains = {}
    for k in keys:
        ks = [q for q in keys if typ[q] == typ[k] and q != k]
        tbest = max(TS, key=lambda t: agg(ks, lambda q: score_mtp(D[q]["mtp7"], t, early)) / agg(ks, lambda q: S_dfl[q]))
        gains[k] = (tbest, score_mtp(D[k]["mtp7"], tbest, early))
    sw = agg(keys, lambda k: max(S_dfl[k], 0) if typ[k] in ("code", "math") else gains[k][1])
    per_type_sw = agg(keys, lambda k: gains[k][1] if typ[k] in ("chat", "agent") else S_dfl[k])
    say(f"  leave-one-prompt-out t per type (chat / agent on mtp7, code / math on dfl): whole set "
        f"{per_type_sw:.1f} ({per_type_sw / agg(keys, lambda k: S_dfl[k]) - 1:+.1%}); t picked: "
        + ", ".join(f"{k.split('/')[-1]} {gains[k][0]}" for k in keys if typ[k] in ("chat", "agent")))
say("\nVerdict line (pre-registered with D-23: a switch is worth a quiet-lock test at >= +10 % whole set; < +4 % = no)")
open(os.path.join(TD, "d23_trunc.txt"), "w").write("\n".join(lines) + "\n")
