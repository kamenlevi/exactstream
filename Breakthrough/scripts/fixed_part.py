"""FIXED-PART (NewIdeas Sun 12:1x; CPU): split the 45 GB default's fixed cost per check by cause, from MTP Head's D-23 dfl
logs (read-only ~/TrainLab/data/d23_runs.jsonl: C 224, gate + REF2-T, t 0.3; 18 prompts). Logged per check: rows, kept,
check ms (snapshot + verify forward + argmax + taps + commit submit), draft ms; per prompt: the MEDIAN misses and SSD io ms
per check, prefill s, steady tok/s. Prompt lengths from the Eval suite's items.
  per check fixed part   F = check ms + draft ms - 7.82 x rows (D-23b's per-row slope)
  start-up excess        the first two checks' cost above the prompt's steady median at the same rows
  run-level rest         decode time - sum(check + draft): bookkeeping, stalls
  steady split           per prompt, the median F of checks 3.. regressed on the prompt's median SSD io ms per check and its
                         context (prompt + half the answer, K tokens): F = c0 + k_io x io + k_ctx x ctx
Agent is the only long-context type (9.1-12K tokens vs 26-184), so k_ctx is identified by agent vs the rest only.
Usage: [BigMoE venv] python fixed_part.py -> stdout + data/trees/fixed_part.txt"""
import collections, json, os
import numpy as np

R = [json.loads(l) for l in open(os.path.expanduser("~/TrainLab/data/d23_runs.jsonl"))]
P = [r for r in R if r["arm"] == "dfl" and not (r["paused"] or r["invalid"] or r["cut"])]
ES = os.path.expanduser("~/EvalSuite/data/items/")
NP = {}
for f in os.listdir(ES):
    if f.startswith("dec_"):
        for l in open(ES + f):
            it = json.loads(l)
            NP[it["key"]] = len(it["prompt_ids"])
SLOPE = 7.82
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


rows = []
for r in P:
    pc = np.array(r["pc"], float)                                     # rows, kept, check ms, draft ms
    dec = 1000 * r["n"] / r["tok_s"]
    F = pc[:, 2] + pc[:, 3] - SLOPE * pc[:, 0]
    st = F[2:] if len(F) > 4 else F
    fs = float(np.median(st))
    start = float((F[:2] - fs).sum()) if len(F) > 4 else 0.0           # the first two checks above the steady fixed part
    rest = dec - pc[:, 2].sum() - pc[:, 3].sum()
    rows.append(dict(type=r["type"], key=r["key"], n_prompt=NP[r["key"]], n=r["n"], checks=len(pc), rows=pc[:, 0].mean(),
                     F_mean_dec=dec / len(pc) - SLOPE * pc[:, 0].mean(), F_steady=fs, F_steady_mean=float(st.mean()),
                     start=start, rest=rest, draft=float(np.median(pc[:, 3])), io=r["io"], miss=r["miss"],
                     ctx=(NP[r["key"]] + r["n"] / 2) / 1000, pf=r["pf"]))
say(f"FIXED-PART: D-23 dfl arm, {len(rows)} prompts, {sum(x['checks'] for x in rows)} checks; F = check + draft - {SLOPE} x rows")
say("  per prompt: F from decode time (as 11:5x) | steady median (checks 3..) | start-up excess of checks 1-2 (total ms) | "
    "run rest (ms) | draft | SSD io / misses per check (medians) | prompt K")
for x in sorted(rows, key=lambda x: (x["type"], x["key"])):
    say(f"    {x['type']:6s} {x['key'][:30]:30s} {x['F_mean_dec']:6.1f} | {x['F_steady']:6.1f} | {x['start']:7.0f} | {x['rest']:6.0f} | "
        f"{x['draft']:5.1f} | {x['io']:6.1f} / {x['miss']:6.0f} | {x['n_prompt'] / 1000:5.2f}")
# steady split across prompts: F = c0 + k_io * io + k_ctx * ctx
X = np.array([[1.0, x["io"], x["ctx"]] for x in rows]); y = np.array([x["F_steady"] for x in rows])
coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
say(f"\n  steady fit over {len(rows)} prompts: F = {coef[0]:.1f} ms + {coef[1]:.3f} x io ms + {coef[2]:.2f} ms per K context; R^2 {r2:.3f}; "
    f"residuals {np.round(y - pred, 1).tolist()}")
Xi = np.array([[1.0, x["io"]] for x in rows]); ci, *_ = np.linalg.lstsq(Xi, y, rcond=None)
say(f"  io alone: F = {ci[0]:.1f} + {ci[1]:.3f} x io; io vs misses: io = {np.polyfit([x['miss'] for x in rows], [x['io'] for x in rows], 1)[1]:.1f} + "
    f"{np.polyfit([x['miss'] for x in rows], [x['io'] for x in rows], 1)[0]:.3f} ms per miss")
say("\n  per type (means over its prompts; ms per check):")
say("    type    F(dec) = steady F + start-up + rest | steady F = constant (dense + draft + syncs) + SSD part + context part | "
    "biggest part")
for t in ("code", "math", "chat", "agent"):
    g = [x for x in rows if x["type"] == t]
    ch = sum(x["checks"] for x in g)
    fd = np.mean([x["F_mean_dec"] for x in g]); fs = np.mean([x["F_steady"] for x in g])
    su = np.mean([x["start"] / x["checks"] for x in g]); rs = np.mean([x["rest"] / x["checks"] for x in g])
    io = np.mean([x["io"] for x in g]); cx = np.mean([x["ctx"] for x in g]); dr = np.mean([x["draft"] for x in g])
    parts = {"constant": coef[0], "SSD": coef[1] * io, "context": coef[2] * cx}
    big = max(parts, key=parts.get)
    say(f"    {t:6s}  {fd:6.1f} = {fs:6.1f} + {su:5.1f} + {rs:5.1f} | {fs:6.1f} = {parts['constant']:5.1f} (draft {dr:4.1f}) + "
        f"{parts['SSD']:5.1f} (io {io:5.1f} ms) + {parts['context']:5.1f} (ctx {cx:5.2f} K) | {big} {100 * parts[big] / sum(parts.values()):.0f} %")
open(os.path.expanduser("~/Breakthrough/data/trees/fixed_part.txt"), "w").write("\n".join(lines) + "\n")
