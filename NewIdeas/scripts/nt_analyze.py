"""NT-1 analysis (NewIdeas, Mon 2026-10-26): reads data/nt/nt_accept_b{BS}.jsonl (from nt_accept.py) and bounds what
near-tie (lossy) acceptance could add to tokens per pass.
Rules: forgive a miss when the drafter's token d has target prob p_d >= rho * p_top1 (rho 0.5 / 0.25 / 0.1), or when d is
the target's #2 and p_top1 < 0.9. Only the FIRST miss of each block has the target's distribution for the drafted prefix
(teacher-forced replay), so the run after a forgiven miss is bounded two ways:
  LB  the forgiven token is kept and the run ends at the next position (+1 token per forgiven block);
  MOD the run continues with the exact replay's conditional survival per depth (P(hit at j | hits before j)), and later
      misses are forgiven with the rule's first-miss rate. MOD assumes the drafter's own branch is as easy to follow as
      the target's; it is an estimate, not a bound.
Quality is not measured here. A rule only goes to a GPU decode + Eval suite test if MOD clears +25 % at block 32."""
import os, sys, json
import numpy as np
BS = int(os.environ.get("NT_BS", "32"))
SRC = os.path.expanduser(f"~/NewIdeas/data/nt/nt_accept_b{BS}.jsonl")
recs = [json.loads(l) for l in open(SRC)]
out = []
def say(s):
    print(s, flush=True); out.append(s)

nb_max = BS - 1
tok_exact = np.mean([r["acc"] for r in recs])
say(f"NT-1 analysis, block {BS}: {len(recs)} blocks from {SRC}")
say(f"  exact tokens per pass {tok_exact:.2f} (Leap cand21_l120 'block 32 selector, truncate t 0.0' = 10.23: harness check)")

# first-miss bins by the target's top-1 p at the missed position
miss = [r for r in recs if r["k"] < r["nb"]]
BINS = [(0.99, 1.01, "p >= 0.99"), (0.9, 0.99, "0.9-0.99"), (0.5, 0.9, "0.5-0.9"), (0.0, 0.5, "< 0.5")]
say(f"  first misses: {len(miss)} of {len(recs)} blocks; by the 125B's top-1 p at the missed position:")
for lo, hi, nm in BINS:
    sub = [r for r in miss if lo <= r["p1"][r["k"]] < hi]
    in32 = [r for r in sub if r["rank"] >= 0]
    say(f"    {nm:10s}: {len(sub):4d} ({len(sub) / max(1, len(miss)):.2f})  drafter token in the 125B's top-32: "
        f"{len(in32) / max(1, len(sub)):.2f}, median p_d {np.median([r['pd'] for r in sub]) if sub else 0:.3f}")

# conditional survival per depth from the exact replay
reach = np.zeros(nb_max); hits = np.zeros(nb_max)
for r in recs:
    for j in range(min(r["nb"], r["k"] + 1)):
        reach[j] += 1; hits[j] += r["hit"][j]
surv = hits / np.maximum(1, reach)
say("  conditional survival P(hit at depth j | hits before), j = 1..: " + " ".join(f"{s:.2f}" for s in surv[:min(nb_max, 31)]))

RULES = [("rho 0.5", lambda r: r["pd"] >= 0.5 * r["p1"][r["k"]]),
         ("rho 0.25", lambda r: r["pd"] >= 0.25 * r["p1"][r["k"]]),
         ("rho 0.1", lambda r: r["pd"] >= 0.1 * r["p1"][r["k"]]),
         ("rank 2, p1 < 0.9", lambda r: r["rank"] == 1 and r["p1"][r["k"]] < 0.9)]
rng = np.random.default_rng(0)
for name, rule in RULES:
    forg = [r for r in miss if rule(r)]
    f = len(forg) / max(1, len(miss))
    lb = np.mean([r["acc"] + (1 if rule(r) and r["k"] < r["nb"] and r["k"] + 2 <= r["rem"] else 0) if r["k"] < r["nb"] else r["acc"]
                  for r in recs])
    tot = 0.0; RUNS = 200
    for _ in range(RUNS):
        for r in recs:
            if r["k"] >= r["nb"] or not rule(r):
                tot += r["acc"]; continue
            j = r["k"] + 1
            while j < r["nb"]:
                if rng.random() < surv[j] or rng.random() < f:
                    j += 1
                else:
                    break
            tot += min(j + 1, r["rem"])
    mod = tot / (RUNS * len(recs))
    say(f"  {name:17s}: forgives {f:.2f} of first misses ({len(forg)}); tokens per pass LB {lb:.2f} ({lb / tok_exact - 1:+.0%}), "
        f"MOD {mod:.2f} ({mod / tok_exact - 1:+.0%})")

# what the forgiven pairs look like (rho 0.5): the 125B's choice vs the drafter's
try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.expanduser("~/BigMoE/models/bare/tokenizer.json"))
    dec = lambda i: tok.decode([i])
except Exception as e:
    dec = lambda i: str(i)
forg = [r for r in miss if RULES[0][1](r)]
RISK = set("0123456789") | {"<", ">", "<=", ">=", "==", "!=", "+", "-", "*", "/", "%", "//", "**", "and", "or", "not",
                            "True", "False", "None", "is", "in", "+=", "-=", "[", "]", "(", ")", ":", "return", "if", "else"}
risky = sum(1 for r in forg if (dec(r["t1"]).strip() in RISK or dec(r["dt"]).strip() in RISK
                                or any(ch.isdigit() for ch in dec(r["t1"]) + dec(r["dt"]))))
say(f"  rho 0.5 forgiven pairs touching digits / operators / brackets / control words: {risky} of {len(forg)}")
say("  examples (125B's token -> drafter's token, p1 / p_d):")
for r in forg[:40]:
    say(f"    {dec(r['t1'])!r} -> {dec(r['dt'])!r}  ({r['p1'][r['k']]:.2f} / {r['pd']:.2f})")
open(os.path.expanduser(f"~/NewIdeas/data/nt/nt_accept_b{BS}.txt"), "w").write("\n".join(out) + "\n")
