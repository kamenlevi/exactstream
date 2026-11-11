#!/bin/bash
# W24d (BigMoE, Mon 9 Nov 23:5x; the orchestrator 23:5x: "yes, option B, with the 14-row config staying armed as the
# fallback"): W24c's diagnostic w13b (141 base + 26 sidecar slots, 13 rows, 4-bit DFlash2 drafter only: 129.9 tok/s, one
# sample) against the 14-row config (140 + 28, 14 rows, the same drafter; = data/stack/card_w24c_env.txt) in a same-run ABBA
# on HumanEval: w14c, w13e, w13f, w14d. The rule: docs/W24_DECISION.md section W24d (committed before this run). No start
# after 00:15 (then option A: the 14-row config runs). Env only on snapshot w24 (016cb7f). Under turn45 (quiet).
cd "$HOME/BigMoE" || exit 1
LOG=data/w24/w24.log
past() { [ $(( (10#$(date +%H) * 60 + 10#$(date +%M) - 10#${1:0:2} * 60 - 10#${1:2:2} + 1440) % 1440 )) -lt 720 ]; }
if past 0015; then echo "$(date '+%m-%d %H:%M:%S') W24d not started: past 00:15 (option A: the 14-row config runs)" >> "$LOG"; exit 0; fi
W24_ARMS="w14c w13e w13f w14d" W24_SETS="h" W24_RUN=w24d W24_DEADLINE=${W24D_DEADLINE:-0030} bash scripts/w24.sh
python3 scripts/w24c_compare.py h w14c w13e w13f w14d > data/w24/w24d_verdict.txt 2>&1
python3 - >> data/w24/w24d_verdict.txt 2>&1 <<'PY'
import json, os
R = os.path.expanduser("~/BigMoE/data/r57")
def runs(a): return json.load(open(f"{R}/{a}h_100_runs.json"))
def wd(rows): return sum(r["n"] for r in rows) / sum(r["n"] / r["tps"] for r in rows)
A = {a: runs(a) for a in ("w14c", "w13e", "w13f", "w14d")}
b13, b14 = wd(A["w13e"] + A["w13f"]), wd(A["w14c"] + A["w14d"])
g = 100 * (b13 / b14 - 1)
pk = {a: max(max(r["mem"][1], r["mem"][-1]) for r in A[a]) for a in A}
bad = sum(1 for a in A for r in A[a] if r["bad"].strip() or r["cut"])
c = [(f"w13b pooled (w13e + w13f) {b13:.2f} vs 14-row pooled (w14c + w14d) {b14:.2f}: {g:+.2f} % > +1 %", g > 1.0),
     (f"w13b peaks w13e {pk['w13e']:.3f} / w13f {pk['w13f']:.3f} < 24.000 GB (14-row: {pk['w14c']:.3f} / {pk['w14d']:.3f})",
      max(pk["w13e"], pk["w13f"]) < 24.0),
     (f"bad or cut rows in all four arms: {bad}", bad == 0)]
print("W24d rule (docs/W24_DECISION.md 'W24d'):")
for t, ok in c: print(f"  [{'PASS' if ok else 'FAIL'}] {t}")
print("  VERDICT: " + ("w13b WINS (still needs card_w24d_env.txt + EvalSuite's dry run + NewIdeas' check by 01:30)"
                       if all(ok for _, ok in c) else "the 14-row config runs"))
PY
echo "$(date '+%m-%d %H:%M:%S') W24d done: data/w24/w24d_verdict.txt" >> "$LOG"
