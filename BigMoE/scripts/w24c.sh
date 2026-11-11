#!/bin/bash
# W24c (BigMoE, Mon 9 Nov; order of the orchestrator 23:0x: W24c first tonight, no start after 00:50, decide before the
# 01:25 gate): can the drafter diet buy the 14th row that 120 tok/s on HumanEval needs, under 24 GB?
# R57_D7B=all keeps only the 4-bit DFlash2 drafter (the gated w13t also holds the bf16 one, 1.14 GB, which drafts only
# math / agent-typed prompts). Arms (scripts/w24.sh armenv): w13c / w13d = w13t as gated (control, first and last),
# w14a / w14b = 140 base + 28 sidecar slots, 14 rows + the diet (twice), w13a = w13t + the diet alone (what it frees),
# w13b = 141 base slots, 13 rows + the diet. DECISION FIRST: (1) HumanEval w13c w14a w14b w13d (ABBA), (2) MBPP + IFEval for
# w14a (validity: every set under 24 GB on the fast path), then the diagnostics (3) HumanEval w13a w13b, (4) MBPP + IFEval
# for w13c. The rule (docs/W24_DECISION.md, W24c): see scripts/w24c_compare.py; the env switch is done by hand.
# Env only on snapshot w24: bigmoe/, snap/w24, frozen_rel1 and the .so stay as gated (EvalSuite's 01:25 gate pins them).
# Needs a held quiet lock (turn45.sh gives HAVE_LOCK=1 KEEP_LOCK=1). ~25 min; lifetime peak ~23.9 GB expected (<= 25).
# usage: NOTAFTER=00:50 WAITMEM=29 MARGIN=4 AFTER="bash scripts/w24c_after.sh" bash scripts/turn45.sh 25 40 "..." bash scripts/w24c.sh
cd "$HOME/BigMoE" || exit 1
LOG=data/w24/w24.log
past() { [ $(( (10#$(date +%H) * 60 + 10#$(date +%M) - 10#${1:0:2} * 60 - 10#${1:2:2} + 1440) % 1440 )) -lt 720 ]; }
if past "${W24C_START_BY:-0050}"; then
  echo "$(date '+%m-%d %H:%M:%S') W24c not started: past ${W24C_START_BY:-0050} (the orchestrator: no start after 00:50)" >> "$LOG"; exit 0
fi
DL=${W24C_DEADLINE:-0102}   # the last arm start for the decision passes (an arm takes ~2 min, HARD caps one at ~11)
W24_ARMS="w13c w14a w14b w13d" W24_SETS="h" W24_RUN=w24c W24_DEADLINE=$DL bash scripts/w24.sh
W24_ARMS="w14a" W24_SETS="m i" W24_RUN=w24c2 W24_DEADLINE=$DL bash scripts/w24.sh
python3 scripts/w24c_compare.py h w13c w14a w14b w13d > data/w24/w24c_verdict.txt 2>&1
python3 scripts/w24c_compare.py m w14a >> data/w24/w24c_verdict.txt 2>&1
python3 scripts/w24c_compare.py i w14a >> data/w24/w24c_verdict.txt 2>&1
echo "$(date '+%m-%d %H:%M:%S') W24c decision passes done: data/w24/w24c_verdict.txt" >> "$LOG"
W24_ARMS="w13a w13b" W24_SETS="h" W24_RUN=w24c3 W24_DEADLINE=${W24C_DEADLINE2:-0100} bash scripts/w24.sh
W24_ARMS="w13c" W24_SETS="m i" W24_RUN=w24c4 W24_DEADLINE=${W24C_DEADLINE2:-0100} bash scripts/w24.sh
