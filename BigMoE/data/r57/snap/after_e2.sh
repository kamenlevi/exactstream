#!/bin/sh
# DAY9 gap order (the orchestrator, Mon ~13:4x): MTP Head's E2 capture first, then BigMoE's A6. Exit 0 once E2's x2 data
# is captured (its own test: >= 16,171 MBPP code tokens in d1102mbpp) or its chain has moved on (train x2/x4, x2 not finished).
f=/Users/kamen/TrainLab/data/dlog/d1102mbpp/index.jsonl
n=$( [ -f $f ] && grep -o '"n": [0-9]*' $f | awk '{s += $2} END {print s + 0}' || echo 0)
[ "${n:-0}" -ge 16171 ] && exit 0
grep -qE '^11-0[23] [0-9:]+ e2_chain: (train x|x2 not finished)' /Users/kamen/TrainLab/logs/turns.log
