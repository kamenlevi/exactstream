#!/bin/sh
# TrainLab turn 1 (10-21). Kamen 17:10: the small-RAM bar = task results, frugal at 10 % qualifies, 5 % fails IFEval
# (-7) -> first healing run = frugal at 5 % (drop_w2_C13 = pol35 C13_w2_ldrop), LoRA incl. the router; then (next
# turns) the stall-hiding flows. Zero-shot table first (per segment). TL_LIMIT (seconds) comes from run_turn.sh.
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
echo "== $(date '+%m-%d %H:%M:%S') turn 1 start (limit $TL_LIMIT s)"
START=$(date +%s)
$PY train35.py eval exact drop_w2_C26 drop_w2_C13 half_w2_C13_x50 half_w2_C26 half late late_B20-40 prew_B8-40 k5 k5_n8 2>&1 | grep -v "Token indices"
USED=$(( $(date +%s) - START ))
$PY train35.py train --flow drop_w2_C13 --run drop_w2_C13_r16 --targets gdn,attn,shared,router --seq 1024 --batch 2 --lr 1e-4 --eval-every 131072 --final-evals drop_w2_C26,half_w2_C13_x50,exact --time-limit $(( TL_LIMIT - USED - 60 )) 2>&1 | grep -v "Token indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 1 end"
