#!/bin/sh
# TrainLab turn 5 (10-21, swapped in 23:10: NewIdeas redirected the drafter to the 125B; this was turn 6): the healing recipe check. Same flow and data as drop_w2_C13_r16b (frugal 5 %,
# LoRA r16 GDN/attn/shared + router) but lr 3e-5 with 4-window gradient accumulation (8K tokens per update): does the
# slope (KLD - floor ~ tokens^-0.135 at lr 1e-4) improve? Evals every 131K tokens; cross-flow evals at the end.
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 5 start (limit $TL_LIMIT s)"
$PY train35.py train --flow drop_w2_C13 --run drop_w2_C13_r16c_lr3e5_acc4 --targets gdn,attn,shared,router --seq 1024 --batch 2 --lr 3e-5 --accum 4 --eval-every 131072 --final-evals drop_w2_C26,exact --time-limit $(( TL_LIMIT - 60 )) 2>&1 | $F
echo "== $(date '+%m-%d %H:%M:%S') turn 5 end"
