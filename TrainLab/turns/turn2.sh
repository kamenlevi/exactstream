#!/bin/sh
# TrainLab turn 2 (10-21): (1) profile a training step (~3 min: where the 10.5 s/step goes); (2) zero-shot of NewIdeas'
# composition = pre-routed frugal (top-2 chosen a layer ahead, rest frugal) at C26/C13 + floor (for the copy-heavy log)
# + prew variants; (3) training to the end of the turn: the run named in turns/train2 (an argument line).
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 2 start (limit $TL_LIMIT s)"
START=$(date +%s)
$PY prof35.py drop_w2_C13 2>&1 | $F
$PY train35.py eval floor prew_drop_w2_C26_B8-40 prew_drop_w2_C13_B8-40 prew_drop_w2_C13 prew pre_B8-40 2>&1 | $F
USED=$(( $(date +%s) - START ))
$PY train35.py $(cat turns/train2) --time-limit $(( TL_LIMIT - USED - 60 )) 2>&1 | $F
echo "== $(date '+%m-%d %H:%M:%S') turn 2 end"
