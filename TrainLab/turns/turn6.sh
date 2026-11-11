#!/bin/sh
# TrainLab turn 6 (10-21/22): the 125B MTP head fine-tune (NewIdeas 23:05). probe (module structure) -> check (my
# autodiff forward == MTPLX's mtp_forward on a real turn) -> if the check agrees (argmax >= 0.99), DeepSeek-style
# depth-4 training on Claude Code turns (streamed trunk prefill per turn, online) for the rest of the turn.
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 6 start (limit $TL_LIMIT s)"
START=$(date +%s)
$PY mtp125.py probe 2>&1 | $F
$PY mtp125.py check 2>&1 | $F | tee logs/mtp125_check.out
AG=$(grep -o "argmax agreement [0-9.]*" logs/mtp125_check.out | awk '{print $3}')
echo "== check argmax agreement: ${AG:-none}"
if [ -n "$AG" ] && awk -v a="$AG" 'BEGIN{exit !(a >= 0.99)}'; then
  USED=$(( $(date +%s) - START ))
  $PY mtp125.py train --run mtp125_d4 --depth 4 --lr 1e-5 --time-limit $(( TL_LIMIT - USED - 60 )) 2>&1 | $F
else
  echo "== check failed: no training this turn (lock released early)"
fi
echo "== $(date '+%m-%d %H:%M:%S') turn 6 end"
