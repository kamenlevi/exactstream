#!/bin/sh
# Runs turns/turnN.sh for the next turn number N (counter in turns/next); if that file does not exist, the default job:
# continue the 125B MTP head's native training (mtp125n_d4) for the whole turn. TL_LIMIT comes from run_turn.sh.
T=$HOME/TrainLab; cd $T
N=$(cat turns/next 2>/dev/null || echo 1)
echo $((N + 1)) > turns/next
if [ -x turns/turn$N.sh ]; then
  echo "== turn $N: turns/turn$N.sh"
  exec turns/turn$N.sh
fi
echo "== turn $N: default = native MTP training (mtp125n_d4)"
TL_PLAIN_MIN=1 ~/BigMoE/.venv/bin/python mtp125.py train --native --run mtp125n_d4 --depth 4 --lr 1e-5 --time-limit $(( TL_LIMIT - 60 )) 2>&1 | grep --line-buffered -v "Token indices" | grep --line-buffered -v "moe-sorted-gather"
