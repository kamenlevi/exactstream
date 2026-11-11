#!/bin/sh
# TrainLab turn 10 (10-22): recipe 2 for the 125B MTP head, from turn 9's diagnosis of mtp125n_d4 (depth-4 training):
# better at chain depths 2-6, worse past depth 8 (never trained there), so the 16-row pass lost on long-chain turns.
# Recipe 2 = every chain depth the 16-row pass uses (15), depth j weighted 0.85^(j-1) (~ the chance a chain reaches it),
# per-depth logits under mx.checkpoint, window 512 (memory), + the 125B's own greedy general text (EvalSuite ref
# decodes, disjoint from tf_gen) 1:1 beside each new transcript; warm start from mtp125n_d4. Turn 11 scores it.
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
G="grep --line-buffered -v moe-sorted-gather"
echo "== $(date '+%m-%d %H:%M:%S') turn 10 start (limit $TL_LIMIT s)"
TL_PLAIN_MIN=1 $PY mtp125.py train --native --run mtp125n_d15 --init mtp125n_d4 --depth 15 --wdecay 0.85 --ck-loss \
  --window 512 --lr 1e-5 --mix-general 1.0 --time-limit $(( TL_LIMIT - 60 )) 2>&1 | $F | $G
echo "== $(date '+%m-%d %H:%M:%S') turn 10 end"
