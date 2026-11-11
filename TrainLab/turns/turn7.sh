#!/bin/sh
# TrainLab turn 7 (10-22): localize the 125B MTP forward mismatch (my forward agreed only 95 % with MTPLX's) and test
# autodiff through MTPLX's own head modules with the custom-kernel QSA/MoE paths off; if gradients pass both, train
# NATIVELY (MTPLX's own head with fp32 trainable projections: exact match to inference by construction).
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 7 start (limit $TL_LIMIT s)"
START=$(date +%s)
$PY mtp125.py check2 2>&1 | $F | grep --line-buffered -v "moe-sorted-gather"
TL_PLAIN_QSA=1 $PY mtp125.py check2 2>&1 | $F | grep --line-buffered -v "moe-sorted-gather" | tee logs/mtp125_check2_plain.out
if grep -q "grad through MTPLX self_attn: OK" logs/mtp125_check2_plain.out && grep -q "grad through MTPLX mlp: OK" logs/mtp125_check2_plain.out; then
  USED=$(( $(date +%s) - START ))
  TL_PLAIN_QSA=1 $PY mtp125.py train --native --run mtp125n_d4 --depth 4 --lr 1e-5 --time-limit $(( TL_LIMIT - USED - 60 )) 2>&1 | $F | grep --line-buffered -v "moe-sorted-gather"
else
  echo "== native autodiff not available: no training this turn"
fi
echo "== $(date '+%m-%d %H:%M:%S') turn 7 end"
