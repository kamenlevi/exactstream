#!/bin/sh
# TrainLab turn 8 (10-22): native training of the 125B MTP head. MTPLX's own QSA attention and HC are differentiable at
# prefill width (turn 7); its MoE call is not (argpartition indices into take_along_axis), so the head's MoE is
# wrapped by NativeMoE (the same plain path with stop_gradient on the indices). check2 first (incl. NativeMoE == MTPLX
# MoE and its gradient), then depth-4 training for the rest of the turn. Turn 9 scores it (agent125_tl.py).
PY=$HOME/BigMoE/.venv/bin/python; T=$HOME/TrainLab
cd $T
F="grep --line-buffered -v Token.indices"
echo "== $(date '+%m-%d %H:%M:%S') turn 8 start (limit $TL_LIMIT s)"
START=$(date +%s)
TL_PLAIN_QSA=1 $PY mtp125.py check2 2>&1 | $F | grep --line-buffered -v "moe-sorted-gather" | tee logs/mtp125_check2_t8.out
if grep -q "grad through MTPLX self_attn: OK" logs/mtp125_check2_t8.out && grep -q "grad through MTPLX NativeMoE: OK" logs/mtp125_check2_t8.out; then
  USED=$(( $(date +%s) - START ))
  TL_PLAIN_QSA=1 $PY mtp125.py train --native --run mtp125n_d4 --depth 4 --lr 1e-5 --time-limit $(( TL_LIMIT - USED - 60 )) 2>&1 | $F | grep --line-buffered -v "moe-sorted-gather"
else
  echo "== native autodiff still not available: no training this turn"
fi
echo "== $(date '+%m-%d %H:%M:%S') turn 8 end"
