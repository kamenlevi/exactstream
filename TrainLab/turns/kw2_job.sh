#!/bin/sh
# Breakthrough's request (11-03 17:08): DFlash2 L1-20's top-16 + teacher-forced selector softmax at HPD-0's anchors on IFEval
# and multi (d02b_walk_match.py TL_WALK_CF + TL_WALK_TOP16 + TL_WALK_ANCHORS=_hpd0; HPD-0's full sample lists, which hold the
# 12 + 12 map windows): data/d02b_walk_kw_ifeval.npz / _multi.npz, the format of d02b_walk_kw_code.npz. MLX ~2.2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk_match.py; pkill -P $$; exit 143' TERM
echo "== KW2 walk $(date +%H:%M:%S)"
TL_WALK_CF=1 TL_WALK_TOP16=1 TL_WALK_ANCHORS=_hpd0 TL_WALK_TAG=_kw \
  TL_WALK_SETS="ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz,multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz" \
  HF_HUB_OFFLINE=1 $PY sticky/d02b_walk_match.py 2>&1 | grep -E "samples|MLX peak|rror"
echo "== KW2 done $(date +%H:%M:%S)"
