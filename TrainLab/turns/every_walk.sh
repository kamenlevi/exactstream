#!/bin/sh
# MTP Head, Tue 11-03: DFlash2 L1-20's own + teacher-forced picks and cf at EVERY position (d02b_walk.py TL_WALK_EVERY +
# TL_WALK_CF; MLX ~2.2 GB, no timing) on code, math, multi, IFEval and the MBPP walk set: DEPTH32's step 2 (same tag and
# sets as turns/d23h_job.sh, which then skips it) + the tables for MTPPICK's exact official-loop simulation (REPORT 'MTPPICK').
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
SW0=$(swap); ( while sleep 10; do if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB: stopping the walk"; pkill -f d02b_walk.py; exit; fi; done ) & G=$!
echo "== every-position walk $(date +%H:%M:%S)"
TL_WALK_EVERY=1 TL_WALK_CF=1 TL_WALK_TAG=_d23h_dfl \
  TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz,multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz,ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz,mbpp=$HOME/TrainLab/data/taps_mbpp_walk/*.npz" \
  HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep -E "samples|rror|MLX peak"
kill $G 2>/dev/null
echo "== every-position walk done $(date +%H:%M:%S)"
