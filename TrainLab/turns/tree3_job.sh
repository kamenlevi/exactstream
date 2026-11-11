#!/bin/sh
# TREE3 (MTP Head, Tue 11-03; the Breakthrough Push): (1) DFlash2 L1-20's top-16 + teacher-forced selector softmax at EVERY
# position of the 36 decision-map windows (d02b_walk_match.py TL_WALK_EVERY + TL_WALK_TOP16; MLX ~2.2 GB); (2) the MTP head
# b5's top-3 per anchor and depth (sticky/tree3_mtp.py; MLX ~10 GB as D-24); (3) the aligned export (CPU).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk_match.py; pkill -f tree3_mtp.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
guard() { SW0=$(swap); ( while sleep 10; do if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB: stopping $1"; pkill -f "$1"; exit; fi; done ) & G=$!; }
if [ ! -f data/d02b_walk_tree3_multi.npz ]; then
  echo "== TREE3 DFlash2 walk $(date +%H:%M:%S)"; guard d02b_walk_match.py
  TL_WALK_EVERY=1 TL_WALK_CF=1 TL_WALK_TOP16=1 TL_WALK_TAG=_tree3 \
    TL_WALK_SETS="code=$HOME/TrainLab/data/tree3_src/code/*.npz,ifeval=$HOME/TrainLab/data/tree3_src/ifeval/*.npz,multi=$HOME/TrainLab/data/tree3_src/multi/*.npz" \
    HF_HUB_OFFLINE=1 $PY sticky/d02b_walk_match.py 2>&1 | grep -E "samples|MLX peak|rror"
  kill $G 2>/dev/null
fi
if [ ! -f data/tree3/mtp_multi.npz ]; then
  echo "== TREE3 MTP top-3 $(date +%H:%M:%S)"; guard tree3_mtp.py
  $PY sticky/tree3_mtp.py --run b5 2>&1 | grep -E "windows|rror|Trace"
  kill $G 2>/dev/null
fi
$PY -I sticky/tree3_export.py
echo "== TREE3 done $(date +%H:%M:%S)"
