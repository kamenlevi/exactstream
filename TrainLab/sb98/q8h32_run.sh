#!/bin/bash
# Q8H32 engine artifact build + bitwise verification (MTP Head, orchestrator 11:0x), ~10-12 GB: run_paused with NOSWAP_MIN 20.
cd "$HOME/TrainLab" || exit 1
export MEMGB=12 NOSWAP_MIN=20
D=$HOME/TrainLab/data/q8h32
mkdir -p $D
for i in 1 2 3; do
  $HOME/TrainLab/sb98/run_paused.sh q8h32_build $D/build.log 20 $HOME/BigMoE/.venv/bin/python -I -B \
    $HOME/TrainLab/sb98/q8h32_artifact.py $HOME/datasets/deepdraft/q8h32 && break
  sleep 120
done
tail -4 $D/build.log
