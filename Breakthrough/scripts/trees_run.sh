#!/bin/sh
# TREES dump runner (Breakthrough Sun 1 Nov 08:3x; lines in SCOREBOARD 'TREES LINES'; chat added at NewIdeas' 08:3x request):
# DFlash2 L1-20 lattices on the CPU (scripts/tree_dump.py, MLX on the CPU, <= 4 threads, nice 10) under a CPU share lock:
#   1. code: Leap's 24 HumanEval captures (the 9.54-per-pass run)
#   2. chat: MTP Head's D-02a tf_ifeval 30 + tf_multi 24 (R-61N's chat set)
#   3. rest: D-02a agent 12, mgsm 30, tf_gsm 60
# (background QoS ran 1.3-115 s per block on the efficiency cores; nice 10 runs ~0.16 s per block.) Waits for no HOLD / PAUSE
# and >= 13 GB available before each stage (unlocking while it waits); a stage that grows swap by > 2 GB is stopped.
# Log: data/trees/run.txt
cd "$HOME/Breakthrough" || exit 1
PY=$HOME/BigMoE/.venv/bin/python; OUT=$HOME/Breakthrough/data/trees; Q=$HOME/.local/share/quiet; D2=$HOME/TrainLab/data/d02a
mkdir -p $OUT
avail() { vm_stat | awk '/Pages free|Pages inactive|Pages speculative|Pages purgeable/ {gsub("\\.", "", $NF); s += $NF} END {print int(s * 16384 / 1e9)}'; }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
export VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 MLX_NUM_THREADS=4
stage() {   # stage TAG MINUTES SRC
  tag=$1; min=$2; src=$3
  until [ ! -e $Q/HOLD ] && [ ! -e $Q/PAUSE ] && [ "$(avail)" -ge 13 ] && \
        peer lock share ${min}m --cpu 400 --gpu 1 "Breakthrough TREES: CPU replay of DFlash2 lattices ($tag, ~$min min, ~5 GB, no GPU)" > /dev/null 2>&1; do
    sleep 60; done
  date "+%H:%M:%S $tag: share granted (available $(avail) GB, swap $(swap_mb) MB)"
  SW0=$(swap_mb)
  TD_SRC="$src" TD_OUT=$OUT/lattice_$tag.npz nice -n 10 $PY scripts/tree_dump.py > $OUT/dump_$tag.txt 2>&1 & P=$!
  while kill -0 $P 2>/dev/null; do sleep 10
    if [ $(( $(swap_mb) - SW0 )) -gt 2048 ]; then echo "swap grew > 2 GB: stopping $tag"; kill $P; sleep 5; kill -9 $P 2>/dev/null; fi
  done
  wait $P; rc=$?
  peer unlock > /dev/null 2>&1
  date "+%H:%M:%S $tag done (rc $rc): $(tail -1 $OUT/dump_$tag.txt | cut -c1-200)"
}
stage code 6 "$HOME/Leap/data/dfl2d_code/*.npz"
stage chat 20 "$D2/tf_ifeval_*.npz,$D2/tf_multi_*.npz"
stage rest 22 "$D2/agent_*.npz,$D2/mgsm_*.npz,$D2/tf_gsm_*.npz"
