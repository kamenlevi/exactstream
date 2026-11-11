#!/bin/sh
# B1 batch (Leap, Sun 2026-10-25): SSD chunk-size microbench (read-only on 4 model shards), then the expert-neuron
# energy test. Run under one quiet lock, cwd ~/Leap.
cd "$HOME/Leap" || exit 1
M=$HOME/BigMoE/models/bare
echo "[b1] ssd $(date '+%H:%M:%S')"
scripts/b1_ssd 1.5 $M/model-00005-of-00017.safetensors $M/model-00007-of-00017.safetensors \
  $M/model-00010-of-00017.safetensors $M/model-00013-of-00017.safetensors > data/b1_ssd.txt 2>&1 || exit 1
echo "[b1] energy $(date '+%H:%M:%S')"
$HOME/BigMoE/.venv/bin/python scripts/b1_energy.py > data/b1_energy.log 2>&1 || exit 1
echo "[b1] done $(date '+%H:%M:%S')"
