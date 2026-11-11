#!/bin/sh
# L1-20 batch 1 (Leap, Sat 2026-10-24; pre-registered in ~/Leap/TABLE_350.md): the 125B's own greedy outputs on the 200
# public L5 prompts (80 unused HumanEval+ problems + 120 GSM8K; C=128, <= 60K tokens or 40 min), then their features and
# top-32 labels (C=96). Resumable: each step skips what is on disk. ~55 min, < 36 GB. Batch 2 trains and scores.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Leap/scripts
D=$HOME/Leap/data
echo "== L1-20 b1 $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
[ -e $D/l120_gen.done ] || { DFL2D_IN=$D/dfl2e_samples.jsonl DFL2D_OUT=$D/l120_windows.jsonl DFL2D_C=128 DFL2D_MAXT=400 \
  $PY $S/dfl2d_gen.py 60000 40 && touch $D/l120_gen.done; } || { echo "== generation FAILED"; exit 1; }
echo "== features $(date +%H:%M:%S)"
DFL2C_IN=$D/l120_windows.jsonl DFL2C_OUT=$D/l120 DFL2C_C=96 $PY $S/dfl2c_extract.py || { echo "== extraction FAILED"; exit 1; }
echo "== done $(date +%H:%M:%S): $(sysctl -n vm.swapusage | cut -d'(' -f1), disk $(df -g / | tail -1 | awk '{print $4}') GB free"
