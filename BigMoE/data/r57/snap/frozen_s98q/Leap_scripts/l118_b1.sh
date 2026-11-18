#!/bin/sh
# L1-18 batch 1 (Leap, Sat 2026-10-24): the 125B's top-32 log-probs on the 24 public HumanEval samples (one streamed
# prefill, C = 96 resident experts per layer, < 36 GB), then the DFlash2 headroom analysis. ~10 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
DFL2C_IN=$HOME/Leap/data/l118_windows.jsonl DFL2C_OUT=$HOME/Leap/data/l118_tgt DFL2C_C=96 $PY scripts/dfl2c_extract.py || exit 1
$PY scripts/headroom18.py
