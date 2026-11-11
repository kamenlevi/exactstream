#!/bin/sh
# L1-15 batch 9: per-prompt acceptance stock vs GDN kernels; matmul accuracy vs an fp32-exact reference. ~3 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/test_qmm_acc.py
$PY scripts/small35_acc.py
