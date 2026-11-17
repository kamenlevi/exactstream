#!/bin/sh
# L1-16 batch 1: weight-space shared structure (layers 12/24/36); then output error at equal bits if Fit12 Q2 inputs exist.
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/wspace16.py
ls $HOME/Fit12/data/q2/x_L*.npy > /dev/null 2>&1 && $PY scripts/act16.py
