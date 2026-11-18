#!/bin/sh
# L1-15 batch 2 (Leap, Sat 2026-10-24): qmm16 (NAX 16-row matmul) accuracy / invariance / speed, prework diagnosis, and
# the recurrence variants. No model load, ~2 min.
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/test_qmm16.py
$PY scripts/test_rec4.py
