#!/bin/sh
# L1-15 batch 3 (Leap, Sat 2026-10-24): the decisive test (30 mixers at 16 rows, ms + spec == plain bitwise), then the
# qmm16 v2 variants. ~3 min, 35B in RAM (~21 GB).
cd "$HOME/Leap" || exit 1
PY=$HOME/SpecDecode/.venv/bin/python
$PY scripts/test_mixer16.py
$PY scripts/test_qmm16b.py
