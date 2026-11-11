#!/bin/sh
# B10(b) batch (Leap, Sun 2026-10-25): trellis-decode GLU kernels vs 3c. Run under one quiet lock, cwd ~/Leap.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
for m in test sweep bench; do
  echo "[b10] $m $(date '+%H:%M:%S')"
  $PY scripts/b10_trellis.py $m > data/b10_$m.log 2>&1 || { echo "[b10] $m failed"; tail -8 data/b10_$m.log; exit 1; }
done
echo "[b10] done $(date '+%H:%M:%S')"
