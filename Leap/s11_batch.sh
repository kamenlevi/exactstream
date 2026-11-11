#!/bin/sh
# S11 batch (Leap, Sun 2026-10-25): grouped-decode 3c vs per-pair 3c. Run under one quiet lock, cwd ~/Leap.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
for m in test bench; do
  echo "[s11] $m $(date '+%H:%M:%S')"
  $PY scripts/s11_grouped.py $m > data/s11_$m.log 2>&1 || { echo "[s11] $m failed"; tail -12 data/s11_$m.log; exit 1; }
done
echo "[s11] done $(date '+%H:%M:%S')"
