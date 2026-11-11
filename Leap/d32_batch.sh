#!/bin/sh
# D32 batch (Leap, Mon 2026-10-26): dense part of an exact check, 32 rows vs 16. Run under one quiet lock, cwd ~/Leap.
# smoke first (2 small shapes: compile + bitwise, ~20 s), then the full set (~3-5 min).
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
echo "[d32] smoke $(date '+%H:%M:%S') swap $(sysctl -n vm.swapusage | cut -d'(' -f1)"
$PY scripts/d32.py smoke > data/d32_smoke.log 2>&1 || { echo "[d32] smoke failed"; tail -15 data/d32_smoke.log; exit 1; }
cat data/d32_smoke.txt
echo "[d32] full $(date '+%H:%M:%S')"
$PY scripts/d32.py > data/d32.log 2>&1 || { echo "[d32] full failed"; tail -15 data/d32.log; exit 1; }
echo "[d32] done $(date '+%H:%M:%S') swap $(sysctl -n vm.swapusage | cut -d'(' -f1)"
