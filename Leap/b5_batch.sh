#!/bin/sh
# B5 batch (Leap, Sun 2026-10-25): 16-row check on a resident 12-layer slice of the 125B. Timings for variants A / B / C,
# then a Metal System Trace of A (GPU busy vs wall). Run under one quiet lock, cwd ~/Leap.
cd "$HOME/Leap" || exit 1
PY=$HOME/BigMoE/.venv/bin/python
TR=/private/tmp/claude-501/-Users-kamen-OjoX/f2971937-6c95-4e86-a049-05beff6a8e32/scratchpad/b5.trace
echo "[b5] time $(date '+%H:%M:%S')"
$PY scripts/b5_slice.py time > data/b5_slice.log 2>&1 || { echo "[b5] time failed"; tail -5 data/b5_slice.log; exit 1; }
echo "[b5] trace $(date '+%H:%M:%S')"
rm -rf "$TR"
xcrun xctrace record --template 'Metal System Trace' --output "$TR" --launch -- $PY scripts/b5_slice.py trace \
  > data/b5_trace.log 2>&1 || { echo "[b5] trace failed"; tail -5 data/b5_trace.log; exit 1; }
xcrun xctrace export --input "$TR" --toc > data/b5_toc.xml 2>&1
echo "[b5] done $(date '+%H:%M:%S')"
