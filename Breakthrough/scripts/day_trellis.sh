#!/bin/sh
# Breakthrough D2 step 1 (detached, 2026-10-20): trellis/scalar/affine NMSE on real 35B and 125B expert weights
# (scripts/trellis_nmse.py, a few GPU minutes). Starts when day_frugal.sh has exited; one quiet-lock ticket.
OUT=$HOME/Breakthrough/data/d2; mkdir -p "$OUT"
R=$(cat $HOME/Breakthrough/data/expA/day_frugal.pid 2>/dev/null)
while [ -n "$R" ] && kill -0 "$R" 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) waiting for the lock" >> "$OUT/summary.txt"
peer lock quiet --queue 90m "Breakthrough: D2 trellis NMSE probe (~5 min)" > /dev/null 2>&1 || { echo "$(date +%H:%M) no lock" >> "$OUT/summary.txt"; exit 1; }
echo "$(date +%H:%M) start trellis_nmse" >> "$OUT/summary.txt"
( PYTHONDONTWRITEBYTECODE=1 $HOME/BigMoE/.venv/bin/python $HOME/Breakthrough/scripts/trellis_nmse.py > "$OUT/trellis_nmse.txt" 2>&1 ) & P=$!
i=0; while kill -0 $P 2>/dev/null && [ $i -lt 1200 ]; do sleep 5; i=$((i + 5)); done
kill -0 $P 2>/dev/null && { kill -9 $P; echo "$(date +%H:%M) TIMEOUT" >> "$OUT/summary.txt"; }
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) done trellis_nmse: $(tail -2 "$OUT/trellis_nmse.txt" | tr '\n' ' ' | cut -c1-300)" >> "$OUT/summary.txt"
