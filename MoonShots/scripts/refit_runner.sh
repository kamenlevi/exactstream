#!/bin/sh
# MoonShots refit_runner: 35B model-level test of the activation-aware 2/3-bit tier refit (scripts/refit35.py, ~25 GB GPU).
# Launch while holding a quiet lock; waits (<= 3 min) until no other process holds > 8 GB and memory is >= 45 % free.
OUT=$HOME/MoonShots/data/day15; PY=$HOME/BigMoE/.venv/bin/python; mkdir -p "$OUT"
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 45 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 18 ]; do
  sleep 10; g=$((g + 1))
done
if [ $g -ge 18 ]; then log "skip refit35 (memory busy: $(memory_pressure | tail -1))"; exit 1; fi
log "refit35 started ($(memory_pressure | tail -1))"
$PY $HOME/MoonShots/scripts/refit35.py > "$OUT/refit35.txt" 2>&1
log "done refit35: $(grep -E '^(refit done|---|4[0-9]{7}:|Traceback|[A-Za-z]*Error)' "$OUT/refit35.txt" | cut -c1-150 | tr '\n' ' ')"
