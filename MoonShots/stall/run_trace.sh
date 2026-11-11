#!/bin/sh
# SSD idle-window trace on BigMoE v3 at 24 GB (launch holding a quiet lock; unlocks at the end).
OUT=$HOME/MoonShots/data/stall; PY=$HOME/BigMoE/.venv/bin/python; M=$HOME/MoonShots
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
for P in code prose edit; do
  STALL_TRACE=1 STALL_TIER=1 $PY $M/stall/anatomy.py $P 160 0.25 > "$OUT/anatomy_${P}_tier_trace.txt" 2>&1
  echo "$(date +%H:%M) trace v3 $P: $(grep -E '^\[[a-z]+_tier|Traceback|[A-Za-z]*Error' "$OUT/anatomy_${P}_tier_trace.txt" | cut -c1-200 | tr '\n' ' ')" >> "$OUT/summary3.txt"
done
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) trace finished" >> "$OUT/summary3.txt"
