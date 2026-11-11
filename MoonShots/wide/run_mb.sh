#!/bin/sh
# Grouped expert kernel microbench (launch holding a quiet lock; unlocks at the end; summary in data/wide/summary.txt):
# wide/microbench.py on a middle / late / early layer and on code / chat / math routing (~3 min each).
M=$HOME/MoonShots; OUT=$M/data/wide; PY=$HOME/BigMoE/.venv/bin/python; mkdir -p "$OUT"
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/summary.txt"; }
log "microbench start"
for cfg in "20 code" "40 chat" "6 math"; do
  set -- $cfg
  $PY $M/wide/microbench.py $1 $2 30 > "$OUT/mb_L$1_$2.txt" 2>&1
  log "L$1 $2: $(grep -E '^S=|bandwidth|check|Error|Traceback' "$OUT/mb_L$1_$2.txt" | tr '\n' ' ' | cut -c1-1500)"
done
$PY $M/wide/dense_scaling.py > "$OUT/dense_scaling.txt" 2>&1
log "dense scaling: $(tr '\n' ' ' < "$OUT/dense_scaling.txt" | cut -c1-1500)"
peer unlock > /dev/null 2>&1
log "microbench finished"
