#!/bin/sh
# Dequant-once multi-row kernel (wide/mr3c.py) validation (launch holding a quiet lock; unlocks at the end; summary in
# data/wide/summary.txt): bit-identity vs 3c at S = 4/8/16/32 + per-layer time of 3c vs mr vs the read-once floor, on the
# natural and budget sets; L20 code with RM 8 and 16, L40 chat and L6 math with RM 8 (~1 min each).
M=$HOME/MoonShots; OUT=$M/data/wide; PY=$HOME/BigMoE/.venv/bin/python; mkdir -p "$OUT"
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/summary.txt"; }
log "mr validation start"
for cfg in "20 code 8" "20 code 16" "40 chat 8" "6 math 8"; do
  set -- $cfg
  env MB_ARMS=3c,mr,3c_distinct MB_RM=$3 MB_TAG=_mr$3 $PY $M/wide/microbench.py $1 $2 30 > "$OUT/mr_L$1_$2_rm$3.txt" 2>&1
  log "mr L$1 $2 RM$3: $(grep -E '^S=|check mr|Error|Traceback' "$OUT/mr_L$1_$2_rm$3.txt" | tr '\n' ' ' | cut -c1-1500)"
done
peer unlock > /dev/null 2>&1
log "mr validation finished"
