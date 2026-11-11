#!/bin/sh
# mr kernel diagnostics (launch holding a quiet lock; unlocks at the end): which part of the per-pair path costs more than
# 3c? (RA, RB, RM) variants at S = 4 and 16 (natural + U=32) on L20 code; RM=1 = the structure with no amortization.
M=$HOME/MoonShots; OUT=$M/data/wide; PY=$HOME/BigMoE/.venv/bin/python
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/summary.txt"; }
log "mr diagnostics start"
for cfg in "4 4 4" "2 2 1" "4 4 1" "4 4 2" "4 4 8"; do
  set -- $cfg
  env MB_ARMS=3c,mr MB_S=4,16 MB_RA=$1 MB_RB=$2 MB_RM=$3 MB_TAG=_ra$1rb$2rm$3 $PY $M/wide/microbench.py 20 code 30 > "$OUT/mr2_ra$1rb$2rm$3.txt" 2>&1
  log "mr RA$1 RB$2 RM$3: $(grep -E '^S=|check mr|Error|Traceback' "$OUT/mr2_ra$1rb$2rm$3.txt" | sed -E 's/3c_3 nan, mlx nan, grouped nan, 3c_grp nan, //; s/, 3c_distinct nan \\| floor distinct.*\\| 3c \\/ mr/ | 3c\\/mr/' | tr '\\n' ' ' | cut -c1-700)"
done
peer unlock > /dev/null 2>&1
log "mr diagnostics finished"
