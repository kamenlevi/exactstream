#!/bin/sh
# Short GPU turn (launch holding a quiet lock; unlocks at the end): tier_glu bit-stability with fixed inputs on all trunk
# layers' real B8 pools, quiet vs under gpuload.py; fused combine (frugal block) vs unfused + MLX sum (v3-like). Log tgtest.txt.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/tgtest.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "tgtest turn start"
for C in n gpu; do
  case $C in gpu*) load_on;; esac
  env MR_FRAC=0.10 B12_LATE3=2,10 MR_ARM=wait MR_TAG=_b12ll_$C $PY $R tgtest 1500 > "$OUT/tgtest_$C.txt" 2>&1
  log "tgtest $C: $(grep -E '^\[tgtest|Error|Traceback' "$OUT/tgtest_$C.txt" | tr '\n' ' ' | cut -c1-400)"
  load_off
done
log "tgtest turn finished"
