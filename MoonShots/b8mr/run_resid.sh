#!/bin/sh
# Short GPU turn (launch holding a quiet lock; unlocks at the end): B8 decode under GPU load diverged from pass 0 (prose) /
# pass 14 (code) in reads. Pool residency (ext.lookup) after prefill, warm-up and every pass, normal vs GPU load; plus v3 exact
# decode under GPU load (sha vs 127b6932ba / d4ef51baea) to see whether the exact engine diverges too.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py; D=$M/defer/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/resid.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "resid turn start"
for C in n gpu gpu2; do
  case $C in gpu*) load_on;; esac
  for P in prose code; do
    env MR_RESID=1 MR_ARM=wait MR_TAG=_res_$C $PY $R gen $P 192 > "$OUT/res_gen_${P}_$C.txt" 2>&1
    log "gen $P wait res $C: $(grep -E '^\[(code|prose)_|Error|Traceback' "$OUT/res_gen_${P}_$C.txt" | tail -1 | cut -c1-260)"
  done
  load_off
done
load_on
for P in code prose; do
  env DH_ARM=v3 DH_TAG=_gpu $PY $D gen $P 160 > "$OUT/v3_gen_${P}_gpu.txt" 2>&1
  log "v3 gen $P gpu: $(grep -E '^\[(code|prose)_|Error|Traceback' "$OUT/v3_gen_${P}_gpu.txt" | tail -1 | cut -c1-260)"
done
load_off
log "resid turn finished"
for C in n gpu gpu2; do                                          # appended ~10:47: v3 EXACT teacher-forced logits, normal vs load
  case $C in gpu*) load_on;; esac
  env DH_ARM=v3 DH_TAG=_$C $PY $D tf 64 4 > "$OUT/v3_tf_$C.txt" 2>&1
  log "v3 tf $C: $(grep -E '^\[tf_|Error|Traceback' "$OUT/v3_tf_$C.txt" | tail -1 | cut -c1-200)"
  load_off
done
log "resid turn finished (appended v3 tf part)"
