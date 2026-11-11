#!/bin/sh
# Short GPU turn (launch holding a quiet lock; unlocks at the end): where does the production wait-all path first diverge
# under GPU load? tfs 192 4 with the per-call recorder (MR_OBS) under normal timing, then twice under gpuload.py; obs.py compares.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/obs.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "obs turn start"
for C in n gpu gpu2; do
  case $C in gpu*) load_on;; esac
  env MR_OBS=1 MR_ARM=wait MR_TAG=_obs_$C $PY $R tfs 192 4 > "$OUT/obs_tfs_$C.txt" 2>&1
  log "tfs wait obs $C: $(grep -E '^\[tfs|Error|Traceback' "$OUT/obs_tfs_$C.txt" | tail -1 | cut -c1-200)"
  load_off
done
for C in gpu gpu2; do
  taskpolicy -b $PY $M/b8mr/obs.py "$OUT/obs_tfs4_wait_obs_n.pkl" "$OUT/obs_tfs4_wait_obs_$C.pkl" > "$OUT/obs_cmp_$C.txt" 2>&1
  log "compare n vs $C: $(tr '\n' ' ' < "$OUT/obs_cmp_$C.txt" | cut -c1-900)"
done
log "obs turn finished"
