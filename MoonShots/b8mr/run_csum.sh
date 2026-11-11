#!/bin/sh
# B12LL under GPU load: per-op checksums (MR_CSUM, b8mr/csum.py) in MTP decode, quiet vs 3 runs under gpuload.py, code and
# prose; csum.py names the first op whose output differs. Launch holding a quiet lock; unlocks at the end. Log csum.txt.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/csum.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "csum turn start"
for C in n gpu gpu2 gpu3; do
  case $C in gpu*) load_on;; esac
  for P in code prose; do
    env MR_FRAC=0.10 B12_LATE3=2,10 MR_CSUM=1 MR_HANDOFF=1 MR_ARM=wait MR_TAG=_cs_$C $PY $R gen $P 192 > "$OUT/cs_gen_${P}_$C.txt" 2>&1
    log "gen $P $C: $(grep -E "^\[(code|prose)_|^\[handoff|Error|Traceback" "$OUT/cs_gen_${P}_$C.txt" | tail -2 | tr "\n" " " | cut -c1-900)"
  done
  load_off
done
for P in code prose; do
  for C in gpu gpu2 gpu3; do
    log "csum $P n vs $C: $(taskpolicy -b $PY $M/b8mr/csum.py "$OUT/csum_${P}_wait_f0.1_cs_n.json" "$OUT/csum_${P}_wait_f0.1_cs_$C.json" 2>&1 | tr '\n' ' ' | cut -c1-1400)"
  done
done
log "csum turn finished"
