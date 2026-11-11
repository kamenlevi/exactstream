#!/bin/sh
# One GPU turn (launch holding a quiet lock; unlocks at the end; summary data/b8mr/stress_a7.txt), NewIdeas' board Fri 10-23:
# (1) slot-reuse stress on the CURRENT wait-all multi-row frugal verify (B8 5 %, bt_b12 as is): teacher-forced S=4 logits
#     (128 positions) and MTP decode shas under normal / loader delay (BIGMOE_DELAY_US=300) / GPU load (gpuload.py);
#     positive control: the dead mr arm (phase-B reads after phase A) under GPU load.
# (2) A7: the MTP drafter's share of a pass = the same passes with the recorded drafts replayed and the MTP head off
#     (MR_REPLAY / DH_REPLAY), bracketed by MTP runs: B8 5 % and v3 24 GB (the deployed adaptive config), code and prose.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py; D=$M/defer/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/stress_a7.txt"; }
pick() { grep -E "$2|Error|Traceback" "$1" | tail -1 | cut -c1-330; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }   # a bare `wait` would wait
                                                                 # for the keepalive too (deadlock, 07:03 Fri)
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "stress + A7 turn start"
for C in ${STRESS_CONDS:-n delay gpu}; do
  E=; [ $C = delay ] && E="BIGMOE_DELAY_US=300"
  [ $C = gpu ] && load_on
  env $E MR_ARM=wait MR_TAG=_$C $PY $R tfs 128 4 > "$OUT/st_tfs_wait_$C.txt" 2>&1
  log "tfs wait $C: $(pick "$OUT/st_tfs_wait_$C.txt" '^\[tfs')"
  for P in code prose; do
    env $E MR_ARM=wait MR_TAG=_$C $PY $R gen $P 192 > "$OUT/st_gen_${P}_wait_$C.txt" 2>&1
    log "gen $P wait $C: $(pick "$OUT/st_gen_${P}_wait_$C.txt" '^\[(code|prose)_')"
  done
  if [ $C != delay ]; then                                       # positive control: the dead mr arm
    env MR_ARM=mr MR_TAG=_$C $PY $R tfs 128 4 > "$OUT/st_tfs_mr_$C.txt" 2>&1
    log "tfs mr $C: $(pick "$OUT/st_tfs_mr_$C.txt" '^\[tfs')"
    [ $C = gpu ] && env MR_ARM=mr MR_TAG=_$C $PY $R gen code 192 > "$OUT/st_gen_code_mr_$C.txt" 2>&1 && \
      log "gen code mr $C: $(pick "$OUT/st_gen_code_mr_$C.txt" '^\[code_')"
  fi
  load_off
done
for P in code prose; do                                          # A7 on B8: recorded = the stress run's normal gen
  env MR_ARM=wait MR_TAG=_replay MR_REPLAY="$OUT/gen_${P}_wait_n.json" $PY $R gen $P 192 > "$OUT/a7_${P}_b8_replay.txt" 2>&1
  log "A7 b8 $P replay: $(pick "$OUT/a7_${P}_b8_replay.txt" '^\[(code|prose)_')"
  env MR_ARM=wait MR_TAG=_mtp2 $PY $R gen $P 192 > "$OUT/a7_${P}_b8_mtp2.txt" 2>&1
  log "A7 b8 $P mtp2: $(pick "$OUT/a7_${P}_b8_mtp2.txt" '^\[(code|prose)_')"
done
for P in code prose; do                                          # A7 on v3 24 GB: MTP (recorded), replay, MTP again
  DD=$M/data/defer
  env DH_ARM=v3 DH_TAG=_a7 $PY $D gen $P 160 > "$OUT/a7_${P}_v3_mtp.txt" 2>&1
  log "A7 v3 $P mtp: $(pick "$OUT/a7_${P}_v3_mtp.txt" '^\[(code|prose)_')"
  env DH_ARM=v3 DH_TAG=_replay DH_REPLAY="$DD/gen_${P}_v3_a7.json" $PY $D gen $P 160 > "$OUT/a7_${P}_v3_replay.txt" 2>&1
  log "A7 v3 $P replay: $(pick "$OUT/a7_${P}_v3_replay.txt" '^\[(code|prose)_')"
  env DH_ARM=v3 DH_TAG=_a7b $PY $D gen $P 160 > "$OUT/a7_${P}_v3_mtp2.txt" 2>&1
  log "A7 v3 $P mtp2: $(pick "$OUT/a7_${P}_v3_mtp2.txt" '^\[(code|prose)_')"
done
log "stress + A7 turn finished"
