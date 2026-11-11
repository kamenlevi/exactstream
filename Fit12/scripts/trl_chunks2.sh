#!/bin/sh
# Fit12 (Wed 4 Nov 14:4x; the orchestrator 14:3x: option A at NORMAL priority, 4 cores, hard stop 21:30, target ~20:00): the
# TRL store encode (/Users/kamen/Fit12/scripts/fit_trl_store.py) LAYER BY LAYER (complete layers first; layer 18 = the
# bit-reference for a Metal encoder), in chunks of N experts that yield to quiet windows. Before each chunk: STOP file, stop
# time, swap growth (< 2 GB since the runner started), a chunk already on disk is skipped without starting Python, then
# `peer wait` (returns once no other session's quiet ticket is queued or held), then a CPU share lock for the chunk only.
# PRIO normal = no taskpolicy (the orchestrator's OK, 14:3x); bg = taskpolicy -b. Resumable (the encoder skips files).
# Usage: sh trl_chunks2.sh LAYERS E0 E1 FORM [CHUNK 4] [STOP_AT HHMM 2130] [PRIO normal|bg]
#   e.g. sh /Users/kamen/Fit12/scripts/trl_chunks2.sh 18,30,7,41 0 511 mix 4 2130 normal
LAYERS=$1; E0=$2; E1=$3; FORM=$4; N=${5:-4}; STOP_AT=${6:-2130}; PRIO=${7:-normal}
F=/Users/kamen/Fit12; OUT=$F/data/trl; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt
ELOG=$OUT/encode_$FORM.log
TP=""; [ "$PRIO" = bg ] && TP="taskpolicy -b"
mkdir -p "$OUT"
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
kfor() { case "$FORM:$1" in mix:down) echo 3;; mix:*) echo 2;; trl2:*) echo 2;; trl3:*) echo 3;; esac; }
chunk_done() {   # chunk_done LAYER E_FROM E_TO: every projection file of every expert exists
  i=$2
  while [ $i -le $3 ]; do
    for p in gate up down; do
      [ -e "$OUT/L$1/$(printf 'e%03d' $i)_${p}_k$(kfor $p).safetensors" ] || return 1
    done
    i=$((i + 1))
  done
  return 0
}
SW0=$(swapmb); why=""
echo "$(date +%H:%M:%S) SCR trl encode start: layers $LAYERS experts $E0-$E1 form $FORM chunk $N stop $STOP_AT prio $PRIO (pid $$, swap $SW0 MB)" >> $LOG
for L in $(echo "$LAYERS" | tr ',' ' '); do
  e=$E0
  while [ $e -le $E1 ]; do
    [ -e $OUT/STOP ] && { why="STOP file"; break 2; }
    [ "$(date +%H%M)" -ge "$STOP_AT" ] && { why="stop time $STOP_AT"; break 2; }
    [ $(( $(swapmb) - SW0 )) -gt 2048 ] && { why="swap grew > 2 GB"; break 2; }
    e2=$((e + N - 1)); [ $e2 -gt $E1 ] && e2=$E1
    chunk_done $L $e $e2 && { e=$((e2 + 1)); continue; }
    peer wait --timeout 30m >/dev/null 2>&1 || continue      # still blocked after 30 min: re-check STOP / time, wait again
    peer lock share 20m --cpu 400 --gpu 0 "Fit12 TRL encode L$L e$e-$e2 ($FORM), CPU only, 4 threads, $PRIO priority, MEM 2 GB (orchestrator 14:3x)" >/dev/null 2>&1
    env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 $TP $PY -B $F/scripts/fit_trl_store.py $OUT $L $e-$e2 $FORM >> "$ELOG" 2>&1
    rc=$?
    peer unlock >/dev/null 2>&1
    [ $rc -ne 0 ] && { why="encoder rc $rc at L$L e$e"; break 2; }
    e=$((e2 + 1))
  done
  echo "$(date +%H:%M:%S) SCR trl encode: layer $L complete ($(ls $OUT/L$L/e*_k*.safetensors 2>/dev/null | wc -l | tr -d ' ') files)" >> $LOG
done
[ -z "$why" ] && why="done"
n=$(ls $OUT/L*/e*_k*.safetensors 2>/dev/null | wc -l | tr -d ' ')
echo "$(date +%H:%M:%S) SCR trl encode end ($why): $n matrix files in $OUT ($(du -sh $OUT | cut -f1)), swap $(swapmb) MB" >> $LOG
echo "end ($why): $n files"
