#!/bin/sh
# Fit12 (Wed 4 Nov 14:4x): the TRL store encode (/Users/kamen/Fit12/scripts/fit_trl_store.py) in short chunks that yield to
# other sessions' quiet windows. CPU only: 4 threads under taskpolicy -b, MLX on the CPU. Before each chunk: STOP file, stop
# time, swap growth (< 2 GB since the runner started), `peer wait` (returns once nothing blocks me), then a CPU share lock
# for the chunk only (released after it). Resumable: the encoder skips finished files.
# Usage: sh trl_chunks.sh LAYERS E0 E1 FORM [CHUNK 4] [STOP_AT HHMM 2145]
#   e.g. sh /Users/kamen/Fit12/scripts/trl_chunks.sh 18 0 511 mix 4 2145
LAYERS=$1; E0=$2; E1=$3; FORM=$4; N=${5:-4}; STOP_AT=${6:-2145}
F=/Users/kamen/Fit12; OUT=$F/data/trl; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt
ELOG=$OUT/encode_$(echo "$LAYERS" | tr ',' '_')_$FORM.log
mkdir -p "$OUT"
swapmb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
SW0=$(swapmb)
echo "$(date +%H:%M:%S) SCR trl encode start: layers $LAYERS experts $E0-$E1 form $FORM chunk $N stop $STOP_AT (pid $$, swap $SW0 MB)" >> $LOG
e=$E0; rc=0
while [ $e -le $E1 ]; do
  [ -e $OUT/STOP ] && { why="STOP file"; break; }
  [ "$(date +%H%M)" -ge "$STOP_AT" ] && { why="stop time $STOP_AT"; break; }
  [ $(( $(swapmb) - SW0 )) -gt 2048 ] && { why="swap grew > 2 GB"; break; }
  peer wait --timeout 30m >/dev/null 2>&1 || continue      # still blocked after 30 min: re-check STOP / time, wait again
  e2=$((e + N - 1)); [ $e2 -gt $E1 ] && e2=$E1
  peer lock share 20m --cpu 400 --gpu 0 "Fit12 TRL encode L$LAYERS e$e-$e2 ($FORM), CPU only, 4 threads, taskpolicy -b, MEM 2 GB" >/dev/null 2>&1
  env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 taskpolicy -b $PY -B $F/scripts/fit_trl_store.py $OUT "$LAYERS" $e-$e2 $FORM >> "$ELOG" 2>&1
  rc=$?
  peer unlock >/dev/null 2>&1
  [ $rc -ne 0 ] && { why="encoder rc $rc"; break; }
  e=$((e2 + 1))
done
[ -z "$why" ] && why="done"
n=$(ls $OUT/L*/e*_k*.safetensors 2>/dev/null | wc -l | tr -d ' ')
echo "$(date +%H:%M:%S) SCR trl encode end ($why): next expert $e, $n matrix files in $OUT, $(du -sh $OUT | cut -f1), swap $(swapmb) MB" >> $LOG
echo "end ($why): next expert $e, $n files"
