#!/bin/sh
# Fit12 (Wed 4 Nov 21:2x; the orchestrator 21:18 hands L41's remaining experts to Leap's batched Metal encoder): re-encode
# a few L41 experts with my own C path (/Users/kamen/Fit12/scripts/fit_trl_store.py, the store runner's exact invocation,
# normal priority) into /Users/kamen/Fit12/data/trl_spot, as the reference that /Users/kamen/Fit12/scripts/trl_import.py
# compares Leap's files with byte for byte. Same inputs as the store: b10rcap R0..R7 (the encoder's default) and a
# byte-identical copy of the store's L41/_hx.npz. CPU only, 4 threads, MEM 2 GB: peer wait + a share lock for the run.
# Refuses while any fit_trl_store.py runs (my runner must have stopped). Usage: sh trl_spot.sh 300 511
F=/Users/kamen/Fit12; S=$F/data/trl_spot; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt
[ $# -ge 1 ] || { echo "usage: trl_spot.sh EXPERT [EXPERT ...]"; exit 2; }
mkdir -p $S/L41
[ -e $S/L41/_hx.npz ] || cp $F/data/trl/L41/_hx.npz $S/L41/_hx.npz
cmp -s $S/L41/_hx.npz $F/data/trl/L41/_hx.npz || { echo "spot: the _hx.npz copy differs from the store's"; exit 2; }
pgrep -f 'fit_trl_stor[e].py' >/dev/null && { echo "spot: an encoder is running; not started"; exit 2; }
peer wait --timeout 15m >/dev/null 2>&1 || { echo "spot: still blocked by a quiet ticket after 15 min; not started"; exit 3; }
peer lock share 10m --cpu 400 --gpu 0 "Fit12 TRL spot re-encode L41 e$* (C path, reference for Leap's Metal files), CPU only, 4 threads, MEM 2 GB" >/dev/null 2>&1
for e in "$@"; do
  env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=4 $PY -B $F/scripts/fit_trl_store.py $S 41 $e-$e mix >> $S/spot.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && { peer unlock >/dev/null 2>&1; echo "spot: encoder rc $rc at e$e"; exit 4; }
done
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR trl spot: L41 experts $* re-encoded by the C path into $S (reference for the Leap import)" >> $LOG
grep -E '^\[trl\] L41 e' $S/spot.log | tail -$#
ls -l $S/L41/ | awk 'NR>1{print $5, $9}'
