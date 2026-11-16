#!/bin/sh
# Fit12: run runaway4.py (desk, ~2 s CPU) only when no other session holds a quiet (benchmarking) lock; give up after 80 min.
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; T0=$(date +%s)
while peer ls 2>/dev/null | grep -q "· benchmarking"; do
  [ $(( $(date +%s) - T0 )) -ge 4800 ] && { echo "$(date +%H:%M:%S) gave up: a quiet lock held for 80 min"; exit 3; }
  sleep 30
done
echo "$(date +%H:%M:%S) no quiet lock held: running runaway4.py"
cd $F && OMP_NUM_THREADS=2 nice -n 10 $PY -I scripts/runaway4.py > data/lossy32/runaway4_run.txt 2>&1; rc=$?
echo "$(date +%H:%M:%S) rc $rc"; tail -45 data/lossy32/runaway4_run.txt
exit $rc
