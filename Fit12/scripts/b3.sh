#!/bin/sh
# Fit12 B3 (2026-10-25): CPU only (no GPU) under a share lock: build the trellis encoder copy, sanity test, B3 analysis.
# Pre-registered in PLAN.md (B3). Launch (background task): cd ~/Fit12 && sh ~/Fit12/scripts/b3.sh
F=$HOME/Fit12; OUT=$F/data/b3; LOG=$F/logs/summary.txt; PY=$HOME/BigMoE/.venv/bin/python
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 B3_THREADS=6
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
n=0
until peer lock share 40m --cpu 600 --gpu 0 "Fit12 B3: 2-bit trellis + LDLQ output error on captured MoE inputs (CPU only, ~10-15 min)" > "$OUT/lock.txt" 2>&1; do
  n=$((n + 1)); [ $n -gt 40 ] && { log "B3: no share lock after 40 waits"; exit 1; }
  peer wait > /dev/null 2>&1 || sleep 60
done
log "B3 share lock taken"
cd $F/scripts && clang -O3 -mcpu=native -shared -fblocks -o fit_trellis.dylib fit_trellis_enc.c > "$OUT/build.txt" 2>&1 || { log "B3: build failed"; peer unlock; exit 1; }
$PY -B $F/scripts/fit_b3.py test > "$OUT/b3_test.txt" 2>&1; log "B3 test rc $?: $(tr '\n' ' ' < $OUT/b3_test.txt | cut -c1-400)"
$PY -B $F/scripts/fit_b3.py > "$OUT/b3.txt" 2>&1; log "B3 rc $?: $(tail -4 $OUT/b3.txt | tr '\n' ' ' | cut -c1-400)"
peer unlock > /dev/null 2>&1
log "B3 done"
