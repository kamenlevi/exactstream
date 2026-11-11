#!/bin/sh
# Stall anatomy, part 1 (launch holding a quiet lock; unlocks at the end): (1) all-hits compute baselines per verify width S
# in BigMoE's exact 24 GB env (fixed-state repeat: true all hits); (2) real streamed MTP decode on code / prose / edit.
OUT=$HOME/MoonShots/data/stall; PY=$HOME/BigMoE/.venv/bin/python; M=$HOME/MoonShots; mkdir -p "$OUT"
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
for S in 2 4 6 8; do
  M1B_ENV=exact M1B_S=$S M1B_REPEAT=1 M1B_FIXED=1 $PY $M/engine/test_m1b_speed.py ref 0.25 48 48 > "$OUT/base_exact_S$S.txt" 2>&1
  echo "$(date +%H:%M) base S=$S: $(grep -E 'verify pass|Traceback|[A-Za-z]*Error' "$OUT/base_exact_S$S.txt" | cut -c1-160 | tr '\n' ' ')" >> "$OUT/summary.txt"
done
for P in code prose edit; do
  $PY $M/stall/anatomy.py $P 160 0.25 > "$OUT/anatomy_${P}_exact.txt" 2>&1
  echo "$(date +%H:%M) anatomy $P: $(grep -E '^\[|Traceback|[A-Za-z]*Error' "$OUT/anatomy_${P}_exact.txt" | cut -c1-220 | tr '\n' ' ')" >> "$OUT/summary.txt"
done
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) part 1 finished" >> "$OUT/summary.txt"
