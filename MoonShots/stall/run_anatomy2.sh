#!/bin/sh
# Stall anatomy v2 (launch holding a quiet lock; unlocks at the end). BigMoE b13 env (+ BIGMOE_GDN_EXACT=1):
# (1) all-hits compute baselines per S: exact 24 GB (S 2/4/6/8), v3 tier 24 GB (S 2/4/6/8), tier 10 % (S 2/3/4, B12 proxy);
# (2) real streamed MTP decode anatomy: exact and v3 on code / prose / edit at 24 GB; (3) B12 + MTP 3:3:0 at 10 % on code / prose.
OUT=$HOME/MoonShots/data/stall; PY=$HOME/BigMoE/.venv/bin/python; M=$HOME/MoonShots; mkdir -p "$OUT"
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary2.txt"; }
base() { tag=$1; env=$2; frac=$3; shift 3
  for S in "$@"; do
    env $env M1B_S=$S M1B_REPEAT=1 M1B_FIXED=1 $PY $M/engine/test_m1b_speed.py ref $frac 48 48 > "$OUT/base_${tag}_S$S.txt" 2>&1
    log "base $tag S=$S: $(grep -oE 'verify pass median [0-9.]+ ms|Traceback|[A-Za-z]*Error' "$OUT/base_${tag}_S$S.txt" | head -2 | tr '\n' ' ')"
  done; }
base exact "M1B_ENV=exact" 0.25 2 4 6 8
base tier "M1B_ENV=tier" 0.25 2 4 6 8
base tier10 "M1B_ENV=tier" 0.10 2 3 4
for P in code prose edit; do
  $PY $M/stall/anatomy.py $P 160 0.25 > "$OUT/anatomy_${P}_exact.txt" 2>&1
  log "anatomy exact $P: $(grep -E '^\[[a-z]+_exact|Traceback|[A-Za-z]*Error' "$OUT/anatomy_${P}_exact.txt" | cut -c1-200 | tr '\n' ' ')"
  STALL_TIER=1 $PY $M/stall/anatomy.py $P 160 0.25 > "$OUT/anatomy_${P}_tier.txt" 2>&1
  log "anatomy v3 $P: $(grep -E '^\[[a-z]+_tier|Traceback|[A-Za-z]*Error' "$OUT/anatomy_${P}_tier.txt" | cut -c1-200 | tr '\n' ' ')"
done
for P in code prose; do
  $PY $M/stall/anatomy_b12.py $P 160 > "$OUT/anatomy_${P}_b12.txt" 2>&1
  log "anatomy b12 $P: $(grep -E '^\[(code|prose)_b12|Traceback|[A-Za-z]*Error' "$OUT/anatomy_${P}_b12.txt" | cut -c1-200 | tr '\n' ' ')"
done
peer unlock > /dev/null 2>&1
log "v2 finished"
