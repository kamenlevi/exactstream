#!/bin/sh
# B8 + MTP multi-row phase B, one GPU turn (launch holding a quiet lock; unlocks at the end; summary data/b8mr/summary.txt):
# (1) bit checks: tf1 wait / mr1 (S=1), tfs4 wait / mr / force / ref, compare.py; (2) quality: S=4 teacher-forced KLD on the
# Eval suite's tf sequences (4 sets, every 8th) for wait and mr; (3) decode A/B: spec125 MTP 3:3:0 on code and prose,
# two rounds in alternating arm order (per-pass verify ms, reads, waited, late; tok/s).
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; mkdir -p "$OUT"
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/summary.txt"; }
log "b8mr turn start (${B8MR_ONLY_GEN:+gen only})"
if [ -z "$B8MR_ONLY_GEN" ]; then
for A in wait mr1; do
  env MR_ARM=$A $PY $M/b8mr/run.py tf1 32 > "$OUT/tf1_$A.txt" 2>&1
  log "tf1 $A: $(grep -E '^\[tf1|^\[b8mr|Error|Traceback' "$OUT/tf1_$A.txt" | tail -2 | cut -c1-220 | tr '\n' ' ')"
done
for A in wait mr force ref; do
  env MR_ARM=$A $PY $M/b8mr/run.py tfs 32 4 > "$OUT/tfs4_$A.txt" 2>&1
  log "tfs4 $A: $(grep -E '^\[tfs|Error|Traceback' "$OUT/tfs4_$A.txt" | tail -2 | cut -c1-220 | tr '\n' ' ')"
done
$PY $M/b8mr/compare.py > "$OUT/compare.txt" 2>&1; ok=$?
log "compare (exit $ok): $(tr '\n' ' ' < "$OUT/compare.txt" | cut -c1-900)"
[ $ok -eq 0 ] || log "WARNING: a bit-identity check failed: the numbers below are UNVERIFIED"
for A in wait mr; do
  env MR_ARM=$A $PY $M/b8mr/run.py kld 8 4 > "$OUT/kld_$A.txt" 2>&1
  log "kld $A: $(grep -E '^\[kld|Error|Traceback' "$OUT/kld_$A.txt" | tail -2 | cut -c1-400 | tr '\n' ' ')"
done
fi
for R in 1 2 3; do                                   # R1, R2: B8 (5 %); R3: B12 + late ranks 2-3 (10 %)
  if [ $R -eq 3 ]; then F=0.10; FT=_f0.1; else F=0.05; FT=; fi
  for P in code prose; do
    if [ $R -eq 2 ]; then ARMS="mr wait"; else ARMS="wait mr"; fi
    for A in $ARMS; do
      env MR_ARM=$A MR_FRAC=$F $PY $M/b8mr/run.py gen $P 192 > "$OUT/gen_${P}_${A}${FT}_r$R.txt" 2>&1
      cp "$OUT/gen_${P}_${A}${FT}.json" "$OUT/gen_${P}_${A}${FT}_r$R.json" 2>/dev/null
      log "gen $P $A f$F r$R: $(grep -E '^\[(code|prose)_|Error|Traceback' "$OUT/gen_${P}_${A}${FT}_r$R.txt" | tail -2 | cut -c1-320 | tr '\n' ' ')"
    done
  done
done
peer unlock > /dev/null 2>&1
log "b8mr turn finished"
