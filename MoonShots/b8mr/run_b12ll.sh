#!/bin/sh
# NewIdeas ~19:00 Fri (12 GB candidate now B12LL, natural top-10 under MTP): B8 diverges under GPU load (~1 per 40 passes; v3 exact stable). Does B12LL (B12 + LATE3=2,10, mass, MTP;
# = run.py MR_FRAC=0.10, wait arm) too? tfs 128 4 logits + MTP decode (code, prose) with residency snapshots, normal vs GPU load
# x2. If it diverges at a similar rate: the two checksum turns; if stable: park the question. Unlocks at the end.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/b12ll.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "B12LL stress start"
for C in n gpu gpu2; do
  case $C in gpu*) load_on;; esac
  env MR_FRAC=0.10 B12_LATE3=2,10 MR_ARM=wait MR_TAG=_b12ll_$C $PY $R tfs 128 4 > "$OUT/b12ll_tfs_$C.txt" 2>&1
  log "tfs $C: $(grep -E '^\[tfs|Error|Traceback' "$OUT/b12ll_tfs_$C.txt" | tail -1 | cut -c1-200)"
  for P in code prose; do
    env MR_FRAC=0.10 B12_LATE3=2,10 MR_RESID=1 MR_ARM=wait MR_TAG=_b12ll_$C $PY $R gen $P 192 > "$OUT/b12ll_gen_${P}_$C.txt" 2>&1
    log "gen $P $C: $(grep -E '^\[(code|prose)_|Error|Traceback' "$OUT/b12ll_gen_${P}_$C.txt" | tail -1 | cut -c1-260)"
  done
  load_off
done
for P in code prose; do
  for C in gpu gpu2; do
    log "resid $P n vs $C: $(taskpolicy -b $PY $M/b8mr/resid_cmp.py "$OUT/resid_${P}_wait_f0.1_b12ll_n.npz" "$OUT/resid_${P}_wait_f0.1_b12ll_$C.npz" \
      "$OUT/gen_${P}_wait_f0.1_b12ll_n.json" "$OUT/gen_${P}_wait_f0.1_b12ll_$C.json" 2>&1 | grep -v '^  layer' | tr '\n' ' ' | cut -c1-500)"
  done
done
log "B12LL stress finished"
env B12_HC=0 MR_ARM=wait MR_TAG=_v3dense $PY $R denseinv > "$OUT/denseinv.txt" 2>&1   # line 1: dense batch invariance
log "denseinv: $(grep -cE '^\[denseinv' "$OUT/denseinv.txt") shapes; $(grep -E 'Error|Traceback' "$OUT/denseinv.txt" | tail -1)"
$PY $M/wide/qmv_mr.py > "$OUT/qmv_mr.txt" 2>&1                     # L1-01 prototype: bits vs MLX + us per call
log "qmv_mr: $(tr '\n' ' ' < "$OUT/qmv_mr.txt" | cut -c1-1500)"
log "turn finished"
