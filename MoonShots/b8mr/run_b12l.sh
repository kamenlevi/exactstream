#!/bin/sh
# NewIdeas 10:5x Fri: B8 diverges under GPU load (~1 per 40 passes; v3 exact stable). Does B12L (B12 + LATE3=2,4, mass, MTP;
# = run.py MR_FRAC=0.10, wait arm) too? tfs 128 4 logits + MTP decode (code, prose) with residency snapshots, normal vs GPU load
# x2. If it diverges at a similar rate: the two checksum turns; if stable: park the question. Unlocks at the end.
M=$HOME/MoonShots; OUT=$M/data/b8mr; PY=$HOME/BigMoE/.venv/bin/python; R=$M/b8mr/run.py
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/b12l.txt"; }
GL=
load_on() { $PY $M/b8mr/gpuload.py 900 0.5 > /dev/null 2>&1 & GL=$!; }
load_off() { if [ -n "$GL" ]; then kill $GL 2>/dev/null; wait $GL 2>/dev/null; fi; GL=; }
trap 'load_off; peer unlock > /dev/null 2>&1' EXIT
log "B12L stress start"
for C in n gpu gpu2; do
  case $C in gpu*) load_on;; esac
  env MR_FRAC=0.10 MR_ARM=wait MR_TAG=_b12l_$C $PY $R tfs 128 4 > "$OUT/b12l_tfs_$C.txt" 2>&1
  log "tfs $C: $(grep -E '^\[tfs|Error|Traceback' "$OUT/b12l_tfs_$C.txt" | tail -1 | cut -c1-200)"
  for P in code prose; do
    env MR_FRAC=0.10 MR_RESID=1 MR_ARM=wait MR_TAG=_b12l_$C $PY $R gen $P 192 > "$OUT/b12l_gen_${P}_$C.txt" 2>&1
    log "gen $P $C: $(grep -E '^\[(code|prose)_|Error|Traceback' "$OUT/b12l_gen_${P}_$C.txt" | tail -1 | cut -c1-260)"
  done
  load_off
done
for P in code prose; do
  for C in gpu gpu2; do
    log "resid $P n vs $C: $(taskpolicy -b $PY $M/b8mr/resid_cmp.py "$OUT/resid_${P}_wait_f0.1_b12l_n.npz" "$OUT/resid_${P}_wait_f0.1_b12l_$C.npz" \
      "$OUT/gen_${P}_wait_f0.1_b12l_n.json" "$OUT/gen_${P}_wait_f0.1_b12l_$C.json" 2>&1 | grep -v '^  layer' | tr '\n' ' ' | cut -c1-500)"
  done
done
DIV=$(cat "$OUT"/b12l_tfs_*.txt "$OUT"/b12l_gen_*.txt 2>/dev/null | grep -oE "(logits )?sha [0-9a-f]{10}" | sort -u | wc -l)
grep -q "first differing snapshot" "$OUT/b12l.txt" && DIV=99
log "B12L divergence check: distinct shas $DIV (4 if stable: tfs logits + top-1, code, prose), resid diffs $(grep -c 'first differing snapshot' "$OUT/b12l.txt")"
if [ "$DIV" -gt 4 ]; then                                        # B12L diverges: Breakthrough's shortcut, B8 with bf16 mixers
  for C in n gpu gpu2; do                                        # (B12_HC=0: MTPLX's fused HC kernels, as v3 exact) under load
    case $C in gpu*) load_on;; esac
    env B12_HC=0 MR_ARM=wait MR_TAG=_hcbf_$C $PY $R tfs 128 4 > "$OUT/hcbf_tfs_$C.txt" 2>&1
    log "B8 bf16-HC tfs $C: $(grep -E '^\[tfs|Error|Traceback' "$OUT/hcbf_tfs_$C.txt" | tail -1 | cut -c1-200)"
    for P in code prose; do
      env B12_HC=0 MR_RESID=1 MR_ARM=wait MR_TAG=_hcbf_$C $PY $R gen $P 192 > "$OUT/hcbf_gen_${P}_$C.txt" 2>&1
      log "B8 bf16-HC gen $P $C: $(grep -E '^\[(code|prose)_|Error|Traceback' "$OUT/hcbf_gen_${P}_$C.txt" | tail -1 | cut -c1-260)"
    done
    load_off
  done
fi
log "B12L stress finished"
