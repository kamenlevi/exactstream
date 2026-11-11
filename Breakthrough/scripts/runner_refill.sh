#!/bin/sh
# Breakthrough R-41F REFILL (NewIdeas Fri 17:5x GO; lines in SCOREBOARD, written before the run): candidate b9d5b491ad =
# 9c3057ce62 (PFR_MIN 512) + B12_PFR_REFILL (default 1): after PFR the re-created pools take each layer's most-used
# experts of the prompt's last MoE chunk (ext.ensure; exact). One quiet lock (scripts/refill_arms.sh): hashes, then
# arms A (today: PFR_MIN 4096, no refill) / B (candidate) / A2 on write14, edit10, text23. PASS: hashes = R-48's AND
# short rows within +3 % of A AND text23 decode +5 % or better. KILL: neither. Log: data/refill/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/refill; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_refill start (pid $$)"
caffeinate -i -w $$ &
LOCK_MIN=15 item refill_arms 870 "R-41F PFR refill: hashes + A / B / A2 on 3 agent rows (~13 min, 12 GB line)" sh scripts/refill_arms.sh
log "refill: $(grep -hE 'HASH VERDICT|^VERDICT|FAILED|failed|skipped|did not|not every' "$OUT"/refill_arms.txt | cut -c1-300 | tr '\n' ' ')"
log "runner_refill done"
