#!/bin/sh
# Breakthrough R-41E PFRMIN (NewIdeas Fri 16:54 GO; lines in SCOREBOARD, written before the run): candidate 9c3057ce62 =
# live fd405861fa + B12_PFR_MIN default 512 (PFR on prefills above 512 tokens instead of 4096). One quiet lock
# (scripts/pfrmin_arms.sh): the hash check, then write14 / edit10 with B12_PFR_MIN 4096 vs 512 (the short-prompt spike and
# the prefill cost). PASS: p2k / p8k / write16 hashes = R-48's (p2k takes PFR) -> the fold-exact default on NewIdeas' word
# and b12llff3 (decision 5). KILL: any hash differs. Log: data/pfrmin/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/pfrmin; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_pfrmin start (pid $$)"
caffeinate -i -w $$ &
LOCK_MIN=15 item pfrmin_arms 870 "R-41E PFR_MIN 512: hashes + short-prompt spike A/B (~10 min, 12 GB line)" sh scripts/pfrmin_arms.sh
log "pfrmin: $(grep -hE 'HASH VERDICT|FAILED|failed|did not' "$OUT"/pfrmin_arms.txt | cut -c1-300 | tr '\n' ' ')"
log "runner_pfrmin done"
