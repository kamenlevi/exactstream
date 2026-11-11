#!/bin/sh
# Breakthrough R-45 v3o (NewIdeas Thu 19:52 GO, QUEUE R-45; Fit12's R-45T PASS named the owner: read_expert's
# thread_local bounce, leaked per GCD-retired worker; lines in SCOREBOARD, written before the run). One 15-min quiet
# lock: build v3o's ext, the hash check of the v3o candidate fd405861fa, then the creep arms v3o and v3n
# (scripts/v3o_arms.sh). PASS: every hash = v3n on p2k / p8k / write16 AND after-gc growth <= 3 MB per request over 6
# requests with >= 10 s idle gaps; KILL >= 15 MB. Log: data/v3o/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/v3o; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_v3o start (pid $$)"
caffeinate -i -w $$ &
LOCK_MIN=15 item v3o_arms 870 "R-45 v3o: build + hashes + creep v3o / v3n (~13 min)" sh scripts/v3o_arms.sh
log "v3o: $(grep -hE 'HASH VERDICT|SUMMARY|skipped|FAILED|failed' "$OUT"/v3o_arms.txt | cut -c1-500 | tr '\n' ' ')"
log "runner_v3o done"
