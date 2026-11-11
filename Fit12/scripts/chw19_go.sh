#!/bin/sh
# Fit12 chw19_go.sh (Mon 9 Nov 12:4x). NewIdeas 12:47: chw19 is GO, but it starts after m1b (EvalSuite amendment 11: m1b's
# last process starts 16:05 and ends by ~16:17); queue it from ~16:10. At 16:10 this swaps
# /Users/kamen/Fit12/data/scr/chw19/WAIT -> GO. chw19.sh's own gate then waits until m1b's MEM lock is gone, the GPU is idle
# and no quiet ticket waits, takes its share lock and starts (start-by Mon 17:30, else NOT STARTED).
O=/Users/kamen/Fit12/data/scr/chw19
TG=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-09 16:10" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
log "chw19_go armed: WAIT -> GO at $(date -r $TG '+%a %H:%M')"
while [ "$(date +%s)" -lt "$TG" ]; do sleep 30; done
[ -e $O/DONE ] && { log "chw19 already has DONE ($(cat $O/DONE)): nothing to do"; exit 3; }
rm -f $O/WAIT; touch $O/GO
log "WAIT removed, GO set; folder: $(ls $O | tr '\n' ' ')"
echo "$(date +%H:%M:%S) SCR chw19 GO set by chw19_go.sh (NewIdeas 12:47: after m1b)" >> /Users/kamen/Fit12/logs/summary.txt
