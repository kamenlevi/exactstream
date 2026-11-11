#!/bin/sh
# Breakthrough R-49 POM12 (NewIdeas Thu 18:3x; lines in SCOREBOARD da2a773, written first): the 12 GB line's own decode
# routing for the CPU price screen. One 10-min quiet lock: scripts/r49_capture_hook.sh -> r49_capture.py (live bt_b12 at
# its defaults; p8k up to 512 tokens, agent bash17 / edit00 / text23 / write16, math prealgebra/1640; each prompt only if
# it fits before the lock's end - 40 s; ~5 min). Then pom12.py on the traces (CPU, after the lock).
# Log: data/r49/summary.txt
cd "$HOME/Breakthrough" || exit 1
OUT=$HOME/Breakthrough/data/r49; mkdir -p "$OUT"; LOG=$OUT/summary.txt
. scripts/lockrun.sh
log "runner_r49 start (pid $$)"
caffeinate -i -w $$ &
LOCK_MIN=10 item r49_capture 570 "R-49 12 GB routing capture (~6 min)" scripts/r49_capture_hook.sh
log "r49_capture: $(grep -E 'RESULT|skipped' "$OUT/r49_capture.txt" | cut -c1-160 | tr '\n' ' ')"
P12_TRACES=r49 taskpolicy -b $PY scripts/pom12.py > "$OUT/pom12_r49.out" 2>&1
log "pom12 r49 (rc $?): $(grep -E 'R2 (10|6|5|4)' "$OUT/pom12_r49.out" | head -8 | cut -c1-110 | tr '\n' ' ')"
log "runner_r49 done"
