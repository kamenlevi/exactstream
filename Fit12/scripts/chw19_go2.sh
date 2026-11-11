#!/bin/sh
# Fit12 chw19_go2.sh (Mon 9 Nov 12:5x). Replaces chw19_go.sh (stopped by task id; it had not swapped anything).
# NewIdeas 12:51: chw19 now goes after Leap's full store conversion, not straight after m1b. The conversion is CPU + SSD
# heavy, runs from >= ~16:17 and has a hard stop at 16:58.
# This swaps /Users/kamen/Fit12/data/scr/chw19/WAIT -> GO when either:
#  - it is 16:10 or later AND Leap's done file exists with an mtime after Mon 16:00; or
#  - it is 17:00 (fallback; NewIdeas: go anyway, since Leap's hard stop is 16:58).
# The done file is /Users/kamen/Leap/data/release/convert_full.done, unless /Users/kamen/Fit12/data/scr/chw19/LEAP_DONE_PATH
# names another one. That file is re-read every 30 s, so a path Leap posts in the lounge needs no script edit.
# chw19.sh's own gate then runs as before: no MEM > 5 GB lock, no quiet ticket for 5 min, an idle GPU, 20 GB free; then a
# share lock; start-by 17:30.
O=/Users/kamen/Fit12/data/scr/chw19
ep() { date -j -f "%Y-%m-%d %H:%M" "$1" +%s; }
T10=$(ep "2026-11-09 16:10") || exit 9; T16=$(ep "2026-11-09 16:00") || exit 9; T17=$(ep "2026-11-09 17:00") || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
dpath() { p=$(head -1 $O/LEAP_DONE_PATH 2>/dev/null); echo "${p:-/Users/kamen/Leap/data/release/convert_full.done}"; }
leap_done() { f=$(dpath); [ -f "$f" ] && [ "$(stat -f %m "$f")" -gt "$T16" ]; }
log "chw19_go2 armed: WAIT -> GO at >= 16:10 once $(dpath) exists with mtime > 16:00, or at 17:00 regardless"
why=""
while :; do
  [ -e $O/DONE ] && { log "chw19 already has DONE ($(cat $O/DONE)): nothing to do"; exit 3; }
  now=$(date +%s)
  if [ "$now" -ge "$T10" ] && leap_done; then why="Leap's done file $(dpath) (mtime $(date -r "$(stat -f %m "$(dpath)")" '+%H:%M:%S'))"; break; fi
  if [ "$now" -ge "$T17" ]; then why="17:00 fallback, no done file at $(dpath)"; break; fi
  sleep 30
done
rm -f $O/WAIT; touch $O/GO
log "WAIT removed, GO set: $why; folder: $(ls $O | tr '\n' ' ')"
echo "$(date +%H:%M:%S) SCR chw19 GO set by chw19_go2.sh: $why" >> /Users/kamen/Fit12/logs/summary.txt
