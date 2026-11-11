#!/bin/sh
# Fit12 s1u_go.sh (Mon 9 Nov 21:1x). NewIdeas 21:16: GO for S1u, plus one start condition and one marker. s1u.sh (task
# be69wq8hc) keeps running unchanged and keeps all its own conditions (>= 23:00, Kamen idle, NO-SWAP >= 20, no quiet ticket
# for 5 min, no MEM > 5 GB lock, idle GPU, start-by Tue 00:58, stop 01:15).
# 1. Start condition: swap /Users/kamen/Fit12/data/scr/s1u/WAIT -> GO only once Leap's /Users/kamen/Leap/data/drafter_1109.done
#    exists with an mtime after this helper's start (Leap touches it when its 23:00 drafter test finishes or is skipped).
# 2. Marker: when s1u.sh ends or gives up at its start-by (it writes $O/DONE either way), touch
#    /Users/kamen/Fit12/logs/s1u.done; Breakthrough's q1u_0_live map arm starts after it. Fallbacks: the runner gone without a
#    DONE, or Tue 01:20 passed -> touch it anyway, so the map arm is never blocked.
O=/Users/kamen/Fit12/data/scr/s1u; LD=/Users/kamen/Leap/data/drafter_1109.done; MK=/Users/kamen/Fit12/logs/s1u.done
T0=$(date +%s); TF=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-10 01:20" +%s) || exit 9
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
runner() { pgrep -f 'sh /Users/kamen/Fit12/scripts/s1u.sh' >/dev/null; }
log "s1u_go armed: WAIT -> GO once $LD is touched after $(date -r $T0 '+%H:%M:%S'); then $MK when $O/DONE appears"
while [ ! -e $O/DONE ]; do
  if [ -e $O/WAIT ] && [ -f "$LD" ] && [ "$(stat -f %m "$LD")" -ge "$T0" ]; then
    rm -f $O/WAIT; touch $O/GO
    log "Leap's done file seen (mtime $(date -r "$(stat -f %m "$LD")" '+%H:%M:%S')): WAIT removed, GO set"
    echo "$(date +%H:%M:%S) SCR s1u GO set by s1u_go.sh (Leap's drafter_1109.done; NewIdeas 21:16)" >> /Users/kamen/Fit12/logs/summary.txt
  fi
  runner || { log "s1u.sh is gone without a DONE"; break; }
  [ "$(date +%s)" -ge "$TF" ] && { log "past Tue 01:20 without a DONE"; break; }
  sleep 20
done
touch $MK
log "marker $MK touched (DONE: $(cat $O/DONE 2>/dev/null || echo none))"
echo "$(date +%H:%M:%S) SCR s1u marker s1u.done touched (DONE: $(cat $O/DONE 2>/dev/null || echo none))" >> /Users/kamen/Fit12/logs/summary.txt
