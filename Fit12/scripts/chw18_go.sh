#!/bin/sh
# Fit12 chw18 GO / WAIT watcher (NewIdeas 16:12): chw18's gap is right after BigMoE's 32 GB chat arm, and chw18 must END by
# 17:07 (EvalSuite's card reference resumes). It reads /Users/kamen/BigMoE/data/turn45.log (read-only):
# - chw18 is held (WAIT) until a NEW 'done rc' line appears there; then WAIT -> GO (chw18's gate needs 1 clear minute).
# - Not started by 16:55 -> WAIT again and exit (NewIdeas places it in the evening's next gap).
# - Exception: BigMoE's PFB queued / granted after arming and not yet done (Chrome closed, PFB first) -> GO stays, no 16:55
#   cut: chw18 goes right after PFB (its gate waits for the quiet lock to end).
O=/Users/kamen/Fit12/data/scr/chw18; TL=/Users/kamen/BigMoE/data/turn45.log
CUT=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 16:55" +%s); END=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-08 22:15" +%s)
nd() { grep -c 'done rc' $TL; }
npq() { grep -cE '\[BigMoE PFB .*\] (queue the quiet ticket|granted)' $TL; }
npd() { grep -cE '\[BigMoE PFB .*\] done rc' $TL; }
log() { echo "[$(date '+%a %H:%M:%S')] $*"; }
N0=$(nd); P0=$(npq); D0=$(npd)
touch $O/WAIT; rm -f $O/GO
log "watcher armed: WAIT set; baseline $N0 'done rc' lines, PFB queued $P0 / done $D0; cut 16:55"
go=0; pfbseen=0
while [ "$(date +%s)" -lt "$END" ]; do
  [ -e $O/MANIFEST.txt ] && { log "watcher: chw18 started"; rm -f $O/GO $O/WAIT; exit 0; }
  [ -e $O/DONE ] && { log "watcher: chw18 DONE without a start: $(cat $O/DONE)"; rm -f $O/GO $O/WAIT; exit 3; }
  pfb=0; [ "$(npq)" -gt "$P0" ] && [ "$(npd)" -eq "$D0" ] && pfb=1
  [ $pfb -eq 1 ] && pfbseen=1                     # PFB went first: no 16:55 cut from here on (go right after PFB)
  if [ $go -eq 0 ] && { [ "$(nd)" -gt "$N0" ] || [ $pfb -eq 1 ]; }; then
    touch $O/GO; rm -f $O/WAIT; go=1
    log "watcher: WAIT -> GO (pfb $pfb; last 'done rc': $(grep 'done rc' $TL | tail -1 | cut -c1-110))"
  fi
  if [ "$(date +%s)" -ge "$CUT" ] && [ $pfbseen -eq 0 ]; then
    touch $O/WAIT; rm -f $O/GO
    log "watcher: chw18 not started by 16:55: WAIT set (NewIdeas places it)"; exit 4
  fi
  sleep 15
done
log "watcher: 22:15 reached"; exit 5
