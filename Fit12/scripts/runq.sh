#!/bin/sh
# Fit12 queue runner (Tue 27 Oct 2026): runs the screens named in /Users/kamen/Fit12/data/queue.txt one at a time, first line
# first, with screens.sh's functions sourced afresh before each, so screens can be queued or edited while it waits (not while
# one runs: its functions are parsed then). Queuing needs no new process (the peers hook blocks launches during others'
# locks). A line 'stop' ends it; so does 23:30 while the queue is empty (nothing of Fit12's waits through the night).
# Launch: cd /Users/kamen/Fit12 && CLAUDE_PID=23445 nohup sh /Users/kamen/Fit12/scripts/runq.sh >/dev/null 2>&1 &
F=$HOME/Fit12; Q=$F/data/queue.txt; SCREENS_LIB=1
. $F/scripts/screens.sh
nohup sh $F/scripts/awake.sh $$ >/dev/null 2>&1 &
log "queue runner start (pid $$): screens from $Q"
while :; do
  s=$(head -1 "$Q" 2>/dev/null | tr -d ' \t')
  if [ -z "$s" ]; then
    [ "$(date +%H%M)" -ge 2330 ] && { log "queue runner: empty at 23:30, ending"; break; }
    sleep 30; continue
  fi
  sed -i '' 1d "$Q"
  [ "$s" = stop ] && { log "queue runner: stop line"; break; }
  . $F/scripts/screens.sh
  log "queue: screen $s"
  scr_$s || log "screen $s stopped early"
  log "queue: screen $s finished"
done
log "queue runner end"
