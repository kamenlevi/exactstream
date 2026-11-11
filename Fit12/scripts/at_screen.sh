#!/bin/sh
# Fit12 at_screen (Tue 3 Nov 19:2x; at_t3r.sh's logic with the screen and times as arguments): start one queued screen at a set
# time from a background process of this session (Bash run_in_background: attached, because a detached runner's quiet lock
# lapses after 8 min), so it fires with nobody awake. Sleeps until GO; then, if the session lives, no other Fit12 runner runs
# and it is not past LATE, queues SCREEN + 'stop' and runs runq.sh as its child. Backstops: at STOP with MARKER absent (the
# screen's first timed process never started) the slot is over: stop the runner tree + `peer unlock` (it also cancels a queued
# ticket); at BACK with the runner alive: the same. Afterwards the SCREEN / 'stop' lines it queued are removed. Heat pauses
# (fanwatch) freeze this process like any research job and thaw it: a start slips by a pause's length.
# Log: /Users/kamen/Fit12/logs/summary.txt ("SCR at_<SCREEN>").
# usage: cd /Users/kamen/Fit12 && CLAUDE_PID=<claude pid> sh /Users/kamen/Fit12/scripts/at_screen.sh SCREEN GO LATE STOP BACK MARKER
#   times as YYYYmmddHHMMSS; MARKER = the screen's first output file (e.g. /Users/kamen/Fit12/data/scr/t3rw/r47.txt)
# F12_AT_LOG / F12_AT_Q / F12_AT_RUNNER / F12_AT_UNLOCK: overrides for the dry tests (/Users/kamen/Fit12/scripts/at_screen_test.sh).
F=/Users/kamen/Fit12
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
: "${CLAUDE_PID:?CLAUDE_PID}"
[ $# -eq 6 ] || { echo "usage: at_screen.sh SCREEN GO LATE STOP BACK MARKER" >&2; exit 64; }
S=$1 GO=$2 LATE=$3 STOP=$4 BACK=$5 MK=$6
LOG=${F12_AT_LOG:-$F/logs/summary.txt}; Q=${F12_AT_Q:-$F/data/queue.txt}; RUNNER=${F12_AT_RUNNER:-$F/scripts/runq.sh}
UNLOCK=${F12_AT_UNLOCK:-peer unlock}
log() { echo "$(date +%H:%M:%S) SCR at_$S $*" >> "$LOG"; }
now() { date +%Y%m%d%H%M%S; }
kill_tree() { for c in $(pgrep -P "$1"); do kill_tree "$c" "$2"; done; kill -"$2" "$1" 2>/dev/null; }
halt() { log "$1: stopping the runner (pid $R) and cancelling any lock or ticket"
  kill_tree "$R" TERM; sleep 3; kill_tree "$R" KILL; $UNLOCK >/dev/null 2>&1; }

log "armed (pid $$, session $CLAUDE_PID): queues $S + runs the queue runner at $GO (late $LATE, stop $STOP, back $BACK)"
while [ "$(now)" -lt "$GO" ]; do
  kill -0 "$CLAUDE_PID" 2>/dev/null || { log "session $CLAUDE_PID gone: nothing started"; exit 1; }
  sleep 30
done
[ "$(now)" -gt "$LATE" ] && { log "woke at $(date +%H:%M:%S), after $LATE: $S not started (slot lost)"; exit 2; }
pgrep -f 'Fit12/scripts/(runq|screens)\.sh' >/dev/null && { log "another Fit12 runner runs: $S not started"; exit 3; }
printf '%s\nstop\n' "$S" > "$Q"
cd "$F" || exit 1
sh "$RUNNER" &
R=$!
log "queue: $S, stop; queue runner started (pid $R)"
st=0
while kill -0 "$R" 2>/dev/null; do
  t=$(now)
  if [ $st -eq 0 ] && [ "$t" -ge "$STOP" ]; then
    st=1; [ -e "$MK" ] || { halt "$(date +%H:%M:%S), $S's first process never started (slot over)"; break; }
  fi
  [ "$t" -ge "$BACK" ] && { halt "$(date +%H:%M:%S), the runner still runs at the hard stop"; break; }
  sleep 5
done
wait "$R" 2>/dev/null; rc=$?
sed -i '' -e "/^$S\$/d" -e '/^stop$/d' "$Q" 2>/dev/null
log "end (runner rc $rc; $(basename "$MK") $([ -e "$MK" ] && echo written || echo 'not started'))"
exit $rc
