#!/bin/sh
# Fit12 at_t3r (Tue 3 Nov 16:2x; the orchestrator asked to "arm it as a real background process ... so it fires with nobody
# awake"): the durable start of T3R + T3R-PF (PLAN.md 'T3R', 'T3R-PF'), replacing the session wake as the trigger. Started as
# a Bash run_in_background child of this session (attached: a detached runner's quiet lock lapses after 8 min), it sleeps
# until Wed 4 Nov 00:23, then queues 't3r' + 'stop' and runs runq.sh as its child. scr_t3r's own waits do the rest (EvalSuite
# idle 5 min, no other listed run, NO-SWAP >= 16 GB, no lock request before 00:30, gate(): PAUSE / ACT / lid / resume ramp,
# a grant after 00:35:30 released, caps that end each process by 00:43:30, p47 only on its gates). Two backstops here:
# 00:37:00 with no T3R process ever started (r47.txt absent) = the slot is over: stop the runner tree, `peer unlock` (it also
# cancels a queued ticket); 00:43:40 with the runner still alive: the same (the GPU must be clear by 00:44, BigMoE at 00:45).
# Nothing starts if the session is gone, another Fit12 runner runs, or it wakes after 00:30 (sleep / lid; es_wait alone would
# pass 00:35). Afterwards the 't3r' / 'stop' lines it queued are removed. Log: /Users/kamen/Fit12/logs/summary.txt ("SCR at_t3r").
# usage: cd /Users/kamen/Fit12 && CLAUDE_PID=18778 sh /Users/kamen/Fit12/scripts/at_t3r.sh   (Bash run_in_background)
# F12_AT_* overrides are for the dry tests only (scratchpad at_test.sh): times as YYYYmmddHHMMSS, a dummy runner, files.
F=/Users/kamen/Fit12
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
: "${CLAUDE_PID:?CLAUDE_PID}"
LOG=${F12_AT_LOG:-$F/logs/summary.txt}; Q=${F12_AT_Q:-$F/data/queue.txt}; RUNNER=${F12_AT_RUNNER:-$F/scripts/runq.sh}
R47=${F12_AT_R47:-$F/data/scr/t3r/r47.txt}; UNLOCK=${F12_AT_UNLOCK:-peer unlock}
GO=${F12_AT_GO:-20261104002300}; LATE=${F12_AT_LATE:-20261104003000}
STOP=${F12_AT_STOP:-20261104003700}; BACK=${F12_AT_BACK:-20261104004340}
log() { echo "$(date +%H:%M:%S) SCR at_t3r $*" >> "$LOG"; }
now() { date +%Y%m%d%H%M%S; }
kill_tree() { for c in $(pgrep -P "$1"); do kill_tree "$c" "$2"; done; kill -"$2" "$1" 2>/dev/null; }
halt() { log "$1: stopping the runner (pid $R) and cancelling any lock or ticket"
  kill_tree "$R" TERM; sleep 3; kill_tree "$R" KILL; $UNLOCK >/dev/null 2>&1; }

log "armed (pid $$, session $CLAUDE_PID): queues t3r + runs the queue runner at $GO"
while [ "$(now)" -lt "$GO" ]; do
  kill -0 "$CLAUDE_PID" 2>/dev/null || { log "session $CLAUDE_PID gone: nothing started"; exit 1; }
  sleep 30
done
[ "$(now)" -gt "$LATE" ] && { log "woke at $(date +%H:%M:%S), after $LATE: T3R not started (slot lost)"; exit 2; }
pgrep -f 'Fit12/scripts/(runq|screens)\.sh' >/dev/null && { log "another Fit12 runner runs: T3R not started"; exit 3; }
printf 't3r\nstop\n' > "$Q"
cd "$F" || exit 1
sh "$RUNNER" &
R=$!
log "queue: t3r, stop; queue runner started (pid $R)"
st=0
while kill -0 "$R" 2>/dev/null; do
  t=$(now)
  if [ $st -eq 0 ] && [ "$t" -ge "$STOP" ]; then
    st=1; [ -e "$R47" ] || { halt "$(date +%H:%M:%S), no T3R process started (slot over)"; break; }
  fi
  [ "$t" -ge "$BACK" ] && { halt "$(date +%H:%M:%S), the runner still runs (GPU clear by 00:44)"; break; }
  sleep 5
done
wait "$R" 2>/dev/null; rc=$?
sed -i '' -e '/^t3r$/d' -e '/^stop$/d' "$Q" 2>/dev/null
log "end (runner rc $rc; r47 $([ -e "$R47" ] && echo ran || echo 'not started'))"
exit $rc
