#!/bin/sh
# Fit12 (Wed 4 Nov 17:1x; the orchestrator 17:1x: "YES. If the store isn't done by 21:30, restart the same command with a
# 22:45 stop. ... Its last chunk must end by 22:45 so BigMoE's 23:00 ticket and RELAX at 23:30 start clean."):
# waits for the running TRL encode runner (pid RUNNER_PID, /Users/kamen/Fit12/scripts/trl_chunks2.sh) to end. Only if it
# ended on its stop time (the last summary line "SCR trl encode end (stop time 2130)") and the store is not whole yet, it
# re-starts the same runner (resumable: finished chunks are skipped without starting Python) with STOP_AT (default 2241: the
# runner checks the clock BEFORE each chunk, and a chunk of 4 experts takes ~41 s, up to ~3 min with a heat freeze, so the
# last chunk ends by 22:45). Any other end (done, STOP file, swap growth, encoder error, no end line) = no restart.
# Run ATTACHED (Bash run_in_background, with CLAUDE_PID set) so the runner's share locks see this session's processes.
# Usage: sh trl_relay.sh RUNNER_PID [STOP_AT HHMM 2241]
#   e.g. CLAUDE_PID=18778 sh /Users/kamen/Fit12/scripts/trl_relay.sh 84909 2241
P=$1; STOP_AT=${2:-2241}; F=/Users/kamen/Fit12; OUT=$F/data/trl; LOG=$F/logs/summary.txt
[ -n "$P" ] || { echo "usage: trl_relay.sh RUNNER_PID [STOP_AT]"; exit 2; }
echo "$(date +%H:%M:%S) SCR trl relay armed: waits for runner $P; re-starts it with stop $STOP_AT only if it ends on its stop time with the store not whole (the orchestrator 17:1x) (pid $$)" >> $LOG
while kill -0 $P 2>/dev/null; do sleep 20; done
end=$(grep 'SCR trl encode end' $LOG | tail -1)
n=$(find $OUT/L18 $OUT/L30 $OUT/L7 $OUT/L41 -name 'e*_k*.safetensors' ! -name '*.tmp.*' 2>/dev/null | wc -l | tr -d ' ')
now=$(date +%H%M)
if ! echo "$end" | grep -q '(stop time 2130)'; then
  why="runner $P did not end on its 21:30 stop time (last end line: ${end:-none})"
elif [ "$n" -ge 6144 ]; then
  why="the store is whole ($n files)"
elif [ "$now" -ge "$STOP_AT" ]; then
  why="it is already $now, past the new stop $STOP_AT"
elif [ -e $OUT/STOP ]; then
  why="STOP file present"
else
  echo "$(date +%H:%M:%S) SCR trl relay: runner $P ended on its stop time with $n of 6144 files; re-starting the same runner with stop $STOP_AT (the orchestrator 17:1x)" >> $LOG
  cd $F && exec sh $F/scripts/trl_chunks2.sh 18,30,7,41 0 511 mix 4 $STOP_AT normal
fi
echo "$(date +%H:%M:%S) SCR trl relay: no re-start: $why" >> $LOG
echo "relay: no re-start: $why"
