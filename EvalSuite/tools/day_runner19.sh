#!/bin/bash
# EvalSuite runner 19, 2026-10-24 (replaces runner 18 at 08:2x: [low] stages now yield) (detached, from a live evalsuite session). Orchestrator 06:10: b8fast.r2 got the GPU at
# 06:00 only because its 04:38 ticket had aged past 60 min, ahead of Fit12's and Leap's research tickets; it was stopped
# (33 rows kept) and is re-queued here as a background run: bin/evalsuite gpu --low puts "[low]" in the lock's why
# text, which the queue ranks behind every research ticket (it moves up only after 180 min). Same chunk rules as
# tools/day_runner6.sh; deadline 18:00. Orchestrator 08:20: a [low] stage gives way while it runs: cli.gpu checks about
# once a minute whether a research ticket (no [low]) waits; if so the stage stops at its next row boundary (exit 75),
# unlocks, and this runner queues it again as [low] (a yield is not counted as a chunk without progress).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day19_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-24 18:00" +%s)
export PEER_QUEUE_TIMEOUT=14400
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
mark() { ls -lT "results/$1" "results/$1/tfstep_parts" 2>/dev/null | md5 -q; }   # changes when a chunk writes anything

job() {   # job NAME VARIANT STAGE ARGS...   (every chunk queued as [low])
  local name=$1 v=$2 stuck=0 rc before
  shift 2
  while [ "$(left)" -gt 900 ]; do
    before=$(mark "$v")
    ./bin/evalsuite gpu "$@" --minutes 45 --low >> "$LOG" 2>&1; rc=$?
    say "$name: chunk rc=$rc"
    [ "$rc" = 75 ] && { say "$name: yielded to a research ticket; queued again as [low]"; continue; }
    if tail -4 "$LOG" | grep -q "can't tell which Claude session\|left the queue"; then
      say "$name: the session that owns this runner's locks is gone: runner stops (relaunch it from a live session)"
      exit 3
    fi
    [ "$rc" = 0 ] && { say "$name: done"; return 0; }
    if [ "$(mark "$v")" = "$before" ]; then
      stuck=$((stuck + 1))
      [ "$stuck" -ge 2 ] && { say "$name: two chunks without progress, skipping"; return 1; }
      sleep 300
    else
      stuck=0
    fi
  done
  say "$name: deadline"
  return 1
}
table() { ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1; say "report $1: $(tail -1 "results/report_$1.txt")"; }

say "runner 19 start (pid $$)"
job "decode b8fast.r2 core [low]" b8fast.r2 decode b8fast.r2 --sets math,ifeval,humaneval; table b8fast.r2
say "runner 19 end"
