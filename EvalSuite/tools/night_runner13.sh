#!/bin/bash
# EvalSuite runner 13, 2026-10-23/24 (detached, from a live evalsuite session; arg 1 = the pid of runner 12's running
# chunk, b12l new: this runner queues only after that chunk has unlocked). B12LL passed 5/5 in its first core run (17:28;
# answer agreement 96.7 % = the floor's 97.1 %, truncations 14 vs 13), so its confirmation comes before B8-fast's second
# run (B8-fast failed IFEval -7): b12l new (resume), b12ll.r2 core, b12ll's MGSM / multilingual / agentic rows,
# b8fast.r2 core, the rest of f24b16.r2 (24 GB fast mode, after the < 12 GB focus; Kamen 14:50). B12LL runs at ~13 tok/s,
# so this runs into the night: deadline 07:05 (night_plan.md). Same chunk rules as tools/day_runner6.sh.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night13_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-24 07:05" +%s)
export PEER_QUEUE_TIMEOUT=14400
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
mark() { ls -lT "results/$1" "results/$1/tfstep_parts" 2>/dev/null | md5 -q; }   # changes when a chunk writes anything

job() {   # job NAME VARIANT STAGE ARGS...
  local name=$1 v=$2 stuck=0 rc before
  shift 2
  while [ "$(left)" -gt 900 ]; do
    before=$(mark "$v")
    ./bin/evalsuite gpu "$@" --minutes 45 >> "$LOG" 2>&1; rc=$?
    say "$name: chunk rc=$rc"
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
vs() { ./bin/evalsuite report --variant "$1" --decode --ref "$2" > "results/report_$1_vs_$2.txt" 2>&1; say "report $1 vs $2: $(tail -1 "results/report_$1_vs_$2.txt")"; }
table() { ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1; say "report $1: $(tail -1 "results/report_$1.txt")"; }

CORE=math,ifeval,humaneval
NEW=mgsm,multi,agent
say "runner 13 start (pid $$), waiting for runner 12's chunk (pid ${1:-none})"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done
say "runner 12's chunk ended"
git -C ~/Breakthrough merge-base --is-ancestor 6fe76c3 HEAD || { say "~/Breakthrough lacks 6fe76c3: stop"; exit 2; }
job "decode b12l new" b12l decode b12l --sets $NEW; table b12l
job "decode b12ll.r2 core" b12ll.r2 decode b12ll.r2 --sets $CORE; table b12ll.r2
job "decode b12ll new" b12ll decode b12ll --sets $NEW; table b12ll
job "decode b8fast.r2 core" b8fast.r2 decode b8fast.r2 --sets $CORE; table b8fast.r2
job "decode f24b16.r2 core" f24b16.r2 decode f24b16.r2 --sets $CORE; table f24b16.r2; vs f24b16.r2 f24nat
say "runner 13 end"
