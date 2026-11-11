#!/bin/bash
# EvalSuite runner 14, 2026-10-23/24 (detached, from a live evalsuite session; arg 1 = the pid of runner 13's running
# chunk, b12ll.r2 chunk 1: this runner queues only after that chunk has unlocked). Orchestrator ~19:00 for Fit12: (1)
# B12LL's shipping env adds BIGMOE_PF_GROUP=64 (11.28 GB instead of 12.85, same tokens by sha) and ASYNC_EVERY=1
# (speed, same shas): in cli.B12_ENV from b12ll.r2's chunk 2 on; (2) a verdict for B12LL-b16 (Fit12's hook
# ~/Fit12/scripts/fit_hook.py: B12LL + the per-pass expert budget, non-exact; b20 the fallback), two core runs, right
# after B12LL's second run. Order: b12ll.r2 core (resume), b12llb16 core, b12llb16.r2 core, b12ll new, b12llb16 new,
# b8fast.r2 core, the rest of f24b16.r2. Same chunk rules as tools/day_runner6.sh; deadline 07:05 (night_plan.md).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night14_runner.log
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
say "runner 14 start (pid $$), waiting for runner 13's chunk (pid ${1:-none})"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done
say "runner 13's chunk ended"
git -C ~/Breakthrough merge-base --is-ancestor 6fe76c3 HEAD || { say "~/Breakthrough lacks 6fe76c3: stop"; exit 2; }
job "decode b12ll.r2 core" b12ll.r2 decode b12ll.r2 --sets $CORE; table b12ll.r2
job "decode b12llb16 core" b12llb16 decode b12llb16 --sets $CORE; table b12llb16; vs b12llb16 b12ll
job "decode b12llb16.r2 core" b12llb16.r2 decode b12llb16.r2 --sets $CORE; table b12llb16.r2; vs b12llb16.r2 b12ll
job "decode b12ll new" b12ll decode b12ll --sets $NEW; table b12ll
job "decode b12llb16 new" b12llb16 decode b12llb16 --sets $NEW; table b12llb16; vs b12llb16 b12ll
job "decode b8fast.r2 core" b8fast.r2 decode b8fast.r2 --sets $CORE; table b8fast.r2
job "decode f24b16.r2 core" f24b16.r2 decode f24b16.r2 --sets $CORE; table f24b16.r2; vs f24b16.r2 f24nat
say "runner 14 end"
