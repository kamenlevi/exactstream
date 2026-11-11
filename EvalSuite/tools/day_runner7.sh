#!/bin/bash
# EvalSuite day runner 7, 2026-10-22 (detached, from a live evalsuite session; arg 1 = the pid of runner 6's running
# chunk, a bin/evalsuite gpu process: this runner queues only after that chunk has unlocked, because `peer unlock`
# also cancels a queued request). Orchestrator 14:30 (Breakthrough + NewIdeas): B8's core sets right after the current
# B12 chunk so B8's verdict lands before 18:00 (B12's is decided: FAIL 3/5; B8 has ~25 % lower hard-text KLD). Order:
# b8 core + table, the rest of b12 new, the ref and floor anchors for the new sets, b8 new, b8fast core + new, w2f_r10
# new, the plan4433332222 filler. Same chunk rules as tools/day_runner6.sh (stops if its owning session is gone).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day7_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-22 23:30" +%s)
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
table() { ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1; say "report $1: $(tail -1 "results/report_$1.txt")"; }

CORE=math,ifeval,humaneval
NEW=mgsm,multi,agent
say "day runner 7 start (pid $$), waiting for runner 6's chunk (pid ${1:-none})"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done
say "runner 6's chunk ended"
job "decode b8 core" b8 decode b8 --sets $CORE; table b8
job "decode b12 new" b12 decode b12 --sets $NEW
job "decode ref new" ref decode ref --sets $NEW --C 256
job "decode floor new" floor decode floor --sets $NEW --C 256; table b12
job "decode b8 new" b8 decode b8 --sets $NEW; table b8
job "decode b8fast all" b8fast decode b8fast --sets $CORE,$NEW; table b8fast
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 7 end"
