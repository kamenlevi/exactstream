#!/bin/bash
# EvalSuite night runner 9, 2026-10-22/23 (detached, from a live evalsuite session; arg 1 = day runner 8's pid: this
# runner queues only after runner 8 has exited, since `peer unlock` also cancels a queued request). Runner 8's 23:30
# deadline cut its queue: B8-fast's core sets (113/210 rows at 22:46), the second runs and the guard-off new rows
# remain. They are not urgent (B12 / B8 verdicts sent 20:30), so they fill the night's GPU gaps behind the research
# sessions (Kamen's queue rule, Thu ~22:40). Same chunk rules as tools/day_runner6.sh; no chunk starts within 15 min of
# the 07:05 deadline. Order: b8fast core + table, b8.r2 core, b12.r2 core, the new sets for b12 / b8 / b8fast (anchors
# ref + floor done), w2f_r10's new sets, the plan4433332222 filler.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night9_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 07:05" +%s)
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
say "night runner 9 start (pid $$), waiting for day runner 8 (pid ${1:-none})"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
say "day runner 8 ended"
job "decode b8fast core" b8fast decode b8fast --sets $CORE; table b8fast
job "decode b8.r2 core" b8.r2 decode b8.r2 --sets $CORE; table b8.r2
job "decode b12.r2 core" b12.r2 decode b12.r2 --sets $CORE; table b12.r2
job "decode b12 new" b12 decode b12 --sets $NEW; table b12
job "decode b8 new" b8 decode b8 --sets $NEW; table b8
job "decode b8fast new" b8fast decode b8fast --sets $NEW; table b8fast
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "night runner 9 end"
