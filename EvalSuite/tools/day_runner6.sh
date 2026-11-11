#!/bin/bash
# EvalSuite day runner 6, 2026-10-22 (detached: nohup ... < /dev/null & disown, FROM A LIVE evalsuite SESSION: peer
# attributes the locks to it, and when the app stopped this session at 10:16 the queue entry went and every later
# lock request failed, "can't tell which Claude session this is"; runner 5 then skipped all its jobs). Now a runner
# stops at the first such answer, so a relaunch resumes the same order. Orchestrator 11:06: B12 again with
# Breakthrough's repetition guard (bt_b12 0418c6c+: load_b12 registers the main track's repguard), all core sets in
# the original order (frugal outputs depend on pool history; Breakthrough's caveat); the 148 no-guard rows are in
# results/b12/decode_noguard_partial.jsonl. Then B12's new sets with both anchors, then B8 and B8-fast (all six
# sets), w2f_r10's new sets, the plan4433332222 filler. Same chunk rules as tools/night_runner2.sh; no chunk starts
# within 15 min of the 23:30 deadline (the 18:00 weekly reset limits Claude usage, not this detached GPU line).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day6_runner.log
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
say "day runner 6 start (pid $$)"
job "decode b12 core" b12 decode b12 --sets $CORE; table b12
job "decode b12 new" b12 decode b12 --sets $NEW
job "decode ref new" ref decode ref --sets $NEW --C 256
job "decode floor new" floor decode floor --sets $NEW --C 256; table b12
job "decode b8 all" b8 decode b8 --sets $CORE,$NEW; table b8
job "decode b8fast all" b8fast decode b8fast --sets $CORE,$NEW; table b8fast
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 6 end"
