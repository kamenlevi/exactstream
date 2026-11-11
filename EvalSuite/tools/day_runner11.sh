#!/bin/bash
# EvalSuite day runner 11, 2026-10-23 (detached, from a live evalsuite session; arg 1 = the pid of runner 10's running
# chunk: this runner queues only after that chunk has unlocked). Adds the 24 GB fast-mode verdict (orchestrator 08:15,
# Kamen's task-results bar for opt-in fast modes; BigMoE's hook ~/BigMoE/scripts/fast24.py pinned at 36c9cc5): f24b16 =
# the per-pass budget b16 + budget-aware JIT, f24nat = the v3 tier plan's natural routing (the speed-up's baseline,
# exact-to-v3), both MTP 3:3:0 fixed widths, guard off; judged against the 4-bit ref (+ floor) and against f24nat. Order:
# B12L's two core runs (as queued), f24b16 core, f24nat core, their new sets, f24b16's second core run, then runner 10's
# rest (b8fast.r2, b12l new, b12ll core, fillers). Same chunk rules as tools/day_runner6.sh; deadline 23:00.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day11_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-23 23:00" +%s)
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
say "day runner 11 start (pid $$), waiting for runner 10's chunk (pid ${1:-none}); BigMoE at $(git -C ~/BigMoE rev-parse --short HEAD)"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done
say "runner 10's chunk ended"
git -C ~/Breakthrough merge-base --is-ancestor 6fe76c3 HEAD || { say "~/Breakthrough lacks 6fe76c3: stop"; exit 2; }
job "decode b12l core" b12l decode b12l --sets $CORE; table b12l
job "decode b12l.r2 core" b12l.r2 decode b12l.r2 --sets $CORE; table b12l.r2
job "decode f24b16 core" f24b16 decode f24b16 --sets $CORE; table f24b16
job "decode f24nat core" f24nat decode f24nat --sets $CORE; table f24nat; vs f24b16 f24nat
job "decode f24b16 new" f24b16 decode f24b16 --sets $NEW; table f24b16
job "decode f24nat new" f24nat decode f24nat --sets $NEW; table f24nat; vs f24b16 f24nat
job "decode f24b16.r2 core" f24b16.r2 decode f24b16.r2 --sets $CORE; table f24b16.r2; vs f24b16.r2 f24nat
job "decode b8fast.r2 core" b8fast.r2 decode b8fast.r2 --sets $CORE; table b8fast.r2
job "decode b12l new" b12l decode b12l --sets $NEW; table b12l
job "decode b12ll core" b12ll decode b12ll --sets $CORE; table b12ll
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 11 end"
