#!/bin/bash
# EvalSuite day runner 10, 2026-10-23 (detached, from a live evalsuite session). After the small-RAM verdict
# (docs/RESULT_smallram_2026-10-22.md): Breakthrough's next 12 GB candidates (02:42, confirmed by NewIdeas), judged like
# B12 (guard off both sides, MTP, two core runs): B12L = B12 + cold ranks 2-3 half a layer late, B12LL = all cold ranks
# late (near-exact, ~half speed); plus B8-fast's missing second core run; then the fillers. ~/Breakthrough must be at
# 6fe76c3 or later (late-write fixes under MTP). Same chunk rules as tools/day_runner6.sh; no chunk starts within 15 min
# of the 23:00 deadline.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day10_runner.log
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
table() { ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1; say "report $1: $(tail -1 "results/report_$1.txt")"; }

CORE=math,ifeval,humaneval
NEW=mgsm,multi,agent
say "day runner 10 start (pid $$); Breakthrough at $(git -C ~/Breakthrough rev-parse --short HEAD)"
git -C ~/Breakthrough merge-base --is-ancestor 6fe76c3 HEAD || { say "~/Breakthrough lacks 6fe76c3: stop"; exit 2; }
job "decode b12l core" b12l decode b12l --sets $CORE; table b12l
job "decode b12l.r2 core" b12l.r2 decode b12l.r2 --sets $CORE; table b12l.r2
job "decode b8fast.r2 core" b8fast.r2 decode b8fast.r2 --sets $CORE; table b8fast.r2
job "decode b12l new" b12l decode b12l --sets $NEW; table b12l
job "decode b12ll core" b12ll decode b12ll --sets $CORE; table b12ll
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 10 end"
