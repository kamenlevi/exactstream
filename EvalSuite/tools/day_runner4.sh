#!/bin/bash
# EvalSuite day runner 4, 2026-10-22 (detached; follows day runner 3, whose pid is arg 1). Runner 3 decodes B12's core
# sets (math, IFEval, HumanEval+); results/HOLD makes it skip its later jobs, so it ends soon after. This runner then
# completes the small-RAM verdict in the order that gives B12 its full verdict first (Kamen judges B12 / B8 / B8-fast
# by math, IFEval, HumanEval, multilingual, agentic): the new sets mgsm (MGSM, 80 problems in 10 languages, scored),
# multi (24 multilingual prompts: agreement with ref) and agent (29 real Claude Code turns, the next 128 tokens:
# agreement with ref) for ref, then b12, then b8 (all its sets), then the floor (the agreement rows' calibration), then
# b8fast (all its sets), then w2f_r10, then the plan4433332222 filler. Tables after each config. Same chunk rules as
# tools/night_runner2.sh; no chunk starts within 15 min of the 17:45 deadline.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day4_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-22 17:45" +%s)
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

NEW=mgsm,multi,agent
say "day runner 4 start (pid $$), waiting for day runner 3 (pid ${1:-none})"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
rm -f results/HOLD                                  # runner 3 is gone; its held jobs are this runner's now
say "day runner 3 ended; HOLD cleared"
job "decode ref $NEW" ref decode ref --sets $NEW --C 256
job "decode b12 $NEW" b12 decode b12 --sets $NEW; table b12
job "decode b8 all" b8 decode b8 --sets math,ifeval,humaneval,$NEW; table b8
job "decode floor $NEW" floor decode floor --sets $NEW --C 256; table b12; table b8
job "decode b8fast all" b8fast decode b8fast --sets math,ifeval,humaneval,$NEW; table b8fast
job "decode w2f_r10 $NEW" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 4 end"
