#!/bin/bash
# EvalSuite day runner 5, 2026-10-22 (detached: nohup ... < /dev/null & disown, from the evalsuite session so peer
# attributes its locks to it). Replaces runners 3 and 4 at 08:2x: Breakthrough + NewIdeas (07:47) judge B12 / B8 /
# B8-fast with B12_NORM=mass (cli.B12_ENV), so runner 3's first B12 chunk (renormalised weights; 1 row kept aside in
# results/b12/decode_normused_partial.jsonl) was stopped inside its lock and this runner's first chunk runs in it.
# The small-RAM verdict as the orchestrator ordered it (B12 -> B8 -> B8-fast, MTP on), each config complete before the
# next: Kamen judges them by math, IFEval, HumanEval, multilingual (MGSM scored + agreement on 24 prompts), agentic
# (29 real Claude Code turns, agreement with ref, read against the floor). Anchors ref and floor decode the new sets.
# Same chunk rules as tools/night_runner2.sh; no chunk starts within 15 min of the 17:45 deadline.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day5_runner.log
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

CORE=math,ifeval,humaneval
NEW=mgsm,multi,agent
say "day runner 5 start (pid $$)"
job "decode b12 core" b12 decode b12 --sets $CORE; table b12
job "decode b12 new" b12 decode b12 --sets $NEW
job "decode ref new" ref decode ref --sets $NEW --C 256; table b12
job "decode b8 all" b8 decode b8 --sets $CORE,$NEW; table b8
job "decode floor new" floor decode floor --sets $NEW --C 256; table b12; table b8
job "decode b8fast all" b8fast decode b8fast --sets $CORE,$NEW; table b8fast
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 5 end"
