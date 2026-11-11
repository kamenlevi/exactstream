#!/bin/bash
# EvalSuite day runner 8, 2026-10-22 (detached, from a live evalsuite session; started inside runner 7's lock).
# 17:0x decisions (Breakthrough's cold replays + NewIdeas' guard replay, both recommending it): the small-RAM verdict is
# judged GUARD-OFF on both sides (the shipped repetition guard armed on a normal formula line in math500 prealgebra/1640
# and steered the arithmetic: a correct answer became a wrong, capped one); the guard runs are kept as b12g / b8fastg.
# B8 decodes eager (its late adds have no MTP path: Breakthrough runs it eager; with MTP it crashed in bt_b12's
# layer_call on the MTP layer). B12 / B8 depend on pool history, so each core verdict gets a second run (.r2;
# NewIdeas: one run's IFEval noise ~2.4 pt SD). Order: b8 core, b12's 62 missing core rows (results/b12 = the morning's
# 148 no-guard rows), b8fast core, b8.r2, b12.r2, then the new sets for b12 / b8 / b8fast (anchors ref + floor done),
# w2f_r10's new sets, the plan4433332222 filler. Same chunk rules as tools/day_runner6.sh.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day8_runner.log
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
say "day runner 8 start (pid $$)"
table b12g
job "decode b8 core" b8 decode b8 --sets $CORE; table b8
job "decode b12 core" b12 decode b12 --sets $CORE; table b12
job "decode b8fast core" b8fast decode b8fast --sets $CORE; table b8fast
job "decode b8.r2 core" b8.r2 decode b8.r2 --sets $CORE; table b8.r2
job "decode b12.r2 core" b12.r2 decode b12.r2 --sets $CORE; table b12.r2
job "decode b12 new" b12 decode b12 --sets $NEW; table b12
job "decode b8 new" b8 decode b8 --sets $NEW; table b8
job "decode b8fast new" b8fast decode b8fast --sets $NEW; table b8fast
job "decode w2f_r10 new" w2f_r10 decode w2f_r10 --sets $NEW; table w2f_r10
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256; table plan4433332222
say "day runner 8 end"
