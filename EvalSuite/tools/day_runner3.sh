#!/bin/bash
# EvalSuite day runner 3, 2026-10-22 (detached: nohup ... < /dev/null & disown, from the evalsuite session so peer
# attributes its locks to it). Orchestrator 07:17: the small-RAM verdicts first, B12 -> B8 -> B8-fast, MTP on (the
# shipping configs; Breakthrough's bt_b12 hook, cli.B12_ENV / SHIP_ENV). Same chunk rules as tools/night_runner2.sh (one
# queued quiet-lock ticket per <= 45-min chunk, resumable; a job ends on a chunk that exits 0, or after two chunks
# in a row that wrote nothing). No chunk starts within 15 min of the 17:45 deadline (the weekly reset is 18:00).
#  1. decode b12, b8, b8fast (math, ifeval, humaneval: 210 rows each), each with its table right after
#  2. filler: the rest of plan4433332222's decode sets (yesterday's deadline cut them)
#  then the tables (CPU): results/report_<variant>.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day3_runner.log
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

say "day runner 3 start (pid $$)"
for v in b12 b8 b8fast; do
  job "decode $v" "$v" decode "$v" --sets math,ifeval,humaneval
  ./bin/evalsuite report --variant "$v" --decode > "results/report_$v.txt" 2>&1
  say "report $v: $(tail -1 "results/report_$v.txt")"
done
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256
./bin/evalsuite report --variant plan4433332222 --decode > results/report_plan4433332222.txt 2>&1
say "day runner 3 end"
