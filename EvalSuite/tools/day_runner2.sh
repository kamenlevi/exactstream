#!/bin/bash
# EvalSuite day runner 2026-10-21 (detached: nohup ... < /dev/null & disown, from the evalsuite session so peer
# attributes its locks to it): the rest of the night batch, which the silent night held. Orchestrator 08:55: frugal's
# decode verdict decides the 6-12 GB line, so its rows go first. Same chunk rules as tools/night_runner2.sh (one queued
# quiet-lock ticket per <= 45-min chunk, resumable; a job ends on a chunk that exits 0, or after two chunks in a row
# that wrote nothing). No chunk starts within 15 min of the 18:00 deadline.
#  1. decode w2f_r10, then w2f_r05 (math, ifeval, humaneval: 210 rows each)
#  2. prefill tf plan4433332222 (its mc ran at night), then its decode sets (all 234 rows)
#  3. filler: RULER for w2f_r10, then w2f_r05
#  then the tables (CPU): results/report_<variant>.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day2_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-21 18:00" +%s)
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

say "day runner 2 start (pid $$)"
job "decode w2f_r10" w2f_r10 decode w2f_r10 --sets math,ifeval,humaneval
job "decode w2f_r05" w2f_r05 decode w2f_r05 --sets math,ifeval,humaneval
for v in w2f_r10 w2f_r05; do
  ./bin/evalsuite report --variant "$v" --decode > "results/report_$v.txt" 2>&1
  say "report $v: $(tail -1 "results/report_$v.txt")"
done
job "prefill tf plan4433332222" plan4433332222 prefill plan4433332222 --tests tf --C 128
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256
job "decode ruler w2f_r10" w2f_r10 decode w2f_r10 --sets ruler
job "decode ruler w2f_r05" w2f_r05 decode w2f_r05 --sets ruler
for v in w2f_r10 w2f_r05 plan4433332222; do
  ./bin/evalsuite report --variant "$v" --decode > "results/report_$v.txt" 2>&1
  say "report $v: $(tail -1 "results/report_$v.txt")"
done
say "day runner 2 end"
