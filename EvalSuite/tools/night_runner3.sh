#!/bin/bash
# EvalSuite night batch 2026-10-20/21, restarted 22:45 after the two-level tier-memo fix (runner 2 did the tfsteps) (detached: nohup ... < /dev/null & disown, started from the evalsuite session so
# peer attributes its locks to it). Jobs in priority order; each job is a chain of <= 45-min chunks and every chunk is one
# bin/evalsuite gpu call: its own queued quiet lock (lease 60 min), memory gate, active_at keepalive every 4 min, hard
# timeout, unlock. A chunk resumes where the last one stopped (decode rows, tfstep parts per sequence; the prefill tests
# mc and tf are separate chunks). A job ends when a chunk exits 0, or after two chunks in a row that wrote nothing. No
# chunk starts within 15 min of the 07:05 deadline (EVALSUITE_DEADLINE; cli.gpu also refuses within 10 min).
#  1. tfstep (the 138 own-output sequences, one token per call): w2f_r10, ref (the step path's own noise), w2f_r05
#  2. prefill tests plan4433332222 (tiers on every token, multi-row calls included): mc, then tf
#  3. decode sets: w2f_r10 (math, ifeval, humaneval), plan4433332222 (all four), w2f_r05 (math, ifeval, humaneval)
#  4. filler: RULER for w2f_r10, then w2f_r05
#  then the tables (CPU): results/report_<variant>.txt
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night2_runner.log   # same log as runner 2
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-21 07:05" +%s)
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

say "night runner 3 start (pid $$): runner 2 jobs 1-3 done, from the plan4433332222 prefill on"
job "prefill mc plan4433332222" plan4433332222 prefill plan4433332222 --tests mc --C 128
job "prefill tf plan4433332222" plan4433332222 prefill plan4433332222 --tests tf --C 128
job "decode w2f_r10" w2f_r10 decode w2f_r10 --sets math,ifeval,humaneval
job "decode plan4433332222" plan4433332222 decode plan4433332222 --C 256
job "decode w2f_r05" w2f_r05 decode w2f_r05 --sets math,ifeval,humaneval
job "decode ruler w2f_r10" w2f_r10 decode w2f_r10 --sets ruler
job "decode ruler w2f_r05" w2f_r05 decode w2f_r05 --sets ruler
for v in w2f_r10 w2f_r05 plan4433332222; do
  ./bin/evalsuite report --variant "$v" --decode > "results/report_$v.txt" 2>&1
  say "report $v: $(tail -1 "results/report_$v.txt")"
done
say "night runner 3 end"
