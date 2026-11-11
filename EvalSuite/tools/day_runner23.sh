#!/bin/bash
# EvalSuite runner 23, 2026-10-24 (detached, from a live evalsuite session). Orchestrator ~22:44: verdict on
# Breakthrough's D1 (B12's frugal wait-top-2: ranks 0-1 read and waited for, missing ranks 2-9 -> resident ranks
# 10-19, every expert kept) at 24 GB residency, C_m 192 (C_l = bt_b12's default 77), with MTP: variant b12c192 =
# load_b12(0.10, mtp=True) + B12_CM=192, mass weights, engine v3f / trims 0, B12LL's exact memory env. Non-exact: vs the
# 4-bit ref and vs B12LL, plus SSD expert reads per generated token (rows now log bt_b12's ext counters, decode only).
# D1's substitutions depend on what earlier rows left resident (B12-style pool history), so run 1 (normal priority)
# is reported alone first, and run 2 (b12c192.r2, [low]: a second run is background work) follows for the pooled
# verdict. Same chunk rules as tools/day_runner22.sh; deadline 2026-10-25 07:05 (the night rule: no Claude wakes in
# 03:08-07:15; detached runners may run through it).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day23_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-25 07:05" +%s)
export PEER_QUEUE_TIMEOUT=14400
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
mark() { ls -lT "results/$1" "results/$1/tfstep_parts" 2>/dev/null | md5 -q; }   # changes when a chunk writes anything

job() {   # job NAME VARIANT STAGE ARGS...   (pass --low in ARGS for a background ticket)
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
    [ "$rc" = 75 ] && { say "$name: paused at a row boundary, queued again"; continue; }
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
tables() {   # tables VARIANT: vs the 4-bit ref and vs B12LL, plus SSD reads per token
  ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1
  say "report $1 vs ref: $(tail -1 "results/report_$1.txt")"
  ./bin/evalsuite report --variant "$1" --decode --ref b12ll > "results/report_$1_vs_b12ll.txt" 2>&1
  say "report $1 vs b12ll: $(tail -1 "results/report_$1_vs_b12ll.txt")"
  say "ssd $1: $(.venv/bin/python tools/ssd_per_token.py "$1" 2>&1 | tail -1)"
}

say "runner 23 start (pid $$)"
job "decode b12c192 core" b12c192 decode b12c192 --sets math,ifeval,humaneval && tables b12c192
job "decode b12c192.r2 core [low]" b12c192.r2 decode b12c192.r2 --sets math,ifeval,humaneval --low && tables b12c192.r2 \
  && say "pooled: $(.venv/bin/python tools/pool_runs.py b12c192,b12c192.r2 2>&1 | grep -v '^Load from' | tr '\n' ';')"
say "runner 23 end"
