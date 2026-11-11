#!/bin/bash
# EvalSuite day runner 2026-10-20 (detached: nohup ... < /dev/null & disown). Decode sets in ~45-min chunks, one queued
# quiet-lock ticket per chunk (RULES "One lock per batch": others interleave between chunks): plan4433333333 until all
# 234 rows are paired, then floor, then ar4. No chunk starts within 10 min of the 14:15 deadline (verdict ~14:30).
# Arg 1: pid of a chunk already running (wait for it first).
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/day_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-20 14:15" +%s)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
rows() { [ -f "results/$1/decode.jsonl" ] && wc -l < "results/$1/decode.jsonl" | tr -d ' ' || echo 0; }
say "day runner start (pid $$), waiting for pid ${1:-none}"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 20; done
for v in plan4433333333 floor ar4; do
  while [ ! -f "results/$v/decode.json" ] && [ "$(left)" -gt 1200 ]; do
    before=$(rows "$v")
    ./bin/evalsuite gpu decode "$v" --C 256 --minutes 45 >> "$LOG" 2>&1; rc=$?
    say "decode $v chunk rc=$rc rows $before -> $(rows "$v")"
    [ "$(rows "$v")" = "$before" ] && { say "decode $v made no progress: next variant"; break; }
  done
done
say "day runner end"
