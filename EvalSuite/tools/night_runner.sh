#!/bin/bash
# EvalSuite night runner, night of 2026-10-19/20 (detached: nohup ... < /dev/null & disown, from the evalsuite session so
# peer attributes its locks to that session). Every GPU stage goes through bin/evalsuite (queued quiet lock with a lease
# longer than the stage, memory gate, active_at keepalive, hard timeout, unlock). No stage starts within 10 min of the
# 07:05 deadline (EVALSUITE_DEADLINE).
#  1. reference outputs + prefill tests ref, floor, plan4433333333 (one load) + report
#  2. accepted anchor: when the AutoRound download (models/ar4_fetch.pid) has all 48 layers: link, prefill ar4, report
#  3. filler between/after: decode sets ref, plan4433333333, floor (70 min slices, resumable), ar4 first whenever ready
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-20 07:05" +%s)
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
left() { echo $(( EVALSUITE_DEADLINE - $(date +%s) )); }
AR4=0

try_ar4() {
  [ "$AR4" = 1 ] && return
  local n; n=$(ls models/ar4/zz-ar-layer-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  local fp; fp=$(cat models/ar4_fetch.pid 2>/dev/null)
  if [ "$n" = 48 ] && ! { [ -n "$fp" ] && kill -0 "$fp" 2>/dev/null; }; then
    say "ar4: 48 layers, linking"
    "$HOME/BigMoE/.venv/bin/python" tools/fetch_autoround.py --link >> "$LOG" 2>&1
    ./bin/evalsuite run --variant plan4433333333 >> "$LOG" 2>&1; say "run (ar4) rc=$?"
    [ -f results/ar4/run.json ] && AR4=1
  fi
}

say "night runner start (pid $$)"
for attempt in 1 2 3; do
  [ "$(left)" -lt 1800 ] && break
  ./bin/evalsuite run --variant plan4433333333 >> "$LOG" 2>&1; rc=$?
  say "run 1 attempt $attempt rc=$rc"
  [ -f results/plan4433333333/run.json ] && [ -f results/floor/run.json ] && break
  sleep 60
done
try_ar4
rows() { [ -f "results/$1/decode.jsonl" ] && wc -l < "results/$1/decode.jsonl" | tr -d ' ' || echo 0; }
for v in ref plan4433333333 floor; do
  while [ ! -f "results/$v/decode.json" ] && [ "$(left)" -gt 1500 ]; do
    try_ar4
    before=$(rows "$v")
    ./bin/evalsuite gpu decode "$v" --minutes 70 >> "$LOG" 2>&1; rc=$?
    say "decode $v slice rc=$rc rows $before -> $(rows "$v")"
    [ "$(rows "$v")" = "$before" ] && { say "decode $v made no progress: skipping"; break; }
  done
done
while [ "$AR4" = 0 ] && [ "$(left)" -gt 2400 ]; do try_ar4; [ "$AR4" = 0 ] && sleep 120; done
say "night runner end"
