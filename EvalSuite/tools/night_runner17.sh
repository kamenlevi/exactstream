#!/bin/bash
# EvalSuite runner 17, 2026-10-24 (detached, from a live evalsuite session; arg 1 = the pid of runner 16's running chunk,
# b12llugb core: this runner queues only after that chunk has unlocked). Orchestrator ~01:45 (Kamen 01:40: run the whole
# night, breakthrough research first): after UGB-0.5's core, only B8-fast's second core run (the 8 GB line), and only
# when no research session waits for the GPU; dropped: b12llb20 new (b20 is out), b12llb16 new (b16 failed core),
# f24b16.r2 (24 GB-only work stopped on Friday). Same chunk rules as tools/day_runner6.sh; deadline 07:05.
export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
cd "$HOME/EvalSuite" || exit 1
LOG=results/night17_runner.log
export EVALSUITE_DEADLINE=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-24 07:05" +%s)
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
vs() { ./bin/evalsuite report --variant "$1" --decode --ref "$2" > "results/report_$1_vs_$2.txt" 2>&1; say "report $1 vs $2: $(tail -1 "results/report_$1_vs_$2.txt")"; }
table() { ./bin/evalsuite report --variant "$1" --decode > "results/report_$1.txt" 2>&1; say "report $1: $(tail -1 "results/report_$1.txt")"; }

CORE=math,ifeval,humaneval
NEW=mgsm,multi,agent
same_as_b12ll() {   # same_as_b12ll VARIANT: rows token-identical to B12LL's, and the peak footprint
  python3 -c "
import json, sys
a = {r['key']: r for r in map(json.loads, open('results/b12ll/decode.jsonl'))}
b = {r['key']: r for r in map(json.loads, open('results/$1/decode.jsonl'))}
ks = [k for k in b if k in a]
fp = json.load(open('results/$1/decode.json')).get('footprint_peak_gb')
print(f'{sum(a[k][\"completion_ids\"] == b[k][\"completion_ids\"] for k in ks)}/{len(ks)} rows token-identical to B12LL; peak footprint {fp:.2f} GB')
" 2>&1
}
research_waits() { peer ls 2>/dev/null | grep -q "waiting for a quiet window"; }   # (our own ticket is not queued then)
say "runner 17 start (pid $$), waiting for runner 16's chunk (pid ${1:-none})"
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 15; done
say "runner 16's chunk ended"
git -C ~/Breakthrough merge-base --is-ancestor 6fe76c3 HEAD || { say "~/Breakthrough lacks 6fe76c3: stop"; exit 2; }
job "decode b12llugb core" b12llugb decode b12llugb --sets $CORE; table b12llugb; vs b12llugb b12ll
say "b12llugb: $(same_as_b12ll b12llugb)"
while [ "$(left)" -gt 900 ] && research_waits; do sleep 120; done   # research tickets go first tonight
job "decode b8fast.r2 core" b8fast.r2 decode b8fast.r2 --sets $CORE; table b8fast.r2
say "runner 17 end"
