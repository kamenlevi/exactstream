#!/bin/sh
# MoonShots day_runner4 (detached): plane-granular streaming prototype for the 125B (patches/planes125), then the queued
# quality arms. Same lock discipline as night_runner.sh: per-job `peer lock quiet --queue`, >= 85 % free-memory guard,
# active_at keepalive. Launched under a quiet lock the session holds: the first job uses it, later jobs queue.
# Plane files: ~72 GB in data/planes125 (Kamen 11:2x: OK, keep >= ~100 GB free; make_planes stops at 110 GB). Delete after.
# Results: ~/MoonShots/data/day4/*.txt, one line per job in day4/summary.txt.   usage: day_runner4.sh
OUT=$HOME/MoonShots/data/day4; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
PL=$HOME/MoonShots/patches/planes125
export PEER_QUEUE_TIMEOUT=14400
export PLANES_MIN_FREE_GB=110
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 18:00" +%s)   # no new job after 18:00
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT CMD...   (sets P = the job's pid: don't use P for anything else)
  name=$1; lim=$2; shift 2
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 18:00)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "MoonShots day_runner4: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P 2>/dev/null; sleep 5; kill -9 $P 2>/dev/null; break; }
    fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  peer unlock > /dev/null 2>&1
  log "done $name: $(grep -E '^(125B|  vs|  noise|planes|bigmoe|decode:|prefill|layers|L[0-9]+: done|fixed|adaptive|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
}
log "day_runner4 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
job make_planes0      900  $PY $PL/make_planes.py $HOME/MoonShots/data/planes_test 0 0
job planes_unit       600  $PY $PL/planes_unit.py $HOME/MoonShots/data/planes_test
if grep -q "planes_unit: OK" "$OUT/planes_unit.txt" 2>/dev/null; then
  rm -rf $HOME/MoonShots/data/planes_test
  job make_planes     3600 sh -c "for r in '0 11' '12 23' '24 35' '36 47'; do $PY $PL/make_planes.py $HOME/MoonShots/data/planes125 \$r & done; wait"
  log "plane files: $(du -sg $HOME/MoonShots/data/planes125 | cut -f1) GB; free $(df -g / | tail -1 | awk '{print $4}') GB"
  if [ "$(ls $HOME/MoonShots/data/planes125/L*.lsb 2>/dev/null | wc -l)" -eq 48 ]; then
    job bench_bigmoe_nopf 1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py bigmoe 256
    job bench_planes_380  1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $HOME/MoonShots/data/planes125 380 90 4
    job bench_planes_420  1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $HOME/MoonShots/data/planes125 420 46 4
    job bench_planes_340  1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $HOME/MoonShots/data/planes125 340 135 4
    FUS="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1"          # BigMoE's dense fusions (env-switched)
    job bench_planes_fus  1800 env $FUS BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $HOME/MoonShots/data/planes125 380 90 4
    job bench_bigmoe_fus  1800 env $FUS BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py bigmoe 256
    job bench_bigmoe_ref  1800 env $FUS BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 $PY $PL/bench_planes125.py bigmoe 256
  else
    log "plane files incomplete: bench skipped"
  fi
else
  log "planes_unit did not pass: plane files and bench skipped"
fi
job expert_tiers35c   900  $PY $HOME/MoonShots/scripts/expert_tiers35c.py
PLAN125=$(cat $HOME/MoonShots/data/next_plan.txt 2>/dev/null || echo 4433333333)
job planes125_plan    5400 env OFFSET=0 $PY $PL/gsm_planes125.py $PLAN125 150
log "day_runner4 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
