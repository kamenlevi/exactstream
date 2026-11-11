#!/bin/sh
# MoonShots day_runner5 (detached): 17 GB benches of the plane-streaming prototype vs BigMoE (Kamen's target: today's speed
# in 24 GB total RAM -> expert cache ~17 GB), then tiers35c and the 125B plan GSM8K. Same lock discipline as
# night_runner.sh: per-job `peer lock quiet --queue`, >= 85 % free-memory guard, active_at keepalive. Launched under a quiet
# lock the session holds: the first job uses it, later jobs queue. Plane files: data/planes125 (67 GB; delete after).
# Results: ~/MoonShots/data/day5/*.txt, one line per job in day5/summary.txt.   usage: day_runner5.sh
OUT=$HOME/MoonShots/data/day5; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
PL=$HOME/MoonShots/patches/planes125
PD=$HOME/MoonShots/data/planes125
export PEER_QUEUE_TIMEOUT=14400
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
  else peer lock quiet --queue 90m "MoonShots day_runner5: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
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
  log "done $name: $(grep -E '^(125B|  vs|  noise|planes|bigmoe|fixed|adaptive|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
}
log "day_runner5 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
# C=128 whole experts = 0.355 GB/layer = 17 GB; plane splits of the same bytes (MSB 1.536 MB, LSB 1.38 MB)
job b17_bigmoe_nopf   1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py bigmoe 128
job b17_planes_130    1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $PD 130 112 4
job b17_planes_150    1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $PD 150 89 4
job b17_planes_110    1800 env BIGMOE_PREFETCH=0 $PY $PL/bench_planes125.py planes $PD 110 134 4
FUS="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1"
job b17_bigmoe_ref    1800 env $FUS BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 $PY $PL/bench_planes125.py bigmoe 128
job expert_tiers35c   900  $PY $HOME/MoonShots/scripts/expert_tiers35c.py
PLAN125=$(cat $HOME/MoonShots/data/next_plan.txt 2>/dev/null || echo 4433333333)
job planes125_plan    5400 env OFFSET=0 $PY $PL/gsm_planes125.py $PLAN125 150
log "day_runner5 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
