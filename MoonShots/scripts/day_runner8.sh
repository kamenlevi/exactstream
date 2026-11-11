#!/bin/sh
# MoonShots day_runner8 (detached): round 22 MTP sweeps with committed history: draft depth 0-6 at 17 GB (C=128), then the
# speed-vs-memory curve (eager and depth 3 at C = 96, 160, 192, 256).
# Same lock discipline as day_runner6. Results: ~/MoonShots/data/day8/*.txt, one line per job in day7/summary.txt.
OUT=$HOME/MoonShots/data/day8; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
PL=$HOME/MoonShots/patches/planes125
export PEER_QUEUE_TIMEOUT=14400
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 21:50" +%s)   # no new job after 21:50
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
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 21:50)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "MoonShots day_runner8: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
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
log "day_runner8 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
S=$HOME/MoonShots/scripts/mtp_sweep125.py
job sweep_depth_C128 1500 $PY $S 0.25 0,1,2,3,4,5,6 192
job curve_C96        900  $PY $S 0.1875 0,3 192
job curve_C160       900  $PY $S 0.3125 0,3 192
job curve_C192       900  $PY $S 0.375 0,3 192
job curve_C256       900  $PY $S 0.5 0,3 192
log "day_runner8 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
