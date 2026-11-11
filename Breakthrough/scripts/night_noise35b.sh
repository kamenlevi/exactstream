#!/bin/sh
# Breakthrough D2 follow-up: null arms for the noise sweep (the KLD floor against the separate reference pass).
# Starts when night_noise35.sh (data/noise35/runner.pid) has exited; no new job after 02:45. Same lock discipline.
OUT=$HOME/Breakthrough/data/noise35; mkdir -p "$OUT"
RUN=$HOME/Breakthrough/data/expA
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Breakthrough/scripts/noise35.py
export PEER_QUEUE_TIMEOUT=14400 PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-20 02:45" +%s)   # no new job after 02:45
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 02:45)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "Breakthrough noise35: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( "$@" >> "$OUT/$name.txt" 2>&1 ) & P=$!
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
  log "done $name: $(grep -E '^(ref|experts|shared|gdn|attn|lm_head|all |Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
}
R3=$(cat "$OUT/runner.pid" 2>/dev/null)
while [ -n "$R3" ] && kill -0 "$R3" 2>/dev/null; do sleep 30; done
log "night_noise35b (null arms) started (pid $$)"
job null_all   600 $PY $S all none none
job null_gu    600 $PY $S experts_gu n0.0
log "night_noise35b finished"
