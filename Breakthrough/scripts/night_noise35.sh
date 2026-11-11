#!/bin/sh
# Breakthrough D2 first experiment (detached): noise-injection sweep on the Qwen3.6-35B-A3B 8-bit proxy (scripts/noise35.py).
# Starts when experiment A's second runner (data/expA/runner2.pid) has exited; no new job after 02:45, so it is done
# before the main-track night runners (03:05). Jobs of ~2-10 min, each under its own queued quiet lock; same lock
# discipline as night_expA.sh (memory gate >= 85 % free, active_at keepalive every 3 min, hard limits).
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
R2=$(cat "$RUN/runner2.pid" 2>/dev/null)
while [ -n "$R2" ] && kill -0 "$R2" 2>/dev/null; do sleep 30; done
log "night_noise35 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
A="q4 n0.0086 n0.012 n0.017 n0.024 n0.034"
job ref           600 $PY $S ref
job experts_gu   1500 $PY $S experts_gu $A
job experts_down 1500 $PY $S experts_down $A
job all_q4        600 $PY $S all q4
job gdn          1200 $PY $S gdn q4 n0.0086 n0.012 n0.017
job attn         1200 $PY $S attn q4 n0.0086 n0.012 n0.017
job shared        900 $PY $S shared q4 n0.012 n0.017
job lm_head       900 $PY $S lm_head q4 n0.012 n0.017
job experts      1200 $PY $S experts q4 n0.012 n0.017
log "night_noise35 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
