#!/bin/sh
# NewIdeas (2026-10-22): 125B same-layer forecast capture (routing only; the forward is exact streaming, unchanged).
# One quiet lock (~15 min, ~13 GB at 10 % residency); keepalive; memory gate. Log: ~/NewIdeas/data/samelayer.log
N=$HOME/NewIdeas; OUT=$N/data/samelayer; LOG=$N/data/samelayer.log
PY=$HOME/BigMoE/.venv/bin/python; S=$N/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=18000 BIGMOE_MAXK=80
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
step() {   # step NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 150)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E 'tf_samelayer125|recall|Error|Traceback' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-1200)"
  return $rc
}
log "queued for the quiet lock"
peer lock quiet --queue 20m "NewIdeas: 125B same-layer routing forecast capture (exact forward, routing logged only; ~15 min, ~13 GB)" > /dev/null 2>&1 || { log "no lock"; exit 1; }
g=0
while [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 25 ] && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
[ $g -ge 120 ] && { log "memory busy for 10 min: giving the lock back"; peer unlock > /dev/null 2>&1; exit 1; }
log "lock taken (memory free $(memory_pressure | tail -1 | grep -o '[0-9]*') %)"
step smoke 600 $PY $S/tf_samelayer125.py 0.10 tf_code 24 || { peer unlock > /dev/null 2>&1; log "smoke failed: lock released"; exit 1; }
step hard 1200 $PY $S/tf_samelayer125.py 0.10 tf_ifeval,tf_multi 16
step easy 1200 $PY $S/tf_samelayer125.py 0.10 tf_gsm,tf_code 16
peer unlock > /dev/null 2>&1; log "lock released"
