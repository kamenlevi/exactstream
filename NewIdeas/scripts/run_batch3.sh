#!/bin/sh
# NewIdeas batch 1 (2026-10-21): I1 deferred-expert KLD arms + I2/I3 MoE-input capture, on the 35B proxy (~21 GB, ~10 min
# of GPU). One quiet lock; active_at keepalive; memory gate. Log: ~/NewIdeas/data/batch3.log
N=$HOME/NewIdeas; OUT=$N/data; LOG=$OUT/batch3.log
PY=$HOME/BigMoE/.venv/bin/python; S=$N/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=18000
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
  log "done $name (rc $rc): $(grep -E 'KLD|capture|Error|Traceback|ref written|load' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
  return $rc
}
log "queued for the quiet lock"
peer lock quiet --queue 30m "NewIdeas: 35B policy frontier: wait-by-mass vs near-tie re-route vs half-layer deferral (~4 min, 25 GB)" > /dev/null 2>&1 || { log "no lock"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
[ $g -ge 120 ] && { log "memory busy for 10 min: giving the lock back"; peer unlock > /dev/null 2>&1; exit 1; }
log "lock taken"
step pol_smoke 600 env NI_SEG=256 $PY $S/pol35.py C26_w2_m70_ldrop C26_b25_w2_m50_lhalf || { peer unlock > /dev/null 2>&1; log "smoke failed: lock released"; exit 1; }
step pol_main 1500 $PY $S/pol35.py C26_w2_ldrop C26_w2_m50_ldrop C26_w2_m70_ldrop C26_w2_m85_ldrop C26_w8_ldrop C26_b25_w8 C26_w2_lhalf C26_w4_lhalf C26_w2_m50_lhalf C26_w2_lhalf_x85 C26_b25_w2_lhalf C26_b25_w2_m50_lhalf C13_w2_m70_ldrop C13_w2_lhalf
peer unlock > /dev/null 2>&1; log "lock released"
