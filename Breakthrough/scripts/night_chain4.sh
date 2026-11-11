#!/bin/sh
# Breakthrough silent-night chain 4 (2026-10-21): re-runs of probes that died on thermostat-induced spin-gate timeouts
# ranks 2-9 (no renormalisation) only in one band of 8 layers and is exact elsewhere, so it measures that band's share of
# the KLD. Small GPU steps (~2 min), each under its own short quiet lock, a 3-min cooling pause after each. Waits for
# night_chain2.sh (pid 32042, same CLAUDE_PID).
B=$HOME/Breakthrough; OUT=$B/data/expS; LOG=$OUT/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 EVALSUITE_FRUGAL=1 PEER_QUEUE_TIMEOUT=7200
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
step() {   # step NAME SECONDS_LIMIT CMD...  (under a lock already taken)
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^[A-Za-z_0-9]+ +C=|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -3 | tr '\n' ' ' | cut -c1-500)"
  return $rc
}
locked() {   # locked WHY NAME LIMIT CMD...: one short quiet lock around one step, then a cooling pause
  why=$1; shift
  peer lock quiet --queue 20m "Breakthrough night: $why" > /dev/null 2>&1 || { log "no lock for $why"; return 1; }
  g=0
  while [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 60 ] && [ $g -lt 60 ]; do sleep 5; g=$((g + 1)); done
  step "$@"; rc=$?
  peer unlock > /dev/null 2>&1
  sleep 180
  return $rc
}
while kill -0 48485 2>/dev/null; do sleep 30; done   # night_chain3.sh (same CLAUDE_PID)
log "night chain 4 started (pid $$)"
for v in exact base noren sere; do
  [ -f $B/data/tfvar/${v}_e8.npz ] && continue
  locked "KLD probe $v (~2 min)" tfvar8_$v 1800 env BIGMOE_MAXK=80 $PY $S/tf_frugal_var.py $v 0.10 tf_ifeval,tf_multi 8
done
log "night chain 4 end"
