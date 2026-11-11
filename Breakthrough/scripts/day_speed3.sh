#!/bin/sh
# Breakthrough Wed 2026-10-21 daytime: real speed of the quality policies (no thermostat after 08:45, so the native spin
# gate is valid). Waits until after 08:45, stops night chain 10 in one of its cooling pauses (two runners of this session
# must never hold locks at once), then one quiet lock (~25 min): bt_bild.py (frugal tau<0, exact tau 99, BiLD 0.1 / 0.5)
# and bt_mass.py (wait-by-mass 0.85 / 0.7 eager and MTP 3:3:0, adaptive theta eager) at 10 %, code + prose; 0.85 at 5 %.
B=$HOME/Breakthrough; OUT=$B/data/day; LOG=$B/data/expS/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$B/scripts
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=10800
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
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
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^\[|^bt_|^[A-Za-z_0-9]+ +C=|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
  return $rc
}
while kill -0 79241 2>/dev/null; do sleep 20; done   # day_speed2.sh (same CLAUDE_PID)
# stop night chain 10 right after one of its probes finished (it then sleeps 180 s without a lock)
held() { $PY -I -S -c 'import json, sys, time
d = json.load(open(sys.argv[1])); sys.exit(0 if d.get("mode") == "quiet" and d.get("until", 0) > time.time() else 1)' "$LOCKF" 2>/dev/null; }
peer lock quiet --queue 45m "Breakthrough: wait-by-mass speed, fixed harness (~15 min)" > /dev/null 2>&1 || { log "no lock (day_speed3)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
log "lock taken (day_speed3)"
for PR in code prose; do
  step sp3_m85_$PR 600 env PROMPT=$PR BT_THETA=0.85 $PY $S/bt_mass.py 0.10 speed ar
  step sp3_m70_$PR 600 env PROMPT=$PR BT_THETA=0.70 $PY $S/bt_mass.py 0.10 speed ar
  step sp3_ada_$PR 600 env PROMPT=$PR BT_ADA=0.5,0.9,0.1 $PY $S/bt_mass.py 0.10 speed ar
  step sp3_m85spec_$PR 600 env PROMPT=$PR BT_THETA=0.85 $PY $S/bt_mass.py 0.10 speed 3:3:0
  step sp3_m70spec_$PR 600 env PROMPT=$PR BT_THETA=0.70 $PY $S/bt_mass.py 0.10 speed 3:3:0
  step sp3_m50spec_$PR 600 env PROMPT=$PR BT_THETA=0.50 $PY $S/bt_mass.py 0.10 speed 3:3:0
done
step sp3_m85spec05_code 600 env PROMPT=code BT_THETA=0.85 $PY $S/bt_mass.py 0.05 speed 3:3:0
step sp3_m70spec05_code 600 env PROMPT=code BT_THETA=0.70 $PY $S/bt_mass.py 0.05 speed 3:3:0
peer unlock > /dev/null 2>&1; log "lock released (day_speed3)"
