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
  log "done $name (rc $rc): $(grep -E '^\[|^bt_|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
  return $rc
}
while [ "$(date +%H%M)" -lt 0845 ]; do sleep 60; done
# stop night chain 10 right after one of its probes finished (it then sleeps 180 s without a lock)
N0=$(grep -c "done tfvar8_" "$LOG")
while pgrep -f "scripts/night_chain10.sh" > /dev/null && [ "$(grep -c 'done tfvar8_' "$LOG")" -eq "$N0" ]; do sleep 10; done
pkill -f "scripts/night_chain10.sh"; log "day_speed: night chain 10 stopped for the daytime speed batch"
peer lock quiet --queue 45m "Breakthrough: speed of frugal quality policies at 10 % / 5 % (~25 min)" > /dev/null 2>&1 || { log "no lock (day_speed)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
log "lock taken (day_speed)"
for P in code prose; do
  step sp_frugal_$P 420 env PROMPT=$P $PY $S/bt_bild.py 0.10 -1 speed
  step sp_exact_$P 600 env PROMPT=$P $PY $S/bt_bild.py 0.10 99 speed
  step sp_bild10_$P 600 env PROMPT=$P $PY $S/bt_bild.py 0.10 0.1 speed
  step sp_bild50_$P 600 env PROMPT=$P $PY $S/bt_bild.py 0.10 0.5 speed
  step sp_m85_$P 600 env PROMPT=$P BT_THETA=0.85 $PY $S/bt_mass.py 0.10 speed ar
  step sp_m70_$P 600 env PROMPT=$P BT_THETA=0.70 $PY $S/bt_mass.py 0.10 speed ar
  step sp_ada_$P 600 env PROMPT=$P BT_ADA=0.5,0.9,0.1 $PY $S/bt_mass.py 0.10 speed ar
  step sp_m85spec_$P 600 env PROMPT=$P BT_THETA=0.85 $PY $S/bt_mass.py 0.10 speed 3:3:0
  step sp_m70spec_$P 600 env PROMPT=$P BT_THETA=0.70 $PY $S/bt_mass.py 0.10 speed 3:3:0
done
step sp_m85spec05_code 600 env PROMPT=code BT_THETA=0.85 $PY $S/bt_mass.py 0.05 speed 3:3:0
peer unlock > /dev/null 2>&1; log "lock released (day_speed)"
