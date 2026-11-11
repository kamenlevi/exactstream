#!/bin/sh
# Breakthrough Wed 2026-10-21: frugal + look-ahead prefetch (BT_PF=1: the next layers' routers on the current hidden state
# predict experts that are read into spare slots while the current layer computes; waited ranks 0-1 wait on in-flight
# prefetches, lower ranks use whatever arrived). Speed vs plain frugal, MTP 3:3:0, 10 % code + prose, 5 % code;
# GSM8K-50 with prefetch at 10 %. Waits for day_speed3.sh (same CLAUDE_PID), then one quiet lock (~20 min).
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
  log "done $name (rc $rc): $(grep -E '^\[|^bt_|^125B|^[A-Za-z_0-9]+ +C=|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -2 | tr '\n' ' ' | cut -c1-400)"
  return $rc
}
while pgrep -f "scripts/day_speed3.sh" > /dev/null; do sleep 20; done   # day_speed3.sh (same CLAUDE_PID)
peer lock quiet --queue 45m "Breakthrough: frugal + look-ahead prefetch, speed + GSM8K-50 (~20 min)" > /dev/null 2>&1 || { log "no lock (day_speed4)"; exit 1; }
g=0
while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 70 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
log "lock taken (day_speed4)"
for PR in code prose; do
  step sp4_fspec_$PR 600 env PROMPT=$PR $PY $S/bt_frugal_spec.py 0.10 speed 3:3:0
  step sp4_fspecpf_$PR 600 env PROMPT=$PR BT_PF=1 $PY $S/bt_frugal_spec.py 0.10 speed 3:3:0
done
step sp4_fspecpf05_code 600 env PROMPT=code BT_PF=1 BT_SPARE=6 $PY $S/bt_frugal_spec.py 0.05 speed 3:3:0
step sp4_fpf_gsm50 1500 env OFFSET=0 BT_PF=1 BT_TAG=r10pf_330 $PY $S/bt_frugal_spec.py 0.10 gsm50 3:3:0
peer unlock > /dev/null 2>&1; log "lock released (day_speed4)"
