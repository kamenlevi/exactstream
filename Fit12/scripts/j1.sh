#!/bin/sh
# Fit12 J1 (2026-10-23): why JIT prefetch loses at C_m 51 (on-time vs late vs victims), and the value of 8 more resident slots.
# Pre-registered in ~/Fit12/PLAN.md (J1). Launch (background task): CLAUDE_PID=23445 sh ~/Fit12/scripts/j1.sh
F=$HOME/Fit12; OUT=$F/data/j1; LOG=$F/logs/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$F/scripts/fit_b12.py
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 BT_N=128 F12_ASYNC=1 BIGMOE_PF_GROUP=64
LOCKF=$HOME/.claude-peers/locks/${CLAUDE_PID:-23445}.json
log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
memok() { free=$(memory_pressure | awk '/free percentage/ {gsub("%","",$NF); print $NF}'); [ "${free:-0}" -ge 30 ]; }
step() {   # step NAME SECONDS_LIMIT CMD...
  name=$1; lim=$2; shift 2
  keepalive || { log "lock lost before $name"; return 125; }
  memok || { log "memory below 30 % free before $name: skipped"; return 126; }
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -cE '^\[.*(k=[0-9]+( AW)?|tf[cg][0-9]|code|lru|csv|p2k|p8k):' "$OUT/$name.txt") arms$(grep -E 'Error|Traceback' "$OUT/$name.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "J1: no lock"; exit 1; }
log "J1 lock taken; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
$PY -m py_compile $S || { log "J1: scripts do not compile"; peer unlock; exit 1; }
step j12 300 env F12_JIT=12 F12_ARMS=code:3,prose:3 $PY $S j12
step j12b16 300 env F12_JIT=12 F12_BUDGET=b16 F12_BUDGET_JIT=1 F12_ARMS=code:3,prose:3 $PY $S j12b16
step ref59 300 env B12_CM=59 B12_CL=20 F12_ARMS=code:3,prose:3 $PY $S ref59
log "J1 done; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
peer unlock >/dev/null 2>&1
