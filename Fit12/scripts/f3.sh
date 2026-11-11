#!/bin/sh
# Fit12 F3 (2026-10-23): budgeted MTP width: b16 with fixed k=3 vs spec125 AdaptiveWidth (max 3 / 5 / 7) and KC 20 / 15 / 10,
# on four prompts (code, prose, math, write). Pre-registered in ~/Fit12/PLAN.md (F3).
# Launch after the lock is granted (background task): CLAUDE_PID=23445 ~/Fit12/scripts/f2.sh
F=$HOME/Fit12; OUT=$F/data/f3; LOG=$F/logs/summary.txt
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
  log "done $name (rc $rc): $(grep -cE '^\[.*k=[0-9]+( AW)?:' "$OUT/$name.txt") arms$(grep -E 'Error|Traceback' "$OUT/$name.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "F3: no lock"; exit 1; }
log "F3 lock taken; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
step a20nat 300 env F12_ARMS=code:3,prose:3,math:3,write:3 $PY $S a20nat
step a20f3 300 env F12_BUDGET=b16 F12_ARMS=code:3,prose:3,math:3,write:3 $PY $S a20f3
step a20aw3 300 env F12_BUDGET=b16 F12_AW=1 F12_ARMS=code:3,prose:3,math:3,write:3 $PY $S a20aw3
step a15f5 300 env F12_KC=15 BIGMOE_MAXK=92 F12_BUDGET=b16 F12_ARMS=code:5,prose:5,math:5,write:5 $PY $S a15f5
step a15aw5 300 env F12_KC=15 BIGMOE_MAXK=92 F12_BUDGET=b16 F12_AW=1 F12_ARMS=code:5,prose:5,math:5,write:5 $PY $S a15aw5
step a10aw7 300 env F12_KC=10 F12_BUDGET=b16 F12_AW=1 F12_ARMS=code:7,prose:7,math:7,write:7 $PY $S a10aw7
log "F3 done; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
peer unlock >/dev/null 2>&1
