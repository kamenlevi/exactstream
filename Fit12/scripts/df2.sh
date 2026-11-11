#!/bin/sh
# Fit12 DF2 (2026-10-23): DFlash2 on B12LL at 12 GB on new-code prompts (code, lru, csv) + 2 GSM8K items, 256 tokens; outputs saved.
# Pre-registered in ~/Fit12/PLAN.md (DF2). Launch (background task): CLAUDE_PID=23445 sh ~/Fit12/scripts/df2.sh
F=$HOME/Fit12; OUT=$F/data/df2; LOG=$F/logs/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$F/scripts/fit_b12.py
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 BT_N=256 F12_SAVE=$HOME/Fit12/data/df2 F12_ASYNC=1 BIGMOE_PF_GROUP=64
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
  log "done $name (rc $rc): $(grep -cE '^\[.*(k=[0-9]+( AW)?|tf[cg][0-9]|code|lru|csv):' "$OUT/$name.txt") arms$(grep -E 'Error|Traceback' "$OUT/$name.txt" | tail -1 | cut -c1-200)"
  return $rc
}
keepalive || { log "DF2: no lock"; exit 1; }
log "DF2 lock taken; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
D=$F/scripts/fit_dfl2.py; A3=code:3,lru:3,csv:3,tfg1:3,tfg2:3; A=code,lru,csv,tfg1,tfg2
$PY -m py_compile $S $D || { log "DF2: scripts do not compile"; peer unlock; exit 1; }
step ref_nat 600 env F12_ARMS=$A3 $PY $S ref_nat
step ref_b16 600 env F12_BUDGET=b16 F12_ARMS=$A3 $PY $S ref_b16
step df8b16 600 env F12_VW=8 F12_BUDGET=b16 F12_ARMS=$A $PY $D df8b16
step df8b20 600 env F12_VW=8 F12_BUDGET=b20 F12_ARMS=$A $PY $D df8b20
log "DF2 done; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
peer unlock >/dev/null 2>&1
