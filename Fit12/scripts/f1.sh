#!/bin/sh
# Fit12 F1 (2026-10-23): B12LL speed levers under one quiet lock (pre-registered in ~/Fit12/PLAN.md). Launch after the
# lock is granted: nohup ~/Fit12/scripts/f1.sh > ~/Fit12/logs/f1.log 2>&1 &
F=$HOME/Fit12; OUT=$F/data/f1; LOG=$F/logs/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$F/scripts/fit_b12.py
mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 BT_N=128
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
  log "done $name (rc $rc): $(grep -E '^\[|Error|Traceback|bigmoe:' "$OUT/$name.txt" | grep -v prefill | tail -3 | tr '\n' ' ' | cut -c1-600)"
  return $rc
}
keepalive || { log "F1: no lock"; exit 1; }
log "F1 lock taken"
V3G=$F/v3g/ext
( cmake -S $V3G -B $V3G/build -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake -DPython_EXECUTABLE=$PY \
    -DCMAKE_BUILD_TYPE=Release && cmake --build $V3G/build -j 4 ) > $F/logs/build_v3g.log 2>&1 \
  && log "v3g ext built (LFU port)" || log "v3g ext BUILD FAILED (see logs/build_v3g.log)"
step base 420 env F12_ARMS=code:3,prose:3,code:2,prose:2,code:1 $PY $S base
step async 240 env F12_ASYNC=1 F12_ARMS=code:3,prose:3 $PY $S async
step jit12 240 env F12_JIT=12 F12_ARMS=code:3,prose:3 $PY $S jit12
step b16 240 env F12_BUDGET=b16 F12_ARMS=code:3,prose:3 $PY $S b16
step b16jit 240 env F12_BUDGET=b16 F12_BUDGET_JIT=1 F12_JIT=12 F12_ARMS=code:3,prose:3 $PY $S b16jit
step jit6 240 env F12_JIT=6 F12_ARMS=code:3,prose:3 $PY $S jit6
if [ -f $V3G/build/bigmoe_ext.cpython-312-darwin.so ]; then
  step lfu 240 env F12_V3F=$F/v3g BIGMOE_EVICT_LFU=0.99 F12_ARMS=code:3,prose:3 $PY $S lfu
  step lfu_off 240 env F12_V3F=$F/v3g F12_ARMS=code:3 $PY $S lfu_off
fi
log "F1 done"
peer unlock >/dev/null 2>&1
