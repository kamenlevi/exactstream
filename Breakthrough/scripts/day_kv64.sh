#!/bin/sh
# Breakthrough Sun 2026-10-25: build v3i (gate stamps) + smoke sha vs v3g, then L2-06 8-bit KV at 64K, one arm per process
B=$HOME/Breakthrough; OUT=$B/data/kv8; LOG=$B/data/kv8/summary.txt
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
keepalive || { log "no lock to reuse (kv64)"; exit 1; }
log "lock taken (kv64)"
step build_v3i 400 sh -c "cd $B/v3i && cmake -S ext -B ext/build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PY -DNB_CMAKE_DIR=$HOME/BigMoE/.nb215/nanobind/cmake > ext/build_cmake.log 2>&1 && cmake --build ext/build -j 4 > ext/build.log 2>&1 && ls ext/build/*.so"
if ls $B/v3i/ext/build/*.so >/dev/null 2>&1; then
  step smoke_v3i 300 env B12_ENGINE=v3i B12_CACHE_MB=64 B12_NORM=mass BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BT_N=128 B12_LATE3=2,10 PROMPT=code $PY $S/bt_b12.py 0.10 spec3:3:0
fi
for A in bf16 kv:64; do
  step mem64k_$(echo $A | tr : _) 600 env B12_CACHE_MB=64 KV8_N=65000 KV8_ARMS=$A $PY $S/kv8_tf.py
done
log "kv64 done"
