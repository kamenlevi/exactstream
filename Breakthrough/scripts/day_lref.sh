#!/bin/sh
# Breakthrough Sun 2026-10-25: lane_ref.py alone (NewIdeas: is the lane the reference behaviour?), shipped env
# lane off vs on, 32K + 60K docs (kv8_long tasks, records long_<L>_g64[_lane].json); then L2-06's lost 60K 8-bit arm in
# the env of the first runs (no grouped prefill), comparable to long_60000.json's bf16 arm.
B=$HOME/Breakthrough; OUT=$B/data/pk64; LOG=$B/data/pk64/summary.txt
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
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w     # (was rc=$? after the test: always 0)
  log "done $name (rc $rc): $(grep -E '^long L=[0-9_a-z]+ \(|^lane_ref [A-D]_.* vs|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -4 | cut -c1-160 | tr '\n' ' ')"
  return $rc
}
keepalive || { log "no lock to reuse (lref)"; exit 1; }
export B12_CACHE_MB=64 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_ASYNC_EVERY=1
step lane_ref 600 $PY $S/lane_ref.py
log "lref done"
