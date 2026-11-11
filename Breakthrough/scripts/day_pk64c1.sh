#!/bin/sh
# Breakthrough Sun 2026-10-25: L2-10 batch 3, shipped env + the prefill lane: persistent grouped-prefill buffers
# (BIGMOE_TIER_PF_PERSIST=1, v3i tier.py opt-in, exact: sha must equal g64_lane's 56c4e038e8 / 1ccb5e8c59) at s_moe
# 4096 and 2048, and the MTP path. Batch 2: lane 12.69 GB, lane + s_moe 2048 12.32 GB (line: < 12.0); the peak sits in
# the post phase right after each MoE, where ~1.0-1.4 GB of just-freed GPU memory is not yet reclaimed.
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
  wait $P; w=$?; [ $rc -eq 0 ] && rc=$w
  log "done $name (rc $rc): $(grep -E '^pk64 |Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -4 | cut -c1-160 | tr '\n' ' ')"
  return $rc
}
keepalive || { log "no lock to reuse (pk64c1)"; exit 1; }
log "lock taken (pk64c1)"
export B12_CACHE_MB=64 PK_N=65000 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BIGMOE_ASYNC_EVERY=1 PK_LANE=1 BIGMOE_TIER_PF_PERSIST=1
step g_lane_ps 600 env PK_TAG=g64_lane_ps $PY $S/pk64.py
log "pk64c1 done"
