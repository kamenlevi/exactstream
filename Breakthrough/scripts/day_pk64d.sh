#!/bin/sh
# Breakthrough Mon 2026-10-26: L2-10r, the last exact lever (pre-registered by NewIdeas 10-25 14:50): the MoE output copy
# drop, BIGMOE_TIER_PF_SCATTER=1 (v3i tier._prefill_grouped opt-in: each group's outputs scattered into place instead of
# the outs list + concatenate + take). Shipped env + the prefill lane at N 65000. PASS = peak <= 12.39 GB (>= 0.3 GB off
# g64_lane's 12.69) with sha / logits = g64_lane's 56c4e038e8 / 1ccb5e8c59; else L2-10 stops. Then the lane arm again as
# a same-window control (same sha expected; shows the run-to-run spread of the footprint peak).
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
keepalive || { log "no lock to reuse (pk64d)"; exit 1; }
log "lock taken (pk64d)"
export B12_CACHE_MB=64 PK_N=65000 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BIGMOE_ASYNC_EVERY=1 PK_LANE=1
step g_lane_sc 600 env PK_TAG=g64_lane_sc BIGMOE_TIER_PF_SCATTER=1 $PY $S/pk64.py
step g_lane_r2 600 env PK_TAG=g64_lane_r2 $PY $S/pk64.py
log "pk64d done"
