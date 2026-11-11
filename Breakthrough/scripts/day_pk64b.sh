#!/bin/sh
# Breakthrough Sun 2026-10-25: L2-10 batch 2, IN THE SHIPPED ENV (batch 1 lacked it): grouped tier prefill
# BIGMOE_TIER_PF_GROUP=64 (2 x 64-expert buffers instead of the ~512-expert full-union scratch per layer call),
# BIGMOE_PF_GROUP=64 + MTP_C 32 for the MTP layer, ASYNC_EVERY 1, cache cap 64 MB. Variants: no lane vs the prefill
# attention lane (+ one-pass growth, s_moe 2048, a 1 GB MLX cache), plain and MTP; the lane run also times decode with
# the opt-in flash-skip decode lane off/on (PK_DECAB). One process per variant (~5 min each). First: lane_ref.py
# (NewIdeas: is the lane the reference behaviour? generate_ar vs b12_prefill 512/512 lane on/off at 4.2K with the
# crossovers lowered to 2049, plus flash-skip decode vs dense decode).
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
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^pk64 |^lane_ref [A-E]_.* vs|Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -5 | cut -c1-160 | tr '\n' ' ')"
  return $rc
}
keepalive || { log "no lock to reuse (pk64b)"; exit 1; }
log "lock taken (pk64b)"
export B12_CACHE_MB=64 PK_N=65000 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BIGMOE_ASYNC_EVERY=1
step lane_ref 600 $PY $S/lane_ref.py
step g_base 600 env PK_TAG=g64_base $PY $S/pk64.py
step g_lane 720 env PK_TAG=g64_lane PK_LANE=1 PK_DECAB=1 $PY $S/pk64.py
step g_lane_m2048 600 env PK_TAG=g64_lane_m2048 PK_LANE=1 PK_MOE=2048 $PY $S/pk64.py
step g_lane_c1024 600 env PK_TAG=g64_lane_c1024 PK_LANE=1 B12_CACHE_MB=1024 $PY $S/pk64.py
step g_mtp 660 env PK_TAG=g64_mtp PK_MTP=1 $PY $S/pk64.py
step g_lane_mtp 660 env PK_TAG=g64_lane_mtp PK_LANE=1 PK_MTP=1 $PY $S/pk64.py
log "pk64b done"
