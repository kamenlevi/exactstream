#!/bin/sh
# Breakthrough Sun 2026-10-25: L2-10 64K prefill footprint peak: stock vs one-pass QSA growth (qsa_rows_target) vs the
# prefill attention phase (sparse lane past 32K), MTP head, s_moe 2048; one process per variant (~6 min each)
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
  log "done $name (rc $rc): $(grep -E '^pk64 |Error|Traceback|bigmoe:' "$OUT/$name.txt" | tail -4 | cut -c1-160 | tr '\n' ' ')"
  return $rc
}
keepalive || { log "no lock to reuse (pk64)"; exit 1; }
log "lock taken (pk64)"
export B12_CACHE_MB=64 PK_N=65000
step base 600 env PK_TARGET=0 $PY $S/pk64.py
step tgt 600 env PK_TARGET=1 $PY $S/pk64.py
step lane 600 env PK_LANE=1 $PY $S/pk64.py
step tgt_lane 600 env PK_TARGET=1 PK_LANE=1 $PY $S/pk64.py
step base_mtp 660 env PK_MTP=1 $PY $S/pk64.py
step tgt_lane_mtp 660 env PK_TARGET=1 PK_LANE=1 PK_MTP=1 $PY $S/pk64.py
log "pk64 done"
