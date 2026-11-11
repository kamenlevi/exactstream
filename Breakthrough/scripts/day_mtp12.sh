#!/bin/sh
# Breakthrough Thu 2026-10-22 01:05: fit B12 + MTP into 12 GB (MTP adds ~2 GB of peak): HC8 (default) with the MTP
# expert pool at 64 / 32 slots and FRSPEC on / off; B8-fast + MTP with HC8. Launched once my session holds the lock.
B=$HOME/Breakthrough; OUT=$B/data/day; LOG=$B/data/expS/summary.txt
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
keepalive || { log "no lock to reuse (mtp12)"; exit 1; }
log "lock taken (mtp12)"
for PR in code prose; do
  step m12_b12_c64_$PR 400 env PROMPT=$PR $PY $S/bt_b12.py 0.10 spec3:3:0
  step m12_b12_c32_$PR 400 env PROMPT=$PR BIGMOE_MTP_C=32 $PY $S/bt_b12.py 0.10 spec3:3:0
  step m12_b8f_$PR 400 env PROMPT=$PR B12_W=3 $PY $S/bt_b12.py 0.05 spec3:3:0
done
step m12_b12_c32_nofr_code 400 env PROMPT=code BIGMOE_MTP_C=32 BIGMOE_FRSPEC=0 $PY $S/bt_b12.py 0.10 spec3:3:0
step m12_b12_c32_cache512_code 400 env PROMPT=code BIGMOE_MTP_C=32 B12_CACHE_MB=512 $PY $S/bt_b12.py 0.10 spec3:3:0
peer unlock > /dev/null 2>&1; log "lock released (mtp12)"
