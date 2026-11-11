#!/bin/sh
# Breakthrough Thu 2026-10-22 04:20: MTP's +1.5 GB prefill = the streamed MTP layer's plain _sync_call scratch (every expert a
# chunk uses): BigMoE's exact grouped path BIGMOE_PF_GROUP=64 caps it; footprint + greedy sha vs without. Held lock.
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
keepalive || { log "no lock to reuse (grp)"; exit 1; }
log "lock taken (grp)"
export PROMPT=code B12_CACHE_MB=64 BIGMOE_MTP_C=32
step grp_mtp_c256_off 400 env B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 spec3:3:0
step grp_mtp_c256_g64 400 env B12_PREFILL_CHUNK=256 BIGMOE_PF_GROUP=64 $PY $S/bt_b12.py 0.10 spec3:3:0
step grp_mtp_c2048_g64 400 env B12_PREFILL_CHUNK=2048 BIGMOE_PF_GROUP=64 $PY $S/bt_b12.py 0.10 spec3:3:0
step grp_mtp_long_c512_g64 400 env PROMPT=long B12_PREFILL_CHUNK=512 BIGMOE_PF_GROUP=64 $PY $S/bt_b12.py 0.10 spec3:3:0
# NewIdeas 04:25: BigMoE's LEDGER (M24d/M24e) found PF_GROUP=64 changes the PROSE sha (kernel shapes / sum order): check prose too
step grp_mtp_prose_c256_off 400 env PROMPT=prose B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 spec3:3:0
step grp_mtp_prose_c256_g64 400 env PROMPT=prose B12_PREFILL_CHUNK=256 BIGMOE_PF_GROUP=64 $PY $S/bt_b12.py 0.10 spec3:3:0
peer unlock > /dev/null 2>&1; log "lock released (grp)"
