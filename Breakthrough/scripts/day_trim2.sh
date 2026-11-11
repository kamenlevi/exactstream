#!/bin/sh
# Breakthrough Thu 2026-10-22 01:35: B12 + MTP below 12 GB of peak FOOTPRINT (13.1 now): MLX cache cap 64 MB, prefill
# chunks of 256, sidecar C_l 12, base C_m 46; B12 eager with cap + chunks. Launched once my session holds the lock.
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
keepalive || { log "no lock to reuse (trim2)"; exit 1; }
log "lock taken (trim2)"
export BIGMOE_MTP_C=32 PROMPT=code
step t2_mtp_cap 400 env B12_CACHE_MB=64 $PY $S/bt_b12.py 0.10 spec3:3:0
step t2_mtp_cap_pf256 400 env B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 spec3:3:0
step t2_mtp_cap_pf256_cl12 400 env B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 B12_CL=12 $PY $S/bt_b12.py 0.10 spec3:3:0
step t2_mtp_cap_pf256_cl12_cm46 400 env B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 B12_CL=12 B12_CM=46 $PY $S/bt_b12.py 0.10 spec3:3:0
step t2_eager_cap_pf256 300 env B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 speed
step t2_mtp_cap_pf256_prose 400 env PROMPT=prose B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 spec3:3:0
# NewIdeas 01:55: v3's exact prefill reads every expert a chunk uses (cache ignored): chunk size sets prefill speed
# (~585 tok/s at 2048 vs ~127 at 256, est.). Prefill tok/s + peak footprint on a ~2K-token prompt at three chunks.
for C in 2048 512 256; do step t2_long_pf$C 400 env PROMPT=long B12_CACHE_MB=64 B12_PREFILL_CHUNK=$C BT_N=48 $PY $S/bt_b12.py 0.10 speed; done
peer unlock > /dev/null 2>&1; log "lock released (trim2)"
