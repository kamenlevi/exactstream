#!/bin/sh
# Breakthrough Sat 2026-10-24: fold check - B12LL + MTP on v3f (as judged) vs v3g + JOINT + SB1 + trims (C_m 51 and 59): shas, tok/s, footprint
B=$HOME/Breakthrough; OUT=$B/data/fold; LOG=$B/data/fold/summary.txt
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
keepalive || { log "no lock to reuse (fold)"; exit 1; }
log "lock taken (fold)"
export B12_CACHE_MB=64 B12_NORM=mass BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BT_N=128 B12_LATE3=2,10
for P_ in code prose; do
  step base_${P_} 300 env PROMPT=$P_ $PY $S/bt_b12.py 0.10 spec3:3:0
  step new_${P_} 300 env PROMPT=$P_ $PY $S/.bt_b12_next.py 0.10 spec3:3:0
  step new59_${P_} 300 env PROMPT=$P_ B12_CM=59 $PY $S/.bt_b12_next.py 0.10 spec3:3:0
done
log "fold done"
