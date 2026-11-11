#!/bin/sh
# Breakthrough Thu 2026-10-22 06:30: D1k skip-residents sizing (NewIdeas): how much of a call's expert union do the decode
# pools already hold (MSB-3 part / both parts)? GSM8K-10 = 10 different prompts, each after the previous decode; + 177 tokens cold / repeat.
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
keepalive || { log "no lock to reuse (res)"; exit 1; }
log "lock taken (res)"
export B12_CACHE_MB=64 BIGMOE_TIER_PF_GROUP=64 BIGMOE_TIER_PF_RESCOUNT=1
step res_gsm10 600 $PY $S/bt_b12.py 0.10 gsm10
step res_177 300 env PROMPT=long B12_LONG_CHARS=450 D1I_RUNS=2 BT_N=128 $PY $S/bt_b12.py 0.10 d1i
# NewIdeas / TrainLab: C_m 56 (load_b12(0.11): C_l 22) for B12 + MTP under 12 GB? my tree, no-copy server path, 2.6K + code
step res_mtp56_long 420 env PROMPT=long B12_CM=56 D1I_FRESH=1 BT_N=128 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 $PY $S/bt_b12.py 0.10 spec3:3:0
step res_mtp56_code 420 env PROMPT=code B12_CM=56 D1I_FRESH=1 BT_N=128 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 $PY $S/bt_b12.py 0.10 spec3:3:0
log "res done"
