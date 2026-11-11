#!/bin/sh
# D1i measurements (TrainLab, 10-22): one quiet lock, one process per variant (peak footprint is a per-process lifetime
# figure). B12 eager (bt_b12 load_b12(0.10) defaults), PROMPT=long (~2K tokens), MLX cache cap 64 MB, 64 decode tokens.
# Task d1i: a cold call right after load (pools as loaded), then a repeat call. Logs: out/<step>.txt, out/summary.txt.
T=$HOME/TrainLab/d1i; OUT=$T/out; mkdir -p $OUT
PY=$HOME/BigMoE/.venv/bin/python; D=$T/bt_b12_d1i.py
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=21600
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
LOG=$OUT/summary.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
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
  for _ in 1 2 3 4 5 6; do sleep 0.5; for q in $P $(pgrep -P $P); do taskpolicy -B -p $q 2>/dev/null; done; done   # normal priority
  i=0; rc=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 120)) -eq 0 ]; then keepalive || { log "lost the lock during $name"; kill $P; sleep 5; kill -9 $P 2>/dev/null; rc=125; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; rc=124; }
  wait $P; [ $rc -eq 0 ] && rc=$?
  log "done $name (rc $rc): $(grep -E '^\[|Traceback|Error|bigmoe:' "$OUT/$name.txt" | tail -3 | tr '\n' ' ' | cut -c1-900)"
  return $rc
}
log "batch c continues in the held lock"
# launched under taskpolicy -b (the queue wait is idle); measured steps run at normal priority, I/O included
# (taskpolicy -B on this shell does not reach its children: each step's process is lifted right after it starts)
export PROMPT=long B12_CACHE_MB=64 BT_N=64
log "batch c: instrumented peak (before / in the MoE), GPU rebuild, one 4096 chunk"
step diag_g64 300 env D1I_NOLOGITS=1 BIGMOE_TIER_PF_GROUP=64 $PY $D 0.10 d1i
step g64gpu 300 env D1I_NOLOGITS=1 BIGMOE_TIER_PF_GROUP=64 BIGMOE_TIER_PF_GPU=1 $PY $D 0.10 d1i
step g64_c4096 300 env D1I_NOLOGITS=1 BIGMOE_TIER_PF_GROUP=64 B12_PREFILL_CHUNK=4096 $PY $D 0.10 d1i
step g64gpu_c4096 300 env D1I_NOLOGITS=1 BIGMOE_TIER_PF_GROUP=64 BIGMOE_TIER_PF_GPU=1 B12_PREFILL_CHUNK=4096 $PY $D 0.10 d1i
step ref_code 300 env PROMPT=code $PY $D 0.10 d1i
step g64_code 300 env PROMPT=code D1I_NOLOGITS=1 BIGMOE_TIER_PF_GROUP=64 $PY $D 0.10 d1i
peer unlock > /dev/null 2>&1; log "lock released"
