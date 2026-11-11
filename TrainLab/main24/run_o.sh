#!/bin/sh
# D1i measurements (TrainLab, 10-22): one quiet lock, one process per variant (peak footprint is a per-process lifetime
# figure). B12 eager (bt_b12 load_b12(0.10) defaults), PROMPT=long (~2K tokens), MLX cache cap 64 MB, 64 decode tokens.
# Task d1i: a cold call right after load (pools as loaded), then a repeat call. Logs: out/<step>.txt, out/summary.txt.
T=$HOME/TrainLab/main24; OUT=$T/out; mkdir -p $OUT
PY=$HOME/BigMoE/.venv/bin/python; D=$T/bench_prefill.py
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
while pgrep -f "bt_b12_d1i.py|bench_prefill.py" > /dev/null; do sleep 2; done
keepalive || { log "batch o: no held lock"; exit 1; }
log "batch o: BigMoE 24 GB main track (v3, BigMoE's flags), prefill per prompt size, unpatched vs D1i's bitwise parts"
X="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1 BIGMOE_GDN_EXACT=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
PR=prose,code,edit,f2k,f4k
step m24_unpatched 600 env $X $V3 $PY $D 0.25 96 $PR
step m24_g64 600 env $X $V3 BIGMOE_TIER_PF_GROUP=64 $PY $D 0.25 96 $PR
step m24_g64_nolog 600 env $X $V3 BIGMOE_TIER_PF_GROUP=64 D1I_NOLOGITS=1 $PY $D 0.25 96 $PR
step m24_g64_gpu_rb2 600 env $X $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_TIER_PF_GPU=1 D1I_RB2=1 D1I_NOLOGITS=1 $PY $D 0.25 96 $PR
peer unlock > /dev/null 2>&1; log "lock released"
