#!/bin/sh
# Breakthrough Thu 2026-10-22 04:25: smoke test before the Eval suite's B12 verdict, with TrainLab's D1i decoupled prefill
# (attention chunks 512, MoE once per 4096 tokens, grouped tier prefill 64) and, for MTP, BIGMOE_PF_GROUP=64 + the d1i
# prefilled state: fluency, peak footprint (<= 12 GB?), prefill + decode tok/s, GSM8K-20 vs 19/20, and the 32K RAM line.
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
keepalive || { log "no lock to reuse (smoke)"; exit 1; }
log "lock taken (smoke)"
export B12_CACHE_MB=64 D1I_ATTN=512 D1I_MOE=4096 D1I_SPLIT=1 BIGMOE_TIER_PF_GROUP=64
step sm_long_eager 420 env PROMPT=long $PY $S/bt_b12.py 0.10 speed
# the Eval suite's own decode path (MTPLX generate_ar, greedy, prefill at MTPLX_PREFILL_CHUNK_SIZE), without / with the
# PF1 flags its env.py forces for non-frugal variants
step sm_ar_c512 420 env MTPLX_PREFILL_CHUNK_SIZE=512 $PY $S/bt_b12_ar.py 0.10 96
step sm_ar_c512_pf1 420 env MTPLX_PREFILL_CHUNK_SIZE=512 BT_AR_PF=1 $PY $S/bt_b12_ar.py 0.10 96
step sm_mtp_code 420 env PROMPT=code BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 $PY $S/bt_b12.py 0.10 spec3:3:0
# task sanity of the MTP decode the release would ship (the suite decodes AR only)
step sm_gsm20_mtp 900 env B12_GSM_MTP=3:3:0 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 $PY $S/bt_b12.py 0.10 gsm20
step sm_mtp_prose 420 env PROMPT=prose BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 $PY $S/bt_b12.py 0.10 spec3:3:0
step sm_ctx32k 600 env PROMPT=ctx32k D1I_RUNS=1 BT_N=64 $PY $S/bt_b12.py 0.10 d1i
log "smoke done"
