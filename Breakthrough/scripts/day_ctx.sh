#!/bin/sh
# Breakthrough Thu 2026-10-22 04:40: RAM per context for the release line (NewIdeas TASKS "Footprint vs CONTEXT"):
# B12 + MTP after a 37K and a ~65K prefill with the decoupled prefill run inside generate (D1I_FRESH: no spec125 state
# deep copy = the server's footprint), and B12 eager at ~65K. Fixed prompts data/prompts/ctx32k|ctx64k. Held lock.
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
keepalive || { log "no lock to reuse (ctx)"; exit 1; }
log "lock taken (ctx)"
export B12_CACHE_MB=64 D1I_ATTN=512 D1I_MOE=4096 D1I_SPLIT=1 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32
step ctx_mtp_long 420 env PROMPT=long D1I_FRESH=1 BT_N=128 $PY $S/bt_b12.py 0.10 spec3:3:0
step ctx_mtp32k 600 env PROMPT=ctx32k D1I_FRESH=1 B12_SPEC_WARM=0 BT_N=64 $PY $S/bt_b12.py 0.10 spec3:3:0
# smoke 04:43: 32K eager = 12.5 GB (binding term: the MoE transient over 4096 rows on top of ~0.9 GB KV): MoE chunk 2048
step ctx32k_moe2048 600 env PROMPT=ctx32k D1I_MOE=2048 D1I_RUNS=1 BT_N=64 $PY $S/bt_b12.py 0.10 d1i
# TrainLab's v3 streaming ported (opt-in): same sha as sm_long_eager (e0420271a0)? footprint (~11.8 ring 6), prefill tok/s
step st_long_b12_r6 420 env PROMPT=long D1I_STREAM=3 D1I_RING=6 D1I_RB2=1 $PY $S/bt_b12.py 0.10 speed
step st_long_b8f_r4 420 env PROMPT=long B12_W=3 D1I_STREAM=3 D1I_RING=4 D1I_RB2=1 $PY $S/bt_b12.py 0.05 speed
# NewIdeas 04:56: a full sweep reads all 512 experts per layer however short the prompt: crossover vs the union read
for CH in 3500 5500; do
  step st_cross_${CH}_union 300 env PROMPT=long B12_LONG_CHARS=$CH $PY $S/bt_b12.py 0.10 speed
  step st_cross_${CH}_r6 300 env PROMPT=long B12_LONG_CHARS=$CH D1I_STREAM=3 D1I_STREAM_MIN=0 D1I_RING=6 D1I_RB2=1 $PY $S/bt_b12.py 0.10 speed
done
log "ctx done"
# 05:01: B12 + MTP 11.3 GB at 2.6K without the state copy; TrainLab ring 4 + RB2 = +0.6 GB eager: does MTP + ring 4 fit 12 GB?
step st_mtp_long_r4 420 env PROMPT=long D1I_FRESH=1 BT_N=128 D1I_STREAM=3 D1I_RING=4 D1I_RB2=1 $PY $S/bt_b12.py 0.10 spec3:3:0
log "ctx extra done"
