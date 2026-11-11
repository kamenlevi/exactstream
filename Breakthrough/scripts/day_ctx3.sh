#!/bin/sh
# Breakthrough Thu 2026-10-22 04:58: RAM per context, part 3 (64K): B12 + MTP and eager after a ~65K-token prefill
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
keepalive || { log "no lock to reuse (ctx3)"; exit 1; }
log "lock taken (ctx3)"
export B12_CACHE_MB=64 D1I_ATTN=512 D1I_MOE=4096 D1I_SPLIT=1 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32
# NewIdeas 05:55: union reads through raw preads (GIL-free pool) + TrainLab's copy-free GPU rebuild (RB2) instead of the CPU
# rebuild: CPU path 177 tokens 3.83 s, 326 tokens 4.23 s (05:49); old GPU path 326 tokens 5.37 s. Shas must be equal.
step pfb_150_gpurb2 300 env PROMPT=long B12_LONG_CHARS=450 D1I_RUNS=2 BT_N=32 BIGMOE_TIER_PF_GPU=1 D1I_RB2=1 $PY $S/bt_b12.py 0.10 d1i
step pfb_code_gpurb2 300 env PROMPT=code D1I_RUNS=2 BT_N=32 BIGMOE_TIER_PF_GPU=1 D1I_RB2=1 $PY $S/bt_b12.py 0.10 d1i
# 05:55: the 32K footprint sits 1.8 GB above MLX's peak active only DURING prefill (after decode both are 11.5): where?
step fps_eager32k 600 env PROMPT=ctx32k B12_FP_SAMPLE=1 D1I_RUNS=1 BT_N=16 $PY $S/bt_b12.py 0.10 d1i
step ng0_he_mtp64k 900 env PROMPT=ctx64k MTPLX_NGRAM_HOT_MB=0 D1I_FRESH=1 B12_SPEC_WARM=0 BT_N=64 $PY $S/bt_b12.py 0.10 spec3:3:0
step ng0_eager64k 900 env PROMPT=ctx64k MTPLX_NGRAM_HOT_MB=0 D1I_RUNS=1 BT_N=64 $PY $S/bt_b12.py 0.10 d1i
log "ctx3 done"
# 06:18: per-phase peaks at 32K: attn 512 -> attention binds (12.23 GB); attn 256 -> MoE 4096 binds (11.88); MoE 2048 -> 11.47: both
step ctx32k_a256_m2048 600 env PROMPT=ctx32k D1I_ATTN=256 D1I_MOE=2048 D1I_RUNS=1 BT_N=64 $PY $S/bt_b12.py 0.10 d1i
log "ctx3 extra done"
