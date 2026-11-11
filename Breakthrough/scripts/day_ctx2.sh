#!/bin/sh
# Breakthrough Thu 2026-10-22 04:58: RAM per context, part 2 (64K): B12 + MTP and eager after a ~65K-token prefill
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
keepalive || { log "no lock to reuse (ctx2)"; exit 1; }
log "lock taken (ctx2)"
export B12_CACHE_MB=64 D1I_ATTN=512 D1I_MOE=4096 D1I_SPLIT=1 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32
# D1k mechanism (SSD only, ~15 s): do F_NOCACHE reads hit pages staged by F_RDADVISE on a cached fd?
step la_advise 90 $PY -I $S/la_advise_test.py 30 200
# D1I_HEAD_EACH=1 (MTP head fed per MoE chunk): exact? sha must equal ctx_mtp_long dc429bc75b / ctx_mtp32k 51c7a0cd52;
# 32K footprint below 13.6 GB?
step he_mtp_long 420 env PROMPT=long D1I_FRESH=1 D1I_HEAD_EACH=1 BT_N=128 $PY $S/bt_b12.py 0.10 spec3:3:0
step he_mtp32k 600 env PROMPT=ctx32k D1I_FRESH=1 D1I_HEAD_EACH=1 B12_SPEC_WARM=0 BT_N=64 $PY $S/bt_b12.py 0.10 spec3:3:0
# NewIdeas 05:11: small-call prefill (53 tokens 3.2 s, 326 ~5 s, cold) may be ~40-50 % of an agentic turn: where does
# it go? d1i task, cold + repeat, PF counters (t_sync = attention + router wait, t_moe = reads + rebuild in prefill_call);
# then the same with D1k look-ahead reads (B12_PF_LA=1: shas must not change); 'long' at 450 chars ~ a median call
for P_ in prose code; do
  step pfb_$P_ 300 env PROMPT=$P_ D1I_RUNS=2 BT_N=32 $PY $S/bt_b12.py 0.10 d1i
  step pfb_${P_}_la 300 env PROMPT=$P_ D1I_RUNS=2 BT_N=32 B12_PF_LA=1 $PY $S/bt_b12.py 0.10 d1i
done
step pfb_150 300 env PROMPT=long B12_LONG_CHARS=450 D1I_RUNS=2 BT_N=32 $PY $S/bt_b12.py 0.10 d1i
step pfb_150_la 300 env PROMPT=long B12_LONG_CHARS=450 D1I_RUNS=2 BT_N=32 B12_PF_LA=1 $PY $S/bt_b12.py 0.10 d1i
# 05:07: at 32K the footprint sits 1.8 GB above MLX's peak active (short context: 1.0): MTPLX's n-gram hot-row LRU
# (MTPLX_NGRAM_HOT_MB, default 1024) fills with a long prompt? Off = pread path (short context: 39.9 vs 40.2 tok/s, same sha)
step ng0_eager32k 600 env PROMPT=ctx32k MTPLX_NGRAM_HOT_MB=0 D1I_RUNS=1 BT_N=64 $PY $S/bt_b12.py 0.10 d1i
step ng0_he_mtp32k 600 env PROMPT=ctx32k MTPLX_NGRAM_HOT_MB=0 D1I_FRESH=1 D1I_HEAD_EACH=1 B12_SPEC_WARM=0 BT_N=64 $PY $S/bt_b12.py 0.10 spec3:3:0
step pfb_prose_gpu 300 env PROMPT=prose D1I_RUNS=2 BT_N=32 BIGMOE_TIER_PF_GPU=1 $PY $S/bt_b12.py 0.10 d1i
step pfb_code_gpu 300 env PROMPT=code D1I_RUNS=2 BT_N=32 BIGMOE_TIER_PF_GPU=1 $PY $S/bt_b12.py 0.10 d1i
log "ctx2 done"
