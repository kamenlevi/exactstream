#!/bin/sh
# Breakthrough Thu 2026-10-22 02:00: prefetch A/B, arm 1 (NewIdeas I8): B12 + MTP code at C=51, baseline vs corrected k=2
# prefetch of both record parts after the demand reads (B12_PF=1, 4 spares from C), interleaved A B A B + prose.
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
keepalive || { log "no lock to reuse (pf)"; exit 1; }
log "lock taken (pf)"
export BIGMOE_MTP_C=32
for r in 1 2; do
  step pfab_base_code_$r 400 env PROMPT=code $PY $S/bt_b12.py 0.10 spec3:3:0
  step pfab_arm1_code_$r 400 env PROMPT=code B12_PF=1 $PY $S/bt_b12.py 0.10 spec3:3:0
done
step pfab_base_prose 400 env PROMPT=prose $PY $S/bt_b12.py 0.10 spec3:3:0
step pfab_arm1_prose 400 env PROMPT=prose B12_PF=1 $PY $S/bt_b12.py 0.10 spec3:3:0
# 02:30: rerun of the lost prefill tok/s rows (~2K-token prompt, eager) + the cache cap now applied after load + the
# n-gram hot cache off on MTP (does it count in the footprint?)
for C in 2048 512 256; do step pf2k_chunk$C 400 env PROMPT=long B12_CACHE_MB=64 B12_PREFILL_CHUNK=$C BT_N=32 $PY $S/bt_b12.py 0.10 speed; done
# 03:31: the chunk-2048 row above started 8 s before bt_b12 stopped building prefill logits (emit_logits=False): rerun it
step pf2k_chunk2048_nologits 400 env PROMPT=long B12_CACHE_MB=64 B12_PREFILL_CHUNK=2048 BT_N=32 $PY $S/bt_b12.py 0.10 speed
step mtp_cap64_after_code 400 env PROMPT=code B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 spec3:3:0
step mtp_cap64_hot0_code 400 env PROMPT=code B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 MTPLX_NGRAM_HOT_MB=0 $PY $S/bt_b12.py 0.10 spec3:3:0
# NewIdeas 02:40: MTP head loaded, drafting off (verify width 1): splits the head's RAM from the 4-row verify's
step mtp_head_nodraft_code 400 env PROMPT=code B12_CACHE_MB=64 B12_PREFILL_CHUNK=256 $PY $S/bt_b12.py 0.10 spec0:0:0
# NewIdeas 02:25: footprint at long context (KV ~25 KB/token: +0.8 GB at 32K): B12 eager and + MTP after a ~32K-token
# prefill (chunk 2048), 16 decoded tokens each; prefill tok/s from the eager row
step ctx32k_b12 900 env PROMPT=ctx32k BT_N=16 $PY $S/bt_b12.py 0.10 speed
step ctx32k_b12_mtp 900 env PROMPT=ctx32k BT_N=16 $PY $S/bt_b12.py 0.10 spec3:3:0
peer unlock > /dev/null 2>&1; log "lock released (pf)"
