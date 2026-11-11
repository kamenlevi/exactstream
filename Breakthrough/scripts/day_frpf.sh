#!/bin/sh
# Breakthrough Thu 2026-10-22 06:50: frugal prefill (NewIdeas B5c item 3), first quality signal before Chat1000's agentic check:
# GSM8K-20 with B12_FRUGAL_PF=2 / 4 (each prompt after the previous decode: warm pools) vs 19/20; 177 tokens cold + repeat.
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
keepalive || { log "no lock to reuse (frpf)"; exit 1; }
log "lock taken (frpf)"
export B12_CACHE_MB=64 BIGMOE_TIER_PF_GROUP=64
# decode weights at the top-10 mass (B12_NORM=mass; Wed TF subnr -10 % hard KLD): GATE for the Eval suite hook (NewIdeas 07:07): runs first
step gsm20_normmass 900 env B12_NORM=mass $PY $S/bt_b12.py 0.10 gsm20
# MoonShots / NewIdeas 07:15 (gates Kamen's main-track decision): v3 tiers + dh_w4 at C=128 on the REAL v3f decoder,
# teacher-forced with the Eval suite's scorer; v3-only control (all top-10 waited) in the same lock, same sequences
step tfb12_v3ctl_c128 900 env B12_W=10 B12_HC=0 $PY $S/tf_b12.py v3ctl_c128 0.25 tf_gsm,tf_code,tf_ifeval,tf_multi 8
step tfb12_dhw4_c128 900 env B12_W=4 B12_LATE3=4,10 B12_HC=0 $PY $S/tf_b12.py dhw4_c128 0.25 tf_gsm,tf_code,tf_ifeval,tf_multi 8
step frpf_gsm20_w2 900 env B12_FRUGAL_PF=2 B12_FRUGAL_PF_NORM=used $PY $S/bt_b12.py 0.10 gsm20
# prior art (search agent 06:5x, unverified): on a 512-expert top-10 model, renormalising over fewer experts costs far more
# than keeping the experts' top-10-mass weights: arm B12_FRUGAL_PF_NORM=mass
step frpf_gsm20_w2_mass 900 env B12_FRUGAL_PF=2 B12_FRUGAL_PF_NORM=mass $PY $S/bt_b12.py 0.10 gsm20
step frpf_177_w2 300 env PROMPT=long B12_LONG_CHARS=450 D1I_RUNS=2 BT_N=64 B12_FRUGAL_PF=2 B12_FRUGAL_PF_NORM=mass $PY $S/bt_b12.py 0.10 d1i
step frpf_gsm20_w4 900 env B12_FRUGAL_PF=4 B12_FRUGAL_PF_NORM=mass $PY $S/bt_b12.py 0.10 gsm20
log "frpf done"
