#!/bin/sh
# Breakthrough Thu 2026-10-22 19:36: IDEAS D1n quality headroom on the REAL v3f path (tf_b12, the Eval suite scorer, every 8):
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
# hard (ifeval + multi) and easy (gsm + code) KLD for B8-mass, B12-mass + LATE3=2,4 and + LATE3=2,10 (dh at 10 %); baseline
# B12-mass hard 0.1117 (10-22 10:30). Kill line for a decode verdict: hard <= ~0.09 and <= 12 GB with MTP at 2.6K.
keepalive || { log "no lock to reuse (d1n)"; exit 1; }
log "lock taken (d1n)"
export B12_CACHE_MB=64 B12_NORM=mass
# MoonShots' bug (late writes under MTP: stale PENDH after commit replays) fixed 1a5eee1: its failing case first, 192 tokens
step d1n_b8mtp_code 420 env PROMPT=code B12_LATE3=2,4 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BT_N=192 $PY $S/bt_b12.py 0.05 spec3:3:0
step d1n_b8mtp_prose 420 env PROMPT=prose B12_LATE3=2,4 BIGMOE_PF_GROUP=64 BIGMOE_MTP_C=32 BT_N=192 $PY $S/bt_b12.py 0.05 spec3:3:0
# NewIdeas 20:10: B8's like-for-like verdict FAILS like B12 (IFEval -6.0): the quality step first, hard text first
step d1n_p4c51_hard 300 env B12_PLAN4=1 $PY $S/tf_b12.py b12m_p4c51_hard 0.10 tf_ifeval,tf_multi 8
step d1n_p4c46_hard 300 env B12_PLAN4=1 B12_CM=46 $PY $S/tf_b12.py b12m_p4c46_hard 0.10 tf_ifeval,tf_multi 8
step d1n_b12l24_hard 300 env B12_LATE3=2,4 $PY $S/tf_b12.py b12m_l24_hard 0.10 tf_ifeval,tf_multi 8
step d1n_b12l210_hard 300 env B12_LATE3=2,10 $PY $S/tf_b12.py b12m_l210_hard 0.10 tf_ifeval,tf_multi 8
step d1n_b8_hard 300 env B12_LATE3=2,4 $PY $S/tf_b12.py b8m_hard 0.05 tf_ifeval,tf_multi 8
step d1n_b12_easy 300 $PY $S/tf_b12.py b12m_easy 0.10 tf_gsm,tf_code 8
step d1n_p4c51_easy 300 env B12_PLAN4=1 $PY $S/tf_b12.py b12m_p4c51_easy 0.10 tf_gsm,tf_code 8
step d1n_p4c46_easy 300 env B12_PLAN4=1 B12_CM=46 $PY $S/tf_b12.py b12m_p4c46_easy 0.10 tf_gsm,tf_code 8
step d1n_b12l24_easy 300 env B12_LATE3=2,4 $PY $S/tf_b12.py b12m_l24_easy 0.10 tf_gsm,tf_code 8
step d1n_b12l210_easy 300 env B12_LATE3=2,10 $PY $S/tf_b12.py b12m_l210_easy 0.10 tf_gsm,tf_code 8
step d1n_b8_easy 300 env B12_LATE3=2,4 $PY $S/tf_b12.py b8m_easy 0.05 tf_gsm,tf_code 8
log "d1n done"
