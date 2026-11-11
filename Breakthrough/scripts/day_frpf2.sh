#!/bin/sh
# Breakthrough Thu 2026-10-22 07:58: frugal prefill rows (cut from day_frpf at the lock's end) + the B12_NORM confirmation on the
# real v3f path (tf_b12.py, C_m 51, HC8 as judged): used vs mass, ifeval + multi every 8 (NewIdeas: for the record).
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
keepalive || { log "no lock to reuse (frpf2)"; exit 1; }
log "lock taken (frpf2)"
export B12_CACHE_MB=64 BIGMOE_TIER_PF_GROUP=64
step tfb12_b12_used 600 env B12_NORM=used $PY $S/tf_b12.py b12_used 0.10 tf_ifeval,tf_multi 8
step tfb12_b12_mass 600 env B12_NORM=mass $PY $S/tf_b12.py b12_mass 0.10 tf_ifeval,tf_multi 8
step frpf_gsm20_w2 900 env B12_FRUGAL_PF=2 B12_FRUGAL_PF_NORM=used $PY $S/bt_b12.py 0.10 gsm20
step frpf_gsm20_w2_mass 900 env B12_FRUGAL_PF=2 B12_FRUGAL_PF_NORM=mass $PY $S/bt_b12.py 0.10 gsm20
step frpf_177_w2 300 env PROMPT=long B12_LONG_CHARS=450 D1I_RUNS=2 BT_N=64 B12_FRUGAL_PF=2 B12_FRUGAL_PF_NORM=mass $PY $S/bt_b12.py 0.10 d1i
step frpf_gsm20_w4 900 env B12_FRUGAL_PF=4 B12_FRUGAL_PF_NORM=mass $PY $S/bt_b12.py 0.10 gsm20
log "frpf2 done"
