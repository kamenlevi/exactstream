#!/bin/sh
# Breakthrough D2 (2026-10-20 13:3x): trellis vs white noise at equal NMSE, on ALL routed experts of the 8-bit 35B.
# Assumes this session already holds the quiet lock (taken 13:34, until 15:04): stamps active_at every 2.5 min,
# runs experts t3 and t4 at L=8 (CPU Viterbi, ~15 min each), then releases the lock.
OUT=$HOME/Breakthrough/data/noise35; LOG=$HOME/Breakthrough/data/d2/summary.txt
PY=$HOME/BigMoE/.venv/bin/python; S=$HOME/Breakthrough/scripts
export PYTHONDONTWRITEBYTECODE=1
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
for arm in t3 t4; do
  keepalive || { log "lock lost before experts $arm"; break; }
  log "start experts $arm (L=8)"
  ( BT_TL=8 $PY $S/noise35.py experts $arm >> "$OUT/experts_${arm}_L8.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt 2700 ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 150)) -eq 0 ]; then keepalive || { log "lost the lock during experts $arm: stopping"; kill $P; sleep 5; kill -9 $P 2>/dev/null; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT experts $arm"; }
  wait $P 2>/dev/null
  log "done experts $arm: $(grep -E '^experts|Error' "$OUT/experts_${arm}_L8.txt" | tail -1 | cut -c1-260)"
done
peer unlock > /dev/null 2>&1; log "lock released (day_d2b)"
