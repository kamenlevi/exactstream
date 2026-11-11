#!/bin/sh
# Breakthrough Wed 2026-10-21: layer-band KLD probes (re-queued after yielding to NewIdeas at 12:49): frugal drop only in
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
while pgrep -f "scripts/day_speed[34].sh" > /dev/null; do sleep 20; done   # day_speed3/4 (same CLAUDE_PID)
# 17:25 restart (Kamen 17:10: the small-RAM line is judged by decode task results; NewIdeas' re-plan): the old runner
# (a 13:00 copy of this file, still open by inode) was stopped; this one reuses the quiet lock it held (until ~18:06)
# for one teacher-forced probe: frugal 5 % with cold ranks >= 2 within half the gate mass added half a layer late
# (dh_w2_m50 at 0.05, ifeval+multi). Its paired baseline (noren at 0.05) and the floors run in the diet turn.
keepalive || { log "no lock to reuse (bands restart)"; exit 1; }
log "lock reused (bands restart)"
step tfdefer8_dh_w2_m50_r05 2340 env $PY $HOME/NewIdeas/scripts/tf_defer125.py dh_w2_m50 0.05 tf_ifeval,tf_multi 8
cp $HOME/NewIdeas/data/tfdefer/dh_w2_m50_e8.npz $B/data/tfvar/dh_w2_m50_r05_e8.npz 2>/dev/null
peer unlock > /dev/null 2>&1; log "lock released (bands restart)"
