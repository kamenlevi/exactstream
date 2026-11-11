#!/bin/sh
# Breakthrough Thu 2026-10-22 00:20: speed of the candidates with HC mixers at 8 bits (free in KLD: +0.0000 hard, +0.0001
# easy): B12 / B8 / B8-fast + B12_HC=8, code + prose. Launched once my session holds the lock (keepalive reuses it).
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
keepalive || { log "no lock to reuse (hc8)"; exit 1; }
log "lock taken (hc8)"
/opt/homebrew/bin/cmake --build $B/v3f/ext/build -j 6 > $B/v3f/ext/build.log 2>&1 || { log "v3f rebuild FAILED"; peer unlock > /dev/null 2>&1; exit 1; }
log "v3f rebuilt (protect-what-fits loader)"
for PR in code prose; do
  step hc8_b12_$PR 300 env PROMPT=$PR B12_HC=8 $PY $S/bt_b12.py 0.10 speed
  step hc8_b8_$PR 300 env PROMPT=$PR B12_HC=8 B12_LATE3=2,4 $PY $S/bt_b12.py 0.05 speed
  step hc8_b8f_$PR 300 env PROMPT=$PR B12_HC=8 B12_W=3 $PY $S/bt_b12.py 0.05 speed
done
# MTP on the v3 copy (NewIdeas 00:10): verify passes of <= 4 rows take the multi-row frugal block (rank-major rows,
# protect-what-fits loader); B8's late ranks are waited for in verify rows
step spec_b12_code 400 env PROMPT=code $PY $S/bt_b12.py 0.10 spec3:3:0
step spec_b12_prose 400 env PROMPT=prose $PY $S/bt_b12.py 0.10 spec3:3:0
step spec_b8f_code 400 env PROMPT=code B12_W=3 $PY $S/bt_b12.py 0.05 spec3:3:0
peer unlock > /dev/null 2>&1; log "lock released (hc8)"
