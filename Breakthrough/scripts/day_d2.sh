#!/bin/sh
# Breakthrough D2 (detached, 2026-10-20): validate the CPU trellis encoder, then trellis fake-quant of ALL routed experts
# of the 8-bit Qwen3.6-35B at k = 3 and k = 4 bits (scripts/noise35.py arms t3 / t4; KLD vs the 8-bit reference,
# comparable with last night's noise sweep). CPU-heavy (~45 min per arm), so each arm runs inside its own quiet-lock batch.
OUT=$HOME/Breakthrough/data/noise35; LOG=$HOME/Breakthrough/data/d2/summary.txt; mkdir -p "$OUT" "$(dirname $LOG)"
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Breakthrough/scripts
export PEER_QUEUE_TIMEOUT=14400 PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date +%H:%M) $*" >> "$LOG"; }
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
batch_start() { peer lock quiet --queue 90m "Breakthrough: $1" > /dev/null 2>&1 || { log "no lock for $1"; return 1; }; log "lock taken: $1"; }
batch_end() { peer unlock > /dev/null 2>&1; log "lock released"; }
run() {   # run NAME SECONDS_LIMIT CMD...  (inside a batch: memory gate, keepalive every 3 min, hard limit)
  name=$1; lim=$2; shift 2
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  [ $g -ge 120 ] && { log "skip $name (memory busy)"; return; }
  log "start $name"
  ( "$@" >> "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P; sleep 5; kill -9 $P 2>/dev/null; break; }; fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  log "done $name: $(grep -E '^(experts|k=|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tail -1 | cut -c1-300)"
}
log "day_d2 started (pid $$)"
if batch_start "D2 trellis test + experts t3 (~50 min, CPU-heavy)"; then
  run trellis_test 600 $PY $S/trellis.py test
  run experts_t3 3600 $PY $S/noise35.py experts t3
  batch_end
fi
if batch_start "D2 experts t4 (~45 min, CPU-heavy)"; then
  run experts_t4 3600 $PY $S/noise35.py experts t4
  batch_end
fi
log "day_d2 finished"
