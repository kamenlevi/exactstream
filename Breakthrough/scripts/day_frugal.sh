#!/bin/sh
# Breakthrough D1 frugal mode, morning batches 2026-10-20 (detached). Frugal = wait for the top-2 ranks only and never
# read a missing lower rank from the SSD: MoEPrefetch's w2 path (nk = 10, which works) with BIGMOE_SPARE=0, so the
# fork's background prefetch has zero capacity (prefetch() returns at cap <= 0). Resident lower ranks are still used
# and LRU-touched. ("bg loads/tok" in the speed line counts rank-uses that WOULD have been fetched; actual SSD reads are
# the waited misses.) RULES 2026-10-20: one quiet-lock ticket per batch, ~45 min per batch, then release and re-queue.
OUT=$HOME/Breakthrough/data/expA; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/Breakthrough/scripts/bt_waittop125.py
export PEER_QUEUE_TIMEOUT=14400 PYTHONDONTWRITEBYTECODE=1 BIGMOE_SPARE=0
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
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
  log "done $name: $(grep -E '^(125B|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tail -1 | cut -c1-300)"
}
compare() {
  $PY -I -c '
import json, sys
ref = json.load(open("/Users/kamen/MoEPrefetch/data/w125_gsm_eager.json"))
d = json.load(open(sys.argv[1])); n = len(d["ok"])
same = sum(a == b for a, b in zip(d["answers"], ref["answers"][:n]))
only_ref = [i for i in range(n) if ref["ok"][i] and not d["ok"][i]]; only_new = [i for i in range(n) if d["ok"][i] and not ref["ok"][i]]
print("%s: GSM8K %d/%d vs exact %d/%d; answers identical %d/%d (same-weights floor 147/150); exact-only right %s, new-only right %s" % (sys.argv[2], sum(d["ok"]), n, sum(ref["ok"][:n]), n, same, n, only_ref, only_new))
' "$OUT/w125_gsm_$1.json" "$1" >> "$OUT/summary.txt" 2>&1; }
log "day_frugal started (pid $$)"
if batch_start "D1 frugal batch 1 (speed x3 + GSM8K-150 at 10 %)"; then
  run speed_w2f_r20 600 $PY $S 0.20 w2 speed
  run speed_w2f_r10 600 $PY $S 0.10 w2 speed
  run speed_w2f_r05 600 $PY $S 0.05 w2 speed
  run gsm_w2f_r10 2700 env W125_TAG=w2f_r10 W125_CHUNK=150 $PY $S 0.10 w2 gsm150
  batch_end
fi
compare w2f_r10
if batch_start "D1 frugal batch 2 (GSM8K-150 at 5 %)"; then
  run gsm_w2f_r05 2700 env W125_TAG=w2f_r05 W125_CHUNK=150 $PY $S 0.05 w2 gsm150
  batch_end
fi
compare w2f_r05
log "day_frugal finished"
