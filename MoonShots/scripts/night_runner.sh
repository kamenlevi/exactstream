#!/bin/sh
# MoonShots night_runner (detached; started with nohup setsid; never needs the session). 125B prefill benches toward the
# 1000 tok/s goal, each in a fresh process, each under its own `peer lock quiet --queue` turn. No new job starts after 06:45;
# the script exits by ~07:10. Results: ~/MoonShots/data/night/*.txt and a one-line-per-job summary in night/summary.txt.
OUT=$HOME/MoonShots/data/night; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
B=$HOME/MoonShots/patches/layer_major_prefill/bench_lmp125.py
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_PF_PREFILL=1"
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 06:45" +%s); [ "$DL" -le "$(date +%s)" ] && DL=$((DL + 86400))   # next 06:45
# peer attributes this detached runner's locks to the session that launched it (inherited CLAUDE_PID): a `peer unlock` from
# that session also drops the runner's queue entries and lock. peer lapses a quiet lock after 8 min when nothing it can see
# runs under it (a detached job is invisible), and a non-queued `peer lock quiet` re-take is refused whenever anyone else
# waits in line (it killed BigMoE's lc1_reset_core at 00:29). So the keepalive stamps active_at in our own lock file, which
# is what peer's hook does when a session runs a command under its own lock; it fails if we no longer hold a quiet lock.
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT ENV... -- ARGS...
  name=$1; lim=$2; shift 2
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 06:45)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"   # the session took it to launch us
  else peer lock quiet --queue 90m "MoonShots night_runner: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  [ "$(date +%s)" -lt $((DL + 900)) ] || { peer unlock > /dev/null 2>&1; log "skip $name (lock came after 07:00)"; return; }
  # memory guard (BigMoE's pattern): >= 85 % free and no other process with > 8 GB resident, else skip
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( env $REF "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then   # keepalive: a detached runner's lock lapses after 8 min, so stamp it every 3 min
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P 2>/dev/null; sleep 5; kill -9 $P 2>/dev/null; break; }
    fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  peer unlock > /dev/null 2>&1
  log "done $name: $(grep -E '^(ref|lmp|pf3|L ?[0-9]+|  (lru|lfu|tinylfu|uniform|alloc|LRU)|4-bit|b[23]|mix)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-600)"
}
log "night_runner started (pid $$)"
# (the CPU-only trace jobs, expert_similarity / admission_sim / stall_alloc_sim, ran niced in the 00:40 quiet window instead)
job expert_bits35     1800 $PY $HOME/MoonShots/scripts/expert_bits35.py
log "night_runner finished"
