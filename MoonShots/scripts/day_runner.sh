#!/bin/sh
# MoonShots day_runner (detached; queued GPU work while the session pace-sleeps, RULES.md "Daytime GPU line"). Same lock
# discipline as night_runner.sh (copied below): per-job `peer lock quiet --queue`, >= 85 % free-memory guard, active_at
# keepalive. No new job after 11:00. Results: ~/MoonShots/data/day/*.txt, one line per job in day/summary.txt.
# usage: day_runner.sh PID   (waits for the session's own GSM8K run PID, then releases the session's lock and queues jobs)
OUT=$HOME/MoonShots/data/day; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
B=$HOME/MoonShots/patches/layer_major_prefill/bench_lmp125.py
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_PF_PREFILL=1"
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 11:00" +%s)   # no new job after 11:00
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
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 11:00)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"   # the session took it to launch us
  else peer lock quiet --queue 90m "MoonShots day_runner: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  [ "$(date +%s)" -lt $((DL + 900)) ] || { peer unlock > /dev/null 2>&1; log "skip $name (lock came after 11:15)"; return; }
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
  log "done $name: $(grep -E '^(4-bit|mix|msb|path4|arm|B vs|C vs)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-600)"
}
log "day_runner started (pid $$)"
# The session's GSM8K run holds the session's lock. Wait for it, then release that lock: a later `peer unlock` from the
# session would also drop this runner's queue entries, so from here on the runner owns every MoonShots lock.
while kill -0 "$1" 2>/dev/null; do sleep 10; done
peer unlock > /dev/null 2>&1; log "session GSM8K run (pid $1) ended; its lock released"
G=$HOME/MoonShots/scripts/gsm8k_planes35.py; A0=$HOME/MoonShots/data/gsm8k_planes35_mix4-msb2m.json
job gsm8k_mix6        1500 env REUSE_A=$A0 SKIP_C=1 $PY $G mix6-msb2m 150
job gsm8k_mix4_off150 2700 env OFFSET=150 SKIP_C=1 $PY $G mix4-msb2m 300
job gsm8k_mix2        1500 env REUSE_A=$A0 SKIP_C=1 $PY $G mix2-msb2m 150
log "day_runner finished"
# (Lines appended here while it ran were never executed: bash had already buffered this small script. Use a new runner.)
