#!/bin/sh
# MoonShots day_runner2 (detached): the 125B nested-plane GSM8K test (patches/planes125) and round 18 PoC-1 (expert_tiers35).
# Same lock discipline as night_runner.sh: per-job `peer lock quiet --queue`, >= 85 % free-memory guard, active_at keepalive.
# Launched under a quiet lock the session holds: the first job uses it, later jobs queue. No new job after 16:00.
# Results: ~/MoonShots/data/day2/*.txt, one line per job in day2/summary.txt.   usage: day_runner2.sh
OUT=$HOME/MoonShots/data/day2; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
REF=""
export PEER_QUEUE_TIMEOUT=14400                       # SpecDecode's long day job holds the Mac until up to 13:15
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 16:00" +%s)   # no new job after 16:00
# peer attributes this detached runner's locks to the session that launched it (inherited CLAUDE_PID): a `peer unlock` from
# that session also drops the runner's queue entries and lock. The keepalive stamps active_at in our own lock file (what
# peer's hook does for commands run under a held lock); a non-queued re-take would be refused whenever anyone waits in line.
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
if sys.argv[2] == "check": sys.exit(0 if time.time() - max(d.get("taken", 0), d.get("active_at", 0)) < 360 else 1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF" "${1:-stamp}"; }
job() {   # job NAME SECONDS_LIMIT CMD...   (sets P = the job's pid: don't use P for anything else)
  name=$1; lim=$2; shift 2
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 16:00)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"   # the session took it to launch us
  else peer lock quiet --queue 90m "MoonShots day_runner2: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  # memory guard (BigMoE's pattern): >= 85 % free and no other process with > 8 GB resident, else skip
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then   # a detached runner's lock lapses after 8 idle minutes in peer's view: stamp every 3
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P 2>/dev/null; sleep 5; kill -9 $P 2>/dev/null; break; }
    fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  peer unlock > /dev/null 2>&1
  log "done $name: $(grep -E '^(125B|  vs|  noise|planes125|path4|msb|mix|T\(|dense|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-700)"
}
log "day_runner2 started (pid $$)"
G125=$HOME/MoonShots/patches/planes125/gsm_planes125.py
job planes125_sanity  900  env OFFSET=0 CHUNK=3 $PY $G125 4 6      # problems 3-5 (0-2 done by the first sanity run, slow slicing)
job expert_tiers35    1500 $PY $HOME/MoonShots/scripts/expert_tiers35.py
job planes125_mix4    5400 env OFFSET=0 $PY $G125 4 150      # resumes after the sanity run's 3 problems
log "day_runner2 finished"
