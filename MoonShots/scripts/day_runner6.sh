#!/bin/sh
# MoonShots day_runner6 (detached, orchestrator 14:10): the bigger-set check of the 3-bit tier plan 4433333333 on the 125B,
# GSM8K problems 150-249, with the unpatched 4-bit arm (mix10) on the same problems as the reference and noise arm.
# Offset-specific checkpoints: data/w125_gsm_planes_{mix10,plan4433333333}_off150.json. Same lock discipline as
# day_runner5 (per-job quiet lock, >= 85 % free-memory guard, active_at keepalive); launched under the session's lock.
# Results: ~/MoonShots/data/day6/*.txt, one line per job in day6/summary.txt.   usage: day_runner6.sh
OUT=$HOME/MoonShots/data/day6; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
PL=$HOME/MoonShots/patches/planes125
export PEER_QUEUE_TIMEOUT=14400
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 17:45" +%s)   # no new job after 17:45
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
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 17:45)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "MoonShots day_runner6: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
  g=0
  while { [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 85 ] || [ "$(ps -axo rss=,pid= | awk -v me=$$ '$1 > 8000000 && $2 != me' | wc -l)" -gt 0 ]; } && [ $g -lt 120 ]; do
    sleep 5; g=$((g + 1)); [ $((g % 36)) -eq 0 ] && keepalive; done
  if [ $g -ge 120 ]; then log "skip $name (memory busy)"; peer unlock > /dev/null 2>&1; return; fi
  log "start $name"
  ( "$@" > "$OUT/$name.txt" 2>&1 ) & P=$!
  i=0
  while kill -0 $P 2>/dev/null && [ $i -lt "$lim" ]; do
    sleep 5; i=$((i + 5))
    if [ $((i % 180)) -eq 0 ]; then
      keepalive || { log "lost the quiet lock during $name: stopping it"; kill $P 2>/dev/null; sleep 5; kill -9 $P 2>/dev/null; break; }
    fi
  done
  kill -0 $P 2>/dev/null && { kill -9 $P; log "TIMEOUT $name"; }
  wait $P 2>/dev/null
  peer unlock > /dev/null 2>&1
  log "done $name: $(grep -E '^(125B|  vs|  noise|planes|bigmoe|fixed|adaptive|Traceback|[A-Za-z]*Error)' "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-900)"
}
log "day_runner6 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
job gsm125_ref_off150  4800 env OFFSET=150 $PY $PL/gsm_planes125.py 10 100
job gsm125_plan_off150 4800 env OFFSET=150 $PY $PL/gsm_planes125.py 4433333333 100
log "day_runner6 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
# extension (orchestrator 15:20): problems 250-449 for both arms, so the 125B check reaches 450 like the 35B
log "day_runner6 extension: GSM8K 250-449, unpatched then plan"
job gsm125_ref_off250  4800 env OFFSET=250 $PY $PL/gsm_planes125.py 10 200
job gsm125_plan_off250 6600 env OFFSET=250 $PY $PL/gsm_planes125.py 4433333333 200
log "day_runner6 extension finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
# round 22 (MoonShots): verify-pass routing trace for the wrong-path expert study (~10 min)
job verify_trace125 1800 $PY $HOME/MoonShots/scripts/verify_trace125.py 0.5 400
log "day_runner6 trace finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
