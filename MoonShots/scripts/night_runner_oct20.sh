#!/bin/sh
# MoonShots night runner, night of Oct 19-20 (Plan B GPU line 03:10-07:00; see ~/.local/share/pacer/night_plan.md).
# Waits until 03:10, then per job: `peer lock quiet --queue 90m`, >= 85 % free memory and no other > 8 GB process,
# active_at keepalive every 3 min (job stopped if the lock is lost), hard per-job limit, no job starts after 06:45.
# Jobs: MTP depth 1-5 with committed history on 6 prompts at 17 GB and 34 GB (BIGMOE_MAXK=60; depth 2 vs 3 was
# prompt-dependent on 3 prompts), 4K-context cycle vs committed at 34 GB, the depth-2 speed-vs-memory curve, filler at 26 GB. Results: ~/MoonShots/data/night20/<job>.txt, summary.txt.
OUT=$HOME/MoonShots/data/night20; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
PL=$HOME/MoonShots/patches/planes125
export PEER_QUEUE_TIMEOUT=14400
log() { echo "$(date +%H:%M) $*" >> "$OUT/summary.txt"; }
DL=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 06:45" +%s); [ "$DL" -le "$(date +%s)" ] && DL=$((DL + 86400))   # next 06:45
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
  [ "$(date +%s)" -lt "$DL" ] || { log "skip $name (past 06:45)"; return; }
  if keepalive check; then keepalive; log "lock already held for $name"
  else peer lock quiet --queue 90m "MoonShots night_runner_oct20: $name" > /dev/null 2>&1 || { log "no lock for $name"; return; }; fi
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
log "night_runner_oct20 started (pid $$); free $(df -g / | tail -1 | awk '{print $4}') GB"
S=$HOME/MoonShots/scripts/mtp_sweep125.py
T310=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%Y-%m-%d) 03:10" +%s); [ "$T310" -le "$(date +%s)" ] && [ "$(date +%H)" -ge 12 ] && T310=$((T310 + 86400))
log "waiting for 03:10"
while [ "$(date +%s)" -lt "$T310" ]; do sleep 30; done
# depth >= 4 needs BIGMOE_MAXK >= 10 x (depth + 1) (default 40 sends those passes to the slow sync gate path)
job n_depth_C128_6p  2400 env BIGMOE_MAXK=60 PROMPTS=6 TAG=_6p $PY $S 0.25 1,2,3,4,5 384
job n_depth_C256_6p  2400 env BIGMOE_MAXK=60 PROMPTS=6 TAG=_6p $PY $S 0.5 1,2,3,4,5 384
job n_longctx_C256   1500 $PY $HOME/MoonShots/scripts/mtp_longctx125.py 0.5 192 4000
job n_curve_d2_C96   900  env TAG=_d2 $PY $S 0.1875 2 192
job n_curve_d2_C160  900  env TAG=_d2 $PY $S 0.3125 2 192
job n_curve_d2_C192  900  env TAG=_d2 $PY $S 0.375 2 192
job n_depth_C192_6p  2400 env BIGMOE_MAXK=60 PROMPTS=6 TAG=_6p $PY $S 0.375 2,3,4 384
log "night_runner_oct20 finished; free $(df -g / | tail -1 | awk '{print $4}') GB"
