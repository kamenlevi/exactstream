#!/bin/sh
# TrainLab GPU turn runner: queue for one quiet lock, wait for memory, then run the job file that exists AT GRANT TIME
# (~/TrainLab/turn_job.sh, editable while queued). Keepalive on the lock; the job's own time limit is TL_LIMIT seconds.
# usage: run_turn.sh <minutes> "<why>"      log: ~/TrainLab/logs/turns.log
T=$HOME/TrainLab; LOG=$T/logs/turns.log; MIN=${1:-45}; WHY=${2:-"TrainLab: LoRA healing of expert dataflows on the 35B"}
PY=$HOME/BigMoE/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PEER_QUEUE_TIMEOUT=${PEER_QUEUE_TIMEOUT:-21600}
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
log "queued: $MIN min, $WHY"
peer lock quiet --queue ${MIN}m "$WHY" > /dev/null 2>&1 || { log "no lock"; exit 1; }
g=0
# memory gate: wait until >= 50 % of RAM is free (the 35B run needs ~25-30 GB), at most 10 min
while [ "$(memory_pressure | tail -1 | grep -o '[0-9]*')" -lt 50 ] && [ $g -lt 120 ]; do
  sleep 5; g=$((g + 1)); [ $((g % 30)) -eq 0 ] && keepalive; done
[ $g -ge 120 ] && { log "memory busy for 10 min: giving the lock back"; peer unlock > /dev/null 2>&1; exit 1; }
[ -x "$T/turn_job.sh" ] || { log "no turn_job.sh: lock released"; peer unlock > /dev/null 2>&1; exit 1; }
log "lock taken; running turn_job.sh"
LIM=$(( MIN * 60 - 90 ))
( TL_LIMIT=$LIM "$T/turn_job.sh" >> "$T/logs/turn_job.out" 2>&1 ) & P=$!
i=0
while kill -0 $P 2>/dev/null && [ $i -lt $LIM ]; do
  sleep 5; i=$((i + 5))
  if [ $((i % 150)) -eq 0 ]; then keepalive || { log "lost the lock: stopping the job"; kill $P; sleep 10; kill -9 $P 2>/dev/null; break; }; fi
done
if kill -0 $P 2>/dev/null; then kill $P; sleep 20; kill -9 $P 2>/dev/null; log "turn time up: job stopped"; fi
wait $P; rc=$?
peer unlock > /dev/null 2>&1; log "turn done (rc $rc), lock released"
