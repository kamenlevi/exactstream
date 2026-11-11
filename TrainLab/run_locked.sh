#!/bin/sh
# TrainLab: run one turn job under a quiet lock that is ALREADY HELD (queue `peer lock quiet --queue` alone first; the
# hook blocks starting jobs during others' locks). Keepalive on the lock; the job's time limit is TL_LIMIT seconds.
# usage: run_locked.sh <minutes> <job>      log: ~/TrainLab/logs/turns.log, job output: logs/turn_job.out
T=$HOME/TrainLab; LOG=$T/logs/turns.log; MIN=${1:-45}; JOB=$2
PY=$HOME/BigMoE/.venv/bin/python
LOCKF=$HOME/.claude-peers/locks/$CLAUDE_PID.json
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$LOG"; }
keepalive() { $PY -I -S -c '
import json, os, sys, time
p = sys.argv[1]
try: d = json.load(open(p))
except Exception: sys.exit(1)
if d.get("mode") != "quiet" or d.get("until", 0) <= time.time(): sys.exit(1)
d["active_at"] = time.time(); t = "%s.tmp%d" % (p, os.getpid()); json.dump(d, open(t, "w")); os.replace(t, p)' "$LOCKF"; }
keepalive || { log "run_locked: no held quiet lock, $JOB not started"; exit 1; }
log "lock held; running $JOB"
LIM=$(( MIN * 60 - 90 ))
set -m                                                   # the job gets its own process group: a stop reaches its python
( TL_LIMIT=$LIM "$JOB" >> "$T/logs/turn_job.out" 2>&1 ) & P=$!   # too (10-27 20:54: an orphaned 16-row run outlived the lock)
i=0
while kill -0 $P 2>/dev/null && [ $i -lt $LIM ]; do
  sleep 5; i=$((i + 5))
  if [ $((i % 150)) -eq 0 ]; then keepalive || { log "lost the lock: stopping the job"; kill -TERM -- -$P; pkill -f "sticky35.py run"; sleep 10; kill -9 -- -$P 2>/dev/null; break; }; fi
done
if kill -0 $P 2>/dev/null; then kill -TERM -- -$P; pkill -f "sticky35.py run"; sleep 20; kill -9 -- -$P 2>/dev/null; log "turn time up: job stopped"; fi
wait $P; rc=$?
peer unlock > /dev/null 2>&1; log "job done (rc $rc), lock released"
