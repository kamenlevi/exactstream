#!/bin/sh
# Leap: run one GPU job under a quiet lock that is ALREADY HELD by this Claude session (TrainLab's run_locked.sh pattern:
# keepalive stamps active_at every 150 s; the job is stopped if the lock is lost; unlock at the end).
# Also stops the job if swap grows by more than 2 GB: swap is written to the SSD and eats the disk floor (orchestrator,
# 10-23, after L2b's run took swap from ~6 to 12.6 GB). Stops kill the job's whole process tree, by PID.
# usage: run_locked.sh <minutes> <command...>     log: ~/Leap/logs/locked.log, job output: ~/Leap/logs/job.out
L=$HOME/Leap; LOG=$L/logs/locked.log; MIN=${1:-30}; shift
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
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}'; }
tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
# SIGTERM first: dfl2c_train.py then checkpoints and exits; whatever is left after 30 s gets SIGKILL
alive() { for p in $T; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
stop() { log "$1: stopping the job"; T=$(tree $P); kill $T 2>/dev/null
  j=0; while [ $j -lt 30 ] && alive; do sleep 1; j=$((j + 1)); done; kill -9 $T 2>/dev/null; }
HEALTH=$HOME/.local/share/quiet/HEALTH.md     # the health monitor: abort on ACT (orchestrator, 10-23)
# Kamen's quiet windows freeze jobs with SIGSTOP (orchestrator, 10-24): time while any process of the job is stopped does
# not count toward the limit
paused() { for p in $(tree $P); do case "$(ps -o stat= -p $p 2>/dev/null)" in *T*) return 0;; esac; done; return 1; }
keepalive || { log "no held quiet lock: $* not started"; exit 1; }
SW0=$(swap_mb)
log "lock held; running $* (swap used ${SW0} MB)"
LIM=$(( MIN * 60 - 60 ))
( "$@" >> "$L/logs/job.out" 2>&1 ) & P=$!
i=0
while kill -0 $P 2>/dev/null && [ $i -lt $LIM ]; do
  sleep 5
  if paused; then continue; fi
  i=$((i + 5))
  if [ $((i % 150)) -eq 0 ]; then keepalive || { stop "lost the lock"; break; }; fi
  SW=$(swap_mb); if [ $((SW - SW0)) -gt 2048 ]; then stop "swap grew ${SW0} -> ${SW} MB (> 2 GB)"; break; fi
  if grep -q '^\*\*Verdict: ACT' "$HEALTH" 2>/dev/null; then stop "health monitor at ACT"; break; fi
done
if kill -0 $P 2>/dev/null; then stop "time up"; fi
wait $P; rc=$?
peer unlock > /dev/null 2>&1; log "job done (rc $rc), swap used $(swap_mb) MB (start ${SW0}), lock released"
exit $rc
