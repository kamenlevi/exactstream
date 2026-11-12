#!/bin/bash
# The agent test's window (EvalSuite's amendment 5 for r120 / r120s): if the runner (pid $1) is still alive at the deadline
# ($3, default 2026-11-13 00:15:00), send it TERM (its cleanup stops serve.py and checks the port), wait up to 4 min, then
# KILL it and stop any serve.py left on its port. Writes to the run folder's runner.log.
# Usage: watchdog_r120.sh <runner pid> <run dir> ["YYYY-MM-DD HH:MM:SS"]
P=$1; RUN=$2; PORT=8122; DLS=${3:-2026-11-13 00:15:00}
DL=$(date -j -f "%Y-%m-%d %H:%M:%S" "$DLS" +%s)
while kill -0 $P 2>/dev/null && [ $(date +%s) -lt $DL ]; do sleep 20; done
kill -0 $P 2>/dev/null || exit 0
echo "$(date '+%m-%d %H:%M:%S') WATCHDOG: the runner is still running at $DLS: TERM" >> $RUN/runner.log
kill -TERM $P
for i in $(seq 1 240); do kill -0 $P 2>/dev/null || break; sleep 1; done
if kill -0 $P 2>/dev/null; then
  echo "$(date '+%m-%d %H:%M:%S') WATCHDOG: no exit after 4 min: KILL" >> $RUN/runner.log; kill -KILL $P
fi
S=$(lsof -nP -t -iTCP:$PORT -sTCP:LISTEN 2>/dev/null)
[ -n "$S" ] && { echo "$(date '+%m-%d %H:%M:%S') WATCHDOG: serve.py pid $S still on port $PORT: TERM" >> $RUN/runner.log; kill -TERM $S; }
exit 0
