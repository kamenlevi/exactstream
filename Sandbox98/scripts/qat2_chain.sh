#!/bin/bash
# One-off hand-over (orchestrator 09:4x). LSQ's 60 min share lock lapsed at 09:24 while the step ran on (~2.8 s/step under
# a 5-way GPU share); it was re-taken by hand until 10:50. This holds it to LSQ's real end (renews 20 min at a time while
# sb/qat2.py --lever lsq runs), then hands FLOOR to a fresh runner whose guard sizes each lock from the step's own s/step.
# The old runner stops by itself after LSQ: a placeholder done-marker for qat2_floor_smoke makes it skip the smoke and
# exit on "no DONE line"; this script then removes the marker and starts LEVERS=floor.
# usage: qat2_chain.sh OLD_RUNNER_PID LOCK_END_EPOCH
cd "$(dirname "$0")/.."
OLD=$1 END=$2
while kill -0 $OLD 2>/dev/null; do
  sleep 10
  now=$(date +%s)
  if [ $((END - now)) -lt 300 ] && pgrep -f "sb/qa[t]2.py --lever lsq" >/dev/null; then
    out=$(peer lock share 20m --cpu 200 --gpu 60 "Sandbox98: step qat2_lsq, MEM 6 GB measured peak" 2>&1 | head -1)
    echo "$(date +%H:%M:%S) LSQ still running at its lock's end: renewed for 20 min: $out"
    END=$((now + 1200))
  fi
done
echo "$(date +%H:%M:%S) old runner $OLD ended; its last lines:"; tail -3 logs/qat2_runner.log
rm -f logs/.done/qat2/qat2_floor_smoke
if [ -f logs/.done/qat2/qat2_lsq ] && tail -1 logs/qat2_runner.log | grep -q "qat2 floor smoke: no DONE line"; then
  pgrep -f "sb/qa[t]2.py" >/dev/null && { echo "a qat2 step is still running: FLOOR not started"; exit 1; }
  peer unlock 2>&1 | tail -1    # drop a renewal that raced LSQ's end, if any
  echo "$(date +%H:%M:%S) starting FLOOR (LEVERS=floor) in a new runner"
  LEVERS=floor nohup bash scripts/qat2_all.sh >> logs/qat2_runner.log 2>&1 &
  echo "new runner pid $!"
else
  echo "$(date +%H:%M:%S) the old runner did not end at the FLOOR hand-over: FLOOR not started"
fi
