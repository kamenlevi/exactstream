#!/bin/zsh
# HC8-1P timing (Leap, Wed 2026-11-04 14:2x): the quiet lock is taken BEFORE this, in its own command (the hook's rule);
# this runs hc8r_screen.py with HC8_TIME=1 under a 300 s wall-clock cap, unlocks, and flags a heat pause. MEM ~3 GB.
cd ~/Leap
LOG=data/hc8r_runner.log
Q=~/.local/share/quiet
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a $LOG; }
if [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then say "PAUSE / HOLD: not running"; peer unlock >> $LOG 2>&1; exit 0; fi
say "run (cap 300 s)"
fw0=$(wc -l < ~/Library/Logs/fanwatch.log)
HC8_TIME=1 HC8_COND="quiet lock${HC8_VARS:+, $HC8_VARS}" ~/BigMoE/.venv/bin/python scripts/hc8r_screen.py >> $LOG 2>&1 &
pid=$! t0=$SECONDS
while kill -0 $pid 2>/dev/null; do
  if [ $((SECONDS - t0)) -ge 300 ]; then say "cap 300 s: killing pid $pid"; kill -KILL $pid 2>/dev/null; break; fi
  sleep 1
done
wait $pid; rc=$?
peer unlock >> $LOG 2>&1
say "exit $rc after $((SECONDS - t0)) s, unlocked"
if tail -n +$((fw0 + 1)) ~/Library/Logs/fanwatch.log | grep -q -E "heat pause|froze"; then say "VOID: the heat watcher paused / froze during the run"; fi
