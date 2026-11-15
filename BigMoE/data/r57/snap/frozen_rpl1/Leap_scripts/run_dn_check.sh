#!/bin/zsh
# DENSE-N (Leap, Wed 2026-11-04 13:5x): the quiet-slot run of dn_check.py. The quiet lock is taken BEFORE this, in its
# own command (the hook's rule); this script only runs the test under a wall-clock cap and unlocks. It skips the run
# if it starts at or after 14:30 (the orchestrator's slot rule) and caps the run so it ends by 14:39. MEM <= 2.5 GB.
cd ~/Leap
LOG=data/dn_runner.log
Q=~/.local/share/quiet
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a $LOG; }
if [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then say "PAUSE / HOLD: not running"; peer unlock >> $LOG 2>&1; exit 0; fi
now=$(date +%H%M)
if [ $now -ge 1430 ]; then say "start at $now >= 14:30: skipped (slot rule)"; peer unlock >> $LOG 2>&1; exit 0; fi
end=$(date -j -f "%H:%M:%S" "14:39:00" +%s); cap=$(( end - $(date +%s) )); [ $cap -gt 540 ] && cap=540
say "run (cap $cap s)"
fw0=$(wc -l < ~/Library/Logs/fanwatch.log)
DN_COND="quiet lock" ~/BigMoE/.venv/bin/python scripts/dn_check.py >> $LOG 2>&1 &
pid=$! t0=$SECONDS
while kill -0 $pid 2>/dev/null; do
  if [ $((SECONDS - t0)) -ge $cap ]; then say "cap $cap s: killing pid $pid"; kill -KILL $pid 2>/dev/null; break; fi
  sleep 1
done
wait $pid; rc=$?
peer unlock >> $LOG 2>&1
say "exit $rc after $((SECONDS - t0)) s, unlocked"
if tail -n +$((fw0 + 1)) ~/Library/Logs/fanwatch.log | grep -q -E "heat pause|froze"; then say "VOID: the heat watcher paused / froze during the run"; fi
