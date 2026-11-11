#!/bin/zsh
# F1-SWITCH (5) quiet re-run (Leap, Tue 2026-11-03 22:3x): no PAUSE / HOLD, a short quiet lock, f1_switch_smoke.py
# --speed (120 s wall-clock cap), unlock. One attention layer, MEM ~3 GB. Run ATTACHED (run_in_background).
cd ~/Leap
LOG=data/f1_runner.log
Q=~/.local/share/quiet
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a $LOG; }
while [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; do sleep 30; done
say "queueing a quiet lock"
PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 5m "Leap F1-SWITCH speed check: one attention layer, fs4 vs REF2-T at 10K, ~1 min, MEM 3 GB" >> $LOG 2>&1
if [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then say "PAUSE / HOLD after the grant: stopping"; peer unlock >> $LOG 2>&1; exit 0; fi
say "run"
fw0=$(wc -l < ~/Library/Logs/fanwatch.log)
F1_COND="quiet lock" ~/BigMoE/.venv/bin/python scripts/f1_switch_smoke.py --speed >> $LOG 2>&1 &
pid=$! t0=$SECONDS
while kill -0 $pid 2>/dev/null; do
  if [ $((SECONDS - t0)) -ge 120 ]; then say "cap 120 s: killing pid $pid"; kill -KILL $pid 2>/dev/null; break; fi
  sleep 1
done
wait $pid; rc=$?
peer unlock >> $LOG 2>&1
say "exit $rc after $((SECONDS - t0)) s, unlocked"
if tail -n +$((fw0 + 1)) ~/Library/Logs/fanwatch.log | grep -q -E "heat pause|froze"; then say "VOID: the heat watcher paused / froze during the run"; fi
