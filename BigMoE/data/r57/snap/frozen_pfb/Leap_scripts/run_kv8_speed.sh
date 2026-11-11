#!/bin/zsh
# KV8 speed part (Leap, Tue 2026-11-03 21:4x; TABLE_350 'KV8'): no PAUSE / HOLD, a short quiet lock, the timing (180 s
# wall-clock cap), unlock. Load: small attention kernels, MEM < 1 GB. Run ATTACHED (run_in_background).
cd ~/Leap
LOG=data/kv8_runner.log
Q=~/.local/share/quiet
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a $LOG; }
while [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; do sleep 30; done
say "queueing a quiet lock"
PEER_QUEUE_TIMEOUT=7200 peer lock quiet --queue 5m "Leap KV8 speed: int8 K/V attention kernels vs bf16 at 2K/10K/20K, ~1 min, MEM <1 GB" >> $LOG 2>&1
if [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then say "PAUSE / HOLD after the grant: stopping"; peer unlock >> $LOG 2>&1; exit 0; fi
say "run"
> (a quote from the project owner, kept private)
pid=$! t0=$SECONDS
while kill -0 $pid 2>/dev/null; do
  if [ $((SECONDS - t0)) -ge 180 ]; then say "cap 180 s: killing pid $pid"; kill -KILL $pid 2>/dev/null; break; fi
  sleep 1
done
wait $pid; rc=$?
peer unlock >> $LOG 2>&1
say "exit $rc after $((SECONDS - t0)) s, unlocked"
