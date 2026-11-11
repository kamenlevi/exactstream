#!/bin/zsh
# W1-PICKUP runner (Leap, Tue 2026-11-03 19:4x; TABLE_350 'W1-PICKUP'): no PAUSE / HOLD, a short quiet lock, the harness
# (all arms interleaved, 300 trials; a 120 s wall-clock cap), unlock. Load: one GPU thread + one CPU core (+ P1's 64 MB
# reads), MEM < 1 GB. Run ATTACHED (run_in_background) so the lock stays with this session.
cd ~/Leap
LOG=data/w1_pickup.log
Q=~/.local/share/quiet
say() { echo "$(date '+%a %H:%M:%S') $*" | tee -a $LOG; }
while [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; do sleep 30; done
say "queueing a quiet lock"
PEER_QUEUE_TIMEOUT=7200 peer lock quiet --queue 5m "Leap W1-PICKUP: release -> GPU pickup latencies (one GPU thread + one CPU core), ~1 min, MEM <1 GB" >> $LOG 2>&1
if [ -e $Q/PAUSE ] || [ -e $Q/HOLD ]; then say "PAUSE / HOLD after the grant: stopping"; peer unlock >> $LOG 2>&1; exit 0; fi
say "run"
W1_MARK=sys native/w1_pickup ${=W1_ARGS:-300 A,B,B1,M,C,N,P0,P1,PS -1,0,50,200,600,1000,2000} > ${W1_OUT:-data/w1_pickup.txt} 2>&1 &
pid=$! t0=$SECONDS
while kill -0 $pid 2>/dev/null; do
  if [ $((SECONDS - t0)) -ge 120 ]; then say "cap 120 s: killing pid $pid"; kill -KILL $pid 2>/dev/null; break; fi
  sleep 1
done
wait $pid; rc=$?
peer unlock >> $LOG 2>&1
say "exit $rc after $((SECONDS - t0)) s, unlocked"
