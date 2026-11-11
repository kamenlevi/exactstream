#!/bin/sh
# Queue the shipped-line A/B (turn53) per the orchestrator 23:5x: a 45 GB-route quiet ticket 15 min after Leap's C 224
# stack (S45STACK) ends; after the grant turn53 settles 30 s and checks vm_stat available >= peak + 4 GB; if short it exits
# (the lock is released) and this re-queues 15 min later, up to 4 tries. Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
i=0; while peer ls 2>/dev/null | grep -q "S45STACK" && [ $i -lt 240 ]; do sleep 30; i=$((i + 1)); done   # Leap's C 224 stack
                                                            # neither running nor queued (at most 2 h)
echo "$(date '+%m-%d %H:%M:%S') turn53q: Leap's S45STACK lock is gone; queueing in 15 min" >> $LOG
sleep 900
for t in 1 2 3 4; do
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-22: trained MTP head b3 in the shipped 45 GB line (H50 A/B, turn53)" >> $LOG 2>&1 || exit 1
  N=$(wc -l < logs/turn_job.out)
  ./run_locked.sh 15 turns/turn53.sh
  tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" || exit 0
  echo "$(date '+%m-%d %H:%M:%S') turn53q: memory short after the grant; re-queueing in 15 min (try $t)" >> $LOG
  sleep 900
done
