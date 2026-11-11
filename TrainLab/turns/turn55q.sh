#!/bin/sh
# The shipped-line A/B again, order-balanced (MTP Head, Sat 10-31 00:4x): the first run (00:22, turns/turn53.sh) always ran
# b3 second on a cache the stock arm had warmed, so its tok/s gains (+8-33 %) are order-confounded. sticky/d22_ship.py now
# runs per prompt a discarded warm-up, then A-B-A. Two quiet tickets after D-20's queuer (turn47q) is done, 15 min apart:
# D22_SHIP_MODE=adapt (the shipped width rule), then k7 (a fixed 7-draft MTP chain). Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
while pgrep -f "turns/turn47q.sh" > /dev/null; do sleep 30; done
for MODE in adapt k7; do
  for t in 1 2 3; do
    echo "$(date '+%m-%d %H:%M:%S') turn55q: shipped-line A/B ($MODE, A-B-A) queueing in 15 min (try $t)" >> $LOG
    sleep 900
    PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-22: shipped-line A/B $MODE, order-balanced A-B-A (turn53)" >> $LOG 2>&1 || exit 1
    N=$(wc -l < logs/turn_job.out)
    D22_SHIP_MODE=$MODE ./run_locked.sh 15 turns/turn53.sh
    tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting" || break
  done
done
