#!/bin/sh
# Queue D-20 (turn47, head b3) after the shipped-line A/B's queuer (turn53q) is done, 15 min apart (orchestrator 23:5x: one
# 45 GB-route ticket at a time, 15-min gaps), then re-queue its chunks (15 min apart) until all 33 runs (11 prompts x
# dfl / mtp7 / mtp15) for head b3 are in data/d20_runs.jsonl; up to 8 tickets. Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
while pgrep -f "turns/turn53q.sh" > /dev/null; do sleep 30; done
done_n() { grep -c '"head": "b3"' data/d20_runs.jsonl 2>/dev/null || echo 0; }
for t in 1 2 3 4 5 6 7 8; do
  [ "$(done_n)" -ge 33 ] && exit 0
  echo "$(date '+%m-%d %H:%M:%S') turn47q: D-20 ($(done_n) / 33 runs saved); queueing in 15 min (ticket $t)" >> $LOG
  sleep 900
  PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-20 SWITCH45: DFlash2 vs the trained MTP head b3 vs the type switch in Leap's live loop (turn47)" >> $LOG 2>&1 || exit 1
  ./run_locked.sh 15 turns/turn47.sh
done
