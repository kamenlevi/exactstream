#!/bin/sh
# MTP Head night queue 2 (Sun 11-01, orchestrator 01:0x: account-2 reset, hold off; night = big tests, 45 GB route first).
# The shipped-line A-B-A redo of D-22 (turn53: b3 vs the head installed untrained, H50 stack, the largest C that fits),
# adapt mode then the fixed 7-draft chain. On 10-31 02:01-02:04 all seven grants found NO-SWAP short after the 30 s
# settle and were released within seconds; now a short grant waits 15 min before the next ticket. Up to 4 tickets per
# mode; one quiet ticket at a time. Run in the background from ~/TrainLab.
cd $HOME/TrainLab
LOG=logs/turns.log
say() { echo "$(date '+%m-%d %H:%M:%S') night2_q: $*" >> $LOG; }
for MODE in adapt k7; do
  for t in 1 2 3 4; do
    [ -s data/d22_ship_b3_${MODE}_aba.json ] && break
    PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-22: shipped-line A/B $MODE (A-B-A, trained head b3; turn53)" >> $LOG 2>&1 || exit 1
    N=$(wc -l < logs/turn_job.out)
    D22_SHIP_MODE=$MODE ./run_locked.sh 15 turns/turn53.sh
    if tail -n +$((N + 1)) logs/turn_job.out | grep -q "not starting"; then
      say "A/B $MODE: memory short after the grant (try $t); next ticket in 15 min"; sleep 900
    fi
  done
done
say "done"
