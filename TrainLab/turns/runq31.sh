#!/bin/sh
# detached runner: wait for my quiet lock, then run turn31 (D-13) under run_locked (MTP Head, Thu 10-29)
cd $HOME/TrainLab
PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 15m "MTP Head D-13: three 21-step drafter arms + walks" >> logs/turns.log 2>&1 || exit 1
./run_locked.sh 15 turns/turn31.sh
