#!/bin/sh
# detached runner: wait for my quiet lock, then run turn30 under run_locked (MTP Head, Thu 10-29)
cd $HOME/TrainLab
PEER_QUEUE_TIMEOUT=21600 peer lock quiet --queue 8m "MTP Head D-02a: incoai-original DFlash2 walk on 5 domains" >> logs/turns.log 2>&1 || exit 1
./run_locked.sh 8 turns/turn30.sh
