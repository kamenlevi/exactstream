#!/bin/sh
# MTP Head's 4-h ticket (Mon 11-02 01:0x; orchestrator: E1 then E5 right after D-23c, Kamen expects the decisive results by
# morning / lunch): D-23c v2 at C 224 (turn68: turn62 with 6 gate retries) then E1 -> E5 on the DLOG set (night8_train in-lock
# mode: no queueing of its own, run_locked holds and keeps the lock). D-23c's verdict and E1 / E5's do not depend on each other.
cd $HOME/TrainLab
sh turns/turn68.sh; echo "== D-23c part rc $?"
touch data/D23C_DONE
NIGHT8_INLOCK=1 sh turns/night8_train.sh >> logs/night8.out 2>&1
echo "== E1 / E5 part rc $?"
