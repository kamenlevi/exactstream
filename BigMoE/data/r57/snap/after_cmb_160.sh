#!/bin/sh
# mission 20 queues only once the combination turn (turn45_cmb.sh 30 150, wrapper pid 27020) has exited
! ps -p 27020 -o command= 2>/dev/null | grep -q turn45_cmb
