#!/bin/sh
# the combination turn queues only once the U3 turn (turn45_u3.sh 30 45, wrapper pid 26930) has exited
! ps -p 26930 -o command= 2>/dev/null | grep -q turn45_u3
