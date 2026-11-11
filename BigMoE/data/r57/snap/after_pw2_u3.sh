#!/bin/sh
# the U3 turn (turn45_u3) queues only once mission 20's turn (turn45_pw.sh 40 90, pid 68957) has exited
! ps -p 68957 -o command= 2>/dev/null | grep -q turn45_pw
