#!/bin/sh
# the look-ahead turn (turn45_la) queues only once the combination turn (turn45_cmb.sh 41 180, pid 69085) has exited
! ps -p 69085 -o command= 2>/dev/null | grep -q turn45_cmb
