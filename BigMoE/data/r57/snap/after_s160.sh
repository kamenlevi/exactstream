#!/bin/sh
# the U3 turn queues only once the wall-1 sprint turn (turn45_s160.sh 30 70, wrapper pid 28947) has exited
! ps -p 28947 -o command= 2>/dev/null | grep -q turn45_s160
