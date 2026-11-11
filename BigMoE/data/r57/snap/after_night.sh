#!/bin/sh
# mission 20's AFTER gate (turn45 AFTER=): queue only once tonight's DAY9 night turn (turn45_n9.sh, pid 11291) has exited
! ps -p 11291 -o command= 2>/dev/null | grep -q turn45_n9
