#!/bin/sh
# the combination turn (turn45_cmb) queues only once mission 20's turn (pid 68957) AND the U3 smoke's turn (pid 82997,
# the orchestrator 11-03 15:5x: U3 right after mission 20) have both exited
! ps -p 68957 -o command= 2>/dev/null | grep -q turn45_pw && ! ps -p 82997 -o command= 2>/dev/null | grep -q turn45_u3
