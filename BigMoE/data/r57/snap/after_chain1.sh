#!/bin/sh
# the continuation ticket queues only once the first night chain (turn45_s160.sh 25 330, bash pid 79984) has exited AND
# that chain gave the night up for memory (its 01:00 cutoff); otherwise it never queues (its NOTAFTER ends it)
! ps -p 79984 -o command= 2>/dev/null | grep -q turn45_s160 && grep -q "NO-SWAP .* < 33 at 01:00: giving the night to FLOOR" "$HOME/BigMoE/data/s160/night.log"
