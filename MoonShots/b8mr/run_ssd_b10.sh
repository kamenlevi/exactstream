#!/bin/sh
# B10 v2 read units (NewIdeas Sun ~07:30): cold F_NOCACHE preads in the loader's pattern, QD 1-16: GB/s and per-call fixed
# cost of 1.29 MB bases and 0.9-1.2 MB residuals vs today's 2.16 MB MSB-3 records (defer/ssd_prefix). Quiet lock; unlocks.
M=$HOME/MoonShots; LOG=$M/data/ssd_b10.txt
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
trap 'peer unlock > /dev/null 2>&1' EXIT
echo "$(date +%H:%M:%S) start" >> $LOG
$M/defer/ssd_prefix $HOME/BigMoE/models/bare/records3 20 >> $LOG 2>&1
echo "$(date +%H:%M:%S) finished" >> $LOG
