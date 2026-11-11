#!/bin/sh
# S10 upper bound (NewIdeas Sun ~10:00): continuous-queue SSD GB/s at the B10 read units, QD 1-32 (defer/ssd_qd2). Quiet
# lock; unlocks.
M=$HOME/MoonShots; LOG=$M/data/ssd_qd2.txt
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
trap 'peer unlock > /dev/null 2>&1' EXIT
echo "$(date +%H:%M:%S) start" >> $LOG
$M/defer/ssd_qd2 $HOME/BigMoE/models/bare/records3 600 5 >> $LOG 2>&1
echo "$(date +%H:%M:%S) finished" >> $LOG
