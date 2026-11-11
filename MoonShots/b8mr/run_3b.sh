#!/bin/sh
# 3B engine prerequisites (NewIdeas Sun ~07:00): (1) SSD prefix vs full reads (+ B10 trellis 1.29 / 1.35 MB) (defer/ssd_prefix, cold records3, loader
# pattern); (2) 2-bit (records4 tierglu4 tier 2) vs 3c 3-bit per pair at M = 4, 8 (bench_tierglu4.py, synthetic pools).
# Launch holding a quiet lock (SSD + GPU timing); unlocks at the end. Log data/3b_prereq.txt.
M=$HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; LOG=$M/data/3b_prereq.txt
sh $M/scripts/keepalive_while.sh $$ > /dev/null 2>&1 &
trap 'peer unlock > /dev/null 2>&1' EXIT
echo "$(date +%H:%M:%S) start" >> $LOG
$M/defer/ssd_prefix $HOME/BigMoE/models/bare/records3 20 >> $LOG 2>&1
echo "$(date +%H:%M:%S) ssd done" >> $LOG
PLANS=3333333333,2222222222,4433333333 MS=4,8 $PY $M/patches/planes_engine/bench_tierglu4.py >> $LOG 2>&1
echo "$(date +%H:%M:%S) finished" >> $LOG
