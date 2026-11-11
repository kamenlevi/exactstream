#!/bin/sh
# Fit12 (Wed 4 Nov 22:5x; NewIdeas 22:33): /Users/kamen/Fit12/scripts/fit_trl_judge_map.py (the screen-7 TRL store vs MSB-3 on the
# map's own answer rows, Breakthrough's xcapture, read-only) for layers 7 18 30 41 under peer wait + a share lock (CPU only,
# 4 threads, MEM 2 GB). Log /Users/kamen/Fit12/data/trl_judge/judge_map.log. Usage: sh trl_judge_map.sh
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; J=$F/data/trl_judge
mkdir -p $J
peer wait --timeout 60m >/dev/null 2>&1 || { echo "map judge: blocked by a quiet ticket for 60 min; not started"; exit 3; }
peer lock share 20m --cpu 400 --gpu 0 "Fit12 TRL judge on the map's own inputs (NewIdeas 22:33), CPU only, 4 threads, MEM 2 GB, ~3 min" >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR trl map judge start: fit_trl_judge_map.py 7,18,30,41 (xcapture rows, NewIdeas 22:33)" >> $LOG
env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 $PY -B $F/scripts/fit_trl_judge_map.py 7,18,30,41 > $J/judge_map.log 2>&1
rc=$?
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR trl map judge end (rc $rc)" >> $LOG
grep -E 'r2-9|windows|Error|Traceback' $J/judge_map.log
exit $rc
