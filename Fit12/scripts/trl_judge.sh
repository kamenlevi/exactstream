#!/bin/sh
# Fit12 (Wed 4 Nov 22:4x; NewIdeas 22:33 / the orchestrator's QUEUE 22:3x): runs /Users/kamen/Fit12/scripts/fit_trl_judge.py
# (the screen-7 TRL store vs MSB-3 on routed held-out inputs per domain) for layers 7 18 30 41, all 512 experts, under
# peer wait + a share lock (CPU only, 4 threads, MEM 2 GB). Log /Users/kamen/Fit12/data/trl_judge/judge.log. Usage: sh trl_judge.sh
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; J=$F/data/trl_judge
mkdir -p $J
peer wait --timeout 60m >/dev/null 2>&1 || { echo "judge: blocked by a quiet ticket for 60 min; not started"; exit 3; }
peer lock share 20m --cpu 400 --gpu 0 "Fit12 TRL routed judge (NewIdeas 22:33): store vs MSB-3 per domain at L7/18/30/41, CPU only, 4 threads, MEM 2 GB, ~5 min" >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR trl judge start: fit_trl_judge.py 7,18,30,41 0-511 (NewIdeas 22:33)" >> $LOG
env VECLIB_MAXIMUM_THREADS=4 OMP_NUM_THREADS=4 $PY -B $F/scripts/fit_trl_judge.py 7,18,30,41 0-511 > $J/judge.log 2>&1
rc=$?
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR trl judge end (rc $rc): $(ls $J/judge_L*.json 2>/dev/null | wc -l | tr -d ' ') layer files in $J" >> $LOG
grep -E 'val (code|multi|instr|all)|U   all|fit all|Error|Traceback' $J/judge.log
exit $rc
