#!/bin/sh
# Fit12 (Thu 5 Nov 18:5x; the orchestrator 18:5x): runs /Users/kamen/Fit12/scripts/fit_halfk_check.py (independent check of
# Fable A's HALF-K numbers) for layers 2 12 24 36 46 under peer wait + a share lock (CPU only, 4 threads, MEM 2 GB,
# python -I). Log /Users/kamen/Fit12/data/halfk/halfk.log. Usage: sh halfk.sh
F=/Users/kamen/Fit12; PY=/Users/kamen/BigMoE/.venv/bin/python; LOG=$F/logs/summary.txt; D=$F/data/halfk
mkdir -p $D
peer wait --timeout 30m >/dev/null 2>&1 || { echo "halfk: blocked by a quiet ticket for 30 min; not started"; exit 3; }
peer lock share 20m --cpu 400 --gpu 0 "Fit12 HALF-K check of Fable A's numbers (the orchestrator 18:5x), CPU only, 4 threads, MEM 2 GB, ~5 min" >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR halfk start: fit_halfk_check.py 2,12,24,36,46 (the orchestrator 18:5x; swap $(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/') MB)" >> $LOG
$PY -I -B $F/scripts/fit_halfk_check.py 2,12,24,36,46 > $D/halfk.log 2>&1
rc=$?
peer unlock >/dev/null 2>&1
echo "$(date +%H:%M:%S) SCR halfk end (rc $rc)" >> $LOG
grep -E '\] L[0-9]+ (val all|map all|map r2-9)|STOP|done|Error|Traceback' $D/halfk.log
exit $rc
