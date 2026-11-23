#!/bin/bash
# S5c (results/LEVERS_A.md): smoke, then the full run. Guarded: yields to quiet tickets and resumes (sb/resume.py).
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=s5c GUARD_KILL_WAIT=15 MEMGB=7 GUARD_NOSWAP=15
guarded s5c_smoke logs/s5c_smoke.log 6 $PY sb/s5c.py --smoke 2 --minutes 0.1 --evals 0.05 || { echo "s5c smoke failed"; exit 1; }
grep -q "DONE verdicts" logs/s5c_smoke.log || { echo "s5c smoke: no DONE line"; exit 1; }
guarded s5c logs/s5c.log 35 $PY sb/s5c.py --minutes 9 --evals 3,6 || exit 1
echo "s5c done: $(grep 'DONE verdicts' logs/s5c.log)"
