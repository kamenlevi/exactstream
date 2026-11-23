#!/bin/bash
# S5b (results/LEVERS_A.md): smoke, then the full run. Guarded: yields to quiet tickets and resumes (sb/resume.py).
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=s5b GUARD_KILL_WAIT=15 MEMGB=7 GUARD_NOSWAP=15
guarded s5b_smoke logs/s5b_smoke.log 6 $PY sb/s5b.py --smoke 2 --minutes 0.1 --evals 0.05 || { echo "s5b smoke failed"; exit 1; }
grep -q "DONE verdict" logs/s5b_smoke.log || { echo "s5b smoke: no DONE line"; exit 1; }
guarded s5b logs/s5b.log 30 $PY sb/s5b.py --minutes 8 --evals 1,2,4 || exit 1
echo "s5b done: $(grep 'DONE verdict' logs/s5b.log)"
