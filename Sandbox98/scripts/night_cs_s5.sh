#!/bin/bash
# WEEK98 CHAIN SIZE (orchestrator 01:3x; results/LEVERS_A.md) then S5 (no-download analog). Starts when the 1c runner
# has exited. Each screen: a smoke first. Guarded: yields to quiet tickets and resumes (logs/.done markers + sb/resume.py;
# the SIGTERM save gets GUARD_KILL_WAIT seconds before the kill).
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=night_cs_s5 GUARD_KILL_WAIT=15 MEMGB=7
while pgrep -f "levers_1c_s[5].sh" > /dev/null; do sleep 15; done
echo "$(date +%H:%M:%S) 1c runner has exited: $(tail -1 logs/levers_1c_s5.log)"
guarded cs_smoke logs/chainsize_smoke.log 6 $PY sb/chainsize.py --smoke || { echo "cs smoke failed"; exit 1; }
grep -q "DONE" logs/chainsize_smoke.log || { echo "cs smoke: no DONE line"; exit 1; }
guarded cs logs/chainsize.log 35 $PY sb/chainsize.py --steps 200 || exit 1
echo "cs done: $(grep 'DONE' logs/chainsize.log)"
guarded s5_smoke logs/s5inject_smoke.log 6 $PY sb/s5inject.py --smoke 2 --minutes 0.1 --evals 0.05 || { echo "s5 smoke failed"; exit 1; }
grep -q "DONE verdict" logs/s5inject_smoke.log || { echo "s5 smoke: no DONE line"; exit 1; }
guarded s5 logs/s5inject.log 25 $PY sb/s5inject.py --minutes 8 --evals 2,4 || exit 1
echo "s5 done: $(grep 'DONE verdict' logs/s5inject.log)"
