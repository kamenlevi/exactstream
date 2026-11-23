#!/bin/bash
# WEEK98 lever 1c (2B LoRA toward the 125B) then the S5 no-download analog (results/LEVERS_A.md, pre-registered 01:3x).
# Each: a smoke on 2-3 files first; stop if it fails. Guarded: yields to quiet tickets, PAUSE, NO-SWAP, swap growth.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 MEMGB=7 GUARD_RUN=levers_1c_s5
guarded 1c_smoke logs/lever1c_smoke.log 5 $PY sb/lever1c.py --smoke 3 --minutes 0.1 --evals 0.05 --tag smoke || { echo "1c smoke failed"; exit 1; }
grep -q "DONE verdict" logs/lever1c_smoke.log || { echo "1c smoke: no DONE line"; exit 1; }
guarded 1c logs/lever1c_a.log 18 $PY sb/lever1c.py --minutes 8 --evals 1,2,4 --tag a || exit 1
echo "1c done: $(grep 'DONE verdict' logs/lever1c_a.log)"
guarded s5_smoke logs/s5inject_smoke.log 5 $PY sb/s5inject.py --smoke 2 --minutes 0.1 --evals 0.05 || { echo "s5 smoke failed"; exit 1; }
grep -q "DONE verdict" logs/s5inject_smoke.log || { echo "s5 smoke: no DONE line"; exit 1; }
guarded s5 logs/s5inject.log 20 $PY sb/s5inject.py --minutes 8 --evals 2,4 || exit 1
echo "s5 done: $(grep 'DONE verdict' logs/s5inject.log)"
