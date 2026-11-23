#!/bin/bash
# QAT 4-bit chain head (results/LEVERS_A.md): smoke, then the full run. Guarded: yields to quiet tickets and resumes.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=qat GUARD_KILL_WAIT=15 MEMGB=6 GUARD_NOSWAP=14
guarded qat_smoke logs/qat_smoke.log 6 $PY sb/qat.py --smoke || { echo "qat smoke failed"; exit 1; }
grep -q "DONE" logs/qat_smoke.log || { echo "qat smoke: no DONE line"; exit 1; }
guarded qat logs/qat.log 30 $PY sb/qat.py --steps 200 || exit 1
echo "qat done: $(grep 'DONE' logs/qat.log)"
