#!/bin/bash
# For Leap (results/LEVERS_A.md): s30 chain + true state walked to depth 63, MBPP-50 + HE-30. Waits for quiet locks.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=30 MEMGB=4.5
guarded deep63 logs/deep63.log 25 $PY sb/evalx.py --runs s30 --sets mbpp,he --nm 50 --nh 30 --depth 64 --mode chain,true --out results/deep63.json || exit 1
echo "deep63 done"
