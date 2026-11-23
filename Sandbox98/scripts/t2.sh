#!/bin/bash
# T2 (results/T2_TRUNCATION.md, lines first): the context-truncation oracle, 50 MBPP windows, K = 0 / 16 / 64 / 256.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=10 MEMGB=5
guarded t2 logs/t2.log 25 $PY sb/trunc.py --ks 0,16,64,256 --n 50 --every 3 || exit 1
echo "t2 done"
