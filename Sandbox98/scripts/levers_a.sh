#!/bin/bash
# LEVERS_A (results/LEVERS_A.md, written first): keyframes, then kNN state memory, then the deep read-out for Leap.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=10 MEMGB=5
guarded keyframes logs/keyframes.log 25 $PY sb/trunc.py --ks "" --keyframes 1,4,8,16 --n 50 --every 3 || exit 1
[ -f sb/knn.py ] && guarded knn logs/knn.log 30 $PY sb/knn.py || echo "knn skipped"
guarded deep63 logs/deep63.log 25 $PY sb/evalx.py --runs s30 --sets mbpp,he --nm 50 --nh 30 --depth 64 --mode chain,true --out results/deep63.json || exit 1
echo "levers_a done"
