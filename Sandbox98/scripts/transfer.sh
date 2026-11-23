#!/bin/bash
# Transfer checks 5 + 6 (TRANSFER.md, lines written first). After the depth-12 calibration; polite guards; the 3-arm batch
# declares MEM 5.4 GB and starts only when no other session runs a memory-heavy (>= 5 GB) job.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
until grep -q "calibration done" logs/calibrate.log 2>/dev/null || ! pgrep -f "scripts/calibrate.sh" >/dev/null; do sleep 30; done
source scripts/guard_lib.sh
export MEMGB=5.4 NO_MEM_OTHERS=1
guarded xfer_train logs/xfer.log 20 $PY sb/train_multi.py --tag xfer --arms ctl:-,cap2x:cap2x,tapsq:tapsq --depth 12 --steps 200 || exit 1
export MEMGB=4.5
R=xfer_ctl,xfer_cap2x@cap2x,xfer_tapsq@tapsq
guarded xfer_eval15 logs/xfer_eval15.log 15 $PY sb/evalx.py --runs $R --sets mbpp,he --nm 50 --nh 30 --depth 15 --mode chain --out results/screens/xfer_eval15.json || exit 1
guarded xfer_eval31 logs/xfer_eval31.log 20 $PY sb/evalx.py --runs $R --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/xfer_eval31.json || exit 1
$PY sb/xferreport.py
echo "transfer done"
