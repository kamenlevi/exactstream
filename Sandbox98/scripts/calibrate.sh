#!/bin/bash
# Fast-screen calibration (results/screens/CALIBRATION.md, written first). Runs after the long-screen chain; polite guards.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
until grep -q "long screens done\|giving up\|stopping" logs/long_screens.log 2>/dev/null || ! pgrep -f "scripts/long_screens.sh" >/dev/null; do sleep 30; done
source scripts/guard_lib.sh
guarded c12 logs/cal_c12.log 25 $PY sb/train_multi.py --tag cal12 --arms ctl:- --depth 12 --steps 1000 --save-at 100,200,300,500,700,1000 || exit 1
guarded c30 logs/cal_c30.log 25 $PY sb/train_multi.py --tag cal30 --arms ctl:- --depth 30 --steps 500 --save-at 100,200,300,500 || exit 1
R12=s30o; R31=s30o
for s in 100 200 300 500 700 1000; do R12="$R12,cal12_ctl_s$s"; done
for s in 100 200 300 500; do R12="$R12,cal30_ctl_s$s"; R31="$R31,cal30_ctl_s$s"; done
guarded eval15 logs/cal_eval15.log 25 $PY sb/evalx.py --runs $R12 --sets mbpp,he --nm 50 --nh 30 --n 1 --depth 15 --mode chain --out results/screens/cal_eval15.json || exit 1
guarded eval31 logs/cal_eval31.log 25 $PY sb/evalx.py --runs $R31 --sets mbpp,he --nm 50 --nh 30 --n 1 --depth 31 --mode chain --out results/screens/cal_eval31.json || exit 1
$PY sb/calreport.py | tee -a results/screens/CALIBRATION.md
echo "calibration done"
