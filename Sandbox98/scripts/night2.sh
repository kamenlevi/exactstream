#!/bin/bash
# 22:5x queue (orchestrator 22:3x): fresh pool -> calibration v2 -> transfer checks 5 + 6 -> fast batch sl1. Lines first
# (CALIBRATION.md v2, TRANSFER.md, FAST_SL1_LINE.md). Polite guards: quiet tickets, PAUSE, NO-SWAP, swap growth.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=15
# 1. 500 fresh Magicoder answers for pool2 (resumable)
export MEMGB=4
guarded capture_mc2 logs/capture_mc2.log 25 $PY sb/capture.py --sets mc2 --n 500 --max-tokens 512 --batch 24 || exit 1
$PY sb/addmc2.py
# 2. calibration v2: each budget its own re-annealed control
export MEMGB=3.6
for B in 100 200 400 800; do
  guarded cal2_d12_b$B logs/cal2_d12_b$B.log 15 $PY sb/train_multi.py --tag cal2_d12_b$B --arms ctl:- --depth 12 --steps $B || exit 1
done
for B in 200 400; do
  guarded cal2_d30_b$B logs/cal2_d30_b$B.log 15 $PY sb/train_multi.py --tag cal2_d30_b$B --arms ctl:- --depth 30 --steps $B || exit 1
done
R15=s30; for B in 100 200 400 800; do R15="$R15,cal2_d12_b${B}_ctl"; done; for B in 200 400; do R15="$R15,cal2_d30_b${B}_ctl"; done
export MEMGB=4.5
guarded cal2_eval15 logs/cal2_eval15.log 20 $PY sb/evalx.py --runs $R15 --sets mbpp,he --nm 50 --nh 30 --depth 15 --mode chain --out results/screens/cal2_eval15.json || exit 1
guarded cal2_eval31 logs/cal2_eval31.log 15 $PY sb/evalx.py --runs s30,cal2_d30_b200_ctl,cal2_d30_b400_ctl --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/cal2_eval31.json || exit 1
$PY sb/calreport2.py | tee -a results/screens/CALIBRATION.md
# 3. transfer checks 5 + 6 (200 steps, the 125B capacity check's length)
export MEMGB=5.4 NO_MEM_OTHERS=1
guarded xfer_train logs/xfer.log 20 $PY sb/train_multi.py --tag xfer --arms ctl:-,cap2x:cap2x,tapsq:tapsq --depth 12 --steps 200 || exit 1
unset NO_MEM_OTHERS; export MEMGB=4.5
R=s30,xfer_ctl,xfer_cap2x@cap2x,xfer_tapsq@tapsq
guarded xfer_eval15 logs/xfer_eval15.log 15 $PY sb/evalx.py --runs $R --sets mbpp,he --nm 50 --nh 30 --depth 15 --mode chain --out results/screens/xfer_eval15.json || exit 1
guarded xfer_eval31 logs/xfer_eval31.log 20 $PY sb/evalx.py --runs $R --sets mbpp,he --nm 50 --nh 30 --depth 31 --mode chain --out results/screens/xfer_eval31.json || exit 1
$PY sb/xferreport.py
# 4. fast batch sl1: distill + noisehead at the calibrated budget
BUD=$(python3 -c "import json; print(json.load(open('results/screens/calibration_v2.json'))['budget'] or '')" 2>/dev/null)
DEP=$(python3 -c "import json; print(json.load(open('results/screens/calibration_v2.json'))['depth'] or 12)" 2>/dev/null)
[ -n "$BUD" ] || { BUD=400; DEP=12; echo "no calibrated budget: sl1 at 400 steps, depth 12 (below / unknown budget)"; }
export MEMGB=5.4 NO_MEM_OTHERS=1
LINE=$(tr '\n' ' ' < results/screens/FAST_SL1_LINE.md | cut -c1-1100)
guarded fast_sl1 logs/fast_sl1.log 30 ./screen_fast.sh sl1 "distill,noisehead" "$LINE" $BUD $DEP
echo "night2 done"
