#!/bin/sh
# Bit check of bundle_b5c7c80 vs the live b5c7c80 tree: r57_draft_test per format, every draft() output digested
# (bundle_check.py). One run at a time; share lock MEM 14 GB.
cd "$(dirname "$0")"
P=$HOME/BigMoE/.venv/bin/python
export MAXGB=14 SWAPGB=1
LM=${LM:?LM = the dir holding the Leap screened_head.py}
date
for f in q8h32 c2 stored; do
  case $f in q8h32) PKV=1; X="--k 8";; c2) PKV=0; X="--k 8";; stored) PKV=0; X="--k 1";; esac
  R57_TREE_PK=$PKV LOG=bundle_check_${f}_live.log sh ../jcc_guard.sh $P -I -B bundle_check.py --live -- --fmt $f --set heplus24 --windows 2,3 --anchors 12 --n 15 $X > /dev/null 2>&1
  R57_TREE_PK=$PKV R57_LEAP_SCRIPTS=$LM LOG=bundle_check_${f}_bundle.log sh ../jcc_guard.sh $P -I -B bundle_check.py --bundle bundle_b5c7c80 -- --fmt $f --set heplus24 --windows 2,3 --anchors 12 --n 15 $X > /dev/null 2>&1
  date
done
grep -h "r57_draft_test\] \(q8h32\|c2\|stored\)\|check_install\|bundle_check\]\|Traceback\|Error" bundle_check_*_live.log bundle_check_*_bundle.log
