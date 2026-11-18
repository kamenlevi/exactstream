#!/bin/sh
# bundle_check.sh's second pass: the live runs (the first pass's live runs had no script dir on sys.path under -I) and
# c2 with S22's checkpoint (dd_t30qa). One run at a time; share lock MEM 14 GB.
cd "$(dirname "$0")"
P=$HOME/BigMoE/.venv/bin/python
export MAXGB=14 SWAPGB=1
LM=${LM:?LM = the dir holding the Leap screened_head.py}
W=$HOME/datasets/deepdraft/ckpt/dd_t30qa.safetensors
A="--set heplus24 --windows 2,3 --anchors 12 --n 15"
date
R57_TREE_PK=1 LOG=bundle_check_q8h32_live.log sh ../jcc_guard.sh $P -I -B bundle_check.py --live -- --fmt q8h32 $A --k 8 > /dev/null 2>&1
R57_TREE_PK=0 LOG=bundle_check_stored_live.log sh ../jcc_guard.sh $P -I -B bundle_check.py --live -- --fmt stored $A --k 1 > /dev/null 2>&1
R57_TREE_PK=0 LOG=bundle_check_c2_live.log sh ../jcc_guard.sh $P -I -B bundle_check.py --live -- --fmt c2 --weights $W $A --k 8 > /dev/null 2>&1
R57_TREE_PK=0 R57_LEAP_SCRIPTS=$LM LOG=bundle_check_c2_bundle.log sh ../jcc_guard.sh $P -I -B bundle_check.py --bundle bundle_b5c7c80 -- --fmt c2 --weights $W $A --k 8 > /dev/null 2>&1
date
for f in q8h32 c2 stored; do for m in live bundle; do echo "== $f $m"; grep -h "r57_draft_test\] \(q8h32\|c2\|stored\)\|check_install\|bundle_check\]\|Traceback\|Error" bundle_check_${f}_${m}.log | cut -c1-400; done; done
