#!/bin/sh
# PK v2 pre-registered bit test (M-46 line, resumed): PK == SF on 120 real trees (HE+ tree-16 48, HE+ tree-32 24,
# MBPP tree-16 48), r57's live Q8H32 head. Share lock (bitwise needs no quiet machine).
cd "$(dirname "$0")"
P=$HOME/BigMoE/.venv/bin/python
export MAXGB=14 SWAPGB=1
LOG=pk_bit120_he16.log sh ../jcc_guard.sh $P -I -B pk_tree_test.py --set heplus24 --windows 0,1,2,3 --anchors 12 --n 15 > /dev/null 2>&1
LOG=pk_bit120_he32.log sh ../jcc_guard.sh $P -I -B pk_tree_test.py --set heplus24 --windows 0,1 --anchors 12 --n 31 > /dev/null 2>&1
LOG=pk_bit120_mb16.log sh ../jcc_guard.sh $P -I -B pk_tree_test.py --set mbppbank --windows 0,1,2,3 --anchors 12 --n 15 > /dev/null 2>&1
grep -h "pk_tree_test\]\|DIFFERENT\|fell back" pk_bit120_he16.log pk_bit120_he32.log pk_bit120_mb16.log
