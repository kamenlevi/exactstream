#!/bin/zsh
# G2-PRICE + HC8 K1M (Leap, Sun 2026-11-08 00:5x): K1M bit checks (kernel + ref2 candidate _gr), the HC8-F timing at
# 1-4 rows, then the expert one-call price at 1-16 rows (NAX 1cq vs 3c-N). Run under Leap's own lock; -> data/*.txt
cd ~/Leap/scripts || exit 1
PY=~/BigMoE/.venv/bin/python
D=~/Leap/data
COND=${G2_COND:-"quiet lock"}
avail() { vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} END {printf "%.1f", (f+i+s)*ps/1e9}'; }
A=$(avail)
echo "run_g2 $(date '+%a %H:%M:%S') ($COND): avail ~$A GB, swap $(sysctl -n vm.swapusage | awk '{print $6}')"
awk -v a="$A" 'BEGIN {exit !(a >= 10)}' || { echo "avail $A GB < 10: not run"; exit 2; }
echo "== hc8f check + K1M check + bench ($(date '+%H:%M:%S'))"
$PY hc8f.py bench 2>&1 | tee -a $D/hc8f_check.txt
echo "== K1M _gr check ($(date '+%H:%M:%S'))"
$PY k1m_gr_check.py 2>&1 | tee -a $D/hc8f/k1m_gr.txt
echo "== HC8-G check + bench ($(date '+%H:%M:%S'))"
$PY hc8g.py bench 2>&1 | tail -8
echo "== G2-PRICE ($(date '+%H:%M:%S'))"
BIGMOE_POM_LP1=1 G2_COND="$COND" $PY g2_bench.py 2>&1 | tail -15
echo "run_g2 done $(date '+%H:%M:%S'), swap $(sysctl -n vm.swapusage | awk '{print $6}')"
