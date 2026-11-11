#!/bin/sh
# D-22 MTPTTT part 1 (MTP Head, Fri 10-30; REPORT 'D-22'). One quiet lock: the chain-reproduction gate (my training-time-
# test forward with the untrained head vs MTPLX's real chains from D-17) and the untrained head's read-out on the
# held-out code / gsm answers (sticky/d22_ttt.py). Trunk experts at the minimum (C 16; the trunk is not run).
# Guards: >= 30 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22a $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d22_ttt.py; exit; fi
  done ) & G=$!
[ -s data/d22_gate.txt ] || HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py gate 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
[ -s data/d22_eval_stock.txt ] || HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run stock --sets code,gsm 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
