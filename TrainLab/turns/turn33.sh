#!/bin/sh
# D-17 MTPDEEP part 1 (MTP Head, Thu 10-29; REPORT 'D-17'). One quiet lock: plain greedy decode of D-05's 32 prompts (8
# code / gsm / ifeval / multi) with the native MTP head drafting a 15-token chain at every position (sticky/
# d17_mtpchain.py, resumable). Guards: >= 30 GB free to start, stop if swap grows > 2 GB. Read-only on ~/BigMoE,
# ~/SpecDecode, ~/Chat1000.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-17a $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d17_mtpchain.py; exit; fi
  done ) & G=$!
$PY sticky/d17_mtpchain.py 0.25 300 2>&1 | grep --line-buffered -v "Token indices"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), $(ls data/d17/*.npz 2>/dev/null | wc -l) of 32 saved, swap $(swap) MB (start $SW0)"
