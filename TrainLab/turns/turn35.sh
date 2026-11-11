#!/bin/sh
# D-16 HCFULL part 1 (MTP Head, Thu 10-29; REPORT 'D-16'). One quiet lock: the 125B's final pre-mixer widened stream for
# L1-20's 109 training windows, the code / gsm walk sets and the 12 agent turns (sticky/d16_extract.py -> data/d16w,
# resumable; ~2 GB). Guards: >= 30 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-16a $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
C=51; [ "$F" -ge 42 ] && C=128                       # more resident experts = faster short prefills, if there is room
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d16_extract.py; exit; fi
  done ) & G=$!
D16_C=$C D16_ONLY="l120:,tf_code,tf_gsm,agent" $PY sticky/d16_extract.py 2>&1 | grep --line-buffered -v "Token indices" | grep -v "moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), $(ls data/d16w/*.npz 2>/dev/null | wc -l) of 205 saved (C=$C), swap $(swap) MB (start $SW0)"
