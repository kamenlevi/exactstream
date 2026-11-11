#!/bin/sh
# D-22 real-runtime read-out (MTP Head, Fri 10-30; REPORT 'D-22'). One quiet lock, resumable across locks: D-05's
# prompts decoded by MTPLX with the trained head 'base' installed (sticky/d22_real.py; the same greedy decode as D-17,
# k_mtp 15, so the answers must equal D-17's token for token), code + gsm first, then ifeval + multi; then the paired
# read-out vs the stock head (data/d17). Guards: >= 30 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22 real $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d22_real.py; exit; fi
  done ) & G=$!
FR=0.10; [ "$F" -ge 42 ] && FR=0.25
HF_HUB_OFFLINE=1 $PY sticky/d22_real.py decode base $FR 300 code,gsm,ifeval,multi 2>&1 | grep --line-buffered "^\[d22r\]\|Error\|Traceback"
taskpolicy -b nice -n 10 $PY sticky/d22_real.py read base 2>&1 | grep -v "Token indices"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
