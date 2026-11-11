#!/bin/sh
# D-24 MTPTOP3 (MTP Head, Sun 11-01): the MTP head's top-1 p and the true token's rank per anchor and depth on the held-out sets
# (sticky/d24_mtptop3.py, head b5) -> data/d24/; the tree / cut policies are scored on the CPU after. Shared GPU, peak ~7-11 GB
# (d22_ttt eval measured 6.8 / 10.8): gate NO-SWAP >= 20 GB, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d24_mtptop3.py"; sleep 10; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-24 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 20 ] || { echo "== needs >= 20 GB NO-SWAP: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f d24_mtptop3.py; exit; fi
  done ) & G=$!
HF_HUB_OFFLINE=1 /usr/bin/time -l $PY sticky/d24_mtptop3.py --run b5 --sets ifeval,multi,mgsm,gsm,code 2>&1 | grep --line-buffered "^  [a-z]* (\|Error\|Traceback\|peak memory footprint"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
