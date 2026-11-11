#!/bin/sh
# D-22 read-out b4 vs b5 (MTP Head, Sun 11-01): b5 = b4's recipe + Spec-AUF (turn60 trained it; its read-out crashed on short
# chat answers, fixed in d22_ttt.py eval). Both heads on the same five held-out sets (teacher-forced chain, walked as D-17):
# b4 vs b5 isolates AUF. Shared GPU (tokens per pass, not speeds). Gate: NO-SWAP >= 20 GB (eval peak unmeasured, <~12 GB:
# measured here by /usr/bin/time), stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d22_ttt.py eval"; sleep 10; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-22 read-out b4 / b5 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 20 ] || { echo "== needs >= 20 GB NO-SWAP: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f "d22_ttt.py eval"; exit; fi
  done ) & G=$!
for RUN in b5 b4; do
  if ! grep -q "^  mgsm" data/d22_eval_$RUN.txt 2>/dev/null; then
    HF_HUB_OFFLINE=1 /usr/bin/time -l $PY sticky/d22_ttt.py eval --run $RUN --sets code,gsm,ifeval,multi,mgsm 2>&1 \
      | grep --line-buffered "^  [a-z]* (\|Error\|Traceback\|peak memory footprint\|^== "
  fi
done
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
