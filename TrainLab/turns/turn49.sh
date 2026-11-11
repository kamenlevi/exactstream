#!/bin/sh
# D-22 run 'b2' (MTP Head, Fri 10-30 19:5x; SHARED GPU run: a training screen, FAST INTEGRATION): depths 1-15 (K15 is the
# target), wdecay 0.9, lr 1e-5 with a cosine decay to 10 %, ONE epoch (99 steps; 'base' over-fit in its second epoch);
# then its read-out on held-out code / gsm (my forward, paired later vs stock / base76). Resumable (the schedule
# continues). Guards: free memory >= 24 GB (peak ~14-16 GB + 8), stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d22_ttt.py"; sleep 20; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22 b2 (shared) $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 24 ] || { echo "== < 24 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f d22_ttt.py; exit; fi
  done ) & G=$!
if ! grep -q '"status": "done"' ckpt/d22_b2.json 2>/dev/null; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py train --run b2 --depth 15 --lr 1e-5 --wdecay 0.9 --epochs 1 --sched cos 2>&1 \
    | grep --line-buffered "^\[d22\]\|Error\|Traceback" &
  wait $!
fi
if grep -q '"status": "done"' ckpt/d22_b2.json 2>/dev/null && [ ! -s data/d22_eval_b2.txt ]; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run b2 --sets code,gsm 2>&1 | grep --line-buffered "^  code\|^  gsm\|Error\|Traceback"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
