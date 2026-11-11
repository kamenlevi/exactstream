#!/bin/sh
# D-22 MTPTTT part 2 (MTP Head, Fri 10-30; REPORT 'D-22'). One quiet lock: the chain-reproduction gate (if not yet
# passed), then training run 'base' (depth 8, lr 1e-5, wdecay 0.8, 2 epochs; resumable across locks) only if the gate
# passed (depth 1 >= 0.98 and depths 1-7 >= 0.95). Guards: >= 30 GB free to start, stop if swap grows > 2 GB; the
# trainer checkpoints on SIGTERM and every 5 min and waits while PAUSE exists.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d22_ttt.py"; sleep 20; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22b $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f d22_ttt.py; exit; fi
  done ) & G=$!
[ -s data/d22_gate.txt ] || HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py gate 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
OK=$(awk '/^  depth 1 / {print ($3 >= 0.98 && $9 >= 0.95) ? 1 : 0}' data/d22_gate.txt 2>/dev/null)
if [ "$OK" = "1" ]; then
  echo "== gate passed: train base $(date +%H:%M:%S)"
  HF_HUB_OFFLINE=1 TL_LIMIT=${TL_LIMIT:-720} $PY sticky/d22_ttt.py train --run base --depth 8 --lr 1e-5 --wdecay 0.8 --epochs 2 2>&1 \
    | grep --line-buffered -v "Token indices\|moe-sorted-gather" &
  wait $!
  if grep -q '"status": "done"' ckpt/d22_base.json 2>/dev/null && [ ! -s data/d22_eval_base.txt ]; then
    echo "== eval base $(date +%H:%M:%S)"
    HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run base --sets code,gsm 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
  fi
else
  echo "== gate NOT passed (or missing): no training"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
