#!/bin/sh
# D-22 run 'b3' (MTP Head, Fri 10-30 21:3x; SHARED GPU run, a training screen, FAST INTEGRATION): the training-time-test
# MTP head on code / math (L1-20's 99 windows) + chat (IFEval 97 + MGSM 50 from sticky/d22_capture.py), depths 1-15,
# wdecay 0.9, lr 1e-5 with a cosine decay to 10 %, ONE epoch; then its read-out (my forward) on held-out code / gsm /
# ifeval / multi / mgsm. Resumable (the schedule continues). Guards: free memory >= 26 GB (peak ~16-18 GB + 8), stop if
# swap grows > 2 GB; the trainer waits while PAUSE exists and checkpoints on SIGTERM / every 5 min.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d22_ttt.py"; sleep 20; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22 b3 (shared) $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 26 ] || { echo "== < 26 GB free: not starting"; exit 1; }
[ "$(ls data/d22c/*.npz 2>/dev/null | wc -l)" -ge 70 ] || { echo "== chat windows not captured yet (turn51)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f d22_ttt.py; exit; fi
  done ) & G=$!
if ! grep -q '"status": "done"' ckpt/d22_b3.json 2>/dev/null; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py train --run b3 --data l120,d22c --depth 15 --lr 1e-5 --wdecay 0.9 --epochs 1 --sched cos 2>&1 \
    | grep --line-buffered "^\[d22\]\|Error\|Traceback" &
  wait $!
fi
if grep -q '"status": "done"' ckpt/d22_b3.json 2>/dev/null && [ ! -s data/d22_eval_b3.txt ]; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run b3 --sets code,gsm 2>&1 | grep --line-buffered "^  [a-z]* (\|Error\|Traceback"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
