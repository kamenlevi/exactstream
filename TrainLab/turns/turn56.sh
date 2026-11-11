#!/bin/sh
# D-22 run 'b4' (MTP Head, Sat 10-31 night): the training-time-test MTP head on L1-20's 99 code / math windows + ALL chat
# windows (data/d22c: IFEval 97 + MGSM 50), depths 1-15, wdecay 0.9, lr 1e-5 cosine, one epoch (b3's recipe with twice the
# chat data); then its read-out with my forward on held-out code / gsm (+ ifeval / multi / mgsm when their streams exist).
# Resumable (the schedule continues). Quiet tickets tonight (orchestrator 01:4x). Guards: NO-SWAP >= 30 GB (peak ~25 + 4),
# stop if swap grows > 2 GB; the trainer waits while PAUSE exists and checkpoints on SIGTERM / every 5 min.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d22_ttt.py"; sleep 20; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-22 b4 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB, $(ls data/d22c/*.npz 2>/dev/null | wc -l) chat windows"
[ "$F" -ge 30 ] || { echo "== needs >= 30 GB NO-SWAP: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f d22_ttt.py; exit; fi
  done ) & G=$!
if ! grep -q '"status": "done"' ckpt/d22_b4.json 2>/dev/null; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py train --run b4 --data l120,d22c --depth 15 --lr 1e-5 --wdecay 0.9 --epochs 1 --sched cos 2>&1 \
    | grep --line-buffered "^\[d22\]\|Error\|Traceback" &
  wait $!
fi
if grep -q '"status": "done"' ckpt/d22_b4.json 2>/dev/null && [ ! -s data/d22_eval_b4.txt ]; then
  S=code,gsm
  [ "$(ls data/d16w/tf_ifeval_*.npz 2>/dev/null | wc -l)" -ge 30 ] && S=$S,ifeval
  [ "$(ls data/d16w/tf_multi_*.npz 2>/dev/null | wc -l)" -ge 24 ] && S=$S,multi
  [ "$(ls data/d16w/mgsm_*.npz 2>/dev/null | wc -l)" -ge 30 ] && S=$S,mgsm
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run b4 --sets $S 2>&1 | grep --line-buffered "^  [a-z]* (\|Error\|Traceback"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
