#!/bin/sh
# D-22 run 'b5' (MTP Head, Sun 11-01; NewIdeas 07:4x RANKED_45GB #17): b4's recipe (L1-20's 99 code / math windows + all 147
# chat windows, depths 1-15, wdecay 0.9, lr 1e-5 cosine, one epoch) + Spec-AUF (arXiv 2607.01893: the loss support stops at
# the chain's first predicted failure; d22_ttt --auf), so b4 vs b5 isolates AUF. Then its read-out with my forward on
# held-out code / gsm / ifeval / multi / mgsm (b4's: data/d22_eval_b4.txt). Shared GPU (training numbers are not speeds).
# Guards: NO-SWAP >= 33 GB (peak ~25 + 8, shared), stop if swap grows > 2 GB; the trainer waits while PAUSE exists and
# checkpoints on SIGTERM / every 5 min (resumable: the schedule continues).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -TERM -f "d22_ttt.py"; sleep 20; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-22 b5 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB, $(ls data/d22c/*.npz 2>/dev/null | wc -l) chat windows"
[ "$F" -ge 33 ] || { echo "== needs >= 33 GB NO-SWAP: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -TERM -f d22_ttt.py; exit; fi
  done ) & G=$!
if ! grep -q '"status": "done"' ckpt/d22_b5.json 2>/dev/null; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py train --run b5 --data l120,d22c --depth 15 --lr 1e-5 --wdecay 0.9 --epochs 1 --sched cos --auf 2>&1 \
    | grep --line-buffered "^\[d22\]\|Error\|Traceback" &
  wait $!
fi
if grep -q '"status": "done"' ckpt/d22_b5.json 2>/dev/null && [ ! -s data/d22_eval_b5.txt ]; then
  HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run b5 --sets code,gsm,ifeval,multi,mgsm 2>&1 | grep --line-buffered "^  [a-z]* (\|Error\|Traceback"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
