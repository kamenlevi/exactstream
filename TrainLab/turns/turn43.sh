#!/bin/sh
# D-22 screen of run 'base' at step 76 (MTP Head, Fri 10-30 16:4x): its online losses rose in steps 1-40. Paired:
# the untrained head's per-window loss on the same 76 windows in the same order and on the 10 validation windows,
# the step-76 head on the same; then the step-76 head's read-out (tokens per pass) on held-out code / gsm.
# Guards: >= 30 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22 screen $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d22_ttt.py; exit; fi
  done ) & G=$!
[ -s ckpt/d22_base76.safetensors ] || cp ckpt/d22_base.safetensors ckpt/d22_base76.safetensors
for R in stock base76; do
  [ -s data/d22_loss_$R.npz ] || HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py loss --run $R --of base --n 76 2>&1 | grep --line-buffered "^\[d22\]\|Error\|Traceback"
done
[ -s data/d22_eval_base76.txt ] || HF_HUB_OFFLINE=1 $PY sticky/d22_ttt.py eval --run base76 --sets code,gsm 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather\|tokenizer\]\|mtplx\]\|qwen4_exp\]"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
