#!/bin/sh
# D-17 MTPDEEP part 2 (MTP Head, Thu 10-29; REPORT 'D-17'). One quiet lock: DFlash2's features + the 125B's top-32 on the
# D-17 decode's own texts (sticky/d02a_extract.py, D02A_SRC=d17 -> data/d17f), then DFlash2 L1-20 drafted at EVERY
# position (sticky/d02b_walk.py, TL_WALK_EVERY) on those texts and on D-02a's agent turns, then the router read-out
# (CPU). Needs part 1 (32 files in data/d17). Guards: >= 30 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-17b $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$(ls data/d17/*.npz 2>/dev/null | wc -l)" -ge 32 ] || { echo "== part 1 incomplete: not starting"; exit 1; }
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d02a_extract.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
echo "== capture $(date +%H:%M:%S)"
D02A_SRC=d17 D02A_OUT=$HOME/TrainLab/data/d17f $PY sticky/d02a_extract.py 2>&1 | grep --line-buffered -v "Token indices" | grep -v "moe-sorted-gather"
[ "$(ls data/d17f/*.npz 2>/dev/null | wc -l)" -ge 32 ] || { echo "== capture incomplete"; kill $G; exit 1; }
echo "== every-position walk $(date +%H:%M:%S)"
D=$HOME/TrainLab/data
for S in code=$D/d17f/tf_code_*.npz gsm=$D/d17f/tf_gsm_*.npz ifeval=$D/d17f/tf_ifeval_*.npz multi=$D/d17f/tf_multi_*.npz \
         agent=$D/d02a/agent_*.npz; do               # one set per walk, skipped when saved (resumable across locks)
  [ -s "$D/d02b_walk_d17e_${S%%=*}.npz" ] && continue
  TL_WALK_EVERY=1 TL_WALK_TAG=_d17e TL_WALK_SETS="$S" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
done
kill $G 2>/dev/null
taskpolicy -b nice -n 10 $PY sticky/d17_route.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
