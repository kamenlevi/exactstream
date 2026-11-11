#!/bin/sh
# D-22 / D-20 captures (MTP Head, Fri 10-30). One quiet lock, resumable: (1) the trunk's widened stream for IFEval /
# multi / MGSM (sticky/d16_extract.py -> data/d16w; D-22's chat read-out needs it), (2) DFlash2's features + the 125B's
# top-32 for MGSM (sticky/d02a_extract.py -> data/d02a), (3) DFlash2 L1-20 on MGSM: the on-path walk (kept per check)
# and the every-position walk (D-19-style drafter choice). MGSM = 30 items, 3 per language, the exact ref's answers.
# Guards: >= 30 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== captures $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 30 ] || { echo "== < 30 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d16_extract.py; pkill -f d02a_extract.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
C=51; [ "$F" -ge 42 ] && C=128
D16_C=$C D16_ONLY="tf_ifeval,tf_multi,mgsm" $PY sticky/d16_extract.py 2>&1 | grep --line-buffered -v "Token indices" | grep -v "moe-sorted-gather"
D02A_C=$C D02A_ONLY="mgsm" $PY sticky/d02a_extract.py 2>&1 | grep --line-buffered -v "Token indices" | grep -v "moe-sorted-gather"
D=$HOME/TrainLab/data
if [ "$(ls $D/d02a/mgsm_*.npz 2>/dev/null | wc -l)" -ge 30 ]; then
  [ -s $D/d02b_walk_d02a_mgsm.npz ] || TL_WALK_TAG=_d02a TL_WALK_SETS="mgsm=$D/d02a/mgsm_*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
  [ -s $D/d02b_walk_d17e_mgsm.npz ] || TL_WALK_EVERY=1 TL_WALK_TAG=_d17e TL_WALK_SETS="mgsm=$D/d02a/mgsm_*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S): d16w $(ls $D/d16w | wc -l) files, d02a mgsm $(ls $D/d02a/mgsm_*.npz 2>/dev/null | wc -l), swap $(swap) MB (start $SW0)"
