#!/bin/sh
# D-22 chat captures (MTP Head, Fri 10-30 21:2x; SHARED GPU run, FAST INTEGRATION; resumable across runs, C 51 ~15 GB):
# (1) chat training windows: IFEval 97 + MGSM 50, the exact ref's answers (sticky/d22_capture.py -> data/d22c + d16w);
# (2) the eval sets' widened streams: tf_ifeval / tf_multi / MGSM-30 (sticky/d16_extract.py -> data/d16w);
# (3) DFlash2's features + the 125B's top-32 for MGSM-30 (sticky/d02a_extract.py) and DFlash2 L1-20's walks on MGSM.
# Guards: free memory >= 24 GB (peak ~15 GB + 8), stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-22 chat captures (shared) $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 24 ] || { echo "== < 24 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d22_capture.py; pkill -f d16_extract.py; pkill -f d02a_extract.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
$PY sticky/d22_capture.py 51 2>&1 | grep --line-buffered "^\[d22c\]\|Error\|Traceback" | grep -v "^\[d22c\] [0-9]*/[0-9]* d22c_ifeval_[0-9]*[1-9]:"
D16_C=51 D16_ONLY="tf_ifeval,tf_multi,mgsm" $PY sticky/d16_extract.py 2>&1 | grep --line-buffered "done\|Error\|Traceback\|stopping"
D02A_C=51 D02A_ONLY="mgsm" $PY sticky/d02a_extract.py 2>&1 | grep --line-buffered "done\|Error\|Traceback\|stopping"
D=$HOME/TrainLab/data
if [ "$(ls $D/d02a/mgsm_*.npz 2>/dev/null | wc -l)" -ge 30 ]; then
  [ -s $D/d02b_walk_d02a_mgsm.npz ] || TL_WALK_TAG=_d02a TL_WALK_SETS="mgsm=$D/d02a/mgsm_*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered "mgsm\|rate per"
  [ -s $D/d02b_walk_d17e_mgsm.npz ] || TL_WALK_EVERY=1 TL_WALK_TAG=_d17e TL_WALK_SETS="mgsm=$D/d02a/mgsm_*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered "mgsm\|rate per"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S): d22c $(ls $D/d22c 2>/dev/null | wc -l) / 147, d16w $(ls $D/d16w | wc -l), d02a mgsm $(ls $D/d02a/mgsm_*.npz 2>/dev/null | wc -l) / 30, swap $(swap) MB (start $SW0)"
