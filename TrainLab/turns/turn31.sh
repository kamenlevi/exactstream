#!/bin/sh
# D-13 DIVERGE (MTP Head, Thu 10-29; REPORT 'D-13'). One quiet lock: three rapid arms from L1-20 (Leap's R-05 protocol:
# 21 steps, L1-20's recipe; sticky/d13_train.py --div fkl | sharp | tvd), each walked at block 16 on code + math, then the
# ratios (CPU). A walked arm is skipped on a re-run; a cut training restarts. Guards: >= 34 GB free to start (L1-20 peaked
# 33.7 GB), stop if swap grows > 2 GB. Read-only on ~/Leap.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-13 $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 34 ] || { echo "== < 34 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d13_train.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
for arm in U S V; do
  case $arm in U) DIV=fkl;; S) DIV=sharp;; V) DIV=tvd;; esac
  W=$HOME/TrainLab/data/d13_$arm.safetensors; ST=data/d13_$arm.state.json
  if [ -s data/d02b_walk_d13$arm.txt ]; then echo "== arm $arm already walked"; continue; fi
  if ! grep -q '"status": "done"' $ST 2>/dev/null; then
    rm -f $ST
    echo "== train $arm ($DIV) $(date +%H:%M:%S)"
    $PY sticky/d13_train.py --data ~/Leap/data/l120 --init $HOME/Leap/data/l120_ft.safetensors --noval --out $W --bs 32 \
      --micro 1 --epochs 0.2 --minutes 6 --pmask 0.5 --div $DIV 2>&1 | grep --line-buffered -v "Token indices"
    grep -q '"status": "done"' $ST 2>/dev/null || { echo "== arm $arm training did not finish"; kill $G; exit 1; }
  fi
  echo "== walk $arm $(date +%H:%M:%S)"
  TL_DRAFT=$W TL_WALK_TAG=_d13$arm HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
done
kill $G 2>/dev/null
taskpolicy -b nice -n 10 $PY sticky/d13_ratio.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
