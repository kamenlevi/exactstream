#!/bin/sh
# D-15a LANGPROBE (MTP Head, Thu 10-29; REPORT 'D-15a'). One quiet lock: pack D-02a's multilingual captures into the
# trainer's format (16 train / 8 held out), a rapid tune from L1-20 on the 16 (L1-20's recipe, 2 epochs), then walks:
# the 8 held-out answers, and code + math for forgetting; then the read-out (CPU). Guards: >= 34 GB free to start,
# stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-15a $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 34 ] || { echo "== < 34 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d13_train.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
HF_HUB_OFFLINE=1 $PY sticky/d15_pack.py || { kill $G; exit 1; }
W=$HOME/TrainLab/data/d15_L.safetensors
if ! grep -q '"status": "done"' data/d15_L.state.json 2>/dev/null; then
  rm -f data/d15_L.state.json
  echo "== train $(date +%H:%M:%S)"
  $PY sticky/d13_train.py --data $HOME/TrainLab/data/d15_multi --init $HOME/Leap/data/l120_ft.safetensors --noval --out $W \
    --bs 32 --micro 1 --epochs 2 --minutes 8 --pmask 0.5 --div fkl 2>&1 | grep --line-buffered -v "Token indices"
  grep -q '"status": "done"' data/d15_L.state.json 2>/dev/null || { echo "== training did not finish"; kill $G; exit 1; }
fi
echo "== walk $(date +%H:%M:%S)"
TL_DRAFT=$W TL_WALK_TAG=_d15L TL_WALK_SETS="mheld=$HOME/TrainLab/data/d15_held/*.npz,code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz" \
  HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
kill $G 2>/dev/null
taskpolicy -b nice -n 10 $PY sticky/d15_read.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
