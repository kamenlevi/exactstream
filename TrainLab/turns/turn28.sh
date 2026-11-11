#!/bin/sh
# D-09 DSCALE (MTP Head, Wed 10-28 22:10; REPORT 'D-09 DSCALE'). One quiet lock: L1-20's recipe (sticky/d09_train.py =
# Leap's dfl2c_train.py verbatim) on the nested 1/4 and 1/2 subsets (data/d09_q, data/d09_h), then my D-02b walk
# (block 16, official 1-step loop) with each new drafter on the 24 held-out HumanEval + 60 GSM8K answers. A finished
# training is skipped on a re-run. Guards: >= 40 GB free to start, stop if swap grows > 2 GB. Read-only on ~/Leap.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-09 $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$F" -ge 40 ] || { echo "== < 40 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d09_train.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
for t in q h; do
  W=$HOME/TrainLab/data/d09_$t.safetensors
  ST=data/d09_$t.state.json
  if grep -q '"status": "done"' $ST 2>/dev/null; then
    echo "== d09_$t already trained"
  else
    R=""; [ -s $ST ] && R="--resume"
    echo "== train d09_$t $(date +%H:%M:%S)"
    $PY sticky/d09_train.py --data data/d09_$t --out $W --bs 32 --micro 1 --epochs 2 --minutes 50 --pmask 0.5 $R 2>&1 \
      | grep --line-buffered -v "Token indices"
    grep -q '"status": "done"' $ST 2>/dev/null || { echo "== d09_$t training did not finish"; kill $G; exit 1; }
  fi
  echo "== walk d09_$t $(date +%H:%M:%S)"
  TL_DRAFT=$W TL_WALK_TAG=_d09$t HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
done
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
