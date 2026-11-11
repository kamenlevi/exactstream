#!/bin/sh
# D-16 HCFULL part 2 (MTP Head, Thu 10-29; REPORT 'D-16'). One quiet lock: arm H = Leap's rapid protocol from L1-20 (21
# steps, L1-20's recipe) with the widened stream as an extra input (sticky/d16_train.py --hc, new projection at lr 1e-3);
# walks of H (code, math, agent; TL_HC) and of D-13's U on agent (U's code / math walks exist); then the ratios (CPU).
# Needs part 1 (data/d16w complete). Guards: >= 34 GB free to start, stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
freegb() { vm_stat | awk '/Pages free/{f=$3} /Pages inactive/{i=$3} /Pages purgeable/{p=$3} END{printf "%d", (f+i+p)*16384/1e9}'; }
SW0=$(swap); F=$(freegb)
echo "== D-16b $(date +%H:%M:%S): ${F} GB free, swap ${SW0} MB"
[ "$(ls data/d16w/*.npz 2>/dev/null | wc -l)" -ge 205 ] || { echo "== part 1 incomplete: not starting"; exit 1; }
[ "$F" -ge 34 ] || { echo "== < 34 GB free: not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then
      echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d16_train.py; pkill -f d02b_walk.py; exit
    fi
  done ) & G=$!
W=$HOME/TrainLab/data/d16_H.safetensors; D=$HOME/TrainLab/data
if ! grep -q '"status": "done"' data/d16_H.state.json 2>/dev/null; then
  rm -f data/d16_H.state.json
  echo "== train H $(date +%H:%M:%S)"
  $PY sticky/d16_train.py --hc $D/d16w --data ~/Leap/data/l120 --init $HOME/Leap/data/l120_ft.safetensors --noval --out $W \
    --bs 32 --micro 1 --epochs 0.2 --minutes 6 --pmask 0.5 --div fkl --hc-lr 1e-3 2>&1 | grep --line-buffered -v "Token indices"
  grep -q '"status": "done"' data/d16_H.state.json 2>/dev/null || { echo "== H training did not finish"; kill $G; exit 1; }
fi
echo "== walk H $(date +%H:%M:%S)"
TL_HC=$D/d16w TL_DRAFT=$W TL_WALK_TAG=_d16H TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz,agent=$D/d02a/agent_*.npz" \
  HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
[ -s data/d02b_walk_d13U_agent.npz ] || { echo "== walk U on agent $(date +%H:%M:%S)"
  TL_DRAFT=$D/d13_U.safetensors TL_WALK_TAG=_d13U TL_WALK_SETS="agent=$D/d02a/agent_*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"; }
kill $G 2>/dev/null
taskpolicy -b nice -n 10 $PY sticky/d16_read.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
