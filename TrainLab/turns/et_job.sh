#!/bin/sh
# EARLYTAP job (MTP Head, Mon 11-02; REPORT 'EARLYTAP'), inside a held SHARE lock; every step resumable / skipped when done:
# 1. sticky/taps_extract.py: L1-20's windows -> data/taps_h5 / h4 / h3 (+ the h5 == ~/Leap/data/l120 self-check), and the 100
#    MBPP walk answers' raw taps -> data/taps_mbpp_walk (C 96 if NO-SWAP >= 33 GB, else 51; ~19-25 GB);
# 2. D-09's protocol on each set -> data/et_h5 / h4 / h3.safetensors (~31 min each, ~15 GB);
# 3. walks (block 16) with the same taps zeroed: HumanEval+, the MBPP walk set, the 12 agent windows; 4. sticky/et_read.py.
# Swap guard +2 GB on every step (stops that step; resumable).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f taps_extract.py; pkill -f "d09_train.py --data data/taps_"; pkill -f d02b_walk_match.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
guard() { SW0=$(swap); ( while sleep 10; do if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping $1"; pkill -f "$1"; exit; fi; done ) & G=$!; }
need() { i=0; while [ "$(noswap)" -lt $1 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== EARLYTAP: NO-SWAP < $1 GB for 10 min: stopping"; exit 1; }; sleep 30; done; }
# 1. extraction
if [ "$(ls data/taps_h3/*.npz 2>/dev/null | wc -l | tr -d ' ')" -lt 109 ] || [ "$(ls data/taps_mbpp_walk/*.npz 2>/dev/null | wc -l | tr -d ' ')" -lt 100 ]; then
  need 27; C=51; [ "$(noswap)" -ge 33 ] && C=96
  echo "== EARLYTAP extract $(date +%H:%M:%S) ($(noswap) GB NO-SWAP, C $C)"
  guard taps_extract.py
  TX_MODE=train TX_C=$C HF_HUB_OFFLINE=1 $PY sticky/taps_extract.py 2>&1 | grep --line-buffered -v "Token indices" | tail -3
  TX_MODE=walk TX_C=$C HF_HUB_OFFLINE=1 $PY sticky/taps_extract.py 2>&1 | grep --line-buffered -v "Token indices" | tail -3
  kill $G 2>/dev/null
fi
[ "$(ls data/taps_h3/*.npz 2>/dev/null | wc -l | tr -d ' ')" = 109 ] && [ "$(ls data/taps_mbpp_walk/*.npz 2>/dev/null | wc -l | tr -d ' ')" = 100 ] \
  || { echo "== EARLYTAP extraction incomplete"; exit 1; }
$PY -I -c "
import numpy as np, glob, os
n = e = 0; mx_d = 0
for p in sorted(glob.glob(os.path.expanduser('~/TrainLab/data/taps_h5/*.npz'))):
    a = np.load(p)['h']; b = np.load(os.path.join(os.path.expanduser('~/Leap/data/l120'), os.path.basename(p)))['h']
    n += 1; e += int(np.array_equal(a, b))
    fa = (a.astype(np.uint32) << 16).view(np.float32); fb = (b.astype(np.uint32) << 16).view(np.float32)
    mx_d = max(mx_d, float(np.abs(fa - fb).max()))
print(f'== EARLYTAP self-check: h5 == l120 h bit for bit on {e} of {n} windows; max |diff| {mx_d:.3g}')"
# 2. training
for v in h5 h4 h3; do
  W=$HOME/TrainLab/data/et_$v.safetensors; ST=data/et_$v.state.json
  grep -q '"status": "done"' $ST 2>/dev/null && continue
  need 23; R=""; [ -s $ST ] && R="--resume"
  echo "== EARLYTAP train $v $(date +%H:%M:%S) ($(noswap) GB NO-SWAP)"
  guard "d09_train.py --data data/taps_$v"
  $PY sticky/d09_train.py --data data/taps_$v --out $W --bs 32 --micro 1 --epochs 2 --minutes 60 --pmask 0.5 --cache-gb 0.5 $R 2>&1 \
    | grep --line-buffered -E "DONE|windows;|rror"
  kill $G 2>/dev/null
  grep -q '"status": "done"' $ST || { echo "== EARLYTAP $v training did not finish (resumable)"; exit 1; }
done
# 3. walks
SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,mbpp=$HOME/TrainLab/data/taps_mbpp_walk/*.npz,agent=$HOME/TrainLab/data/d02a/agent_*.npz"
for v in h5 h4 h3; do
  [ -f data/d02b_walk_et_$v.txt ] && continue
  case $v in h5) M="1,1,1,1,1";; h4) M="1,1,1,1,0";; h3) M="1,1,1,0,0";; esac
  need 13; echo "== EARLYTAP walk $v $(date +%H:%M:%S)"
  guard d02b_walk_match.py
  TL_DRAFT=$HOME/TrainLab/data/et_$v.safetensors TL_WALK_TAPMASK=$M TL_WALK_TAG=_et_$v TL_WALK_SETS="$SETS" HF_HUB_OFFLINE=1 \
    $PY sticky/d02b_walk_match.py 2>&1 | grep --line-buffered -E "samples|rror"
  kill $G 2>/dev/null
done
$PY -I sticky/et_read.py
echo "== EARLYTAP done $(date +%H:%M:%S)"
