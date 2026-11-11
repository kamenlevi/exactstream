#!/bin/sh
# E2 x2 job (MTP Head, Mon 11-02; REPORT 'E2 / D2'), inside a held SHARE lock: D-09's protocol (sticky/d09_train.py = Leap's
# dfl2c_train.py, incoai init, 2 epochs, pmask 0.5) on data/e2_x2, resumable (--minutes 105 is a cap only), then the D-02b walk
# (code + math, D-09's sets) and sticky/e2_read.py. Gate NO-SWAP >= 23 GB (peak ~15 + 8, shared); swap guard +2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d09_train.py; pkill -f d02b_walk.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 23 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== E2 x2: NO-SWAP < 23 GB for 10 min: not starting"; exit 1; }; sleep 30; done
SW0=$(swap)
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d09_train.py; pkill -f d02b_walk.py; exit; fi
  done ) & G=$!
W=$HOME/TrainLab/data/e2_x2.safetensors; ST=data/e2_x2.state.json
if ! grep -q '"status": "done"' $ST 2>/dev/null; then
  R=""; [ -s $ST ] && R="--resume"
  echo "== E2 train x2 $(date +%H:%M:%S) ($(noswap) GB NO-SWAP, swap ${SW0} MB)"
  $PY sticky/d09_train.py --data data/e2_x2 --out $W --bs 32 --micro 1 --epochs 2 --minutes 105 --pmask 0.5 --cache-gb 0.5 $R 2>&1 \
    | grep --line-buffered -v "Token indices"
  grep -q '"status": "done"' $ST 2>/dev/null || { echo "== E2 x2 training did not finish (resumable)"; kill $G; exit 1; }
fi
echo "== E2 walk x2 $(date +%H:%M:%S)"
TL_DRAFT=$W TL_WALK_TAG=_e2x2 HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
kill $G 2>/dev/null
$PY sticky/e2_read.py
echo "== E2 x2 done $(date +%H:%M:%S)"
