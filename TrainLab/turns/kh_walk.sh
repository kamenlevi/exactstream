#!/bin/sh
# K-HEAD walk job (MTP Head, Mon 11-02; REPORT 'K-HEAD'), inside a held SHARE lock: DFlash2 L1-20 on stored taps at the HPD-0
# walk's anchors with the k-head features (TL_WALK_KH=1) on code / multi / IFEval -> data/d02b_walk_kh_<set>.npz.
# Gate NO-SWAP >= 13 GB (~5 + 8, shared); swap guard +2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 13 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== K-HEAD walk: NO-SWAP < 13 GB for 10 min: not starting"; exit 1; }; sleep 30; done
SW0=$(swap)
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d02b_walk.py; exit; fi
  done ) & G=$!
echo "== K-HEAD walk $(date +%H:%M:%S) ($(noswap) GB NO-SWAP)"
TL_WALK_CF=1 TL_WALK_KH=1 TL_WALK_ANCHORS=_hpd0 TL_WALK_TAG=_kh \
  TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,multi=$HOME/TrainLab/data/d02a/tf_multi_*.npz,ifeval=$HOME/TrainLab/data/d02a/tf_ifeval_*.npz" \
  HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -v "Token indices"
kill $G 2>/dev/null
echo "== K-HEAD walk done $(date +%H:%M:%S)"
