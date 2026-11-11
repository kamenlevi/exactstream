#!/bin/sh
# DUALDRAFT job (MTP Head; REPORT 'DUALDRAFT'), inside a held SHARE lock: (1) the self-check (TL_WALK_ENS = L1-20 itself:
# must give 9.57 / 7.00 and 0 % second-drafter depths), (2) the ensemble L1-20 + E2's x2, (3) x2's teacher-forced picks at
# HPD-0's anchors for the overlap q (DIVERSE2's baseline), (4) sticky/dd_read.py. Gate NO-SWAP >= 14 GB; swap guard +2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk_ens.py; pkill -f "d02b_walk.py"; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 14 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== DUALDRAFT: NO-SWAP < 14 GB for 10 min: not starting"; exit 1; }; sleep 30; done
SW0=$(swap)
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d02b_walk_ens.py; pkill -f d02b_walk.py; exit; fi
  done ) & G=$!
echo "== DUALDRAFT $(date +%H:%M:%S) ($(noswap) GB NO-SWAP)"
[ -f data/d02b_walk_dd_self.txt ] || TL_WALK_ENS=$HOME/Leap/data/l120_ft.safetensors TL_WALK_TAG=_dd_self HF_HUB_OFFLINE=1 \
  $PY sticky/d02b_walk_ens.py 2>&1 | grep --line-buffered -E "samples|DUALDRAFT|rror"
[ -f data/d02b_walk_dd_x2.txt ] || TL_WALK_ENS=$HOME/TrainLab/data/e2_x2.safetensors TL_WALK_TAG=_dd_x2 HF_HUB_OFFLINE=1 \
  $PY sticky/d02b_walk_ens.py 2>&1 | grep --line-buffered -E "samples|DUALDRAFT|rror"
[ -f data/d02b_walk_dd_x2cf_code.npz ] || TL_DRAFT=$HOME/TrainLab/data/e2_x2.safetensors TL_WALK_CF=1 TL_WALK_ANCHORS=_hpd0 TL_WALK_TAG=_dd_x2cf \
  TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,math=$HOME/Leap/data/dfl2d_math/*.npz" HF_HUB_OFFLINE=1 \
  $PY sticky/d02b_walk.py 2>&1 | grep --line-buffered -E "samples|rror"
kill $G 2>/dev/null
$PY -I sticky/dd_read.py
echo "== DUALDRAFT done $(date +%H:%M:%S)"
