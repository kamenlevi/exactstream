#!/bin/sh
# DEPTH32 job (MTP Head; REPORT 'DEPTH32'), inside a held SHARE lock, the big window: (1) the MTP head's hits to depth 31
# (sticky/d23h_mtp31.py; D-24 was ~7-11 GB at depth 15: gate 23 GB, its MLX peak printed), (2) DFlash2's own picks at every
# position (d02b_walk.py TL_WALK_EVERY + TL_WALK_CF) on HumanEval+ and the MBPP walk set, (3) sticky/d23h_sim.py (CPU).
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d23h_mtp31.py; pkill -f d02b_walk.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
guard() { SW0=$(swap); ( while sleep 10; do if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping $1"; pkill -f "$1"; exit; fi; done ) & G=$!; }
need() { i=0; while [ "$(noswap)" -lt $1 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== DEPTH32: NO-SWAP < $1 GB for 10 min: stopping"; exit 1; }; sleep 30; done; }
if [ ! -f data/d23h/mbpp_b5.npz ]; then
  need 23; echo "== DEPTH32 MTP hits $(date +%H:%M:%S) ($(noswap) GB NO-SWAP)"; guard d23h_mtp31.py
  $PY sticky/d23h_mtp31.py --run b5 --sets code,mbpp 2>&1 | grep -E "answers|rror|Trace"
  kill $G 2>/dev/null
fi
if [ ! -f data/d02b_walk_d23h_dfl_mbpp.npz ]; then
  need 14; echo "== DEPTH32 DFlash2 every-position walk $(date +%H:%M:%S)"; guard d02b_walk.py
  TL_WALK_EVERY=1 TL_WALK_CF=1 TL_WALK_TAG=_d23h_dfl TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,mbpp=$HOME/TrainLab/data/taps_mbpp_walk/*.npz" \
    HF_HUB_OFFLINE=1 $PY sticky/d02b_walk.py 2>&1 | grep -E "samples|rror"
  kill $G 2>/dev/null
fi
$PY -I sticky/d23h_sim.py
echo "== DEPTH32 done $(date +%H:%M:%S)"
