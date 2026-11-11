#!/bin/sh
# Breakthrough's request (Tue 06:48, for its second FLOOR night): L1-20's HPD-0-style CF walks (official loop, TL_WALK_CF, tag
# _hpd0) on (2) the 12 agent windows and (3) 100 MBPP answers of E2's capture (h format: fc + hidden_norm as identity, in
# sticky/d02b_walk_match.py); (1) math exists (d02b_walk_hpd0_math.npz, the same 60 tf_gsm windows). Inside a held SHARE
# lock; the walk's txt is backed up first (the _hpd0 tag rewrites d02b_walk_hpd0.txt). Gate 14 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk_match.py; pkill -P $$; exit 143' TERM
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 14 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== BT walks: NO-SWAP < 14 GB: not starting"; exit 1; }; sleep 30; done
[ -f data/d02b_walk_hpd0_txt_before_bt.txt ] || cp data/d02b_walk_hpd0.txt data/d02b_walk_hpd0_txt_before_bt.txt
echo "== BT walks $(date +%H:%M:%S)"
TL_WALK_CF=1 TL_WALK_TAG=_hpd0 TL_WALK_MAXN=100 \
  TL_WALK_SETS="agent=$HOME/TrainLab/data/d02a/agent_*.npz,mbpp=$HOME/TrainLab/data/dlog/d1102mbpp/dlog__mbpp*.npz" HF_HUB_OFFLINE=1 \
  $PY sticky/d02b_walk_match.py 2>&1 | grep -E "samples|MLX peak|rror"
cp data/d02b_walk_hpd0.txt data/d02b_walk_hpd0_bt.txt; cp data/d02b_walk_hpd0_txt_before_bt.txt data/d02b_walk_hpd0.txt
ls -la data/d02b_walk_hpd0_agent.npz data/d02b_walk_hpd0_mbpp.npz | awk '{print $5, $9}'
