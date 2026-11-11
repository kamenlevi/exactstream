#!/bin/sh
# KWALL (a) job (MTP Head; REPORT 'KWALL'), inside a held SHARE lock: DFlash2 L1-20 at HPD-0's anchors on the 24 HumanEval+
# windows with TL_WALK_TOP16 (the 16 candidates + the teacher-forced selector softmax), then sticky/kw_read.py. Gate 14 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk_match.py; pkill -P $$; exit 143' TERM
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 14 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== KWALL: NO-SWAP < 14 GB: not starting"; exit 1; }; sleep 30; done
echo "== KWALL walk $(date +%H:%M:%S)"
[ -f data/d02b_walk_kw_code.npz ] || TL_WALK_CF=1 TL_WALK_TOP16=1 TL_WALK_ANCHORS=_hpd0 TL_WALK_TAG=_kw \
  TL_WALK_SETS="code=$HOME/Leap/data/dfl2d_code/*.npz" HF_HUB_OFFLINE=1 $PY sticky/d02b_walk_match.py 2>&1 | grep -E "samples|MLX peak|rror"
$PY -I sticky/kw_read.py
