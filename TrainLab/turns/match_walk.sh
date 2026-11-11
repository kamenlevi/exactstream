#!/bin/sh
# MATCH walk job (MTP Head, Mon 11-02; REPORT 'MATCH'), inside a held SHARE lock: DFlash2 L1-20 walked with and without
# Fable's session match override, at 16 and 32 rows, plus the 2.5 length rule, on HumanEval+ (24), 100 of E2's MBPP answers
# and the 12 agent windows; then sticky/match_read.py. Gate NO-SWAP >= 13 GB (~5 + 8, shared); swap guard +2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d02b_walk_match.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 13 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== MATCH walk: NO-SWAP < 13 GB for 10 min: not starting"; exit 1; }; sleep 30; done
SW0=$(swap)
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d02b_walk_match.py; exit; fi
  done ) & G=$!
SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,mbpp=$HOME/TrainLab/data/dlog/d1102mbpp/dlog__mbpp*.npz,agent=$HOME/TrainLab/data/d02a/agent_*.npz"
for arm in base16 match16 base32 match32 len25; do
  [ -f data/d02b_walk_m_$arm.txt ] && { echo "== MATCH $arm already walked"; continue; }
  M=""; B=16; L=""
  case $arm in match16) M=1;; base32) B=32;; match32) M=1; B=32;; len25) M=1; B=32; L=1;; esac
  echo "== MATCH walk $arm $(date +%H:%M:%S)"
  TL_WALK_MATCH=$M TL_WALK_BS=$B TL_WALK_LEN25=$L TL_WALK_MAXN=100 TL_WALK_TAG=_m_$arm TL_WALK_SETS="$SETS" HF_HUB_OFFLINE=1 \
    $PY sticky/d02b_walk_match.py 2>&1 | grep --line-buffered -v "Token indices"
done
kill $G 2>/dev/null
$PY -I sticky/match_read.py
echo "== MATCH done $(date +%H:%M:%S)"
