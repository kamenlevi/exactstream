#!/bin/sh
# COPYGATE walk (FableGod ~/FableGod/copygate/GATE.md, pre-registered Mon 11-02 18:3x EET), for MTP Head to run inside a
# SHARE lock, like turns/match_walk.sh: DFlash2 L1-20, the official loop, the same three sets, with the gated override
# (~/FableGod/copygate/d02b_walk_gate.py = sticky/d02b_walk_match.py + TL_WALK_GATE; diff: d02b_walk_gate.patch).
# Arms: gall16 (TL_WALK_GATE=all: must equal m_match16 bit for bit), g2_16 / g2_32 (copygate, the decision),
# g1_16 / g1_32 (pduel), g0_16 (prompt). Base arms are reused (m_base16 / m_base32, deterministic). Then gate_read.py.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
W=$HOME/FableGod/copygate
trap 'pkill -f d02b_walk_gate.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
i=0; while [ "$(noswap)" -lt 13 ]; do i=$((i + 1)); [ $i -gt 20 ] && { echo "== COPYGATE walk: NO-SWAP < 13 GB for 10 min: not starting"; exit 1; }; sleep 30; done
SW0=$(swap)
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d02b_walk_gate.py; exit; fi
  done ) & G=$!
SETS="code=$HOME/Leap/data/dfl2d_code/*.npz,mbpp=$HOME/TrainLab/data/dlog/d1102mbpp/dlog__mbpp*.npz,agent=$HOME/TrainLab/data/d02a/agent_*.npz"
for arm in gall16 g2_16 g2_32 g1_16 g1_32 g0_16; do
  [ -f data/d02b_walk_g_$arm.txt ] && { echo "== COPYGATE $arm already walked"; continue; }
  case $arm in gall*) M=all;; g2*) M=copygate;; g1*) M=pduel;; g0*) M=prompt;; esac
  B=${arm##*_}; [ "$arm" = gall16 ] && B=16
  echo "== COPYGATE walk $arm ($M, $B rows) $(date +%H:%M:%S)"
  TL_WALK_GATE=$M TL_WALK_BS=$B TL_WALK_MAXN=100 TL_WALK_TAG=_g_$arm TL_WALK_SETS="$SETS" HF_HUB_OFFLINE=1 \
    $PY $W/d02b_walk_gate.py 2>&1 | grep --line-buffered -v "Token indices"
done
kill $G 2>/dev/null
$PY -I $W/gate_read.py
echo "== COPYGATE done $(date +%H:%M:%S)"
