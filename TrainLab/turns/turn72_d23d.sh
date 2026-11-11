#!/bin/sh
# D-23d CHATSWITCH (MTP Head, Mon 11-02; REPORT "D-23d CHATSWITCH: pre-registration" + amendment 1): the D-23c' harness with
# D23_ORDER=sep (each arm its own pass over the set) on 23 prompts outside every training set (16 multi, 3 IFEval, 4 agent),
# C 192, head b5, gate 42 GB, budget 2000 / 2100 s (one 40-min ticket), output logs/d23d.out, read-out sticky/d23d_read.py ->
# data/d23d_read.txt. Gate: settle 30 s, NO-SWAP >= 42 GB; stop if swap grows > 2 GB.
cd $HOME/TrainLab
KEYS="multi/bg0,ifeval/1220,multi/bg2,agent/bash17,multi/de3,multi/de4,ifeval/3280,multi/fr7,agent/edit00,multi/fr8,multi/es10,ifeval/1561,multi/es11,agent/text23,multi/ru12,multi/ru13,multi/zh15,agent/write11,multi/zh16,multi/ja18,multi/ja19,multi/ar21,multi/ar22"
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-23d $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 42 ] || { echo "== needs >= 42 GB NO-SWAP (C 192): not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d23c_mtptrunc.py; exit; fi
  done ) & G=$!
D23_KEYS="$KEYS" D23_ORDER=sep D23_HEAD=b5 D23_NEWDEF=1 D23_BUDGET=2000 D23_HARD=2100 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 192 256 2>&1 \
  | tee logs/d23d.out | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
D23_KEYS="$KEYS" $PY sticky/d23d_read.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
