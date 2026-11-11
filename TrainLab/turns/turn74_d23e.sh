#!/bin/sh
# D-23e (MTP Head, Mon 11-02; REPORT "D-23e: the chat switch at C 224 -- pre-registration"): D-23d unchanged but C 224 (the default's C):
# D23_ORDER=sep (each arm its own pass over the set) on 23 prompts outside every training set (16 multi, 3 IFEval, 4 agent),
# C 224, head b5, gate 47 GB (amendment 1; v2b swapped at 44), budget 1900 / 2000 s (one 40-min ticket), output logs/d23e.out, read-out d23d_read.py (D23_C=224) ->
# data/d23d_read.txt. Gate: settle 30 s, NO-SWAP >= 47 GB; stop if swap grows > 2 GB.
cd $HOME/TrainLab
KEYS="multi/bg0,ifeval/1220,multi/bg2,agent/bash17,multi/de3,multi/de4,ifeval/3280,multi/fr7,agent/edit00,multi/fr8,multi/es10,ifeval/1561,multi/es11,agent/text23,multi/ru12,multi/ru13,multi/zh15,agent/write11,multi/zh16,multi/ja18,multi/ja19,multi/ar21,multi/ar22"
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-23e $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 47 ] || { echo "== needs >= 47 GB NO-SWAP (C 224): not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d23c_mtptrunc.py; exit; fi
  done ) & G=$!
D23_KEYS="$KEYS" D23_ORDER=sep D23_HEAD=b5 D23_NEWDEF=1 D23_BUDGET=1900 D23_HARD=2000 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 224 256 2>&1 \
  | tee logs/d23e.out | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
D23_C=224 D23_KEYS="$KEYS" $PY sticky/d23d_read.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
