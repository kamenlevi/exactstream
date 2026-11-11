#!/bin/sh
# D-23x (MTP Head, Mon 11-02; NewIdeas 09:5x): why the first check after a long prefill costs 1.5-2.8 s (REPORT "D-23d note").
# The D-23d harness at C 192 (D-23d's condition) with read tracing (BIGMOE_TRACE=1, D23_TRACE=1): each prompt's prefill and
# first 6 checks get their expert reads split by pool (base MSB-3 vs LSB-1 sidecar), tag (demand / prefetch / joint) and layer.
# 4 agent prompts + 2 short chat prompts as the contrast, both arms, separate passes. Own files: data/d23x_runs.jsonl,
# data/d23x_mtptrunc.txt, logs/d23x.out; read-out sticky/d23x_read.py -> data/d23x_read.txt. Not a timing result (tracing
# adds a little per read). Gate: settle 30 s, NO-SWAP >= 42 GB; stop if swap grows > 2 GB.
cd $HOME/TrainLab
KEYS="agent/bash17,multi/bg0,agent/edit00,agent/text23,ifeval/1220,agent/write11"
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-23x $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 42 ] || { echo "== needs >= 42 GB NO-SWAP (C 192): not starting"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d23c_mtptrunc.py; exit; fi
  done ) & G=$!
BIGMOE_TRACE=1 D23_TRACE=1 D23_RUNS=$HOME/TrainLab/data/d23x_runs.jsonl D23_TXT=$HOME/TrainLab/data/d23x_mtptrunc.txt \
D23_KEYS="$KEYS" D23_ORDER=sep D23_HEAD=b5 D23_NEWDEF=1 D23_BUDGET=1100 D23_HARD=1200 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 192 256 2>&1 \
  | tee logs/d23x.out | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
$PY sticky/d23x_read.py
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
