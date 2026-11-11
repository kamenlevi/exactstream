#!/bin/sh
# D-20 SWITCH45 (MTP Head, Fri 10-30; NewIdeas' row D-20, three arms per NewIdeas 23:4x; Leap reviewed the harness 18:36).
# One QUIET lock per chunk (a 45 GB-route load), resumable across locks: sticky/d20_switch45.py, 256 tokens, arms dfl /
# mtp7 / mtp15 interleaved per prompt (code x2, math x2, ifeval x3, multi x2, agent x2), MTP head D20_HEAD (default b3).
# Gate (orchestrator 23:5x, RULES.md): after the grant settle 30 s, then NO-SWAP (free + speculative + purgeable +
# file-backed) >= peak + 4 GB: C 224 (~40 GB) needs 44, else C 160 (~28 GB) needs 32; stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-20 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
C=224; NEED=44; [ "$F" -lt 44 ] && { C=160; NEED=32; }
[ "$F" -ge $NEED ] || { echo "== needs >= $NEED GB NO-SWAP: not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d20_switch45.py; exit; fi
  done ) & G=$!
D20_HEAD=${D20_HEAD:-b3} HF_HUB_OFFLINE=1 $PY sticky/d20_switch45.py $C 256 dfl,mtp7,mtp15 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
