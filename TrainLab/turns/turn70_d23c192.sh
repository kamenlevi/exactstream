#!/bin/sh
# D-23c' at C 192 (MTP Head, Mon 11-02; REPORT "D-23c' at C 192: pre-registration"): turn62 with C 192, gate 42 GB, budget 2300 / 2400 s
# (one 45-min ticket), output logs/d23c_c192.out; everything else as D-23c v2b.
# Gate: settle 30 s, NO-SWAP >= 42 GB (C 192 peak ~38 GB + 4); stop if swap grows > 2 GB.
cd $HOME/TrainLab
KEYS="humaneval/HumanEval/16,math500/test/prealgebra/1640.json,ifeval/2853,agent/bash18,multi/bg1,humaneval/HumanEval/6,math500/test/intermediate_algebra/1422.json,ifeval/374,agent/edit01,multi/fr6,humaneval/HumanEval/79,math500/test/algebra/694.json,ifeval/3407,agent/bash19,multi/es9,humaneval/HumanEval/19,math500/test/number_theory/427.json,agent/edit02"
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-23c' C 192 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 42 ] || { echo "== needs >= 42 GB NO-SWAP (C 192): not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d23c_mtptrunc.py; exit; fi
  done ) & G=$!
# v2b: the current default (OOO + TIER_COMBINE via BigMoE's scores hook). If that path crashes, the same ticket retries on D-23's
# preset (D23_NEWDEF=0) with the time left; the first attempt's traceback is relabelled so the chain's crash check sees only the retry.
OUTF=logs/d23c_c192.out
D23_KEYS="$KEYS" D23_HEAD=${D23_HEAD:-b5} D23_NEWDEF=1 D23_BUDGET=2300 D23_HARD=2400 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 192 256 2>&1 \
  | tee $OUTF | grep --line-buffered -v "Token indices\|moe-sorted-gather" | sed -l 's/Traceback/TRACEBACK-v2b/'
if grep -q "Traceback" $OUTF; then
  echo "== v2b (OOO + COMBINE) crashed $(date +%H:%M:%S): retrying on D-23's preset (D23_NEWDEF=0) in this ticket"
  D23_KEYS="$KEYS" D23_HEAD=${D23_HEAD:-b5} D23_NEWDEF=0 D23_BUDGET=1100 D23_HARD=1160 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 192 256 2>&1 \
    | grep --line-buffered -v "Token indices\|moe-sorted-gather"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
