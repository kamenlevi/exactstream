#!/bin/sh
# D-23c MTPTRUNC v2 (MTP Head, Sun 11-01): D-23's harness (the 45 GB default as D-23 ran it, REF2-T, C 224) with the MTP chain
# cut by the default's own running-confidence rule (t 0.3) and an early exit, K15, head b5: dfl vs mtp15e on all 18 of D-23's
# prompts in one 30-min ticket (sticky/d23c_mtptrunc.py v2; D-23b's replay: chat +50 %, the switch +19 %; D-24: K15 > K7
# once cut). Quiet, resumable.
# Gate as D-23: after the grant settle 30 s, then NO-SWAP >= 44 GB (REF2-T C 224 peak ~42 GB with the MTP head + 4 GB is
# tighter, but D-23's own run held at 44); stop if swap grows > 2 GB.
cd $HOME/TrainLab
KEYS="humaneval/HumanEval/16,math500/test/prealgebra/1640.json,ifeval/2853,agent/bash18,multi/bg1,humaneval/HumanEval/6,math500/test/intermediate_algebra/1422.json,ifeval/374,agent/edit01,multi/fr6,humaneval/HumanEval/79,math500/test/algebra/694.json,ifeval/3407,agent/bash19,multi/es9,humaneval/HumanEval/19,math500/test/number_theory/427.json,agent/edit02"
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -f d23c_mtptrunc.py; pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-23c $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 44 ] || { echo "== needs >= 44 GB NO-SWAP (C 224): not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d23c_mtptrunc.py; exit; fi
  done ) & G=$!
# v2b: the current default (OOO + TIER_COMBINE via BigMoE's scores hook). If that path crashes, the same ticket retries on D-23's
# preset (D23_NEWDEF=0) with the time left; the first attempt's traceback is relabelled so the chain's crash check sees only the retry.
OUTF=logs/d23c_v2b.out
D23_KEYS="$KEYS" D23_HEAD=${D23_HEAD:-b5} D23_NEWDEF=1 D23_BUDGET=1500 D23_HARD=1560 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 224 256 2>&1 \
  | tee $OUTF | grep --line-buffered -v "Token indices\|moe-sorted-gather" | sed -l 's/Traceback/TRACEBACK-v2b/'
if grep -q "Traceback" $OUTF; then
  echo "== v2b (OOO + COMBINE) crashed $(date +%H:%M:%S): retrying on D-23's preset (D23_NEWDEF=0) in this ticket"
  D23_KEYS="$KEYS" D23_HEAD=${D23_HEAD:-b5} D23_NEWDEF=0 D23_BUDGET=1250 D23_HARD=1310 HF_HUB_OFFLINE=1 $PY sticky/d23c_mtptrunc.py 224 256 2>&1 \
    | grep --line-buffered -v "Token indices\|moe-sorted-gather"
fi
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
