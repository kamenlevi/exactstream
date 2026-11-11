#!/bin/sh
# D-23 SWITCHDEF (MTP Head, Sun 11-01; NewIdeas 08:1x, RANKED_45GB #5 on the 45 GB DEFAULT): the drafter switch on the default's
# live loop (sticky/d23_switchdef.py = Leap's k69_live R57_PRESET=45 + D-20's MTP chain): dfl (the default) vs mtp7 (b3), D-20b's
# 18 prompts in type-interleaved order, C 224 only (the 45 GB tier: no C 160 fallback), quiet, resumable.
# Gate (orchestrator 23:5x, RULES.md): after the grant settle 30 s, then NO-SWAP (free + speculative + purgeable + file-backed)
# >= peak + 4 GB (REF2-T C 224 lifetime peak ~38 GB + the resident MTP head ~1.5 GB -> 44); stop if swap grows > 2 GB.
cd $HOME/TrainLab
KEYS="humaneval/HumanEval/16,math500/test/prealgebra/1640.json,ifeval/2853,agent/bash18,multi/bg1,humaneval/HumanEval/6,math500/test/intermediate_algebra/1422.json,ifeval/374,agent/edit01,multi/fr6,humaneval/HumanEval/79,math500/test/algebra/694.json,ifeval/3407,agent/bash19,multi/es9,humaneval/HumanEval/19,math500/test/number_theory/427.json,agent/edit02"
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
noswap() { vm_stat | awk '/page size of/ {ps=$8} /^Pages (free|speculative|purgeable):/ || /^File-backed pages:/ {v=$NF; sub("[.]","",v); s+=v} END {printf "%d", s*ps/1e9}'; }
sleep 30
SW0=$(swap); F=$(noswap)
echo "== D-23 $(date +%H:%M:%S): ${F} GB NO-SWAP, swap ${SW0} MB"
[ "$F" -ge 44 ] || { echo "== needs >= 44 GB NO-SWAP (C 224): not starting (re-queue in 15 min)"; exit 1; }
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f d23_switchdef.py; exit; fi
  done ) & G=$!
D23_KEYS="$KEYS" D23_HEAD=${D23_HEAD:-b3} HF_HUB_OFFLINE=1 $PY sticky/d23_switchdef.py 224 256 2>&1 | grep --line-buffered -v "Token indices\|moe-sorted-gather"
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
