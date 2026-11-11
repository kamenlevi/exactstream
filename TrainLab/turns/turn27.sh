#!/bin/sh
# D-05 like-with-like (MTP Head, Wed 10-28 19:4x; NewIdeas' 18:49 asks; REPORT 'D-05 like-with-like'). One quiet lock:
# (1) the native MTP chain (k = 3) in BigMoE's EXACT 24 GB mode (plane pools 4444444444, bitwise to the reference; turn26
#     ran v3 plan mode, whose gsm text differs from the reference's at token 0-1) on the 8 code + 8 gsm D-05 prompts,
#     300 tokens, saving the decode and spec125's timers: the same text DFlash2's walk (dfl2d) runs on;
# (2) drafting cost with the MTP layer resident (no MTP_STREAM, as at 45 GB): k = 3 and k = 1 on 4 code prompts.
# Read-only on ~/BigMoE, ~/SpecDecode. Swap guard: stop if swap grows > 2 GB.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
trap 'pkill -P $$; exit 143' TERM                     # run_locked's stop reaches the running python too
eval "$(grep -E '^(REF|X|V3|FULL)=' $HOME/BigMoE/scripts/mtp_depth.sh)"
EXACT="BIGMOE_GDN_EXACT=1 BIGMOE_TIER=4444444444 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_JOINT=1"
XR=$(echo "$X" | sed 's/BIGMOE_MTP_STREAM=1//')
swap() { sysctl -n vm.swapusage | awk '{v = $6; sub("M", "", v); print int(v)}'; }
SW0=$(swap)
( while sleep 10; do
    if [ $(( $(swap) - SW0 )) -gt 2048 ]; then echo "== swap grew > 2 GB ($(swap) MB): stopping"; pkill -f mtp_depth_ids.py; exit; fi
  done ) & G=$!
run() { echo "== $1 $(date +%H:%M:%S)"; shift; env "$@" 2>&1 | grep --line-buffered -v "Token indices"; }
run "(1) exact 24 GB, k = 3, code + gsm" $X $EXACT TL_DOMAINS=code,gsm \
  $PY sticky/mtp_depth_ids.py data/d05_prompts.jsonl data/d05_mtp_exact.json 3 300
run "(2) MTP resident, k = 3" $XR $EXACT TL_DOMAINS=code TL_PER_DOMAIN=4 \
  $PY sticky/mtp_depth_ids.py data/d05_prompts.jsonl data/d05_mtp_cost_k3.json 3 200
run "(2) MTP resident, k = 1" $XR $EXACT TL_DOMAINS=code TL_PER_DOMAIN=4 \
  $PY sticky/mtp_depth_ids.py data/d05_prompts.jsonl data/d05_mtp_cost_k1.json 1 200
kill $G 2>/dev/null
echo "== done $(date +%H:%M:%S), swap $(swap) MB (start $SW0)"
