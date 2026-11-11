#!/bin/sh
# MoonShots day_runner14b: follow-ups to day_runner14 (launch while holding a granted quiet lock).
#  1. the chain under the reference flags with BIGMOE_ASYNC_EVERY=0 (async eval cuts the graph per layer otherwise)
#  2. the exact GDN speed bench (14's bench imported the test module, which exited at import)
OUT=$HOME/MoonShots/data/day14; mkdir -p "$OUT"
PY=$HOME/BigMoE/.venv/bin/python
E=$HOME/MoonShots/patches/planes_engine
S=$HOME/MoonShots/scripts
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=0 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1 BIGMOE_RECORDS=3 MTPLX_NGRAM_PREWARM=off MTPLX_FAMILY_CAPTURE_COMMIT=1 MTPLX_DROP_EVENTS=1 MTPLX_LAZY_MTP_HISTORY_APPEND=1 MTPLX_DEFER_REPAIR_EVAL=1 MTPLX_COMPILED_VERIFY=0 MTPLX_QWEN4EXP_COMPILE=0 MTPLX_COMPILE_AR_FORWARD=0 MTPLX_COMPILED_TARGET_PREFIX=0"
echo "$(date +%H:%M) day_runner14b started" >> "$OUT/summary.txt"
env $REF $PY $S/export_pass_graph125.py 0.25 $HOME/MoonShots/data/pass_graph125_ref_sync.dot > "$OUT/chain_ref_sync.txt" 2>&1
echo "$(date +%H:%M) done chain_ref_sync: $(grep -E 'critical chain' "$OUT/chain_ref_sync.txt" | head -1)" >> "$OUT/summary.txt"
$PY $E/bench_gdn_exact.py > "$OUT/gdn_bench2.txt" 2>&1
echo "$(date +%H:%M) done gdn_bench2: $(grep -E '^S=' "$OUT/gdn_bench2.txt" | tr '\n' ' ')" >> "$OUT/summary.txt"
REFA=$(echo "$REF" | sed "s/BIGMOE_ASYNC_EVERY=0/BIGMOE_ASYNC_EVERY=1/")   # the true reference (async eval per layer)
env $REFA VERIFY_TIMES=1 PROFILE_OUT=$HOME/MoonShots/data/pass_profile125_vt.json $PY $S/pass_profile125.py 0.25 160 > "$OUT/verify_times.txt" 2>&1
echo "$(date +%H:%M) done verify_times" >> "$OUT/summary.txt"
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) day_runner14b finished" >> "$OUT/summary.txt"
