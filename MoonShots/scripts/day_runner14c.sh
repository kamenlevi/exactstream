#!/bin/sh
# MoonShots day_runner14c: resident-only MTP drafts A/B under the true reference (24 GB mode). Launch while holding a lock.
OUT=$HOME/MoonShots/data/day14
PY=$HOME/BigMoE/.venv/bin/python
S=$HOME/MoonShots/scripts
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1 BIGMOE_RECORDS=3 MTPLX_NGRAM_PREWARM=off MTPLX_FAMILY_CAPTURE_COMMIT=1 MTPLX_DROP_EVENTS=1 MTPLX_LAZY_MTP_HISTORY_APPEND=1 MTPLX_DEFER_REPAIR_EVAL=1 MTPLX_COMPILED_VERIFY=0 MTPLX_QWEN4EXP_COMPILE=0 MTPLX_COMPILE_AR_FORWARD=0 MTPLX_COMPILED_TARGET_PREFIX=0"
echo "$(date +%H:%M) day_runner14c started" >> "$OUT/summary.txt"
env $REF $PY $S/mtp_resident_draft125.py 0.25 192 > "$OUT/mtp_resident_draft.txt" 2>&1
echo "$(date +%H:%M) done mtp_resident_draft: $(grep -E '^(code|math|chat):' "$OUT/mtp_resident_draft.txt" | tr '\n' ' ' | cut -c1-700)" >> "$OUT/summary.txt"
$PY $HOME/MoonShots/patches/planes_engine/test_tierglu4.py > "$OUT/test_tierglu4.txt" 2>&1
echo "$(date +%H:%M) done test_tierglu4: $(grep -E '^(PASS|FAIL|plan .* M=8)' "$OUT/test_tierglu4.txt" | tr '\n' ' ' | cut -c1-400)" >> "$OUT/summary.txt"
peer unlock > /dev/null 2>&1
echo "$(date +%H:%M) day_runner14c finished" >> "$OUT/summary.txt"
