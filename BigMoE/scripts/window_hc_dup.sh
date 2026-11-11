#!/bin/sh
# Window: (1) compiled-HC A/B (noise-robust), (2) model-level band probe with dup-row, (3) dup-row cost A/B.
cd ~/BigMoE
memory_pressure | tail -1
./scripts/ab125.sh "BIGMOE_COMPILE_HC=1" | sed 's/^/[HC] /'
( BIGMOE_DUPROW=1 MTPLX_NGRAM_PREWARM=off MTPLX_COMPILED_VERIFY=0 MTPLX_QWEN4EXP_COMPILE=0 MTPLX_COMPILE_AR_FORWARD=0 MTPLX_COMPILED_TARGET_PREFIX=0 .venv/bin/python ~/SpecDecode/band_probe_model.py ~/BigMoE/models/bare 256 6 2>&1 | grep -E "bands|fusions" | sed 's/^/[DUPROW probe] /' ) &
P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 400 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait
