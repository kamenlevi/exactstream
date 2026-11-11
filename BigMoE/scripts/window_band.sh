#!/bin/sh
# B6: model-level bands vs HC handling (SpecDecode found the S=6 break at layer 28's HC read).
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
for e in "BIGMOE_DUPROW=1" "BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1" "BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done
  sh -c "env MTPLX_NGRAM_PREWARM=off MTPLX_COMPILED_VERIFY=0 MTPLX_QWEN4EXP_COMPILE=0 MTPLX_COMPILE_AR_FORWARD=0 MTPLX_COMPILED_TARGET_PREFIX=0 $e .venv/bin/python ~/SpecDecode/band_probe_model.py models/bare 256 12 2>&1 | grep -iE 'band|diff|Error|Traceback' | tail -4 | sed 's|^|[$e] |'"
done
