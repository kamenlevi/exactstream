#!/bin/sh
# cheap residency: n-gram hot cache 1 GB -> 256 MB + MLX cache cap 1 GB -> 512 MB, at 50 % (cost) and 53 % (gain); spec k=3.
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
for c in "0.50 X=1" "0.50 MTPLX_NGRAM_HOT_MB=256 BIGMOE_MLX_CACHE_MB=512" "0.53 MTPLX_NGRAM_HOT_MB=256 BIGMOE_MLX_CACHE_MB=512"; do
  f=${c%% *}; e=${c#* }
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $R $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py $f 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|^peak' | sed -E 's/max_draft=5 lookup=0 p_min=0.0://; s/commit refused 0, //; s/\[code\] //' | sed 's|^|[$f $e] |'"
done
