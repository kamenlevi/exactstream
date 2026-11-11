#!/bin/sh
# plain decode (ar2 64, 50 %): J1 vs MoonShots' ar setting (JIT+QD4+VICTIM24, look-ahead 12/6/4), 2 rounds; sha c09a8163f9.
cd ~/BigMoE
R0="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
F='s/.*decode +([0-9.]+) tok.*stalled layers +([0-9.]+).*sha ([0-9a-f]+).*/decode \1 tok\/s stalled \2 sha \3/'
for r in 1 2; do for e in "BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12" "BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_VICTIM=24 BIGMOE_PREFETCH=12 BIGMOE_PREFETCH2=6 BIGMOE_PREFETCH3=4"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $R0 $e .venv/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|Traceback|bigmoe:' | sed -E '$F' | sed 's|^|[$e] |; s/BIGMOE_//g'"
done; done
