#!/bin/sh
# prefetch width sweep on spec125 k=3 (first spec run per process is warm-up).
cd ~/BigMoE
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
for e in "BIGMOE_SPARE=12 BIGMOE_PREFETCH=16" "BIGMOE_SPARE=16 BIGMOE_PREFETCH=16" "BIGMOE_SPARE=20 BIGMOE_PREFETCH=20"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; memory_pressure | tail -1
  ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $REF $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0 2>&1 | grep -E 'tok/s|IDENT|DIFF|Error|Traceback' | sed 's/^/[$e] /'"
done
