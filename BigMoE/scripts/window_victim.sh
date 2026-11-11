#!/bin/sh
# victim-slot prefetch (no spares) vs 12/12 spares, spec125 k=3 at 50 %.
cd ~/BigMoE
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
for e in "BIGMOE_SPARE=12 BIGMOE_PREFETCH=12" "BIGMOE_PF_VICTIM=12 BIGMOE_PREFETCH=12" "BIGMOE_PF_VICTIM=16 BIGMOE_PREFETCH=16" "BIGMOE_PF_VICTIM=24 BIGMOE_PREFETCH=24"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; memory_pressure | tail -1
  ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $REF $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|IDENT|DIFF|Error|Traceback|bigmoe:|abort' | sed 's/^/[$e] /'"
done
