#!/bin/sh
# LC1: plain -> plain, and plain -> reset QSA state -> spec3, in one process each (fresh = a67851027e for both).
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
for m in plain2 reset; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done
  sh -c "env $R .venv/bin/python scripts/lc1_reset.py $m 128 2>&1 | grep -vi warn | grep -E 'mode|sha|reset|cleared|Error|Traceback' | sed 's|^|[$m] |'"
done
