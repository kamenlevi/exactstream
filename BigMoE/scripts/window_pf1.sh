#!/bin/sh
# PF1 staged prefill A/B: prefill time + identical tokens (ar2 64 at 50 %), 2 rounds.
cd ~/BigMoE/ext/build && make 2>&1 | grep -E "error|\[100" | tail -3
cd ~/BigMoE
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_SPARE=12 BIGMOE_PREFETCH=12"
for r in 1 2; do for e in "X=1" "BIGMOE_PF_PREFILL=1"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done
  ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $REF $e .venv/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|Traceback|bigmoe:|assert' | sed -E 's/.*decode +([0-9.]+) tok.*prefill\+first +([0-9.]+)s.*peak ([0-9.]+) GB.*sha ([0-9a-f]+).*/decode \1 tok\/s prefill+first \2 s peak \3 GB sha \4/' | sed 's|^|[$e] |'"
done; done
