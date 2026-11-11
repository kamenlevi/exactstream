#!/bin/sh
# HC pack A/B on the reference with the pipelined loop, 50 %, 2 rounds: tokens must equal c09a8163f9.
cd ~/BigMoE
PY=.venv/bin/python; REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_ASYNC_EVERY=1"
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 420 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
memory_pressure | tail -1
for r in 1 2; do
  for e in "X=1" "BIGMOE_HC_PACK=1"; do
    ./scripts/gpu_quiet.sh > /dev/null
    lim sh -c "env $REF $e $PY scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|fusions' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/\1 tok\/s sha \2/' | sed 's/^/$e: /'"
  done
done
