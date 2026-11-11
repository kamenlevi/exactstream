#!/bin/sh
cd ~/BigMoE
PY=.venv/bin/python; REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1"
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 360 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
for v in "ar:X=1" "ar2:BIGMOE_ASYNC_EVERY=1" "ar2:BIGMOE_ASYNC_EVERY=1"; do
  m=${v%%:*}; e=${v#*:}; ./scripts/gpu_quiet.sh > /dev/null
  lim sh -c "env $REF $e $PY scripts/bench.py 0.50 $m 64 2>&1 | grep -E '^ar|Error' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/\1 tok\/s sha \2/' | sed 's/^/$m $e: /'"
done
