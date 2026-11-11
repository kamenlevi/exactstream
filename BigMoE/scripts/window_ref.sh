#!/bin/sh
# Window: reference config cost and speculation identity.
#  R  = DUPROW + COMPILE_HC (candidate reference), plain decode
#  RM = same + MTP k=2 (capture-commit): must give tokens identical to R's plain decode
cd ~/BigMoE
PY=.venv/bin/python
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1"
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 360 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
memory_pressure | tail -1
for r in 1 2; do
  ./scripts/gpu_quiet.sh > /dev/null
  lim sh -c "env $REF $PY scripts/bench.py 0.50 ar 64 2>&1 | grep -E '^ar ' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/R  plain \1 tok\/s sha \2/'"
  ./scripts/gpu_quiet.sh > /dev/null
  lim sh -c "env $REF $PY scripts/bench.py 0.50 mtp2 64 2>&1 | grep -E '^mtp' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/RM mtp2  \1 tok\/s sha \2/'"
done
./scripts/gpu_quiet.sh > /dev/null
lim sh -c "BIGMOE_DUPROW=1 $PY scripts/bench.py 0.50 ar 64 2>&1 | grep -E '^ar ' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/D  duprow-only \1 tok\/s sha \2/'"
