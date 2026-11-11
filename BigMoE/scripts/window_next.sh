#!/bin/sh
# (1) look-ahead recall on qwen4 (decides prefetch), (2) MTP k=3 on the reference, (3) pipelined plain at 55 %.
cd ~/BigMoE
PY=.venv/bin/python; REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1"
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 420 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
memory_pressure | tail -1
./scripts/gpu_quiet.sh > /dev/null
lim sh -c "$PY scripts/recall_target.py 0.50 48 2>&1 | grep -E '^recall' "
./scripts/gpu_quiet.sh > /dev/null
lim sh -c "env $REF $PY scripts/bench.py 0.50 mtp3 64 2>&1 | grep -E '^mtp' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/ref mtp3 50%: \1 tok\/s sha \2/'"
memory_pressure | tail -1
./scripts/gpu_quiet.sh > /dev/null
lim sh -c "env $REF BIGMOE_ASYNC_EVERY=1 $PY scripts/bench.py 0.55 ar2 64 2>&1 | grep -E '^ar' | sed -E 's/.*decode +([0-9.]+) tok.*peak ([0-9.]+) GB.*sha ([0-9a-f]+).*/ref ar2 55%: \1 tok\/s peak \2 GB sha \3/'"
