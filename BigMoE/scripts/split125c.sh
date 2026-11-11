#!/bin/sh
# Next 125B window at 50 % (memory margin). Each run has a hard time limit.
cd ~/BigMoE
PY=.venv/bin/python
lim() { t=$1; shift; ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt $t ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
memory_pressure | tail -1
$PY scripts/evict.py models/bare > /dev/null 2>&1
lim 300 sh -c "$PY scripts/bench.py 0.50 ar 64 2>&1 | grep -E '^ar ' | cut -c1-200 | sed 's/^/base50   /'"
$PY scripts/evict.py models/bare > /dev/null 2>&1
lim 300 sh -c "MTPLX_FUSED_HC=1 $PY scripts/bench.py 0.50 ar 64 2>&1 | grep -E '^ar ' | cut -c1-200 | sed 's/^/FUSED_HC /'"
lim 300 sh -c "$PY scripts/kcount125.py 0.50 2>&1 | grep -A26 'one decode step'"
lim 400 sh -c "$PY scripts/diverge125.py 0.50 64 2 2>&1 | grep -E '^plain' | cut -c1-320"
lim 400 sh -c "$PY ~/SpecDecode/band_probe_model.py ~/BigMoE/models/bare 256 6 2>&1 | tail -12"
