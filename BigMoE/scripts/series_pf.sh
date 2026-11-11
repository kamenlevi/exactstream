#!/bin/sh
# Mid-token prefetch test on 35B: correctness, then cold-SSD timing with/without prefetch (same memory).
cd ~/BigMoE
PY=~/BigMoE/.venv/bin/python; M=~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit
( REPS=10 $PY scripts/diag_gate.py 2>&1 | tail -1 )
( BIGMOE_SPARE=12 BIGMOE_PREFETCH=12 $PY scripts/diag35.py 0.6 2>&1 | grep -E "C=153|fast-path" | cut -c1-170 | sed 's/^/[pf] /' )
for args in "bench35.py stream 0.6 200" "bench35_mtp.py stream 0.6 200 2" "bench35_mtp.py stream 0.6 200 3"; do
  $PY scripts/evict.py $M > /dev/null
  ( BIGMOE_SPARE=12 BIGMOE_PREFETCH=12 $PY scripts/$args 2>&1 | grep -v -i warn | tail -2 | sed "s/^/COLD+PF12+PUB /" )
done
