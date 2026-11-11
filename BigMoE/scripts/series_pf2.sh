#!/bin/sh
# Depth-2 mid-token prefetch on 35B (same memory as the 60 % runs): correctness, then cold-SSD timing.
# usage: series_pf2.sh SPARE B1 B2
cd ~/BigMoE
PY=~/BigMoE/.venv/bin/python; M=~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit
SP=${1:-20}; B1=${2:-10}; B2=${3:-10}
export BIGMOE_SPARE=$SP BIGMOE_PREFETCH=$B1 BIGMOE_PREFETCH2=$B2
( REPS=10 $PY scripts/diag_gate.py 2>&1 | tail -1 )
( $PY scripts/diag35.py 0.6 2>&1 | grep -E "C=153|fast-path" | cut -c1-170 | sed "s/^/[pf2 $SP\/$B1\/$B2] /" )
for args in "bench35.py stream 0.6 200" "bench35_mtp.py stream 0.6 200 2" "bench35_mtp.py stream 0.6 200 3"; do
  $PY scripts/evict.py $M > /dev/null
  ( $PY scripts/$args 2>&1 | grep -v -i warn | tail -2 | sed "s/^/COLD+PF2 $SP\/$B1\/$B2 /" )
done
