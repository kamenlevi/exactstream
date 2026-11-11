#!/bin/sh
# 125B 55 %: build (trace tags), per-token budget from the trace, then MTP k=1/k=2 with capture-commit (no fused flags).
cd ~/BigMoE
PY=.venv/bin/python
(cd ext/build && make -j4 2>&1 | grep -E "error|Built target bigmoe")
$PY scripts/evict.py models/bare > /dev/null 2>&1
( $PY scripts/target_diag.py 0.55 2>&1 | grep -E "ms/token|per token" )
for m in mtp1 mtp2; do
  $PY scripts/evict.py models/bare > /dev/null 2>&1
  ( $PY scripts/bench.py 0.55 $m 64 2>&1 | grep -E "^mtp" | cut -c1-270 | sed "s/^/capture-commit /" )
done
