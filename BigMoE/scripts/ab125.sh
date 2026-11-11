#!/bin/sh
# Noise-robust A/B on the 125B at 50 %: base and variant interleaved, 3 rounds, 48 tokens each.
# Compare best-of-3 tok/s; tokens (sha) must match base.  usage: ab125.sh "ENV=1 ENV2=1"
cd ~/BigMoE
PY=.venv/bin/python; V="$1"
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 300 ]; do sleep 1; i=$((i+1)); done; kill $P 2>/dev/null; wait; }
for r in 1 2 3; do
  ./scripts/gpu_quiet.sh || { echo "round $r skipped: GPU busy"; continue; }
  lim sh -c "$PY scripts/bench.py 0.50 ar 48 2>&1 | grep -E '^ar ' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/base    \1 tok\/s sha \2/'"
  ./scripts/gpu_quiet.sh > /dev/null || echo "  (variant round $r started with GPU busy)"
  lim sh -c "env $V $PY scripts/bench.py 0.50 ar 48 2>&1 | grep -E '^ar |fusions' | sed -E 's/.*decode +([0-9.]+) tok.*sha ([0-9a-f]+).*/variant \1 tok\/s sha \2/'"
done
