#!/bin/sh
# Slow-path round-trip decomposition on 35B, prefetch v1, plain decode:
#  A = normal (exact, cold SSD); B = same but the GPU never waits (wrong output, timing only).
#  wait cost per slow layer = (T_A - T_B) / slow calls; compare with the CPU's time per slow call.
cd ~/BigMoE
PY=~/BigMoE/.venv/bin/python; M=~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit
export BIGMOE_SPARE=12 BIGMOE_PREFETCH=12
(cd ext/build && make -j4 2>&1 | grep -E "error|Built target bigmoe")
$PY scripts/evict.py $M > /dev/null
( $PY scripts/bench35.py stream 0.6 200 2>&1 | grep -v -i warn | tail -2 | sed 's/^/A exact /' )
$PY scripts/evict.py $M > /dev/null
( BIGMOE_SKIPSPIN=1 $PY scripts/bench35.py stream 0.6 200 2>&1 | grep -v -i warn | tail -2 | sed 's/^/B nowait /' )
