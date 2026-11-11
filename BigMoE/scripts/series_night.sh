#!/bin/sh
# One quiet-lock window: build, depth-2 prefetch series, then dry runs of the 125B scripts on the 35B.
cd ~/BigMoE
PY=~/BigMoE/.venv/bin/python; M=~/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit
(cd ext/build && make -j4 2>&1 | grep -E "error|Built target bigmoe")
./scripts/series_pf2.sh 20 10 10
echo "== dry run: check_layer.py on 35B"
( $PY scripts/check_layer.py $M 0.6 0,20,39 2>&1 | grep -v -i warn | tail -3 )
echo "== dry run: bench.py (MTPLX load path) on 35B, plain decode"
( BIGMOE_MD=$M BIGMOE_NOMTP=1 $PY scripts/bench.py 0.6 ar 32 2>&1 | grep -v -i warn | tail -3 )
