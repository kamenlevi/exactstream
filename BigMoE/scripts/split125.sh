#!/bin/sh
# 125B at 55 %, plain decode, cold SSD, each an A/B against A (identical tokens required except B):
#  A exact baseline | B GPU never waits (timing only) | C async_eval every 6 layers | D MTPLX exact fused flags
#  E = C + D | then MTP k=2/k=1 with capture-commit on the best plain config
cd ~/BigMoE
PY=.venv/bin/python
FUSED="MTPLX_FUSED_GDN_INPROJ=1 MTPLX_FUSED_QSA_QKV=1 MTPLX_QWEN4_OPDIET=1 MTPLX_FUSED_QSA_INDEXER=1"
run() { tag=$1; shift; $PY scripts/evict.py models/bare > /dev/null 2>&1
        ( env "$@" $PY scripts/bench.py 0.55 ${MODE:-ar} 64 2>&1 | grep -E "^(ar|mtp)" | cut -c1-270 | sed "s/^/$tag /" ); }
run "A base    " X=1
run "B nowait  " BIGMOE_SKIPSPIN=1
run "C async6  " BIGMOE_ASYNC_EVERY=6
run "D fused   " $FUSED
run "E async+f " BIGMOE_ASYNC_EVERY=6 $FUSED
MODE=mtp2 run "F E+mtp2  " BIGMOE_ASYNC_EVERY=6 $FUSED
MODE=mtp1 run "G E+mtp1  " BIGMOE_ASYNC_EVERY=6 $FUSED
