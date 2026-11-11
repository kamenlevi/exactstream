#!/bin/bash
# b45 smoke (one arm L = M-01 + POM 4 + LP1, prompt 4 of each domain, 400 tok) inside the 17:56 lock: crashes / garbage
# before the A/B (a broken slot would show as a collapsed tokens-per-pass), and the pom_stats shares.
cd "$HOME/BigMoE" || exit 1
PY="$HOME/BigMoE/.venv/bin/python"
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
X="$REF BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
FULL="BIGMOE_GDN_EXACT=1 $V3 BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1 BIGMOE_EVICT_LFU=0.99 BIGMOE_VERIFY_ROWS_SDPA=1"
L_ENV="${ARM_ENV:-BIGMOE_MTP_PF_GROUP=48 BIGMOE_M01=1 BIGMOE_POM=4 BIGMOE_POM_LP1=1}"
T=data/b45s/${ARM:-L}
set -m
perl -e 'alarm shift; exec @ARGV' 220 env BIGMOE_EXT_BUILD=build_m01 $X $FULL BIGMOE_JOINT=1 BIGMOE_TRACE=1 S10B_IDX=4 S10B_X_IDX=9 $L_ENV BIGMOE_S10B=1 BIGMOE_S10B_MISS=$T/miss.bin $PY scripts/s10b_trace.py 0.25 400 $T > data/b45s/${ARM:-L}.txt 2>&1 &
jp=$!; wait $jp; rc=$?; kill -9 -- -$jp 2>/dev/null; echo "rc=$rc"
