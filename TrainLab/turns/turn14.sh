#!/bin/sh
# NewIdeas 10-22 13:22 (approved): BigMoE v3 24 GB main-track MTP decode (spec 3:3:0, JIT prefetch on) with the
# per-pass expert budget (sticky/budget_hook.py via sticky/bench_budget24.py; opt-in, NON-EXACT) vs natural routing.
# Per prompt: natural, b18, b16, natural again (timing noise). Env = BigMoE's b15_runner.sh v3 stack (read-only use).
# Output: logs/turn_job.out; the [budget] / spec lines also in data/budget24.txt.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
X="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
G="BIGMOE_GDN_EXACT=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
SUM=$HOME/TrainLab/data/budget24.txt
echo "== $(date '+%m-%d %H:%M') turn14: v3 24 GB, spec 3:3:0, 256 tokens" >> $SUM
for P in code long; do
  for B in "" b18 b16 ""; do
    echo "== $P ${B:-natural} $(date +%H:%M:%S)"
    env $X $G $V3 $PG PROMPT=$P TL_BUDGET=$B $PY sticky/bench_budget24.py 0.25 256 3:3:0 2>&1 \
      | grep --line-buffered -v "Token indices" | tee /dev/stderr | grep --line-buffered -E "^\[budget\]|^\[$P\]|^peak" >> $SUM
  done
done
echo "== $(date '+%m-%d %H:%M') turn14 done" >> $SUM
