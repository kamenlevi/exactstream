#!/bin/sh
# NewIdeas 10-22 16:55 / 19:40, steps 2 + 3: v3 24 GB MTP tok/s per arm (TL_ARMS, default natural b16 b18 b16+jit) on six prompts:
# code + edit (code), long + prose, agent (a real Claude Code turn, Chat1000 bash17), math (GSM8K 0): natural, budget, and
# budget + the budget-aware JIT prefetch (TL_BUDGET_JIT=1: look-ahead routers masked to the predicted pass set), once each
# (natural reproduces to 0.04 %). ~10 min per arm (long / agent prefills dominate): 4 arms ~40 min.
# Output: logs/turn_job.out, data/budget24.txt.
cd $HOME/TrainLab
PY=$HOME/BigMoE/.venv/bin/python
X="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_FRSPEC=1 BIGMOE_SCOUT_BUNDLE=1 MTPLX_QWEN4_VERIFY_GLUE=1 BIGMOE_MTP_STREAM=1"
G="BIGMOE_GDN_EXACT=1"
V3="BIGMOE_TIER=4433333333 BIGMOE_TIER_GLU=3c BIGMOE_TIER_COMBINE=1 BIGMOE_OOO=1"
PG="BIGMOE_TIER_PF_GROUP=64 BIGMOE_PF_NOLOGITS=1"
ARMS=${TL_ARMS:-"natural b16 b18 b16+jit"}        # NewIdeas 19:40: b16 / b18 speed for Kamen's task-result question
SUM=$HOME/TrainLab/data/budget24.txt
echo "== $(date '+%m-%d %H:%M') turn16: v3 24 GB, spec 3:3:0, 256 tokens, six prompts, arms: $ARMS" >> $SUM
for P in code edit long prose agent math; do
  for A in $ARMS; do
    B=$A; J=0; case $B in natural) B="";; *+jit) J=1; B=${B%+jit};; esac
    echo "== $P ${B:-natural} jit=$J $(date +%H:%M:%S)"
    env $X $G $V3 $PG PROMPT=$P TL_BUDGET=$B TL_BUDGET_JIT=$J $PY sticky/bench_budget24.py 0.25 256 3:3:0 2>&1 \
      | grep --line-buffered -v "Token indices" | tee /dev/stderr | grep --line-buffered -E "^\[budget\]|^\[$P\]|^peak" >> $SUM
  done
done
echo "== $(date '+%m-%d %H:%M') turn16 done" >> $SUM
