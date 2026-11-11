#!/bin/sh
# Scheduler on BigMoE's 4-line row (61 predictions; BIGMOE_PRED_LINES=3 widens to 123) on the streamed 35B.
# Exactness: same shas as before (K=3 70904c380e, K=0 3cb379f49e at 0.55, 64 tokens). Quality only: `peer lock share`.
cd "$(dirname "$0")"
F=${1:-0.55}; N=${2:-64}; PY=$HOME/BigMoE/.venv/bin/python
OR=/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/oracle35.bin
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 300 ]; do sleep 1; i=$((i+1)); done
        kill -0 $P 2>/dev/null && { echo "TIMEOUT: $*"; kill -9 $P; }; wait $P 2>/dev/null; }
run() { code=$1; k=$2; shift 2; lim env "$@" $PY test_sched35.py $code $F $N $k 2>&1 | grep -E "^(base|sched) |^   |Error|error|abort|Traceback|TIMEOUT"; }
J="BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4"
run sched 3 BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12   # = BigMoE 20:11 behaviour (scheduler flags off)
run sched 3 $J BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12
run sched 3 $J BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=24
run sched 3 $J BIGMOE_PRED_LINES=3 BIGMOE_PREFETCH=12 BIGMOE_PREFETCH2=8 BIGMOE_PREFETCH3=6 BIGMOE_PF_VICTIM=24
run sched 3 $J BIGMOE_PRED_LINES=3 BIGMOE_PF_RANKED=1 BIGMOE_PREFETCH=40 BIGMOE_PREFETCH2=30 BIGMOE_PREFETCH3=20 BIGMOE_PF_VICTIM=24
run sched 3 $J BIGMOE_PRED_LINES=3 BIGMOE_PREFETCH=12 BIGMOE_PREFETCH2=8 BIGMOE_PREFETCH3=6 BIGMOE_PF_VICTIM=24 BIGMOE_PF_THETA=0.25
run sched 3 $J BIGMOE_PREFETCH=0 BIGMOE_PF_VICTIM=12 BIGMOE_ORACLE_PLAY=$OR BIGMOE_ORACLE_DIST=3
run sched 0 BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12
run sched 0 $J BIGMOE_PREFETCH=24 BIGMOE_PREFETCH2=12 BIGMOE_PREFETCH3=8 BIGMOE_PF_VICTIM=24
