#!/bin/sh
# Exactness matrix on the streamed 35B (quality only: run under `peer lock share`). Every sha must equal base's.
# usage: run35.sh [FRAC] [N]     -> one summary line per run (+ scheduler counters), JSON in ~/MoonShots/data/sched35/
cd "$(dirname "$0")"
F=${1:-0.55}; N=${2:-64}; PY=$HOME/BigMoE/.venv/bin/python
OR=/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/oracle35.bin
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 300 ]; do sleep 1; i=$((i+1)); done
        kill -0 $P 2>/dev/null && { echo "TIMEOUT: $*"; kill -9 $P; }; wait $P 2>/dev/null; }
run() { code=$1; k=$2; shift 2; lim env "$@" $PY test_sched35.py $code $F $N $k 2>&1 | grep -E "^(base|sched) |^   |Error|error|abort|Traceback|TIMEOUT"; }
B="BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12"
R3="BIGMOE_PF_RANKED=1 BIGMOE_PREFETCH=16 BIGMOE_PREFETCH2=9 BIGMOE_PREFETCH3=5 BIGMOE_PF_VICTIM=12"
run base 3 $B BIGMOE_ORACLE_REC=$OR
run sched 3 $B BIGMOE_PF_JIT=1
run sched 3 $B BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4
run sched 3 $R3 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4
run sched 3 $R3 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_THETA=0.3
run sched 3 $R3 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_EARLY=1
run sched 3 BIGMOE_PREFETCH=0 BIGMOE_PF_VICTIM=12 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_ORACLE_PLAY=$OR BIGMOE_ORACLE_DIST=1
run sched 3 BIGMOE_PREFETCH=0 BIGMOE_PF_VICTIM=12 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_ORACLE_PLAY=$OR BIGMOE_ORACLE_DIST=3
run sched 3 BIGMOE_PREFETCH=0 BIGMOE_PF_VICTIM=12 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_ORACLE_PLAY=$OR BIGMOE_ORACLE_DIST=48
run base 0 $B
run sched 0 BIGMOE_PREFETCH=12 BIGMOE_PREFETCH2=6 BIGMOE_PREFETCH3=4 BIGMOE_PF_VICTIM=12 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4
run sched 0 $R3 BIGMOE_PF_JIT=1
