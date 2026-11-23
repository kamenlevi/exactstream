#!/bin/bash
# QAT 4-bit chain head at FULL BUDGET, group 32 (results/LEVERS_A.md; orchestrator 06:3x). A full-length smoke (the
# longest windows and files, the stop line, a state save + reload, every read-out form) measures the lifetime peak
# footprint (/usr/bin/time -l); the full run then takes a share lock with that MEM and needs peak + 8 GB free to start.
# Guarded: yields to quiet tickets and resumes from its saved state; kills and stops on swap +1 GB or NO-SWAP < 8 GB.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=qatfull GUARD_KILL_WAIT=90 GUARD_SWAP_KILL_MB=1000 GUARD_NOSWAP_KILL=8 GUARD_GPU=60
export MEMGB=7 GUARD_NOSWAP=15
guarded qatfull_smoke logs/qat_full_smoke.log 10 /usr/bin/time -l $PY sb/qat.py --full --gs 32 --smoke-long \
  || { echo "qat full smoke failed"; exit 1; }
grep -q "DONE" logs/qat_full_smoke.log || { echo "qat full smoke: no DONE line"; exit 1; }
PB=$(awk '/peak memory footprint/ {print $1}' logs/qat_full_smoke.log | tail -1)
[ -n "$PB" ] || { echo "qat full smoke: no peak footprint line"; exit 1; }
PK=$(( (PB + 999999999) / 1000000000 ))
echo "$(date +%H:%M:%S) smoke lifetime peak footprint $(awk -v b=$PB 'BEGIN {printf "%.2f", b / 1e9}') GB -> share lock MEM $PK GB, start gate NO-SWAP >= $((PK + 8)) GB"
export MEMGB=$PK GUARD_NOSWAP=$((PK + 8))
guarded qatfull logs/qat_full.log 90 /usr/bin/time -l $PY sb/qat.py --full --gs 32 --save-every 500 || exit 1
echo "qat full done: $(grep -E 'VERDICT|STOPPED' logs/qat_full.log | tail -1)"
