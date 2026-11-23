#!/bin/bash
# QAT follow-ups (results/LEVERS_A.md; orchestrator 07:3x): MIX, LSQ, FLOOR one at a time, each one arm against s30.
# Each lever: a full-length smoke measures its lifetime peak footprint (/usr/bin/time -l); the run then takes a share
# lock with that MEM and needs peak + 8 GB free to start. Guarded: yields (and resumes) on a quiet ticket, PAUSE or a big
# slot (MEM >= 30 GB elsewhere); kills and stops on swap +1 GB or NO-SWAP < 8 GB.
# Lock length (orchestrator 09:4x: LSQ's 60 min lock lapsed mid-run under a 5-way GPU share): the guard re-sizes each
# step's lock from the run's own s/step at step 40 and holds it to the real end (guard_lib.sh). GUARD_TAIL_STEPS = the
# work after training in step-equivalents: MIX's read-outs took 487 s at its 0.59 s/step mean -> ~830, so 900 for MIX and
# FLOOR (same forms); LSQ's read-outs are ~1.8x MIX's (smokes 53 vs 29 s) while its step is ~5x heavier -> 400.
# LEVERS picks the levers (default all three); done markers in logs/.done/qat2 skip finished steps.
cd "$(dirname "$0")/.."
PY=$HOME/BigMoE/.venv/bin/python
source scripts/guard_lib.sh
export GUARD_SETTLE=20 GUARD_RUN=qat2 GUARD_KILL_WAIT=90 GUARD_SWAP_KILL_MB=1000 GUARD_NOSWAP_KILL=8 GUARD_GPU=60 GUARD_BIG_GB=30
for L in ${LEVERS:-mix lsq floor}; do
  export MEMGB=7 GUARD_NOSWAP=15 GUARD_TAIL_STEPS=0
  guarded qat2_${L}_smoke logs/qat2_${L}_smoke.log 10 /usr/bin/time -l $PY sb/qat2.py --lever $L --smoke-long \
    || { echo "qat2 $L smoke failed"; exit 1; }
  grep -q "DONE" logs/qat2_${L}_smoke.log || { echo "qat2 $L smoke: no DONE line"; exit 1; }
  PB=$(awk '/peak memory footprint/ {print $1}' logs/qat2_${L}_smoke.log | tail -1)
  [ -n "$PB" ] || { echo "qat2 $L smoke: no peak footprint line"; exit 1; }
  PK=$(( (PB + 999999999) / 1000000000 ))
  echo "$(date +%H:%M:%S) $L smoke lifetime peak $(awk -v b=$PB 'BEGIN {printf "%.2f", b / 1e9}') GB -> share lock MEM $PK GB, start gate NO-SWAP >= $((PK + 8)) GB"
  case $L in lsq) TS=400 ;; *) TS=900 ;; esac
  export MEMGB=$PK GUARD_NOSWAP=$((PK + 8)) GUARD_TAIL_STEPS=$TS
  # 120 min to start (LSQ took ~2 h under this load); re-sized from the run's own s/step at step 40
  guarded qat2_$L logs/qat2_$L.log 120 /usr/bin/time -l $PY sb/qat2.py --lever $L --save-every 500 || exit 1
  echo "$(date +%H:%M:%S) qat2 $L done: $(grep -E 'VERDICT|STOPPED|KILL at phase A' logs/qat2_$L.log | tail -1)"
done
echo "$(date +%H:%M:%S) qat2 all done"
