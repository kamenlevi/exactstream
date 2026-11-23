#!/bin/bash
# L1CF smoke (results/LEVERS_A.md, L1CF): harness + memory check before the Monday run. The 20 longest training files,
# the full eval sets (the control must reproduce 94.42 / 85.67 / 87.29), reads at steps 10 and 20. Lifetime peak
# footprint from /usr/bin/time -l. Share lock "MEM 7 GB" (the orchestrator's cap); a HOLD stops it.
cd "$(dirname "$0")/.."
source scripts/guard_lib.sh
> (a quote from the project owner, kept private)
export MEMGB=7 GUARD_MEMNOTE="cap (smoke)" GUARD_NOSWAP=14 NO_MEM_OTHERS=1 GUARD_MEM_OTHERS_GT=4 GUARD_GPU_OTHERS_MAX=50 \
  GUARD_SETTLE=10 GUARD_KILL_WAIT=120 GUARD_SWAP_KILL_MB=1000 GUARD_NOSWAP_KILL=5 GUARD_GPU=60 GUARD_TAIL_STEPS=0 \
  GUARD_STOP_FN=hold_stop
rm -f ckpt/lever1cf_a_smoke_state.safetensors
guarded l1cf_smoke logs/lever1cf_smoke.log 10 /usr/bin/time -l $PY sb/lever1cf.py --smoke 20 --tag a; rc=$?
PB=$(awk '/peak memory footprint/ {print $1}' logs/lever1cf_smoke.log | tail -1)
echo "$(date +%H:%M:%S) smoke rc $rc, lifetime peak $(awk -v b=${PB:-0} 'BEGIN {printf "%.2f", b / 1e9}') GB; $(grep -E 'DONE|Traceback|Error' logs/lever1cf_smoke.log | tail -1)"
exit $rc
