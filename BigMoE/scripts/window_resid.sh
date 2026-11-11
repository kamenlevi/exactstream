#!/bin/sh
# residency sweep with the best stack (spec125 k=3, prefetch 12/12, HC_VREAD); ascending, stop at the first failure.
cd ~/BigMoE
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_SPARE=12 BIGMOE_PREFETCH=12"
lim() { ( "$@" ) & P=$!; i=0; while kill -0 $P 2>/dev/null && [ $i -lt 480 ]; do sleep 1; i=$((i+1)); done; pkill -P $P 2>/dev/null; pkill -f test_spec125.py 2>/dev/null; kill $P 2>/dev/null; wait; }
for f in 0.55 0.60 0.65; do
  echo "== $f: $(memory_pressure | tail -1)"
  ./scripts/gpu_quiet.sh > /dev/null
  lim sh -c "env $REF .venv/bin/python ~/SpecDecode/scripts/test_spec125.py $f 64 3:5:0,3:5:0 > /private/tmp/claude-501/-Users-kamen-SpecDecode/1aae8c59-1c84-4354-9548-b395fab98605/scratchpad/resid_$f.log 2>&1"
  grep -E 'tok/s|IDENT|DIFF|Error|Traceback|working set|Insufficient|peak' /private/tmp/claude-501/-Users-kamen-SpecDecode/1aae8c59-1c84-4354-9548-b395fab98605/scratchpad/resid_$f.log | sed "s/^/[$f] /"
  grep -q 'k=3.*IDENT' /private/tmp/claude-501/-Users-kamen-SpecDecode/1aae8c59-1c84-4354-9548-b395fab98605/scratchpad/resid_$f.log || { echo "[$f] failed or incomplete; stopping"; tail -3 /private/tmp/claude-501/-Users-kamen-SpecDecode/1aae8c59-1c84-4354-9548-b395fab98605/scratchpad/resid_$f.log; break; }
done
echo "== end: $(memory_pressure | tail -1)"
