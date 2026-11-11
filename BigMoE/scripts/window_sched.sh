#!/bin/sh
# prefetch_sched (MoonShots, JIT table publication) A/B on spec125 k=3 at 50 %: base W1 vs J1/J2/J3; tokens must match.
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
R0="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
for e in "BIGMOE_SPARE=12 BIGMOE_PF_VICTIM=24 BIGMOE_PREFETCH=12" \
         "BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12" \
         "BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_VICTIM=24 BIGMOE_PREFETCH=12" \
         "BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PF_VICTIM=24 BIGMOE_PREFETCH=12 BIGMOE_PRED_LINES=3 BIGMOE_PREFETCH2=8 BIGMOE_PREFETCH3=6"; do
  w; sh -c "env $R0 $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|bigmoe:' | sed -E 's/max_draft=5 lookup=0 p_min=0.0://; s/commit refused 0, //; s/BIGMOE_//g' | sed 's|^|[$e] |'"
done
