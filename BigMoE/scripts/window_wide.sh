#!/bin/sh
# 61-slot prediction line (top-12 per row at S=4): identity + spec k=3 + miss split; then fixed PF2 prefill at 4K.
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
R0="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
for e in "BIGMOE_SPARE=12 BIGMOE_PREFETCH=12" "BIGMOE_SPARE=12 BIGMOE_PF_VICTIM=24 BIGMOE_PREFETCH=12"; do
  w; sh -c "env $R0 $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|bigmoe:' | sed -E 's/max_draft=5 lookup=0 p_min=0.0://; s/commit refused 0, //' | sed 's|^|[wide $e] |'"
  w; sh -c "env $R0 $e .venv/bin/python scripts/miss_split.py 0.50 256 2>&1 | grep -E '^plain|^spec3|Error|Traceback' | sed 's|^|[split-wide $e] |'"
done
w; sh -c "env $R0 BIGMOE_SPARE=12 BIGMOE_PREFETCH=12 BIGMOE_PF_PREFILL=1 .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 4096 2>&1 | grep -E '^prompt|^ref|^lmp|Error|Traceback' | sed 's|^|[PF2 4096] |'"
