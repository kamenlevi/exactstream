#!/bin/sh
# PF2 at 16K, then long-context decode (4K/16K/32K, plain + spec k=3) with the best stack.
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
w; sh -c "env $R BIGMOE_PF_PREFILL=1 .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 16384 2>&1 | grep -E '^prompt|^ref|^lmp|Error|Traceback' | sed 's|^|[PF2 16K] |'"
w; sh -c "env $R .venv/bin/python scripts/longctx.py 0.50 128 4096,16384,32768 2>&1 | grep -E '^corpus|^T=|Error|Traceback' | sed 's|^|[long] |'"
