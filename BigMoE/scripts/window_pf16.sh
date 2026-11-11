#!/bin/sh
# 16K prefill time split (PF1 timers, per layer) for layer-major (lmp) and PF3 (MoE over 8K-token groups), PF3 twice (determinism).
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_PF_PREFILL=1 BIGMOE_PF_PROF=1"
until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
sh -c "env $R LMP_MODES=lmp,pf3,pf3 .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 16384 8 2>&1 | grep -E '^prompt|^ref|^lmp|^pf3|PF1|Error|Traceback'"
