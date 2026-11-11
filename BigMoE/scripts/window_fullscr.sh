#!/bin/sh
# BIGMOE_PF_FULLSCR=1 (one full-E scratch per layer, reused across chunks): 4K and 16K, ref + lmp, sha must equal the plain run.
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_PF_PREFILL=1 BIGMOE_PF_PROF=1"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
w; sh -c "env $R BIGMOE_PF_FULLSCR=1 LMP_MODES=ref,lmp .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 4096 32 2>&1 | grep -E '^prompt|^ref|^lmp|PF1 prefill|Error|Traceback|assert' | sed 's|^|[fullscr 4K] |'"
w; sh -c "env $R BIGMOE_PF_FULLSCR=1 LMP_MODES=lmp .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 16384 8 2>&1 | grep -E '^prompt|^lmp|PF1 prefill|Error|Traceback|assert' | sed 's|^|[fullscr 16K] |'"
