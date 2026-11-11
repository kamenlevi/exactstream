#!/bin/sh
# FULLSCR v2 (C++ parallel memcpy into one reused buffer): lmp 4K (sha bcd1187bb0) and 16K (sha 50536dae90).
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_PF_PREFILL=1 BIGMOE_PF_PROF=1 BIGMOE_PF_FULLSCR=1"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
for T in 4096 16384; do
  w; sh -c "env $R LMP_MODES=lmp,pf3 .venv/bin/python ~/MoonShots/patches/layer_major_prefill/bench_lmp125.py 0.5 $T 8 2>&1 | grep -E '^prompt|^lmp|^pf3|PF1 prefill|Error|Traceback|abort|bigmoe:' | sed 's|^|[fullscr2 $T] |'"
done
