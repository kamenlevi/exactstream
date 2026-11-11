#!/bin/sh
# (1) best exact stack on the code prompt (k=3 MTP, and lookup md7), 2 processes; (2) LC1 32K clear test (plain, clear_cache, spec3).
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
for r in 1 2; do
  w; sh -c "env $R .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:7:1,3:7:1 2>&1 | grep -E 'tok/s|Error|Traceback' | sed -E 's/p_min=0.0://; s/commit refused 0, //; s/\[code\] //' | sed 's|^|[best r$r] |'"
done
export LC1_OUT=$HOME/BigMoE/data/lc1
w; sh -c "env $R .venv/bin/python ~/MoonShots/patches/lc1/lc1_leak.py clear 32768 128 2>&1 | grep -vi warn | grep -E 'sha|Error|Traceback' | tail -4 | sed 's|^|[lc1 clear] |'"
