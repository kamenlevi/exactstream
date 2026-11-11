#!/bin/sh
# GDN fusion (MTPLX flags): speed + exactness. INPROJ (dup-row wrapped) must equal the reference c09a8163f9;
# CONVNORM(+VERIFY), STEP, OUT checked for spec == own ar2 (v2 candidates).
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
for e in "X=1" "MTPLX_FUSED_GDN_INPROJ=1" "MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1" "MTPLX_FUSED_GDN_STEP=1" "MTPLX_FUSED_GDN_OUT=1"; do
  w; sh -c "env $R $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|Error|Traceback' | sed -E 's/max_draft=5 lookup=0 p_min=0.0://; s/commit refused 0, //; s/\[code\] //' | sed 's|^|[$e] |'"
done
