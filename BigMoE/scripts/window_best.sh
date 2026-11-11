#!/bin/sh
# Clean-window confirmation: base stack vs + GDN CONVNORM (exact), both with the S<=8 chunked band; spec k=3 and lookup md7.
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
for r in 1 2; do for e in "X=1" "MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $R $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:7:1 2>&1 | grep -E 'tok/s|Error|Traceback' | sed -E 's/p_min=0.0://; s/commit refused 0, //; s/\[code\] //; s/MTPLX_FUSED_//g' | sed 's|^|[r$r $e] |'"
done; done
