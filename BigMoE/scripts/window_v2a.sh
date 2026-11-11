#!/bin/sh
# v2a fast-path check: spec == v2a's own ar2 at k=3/5/7 (S<=8: CHUNK=5, MAXK=80), code prompt, 50 %.
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1 MTPLX_FUSED_GDN_INPROJ=1"
until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
sh -c "env $R .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:7:1,5:5:0,7:7:0 2>&1 | grep -E 'tok/s|Error|Traceback' | sed -E 's/p_min=0.0://; s/commit refused 0, //; s/\[code\] //' | sed 's|^|[v2a] |'"
# stacked build: + FR-Spec draft head + PKD1 grouped projections, on v1 (reference) and on v2a
V1="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
for e in "$V1" "$R"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $e BIGMOE_FRSPEC=1 BIGMOE_PKD1=1 .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:7:1,5:5:0 2>&1 | grep -E 'fusions|tok/s|Error|Traceback' | sed -E 's/p_min=0.0://; s/commit refused 0, //; s/\[code\] //' | sed 's|^|[stack] |'"
done
# lmp4pd prefill in decode.greedy: plain decode sha must stay c09a8163f9
until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
sh -c "env $V1 BIGMOE_PF_PREFILL=1 BIGMOE_PF_FULLSCR=1 BIGMOE_PREFILL=lmp4pd .venv/bin/python scripts/bench.py 0.50 ar2 64 2>&1 | grep -E '^ar|Error|Traceback|assert' | sed -E 's/.*decode +([0-9.]+) tok.*prefill\+first +([0-9.]+)s.*sha ([0-9a-f]+).*/decode \1 tok\/s prefill+first \2 s sha \3/' | sed 's|^|[lmp4pd] |'"
