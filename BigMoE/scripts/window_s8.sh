#!/bin/sh
# 80-id gate: (1) layout check on the accepted reference (k=3, MAXK=80, sha must be c09a8163f9);
# (2) v2 candidate (dup-row + eager HC) with MTP k=5 and k=7 (S=6, S=8): spec must equal its own ar2 (eb5801b31c).
cd ~/BigMoE
(cd ext/build && make 2>&1 | grep -E "error|\[100" | tail -1)
P="BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_MAXK=80"
w() { until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null; }
w; sh -c "env BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 $P .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|bigmoe:' | sed -E 's/max_draft=[0-9]+ lookup=0 p_min=0.0://; s/commit refused 0, //' | sed 's|^|[ref MAXK=80] |'"
w; sh -c "env BIGMOE_DUPROW=1 $P .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,5:5:0,7:7:0,7:7:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|bigmoe:' | sed -E 's/lookup=0 p_min=0.0://; s/commit refused 0, //' | sed 's|^|[v2 eagerHC MAXK=80] |'"
