#!/bin/sh
# HC lane pinning: does the accepted reference stay c09a8163f9, and do S=6/S=8 (k=5/k=7, MAXK=80) become identical?
cd ~/BigMoE
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_HC_PIN=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12 BIGMOE_MAXK=80"
until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done; ./scripts/gpu_quiet.sh > /dev/null
sh -c "env $R .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,5:5:0,7:7:0 2>&1 | grep -E 'tok/s|DIFF|Error|Traceback|bigmoe:|fusions' | sed -E 's/lookup=0 p_min=0.0://; s/commit refused 0, //; s/\[code\] //' | sed 's|^|[pin] |'"
