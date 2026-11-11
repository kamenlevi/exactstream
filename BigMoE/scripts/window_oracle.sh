#!/bin/sh
# Replay oracle (runahead bound): record true ids with 12/12 spares (also the baseline), then replay L+1's true ids
# into victim slots while L stalls, reads delayed 600 us (realistic runahead) and 0 us (ideal). spec125 k=3, 50 %.
cd ~/BigMoE/ext/build && make 2>&1 | grep -E "error|\[100" | tail -1
cd ~/BigMoE
REF="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1"
O=data/oracle125.bin
for e in "BIGMOE_SPARE=12 BIGMOE_PREFETCH=12 BIGMOE_ORACLE_REC=$O" "BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=48 BIGMOE_ORACLE_PLAY=$O BIGMOE_ORACLE_DELAY_US=600" "BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=48 BIGMOE_ORACLE_PLAY=$O BIGMOE_ORACLE_DELAY_US=0"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done
  ./scripts/gpu_quiet.sh > /dev/null
  sh -c "env $REF $e .venv/bin/python ~/SpecDecode/scripts/test_spec125.py 0.50 64 3:5:0,3:5:0,3:5:0 2>&1 | grep -E 'tok/s|IDENT|DIFF|Error|Traceback|bigmoe' | grep -v fusions | sed -E 's/max_draft=5 lookup=0://; s/commit refused 0, //' | sed 's|^|[$e] |'"
done
