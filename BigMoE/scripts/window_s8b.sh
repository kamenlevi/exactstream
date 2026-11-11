#!/bin/sh
# rerun: 80-id gate checks (SpinWait stride fix) + LC1 poison tests at 4K (fresh plain baseline first: builds ids_4096).
cd ~/BigMoE
./scripts/window_s8.sh
export LC1_OUT=$HOME/BigMoE/data/lc1
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
sh -c "env $R .venv/bin/python ~/MoonShots/patches/lc1/lc1_fresh.py plain 4096 64 2>&1 | grep -vi warn | tail -1 | sed 's|^|[fresh plain 4K] |'"
for m in poison-nan-spec3 poison-ff-plain; do
  sh -c "env $R .venv/bin/python ~/MoonShots/patches/lc1/lc1_leak.py $m 4096 64 2>&1 | grep -vi warn | grep -E 'sha|poison|Error|Traceback' | tail -3 | sed 's|^|[$m] |'"
done
.venv/bin/python ~/MoonShots/patches/lc1/lc1_fresh.py cmp 4096
