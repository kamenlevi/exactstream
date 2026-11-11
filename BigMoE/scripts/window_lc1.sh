#!/bin/sh
# LC1 discriminator (MoonShots lc1_fresh.py): fresh-process plain vs fresh-process spec3 on one frozen 32K prompt.
cd ~/BigMoE
export LC1_OUT=$HOME/BigMoE/data/lc1
R="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
for m in plain spec3; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done
  sh -c "env $R .venv/bin/python ~/MoonShots/patches/lc1/lc1_fresh.py $m 32768 128 2>&1 | grep -vi warn | tail -2 | sed 's|^|[$m] |'"
done
.venv/bin/python ~/MoonShots/patches/lc1/lc1_fresh.py cmp 32768
