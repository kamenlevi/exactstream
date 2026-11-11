#!/bin/sh
# Which env piece changes the FRESH 32K plain result (a67851027e before; a47168003e with CHUNK/MAXK80/CONVNORM)?
cd ~/BigMoE
export LC1_OUT=$HOME/BigMoE/data/lc1b
mkdir -p $LC1_OUT; cp data/lc1/ids_32768.json $LC1_OUT/
B="BIGMOE_DUPROW=1 BIGMOE_COMPILE_HC=1 BIGMOE_HC_VREAD=1 BIGMOE_ASYNC_EVERY=1 BIGMOE_PF_JIT=1 BIGMOE_PF_QD=4 BIGMOE_PREFETCH=12"
for e in "BIGMOE_DUPROW_CHUNK=5 BIGMOE_MAXK=80" "MTPLX_FUSED_GDN_CONVNORM=1 MTPLX_FUSED_CONVNORM_VERIFY=1"; do
  until [ "$(memory_pressure | tail -1 | grep -o "[0-9]*")" -ge 85 ]; do sleep 5; done
  sh -c "env $B $e .venv/bin/python ~/MoonShots/patches/lc1/lc1_fresh.py plain 32768 128 2>&1 | grep -vi warn | tail -1 | sed 's|^|[$e] |'"
done
