#!/bin/sh
# Stall decomposition for the 24 GB problem (MoonShots round 20): BigMoE eager decode at C=128 (17 GB) and C=256 (34 GB),
# no prefetch, with full ext.stats (io = pread time, proc = gate processing, cpu_wait = loader idle waiting for ids), and
# the same runs with BIGMOE_SKIPSPIN=1 (BigMoE's timing switch: the gate does not wait, wrong slots) = compute-only time.
# stall per token = wall(normal) - wall(skipspin); compare with io per token.   usage: stall_breakdown.sh (under a quiet lock)
cd $HOME/MoonShots; PY=$HOME/BigMoE/.venv/bin/python; B=patches/planes125/bench_planes125.py
for C in 128 256; do
  env BIGMOE_PREFETCH=0 $PY $B bigmoe $C > data/stall_c$C.txt 2>&1
  env BIGMOE_PREFETCH=0 BIGMOE_SKIPSPIN=1 $PY $B bigmoe $C > data/stall_c${C}_skip.txt 2>&1
done
grep -h "per token ms\|decode" data/stall_c128.txt data/stall_c128_skip.txt data/stall_c256.txt data/stall_c256_skip.txt | cut -c1-260
