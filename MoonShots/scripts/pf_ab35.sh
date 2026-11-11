#!/bin/sh
# 35B 16K prefill A/B: PF2 (lmp), PF3 sync, PF3 async, PF3 async + FULLSCR, PF2 + FULLSCR. Cache arrays vs the previous mode's file.
cd ~/MoonShots/patches/layer_major_prefill; PY=~/BigMoE/.venv/bin/python
run() { tag=$1; shift; env "$@" BIGMOE_PF_PREFILL=1 $PY bench_lmp35.py ${M:-pf3} 0.55 16384 32 2>&1 | grep -E "prefill|bit-identical|Error|Traceback" | sed "s/^/$tag: /"; }
{ M=lmp run "PF2" ; M=lmp run "PF2+FULLSCR" BIGMOE_PF_FULLSCR=1
  M=pf3 run "PF3 sync" PF3_ASYNC=0; M=pf3 run "PF3 async" PF3_ASYNC=1; M=pf3 run "PF3 async+FULLSCR" PF3_ASYNC=1 BIGMOE_PF_FULLSCR=1
  M=pf3 run "PF3 async+FULLSCR 16K groups" PF3_ASYNC=1 BIGMOE_PF_FULLSCR=1 PF3_MOE_TOKENS=16384; } > ~/MoonShots/data/pf_ab35.txt 2>&1
$PY ~/MoonShots/scripts/nax_gemm.py > ~/MoonShots/data/nax_gemm.txt 2>&1
peer unlock > /dev/null 2>&1
