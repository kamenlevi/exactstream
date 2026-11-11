#!/bin/sh
# Inside a quiet lock: rebuild the patched tree (prefetch_sched ext + stream.py with ext.gate_glue), then the exposure probe
# and the C++ glue A/B on the streamed 35B (base config BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12). -> ~/MoonShots/data/host35.txt
T=/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/bm
{ date +%H:%M; ~/MoonShots/patches/prefetch_sched/mk_tree.sh 2>&1 | grep -E "error|Error" ; date +%H:%M
  HOST_TREE=$T BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12 ~/BigMoE/.venv/bin/python ~/MoonShots/patches/host_build/host35.py 0.55 64 100 3 0 2>&1 \
    | grep -E "^K=|Error|Traceback|error|abort"; date +%H:%M; } > ~/MoonShots/data/host35.txt 2>&1
{ date +%H:%M; ~/BigMoE/.venv/bin/python ~/MoonShots/patches/expert_kernel/bench_xk2.py 3 2>&1 | grep -v "^\s*$"; date +%H:%M; } > ~/MoonShots/data/bench_xk2.txt 2>&1
peer unlock > /dev/null 2>&1
