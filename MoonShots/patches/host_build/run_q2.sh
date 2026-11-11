#!/bin/sh
# Quiet-lock batch: (1) C++ gate glue exactness, one fresh process per config (K=3 and K=0); (2) repeated runs in one process
# without glue (determinism across runs); (3) expert kernel bench with the chunked SwitchGLU + dependent qmv chain.
T=/private/tmp/claude-501/-Users-kamen-MoonShots/3c6677ac-9553-4d46-9a7d-3f8008b167c0/scratchpad/bm
PY=$HOME/BigMoE/.venv/bin/python; P=$HOME/MoonShots/patches
{ date +%H:%M
  for k in 3 0; do for g in 0 1; do
    BIGMOE_CGLUE=$g BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12 SCHED_TREE=$T $PY $P/prefetch_sched/test_sched35.py sched 0.55 64 $k 2>&1 \
      | grep -E "^sched |Error|Traceback" | sed "s/^/cglue=$g /" | cut -c1-200
  done; done
  HOST_TREE=$T BIGMOE_PREFETCH=12 BIGMOE_PF_VICTIM=12 $PY $P/host_build/corrupt35.py "p p p p" 2>&1 | grep -E "^[pgd]:|Error|Traceback" | cut -c1-160
  date +%H:%M; } > $HOME/MoonShots/data/host35_exact.txt 2>&1
{ date +%H:%M; $PY $P/expert_kernel/bench_xk2.py 3 2>&1 | grep -v "^\s*$"; date +%H:%M; } > $HOME/MoonShots/data/bench_xk2b.txt 2>&1
peer unlock > /dev/null 2>&1
