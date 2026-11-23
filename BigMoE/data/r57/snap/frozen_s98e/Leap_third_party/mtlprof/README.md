# mtlprof (copy)

Breakthrough's libmtlprof, copied read-only on Mon 16 Nov 2026 13:5x from ~/Breakthrough/scripts/m1/check55 (mtlprof.m last
commit there: 282a0da). It is an in-process hook on the Metal classes: it counts every dispatch / barrier / encoder /
command buffer, names pipelines as MLX creates them, and reads each command buffer's GPUStartTime / GPUEndTime. Its cost is
<= 0.01 us per dispatch (hook_overhead.py). Load it BEFORE `import mlx.core`.

Used by Leap's c25 engine (C25_PROF=1, patch_c25_prof.py): per check, GPU busy vs wall, CPU per phase, dispatches.

md5s:
    f860a12310d551e993350d1871446426 mtlprof.py
    8e40093893c08379662c443478969b06 mtlprof.m
    9596e4e3abb625290b17fcfdf1d05f81 libmtlprof_v2.dylib
