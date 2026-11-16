#!/bin/bash
# gpub_arm.sh (Leap, Mon 16 Nov 2026 18:2x): tonight's GPU-busy measurement of the lossy 32-row check (the orchestrator
# 18:0x: Kamen asked for the CPU-vs-GPU question to be settled tonight). Arms F32c6i / F16 / F32s / F32o on the 3 G32T
# windows, 2 warm-ups + 8 rounds, with libmtlprof (GPU busy per phase, thread CPU, dispatches) and c25_cpusplit (the
# forward's wall = Python build + MLX encode + blocked in eval + other waits). QUIET (the runner's quiet ticket).
# Chain: after BigMoE's g1b key test, which writes ~/BigMoE/data/r57/G1A_KEYTEST_DONE at any end (g1_follow.sh). The marker
# must be newer than this arming, and no other process may be over 16 GB RSS. NO-SWAP 39 on two reads. Latest start 00:10
# (hard end 00:25), so E14 (00:30) is never touched; if it cannot start by then, Leap re-arms it after E14.
cd /Users/kamen/Leap || exit 2
export C25_TAG=gpub C25_C=160 C25_GATE=39 C25_ROUNDS=8 C25_WARM=2 C25_EST_MIN=15 C25_CAP_MIN=20
export C25_PROF=1 C25_CPUSPLIT=1 C25_ARMS=F32c6i,F16,F32s,F32o
export C25_AFTER=/Users/kamen/BigMoE/data/r57/G1A_KEYTEST_DONE
export C25_WAIT_NOSWAP=39 C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_gpub.sh "2026-11-16 18:30:00" "2026-11-17 00:25:00"
