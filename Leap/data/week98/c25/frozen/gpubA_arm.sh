#!/bin/bash
# gpubA_arm.sh (Leap, Mon 16 Nov 2026 18:2x): the GPU-busy measurement, try A. NewIdeas 18:1x: g1b's key test likely runs
# only after M4, so the old g1b chain (gpub_arm.sh, stopped 18:15) would lapse.
# Starts after Breakthrough's map screen ends (G1G2_MAP_END, newer than this arming). BigMoE's key test goes first if it
# can: NO-SWAP 39 must hold on two reads 180 s apart (the grace in which the key test's follower, gate 40, starts first)
# and no other process may be over 16 GB RSS; then the quiet ticket.
# Latest start 22:35, hard end 22:50 (M4 at 23:05). If not run by then: try B goes first in Leap's own 00:30 slot, ahead
# of E14.
cd /Users/kamen/Leap || exit 2
export C25_TAG=gpubA C25_C=160 C25_GATE=39 C25_ROUNDS=8 C25_WARM=2 C25_EST_MIN=15 C25_CAP_MIN=20
export C25_PROF=1 C25_CPUSPLIT=1 C25_ARMS=F32c6i,F16,F32s,F32o
export C25_AFTER=/Users/kamen/Breakthrough/data/m1/g1g2_map/G1G2_MAP_END
export C25_WAIT_NOSWAP=39 C25_WAIT_NOBIG=16 C25_POLL=20 C25_GAP=180
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_gpub.sh "2026-11-16 18:30:00" "2026-11-16 22:50:00"
