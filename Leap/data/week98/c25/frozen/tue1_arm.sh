#!/bin/bash
# tue1_arm.sh (Leap, Mon 16 Nov 2026 14:1x): Tue 17 Nov slot #1 (NewIdeas 13:1x/13:2x): the c25 engine A/B with the profile
# pass (C25_PROF: GPU busy / CPU / kernels) + pass K (per-class GPU ms, first window) + the post benches (tue1_post.sh).
# Gate (NewIdeas): ~/EvalSuite/results/T0A_1117_END newer than this arm (night1117's end, any end) + no other > 16 GB
# process + NO-SWAP >= 39 (C 160: peak 31.4 + ~4; two reads 20 s apart) ; not before 09:30; latest start 11:30 (hard end
# 12:00); the quiet window is cut 30 min after the grant. Frozen runner: run_c25_tue1.sh (never edit it while armed).
export C25_TAG=tue1 C25_C=160 C25_GATE=39 C25_ROUNDS=12 C25_WARM=2 C25_EST_MIN=30 C25_CAP_MIN=30
export C25_PROF=1 C25_PASSK=1 C25_KROUNDS=3 C25_ARMS=F16,F32s,F32o,F32g,F32df,F32if,F32c6i,F32c6g
export C25_AFTER=/Users/kamen/EvalSuite/results/T0A_1117_END
export C25_WAIT_NOSWAP=39 C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
export C25_POST="bash /Users/kamen/Leap/scripts/week98/c25/tue1_post.sh"
cd /Users/kamen/Leap || exit 9
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_tue1.sh "2026-11-17 09:30:00" "2026-11-17 12:00:00"
