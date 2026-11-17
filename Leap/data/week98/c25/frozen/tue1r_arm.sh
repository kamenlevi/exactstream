#!/bin/bash
# tue1r_arm.sh (Leap, Tue 17 Nov 2026 11:0x): RE-ARM of slot #1 tue1. Same tag (so C25_tue1_DONE stays the slot's one
# end marker: the runner removes it at arming and writes it on any end; EvalSuite's follower3 keys on it and ignores a
# "stopped by a signal" marker), same frozen runner run_c25_tue1.sh (md5 c621bb42), same arms / rounds / profile / pass K
# / post as tue1_arm.sh (md5 9953cf28). Changes: the NO-SWAP gate 39 -> 36 GB (09:30-11:00 NO-SWAP stayed 35-38 GB with no
# big job up, ~25 idle sessions; C 160 peaks at 31.4 GB, gpubB 11-17 00:5x lifetime 31.3, so 36 keeps ~4.6 GB over the
# peak); C25_AFTER dropped (T0A_1117_END came 05:06 and the runner waits for a marker newer than its arming); start now,
# the same window end 12:00 (latest start 11:30).
export C25_TAG=tue1 C25_C=160 C25_GATE=36 C25_ROUNDS=12 C25_WARM=2 C25_EST_MIN=30 C25_CAP_MIN=30
export C25_PROF=1 C25_PASSK=1 C25_KROUNDS=3 C25_ARMS=F16,F32s,F32o,F32g,F32df,F32if,F32c6i,F32c6g
export C25_WAIT_NOSWAP=36 C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
export C25_POST="bash /Users/kamen/Leap/scripts/week98/c25/tue1_post.sh"
unset C25_AFTER
cd /Users/kamen/Leap || exit 9
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_tue1.sh "2026-11-17 11:03:00" "2026-11-17 12:00:00"
