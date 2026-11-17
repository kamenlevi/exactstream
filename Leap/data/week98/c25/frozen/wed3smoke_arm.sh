#!/bin/bash
# wed3smoke_arm.sh (Leap, Tue 17 Nov 2026 11:10, re-armed: EvalSuite's V2 (34.6 GB) started 11:07:56, 13 s before
# the first arming): SMOKE of slot #3's new c25 arms before slot #2 / #3 (patch_c25_wed3.py applied 7ef75f6): one window,
# 1 warm-up + 2 timed rounds, no profile / pass K / post. Reads: the bits pairs (F16gd == F16g, F32c6p / F32c6x / F32c6k ==
# F32c6i bit for bit; F16gu / F32c6u FAST), m16 / int8fold / xknob COUNTS, the one-launch HC spin-cap fails (must be 0).
# Timings are NOT quoted (1 window). Runner: run_c25_smoke.sh (= run_c25.sh, md5 c621bb42; a name EvalSuite's follower3
# does not take for slot #1). Window: starts in the first gap (no process over 16 GB, NO-SWAP >= 36), until the day rule's 15:00.
export C25_TAG=wed3smoke C25_C=160 C25_GATE=36 C25_ROUNDS=2 C25_WARM=1 C25_EST_MIN=10 C25_CAP_MIN=15
export C25_KEYS=dd__mc_27921
export C25_ARMS=F16,F16g,F16gd,F16gk,F16gu,F32s,F32c6i,F32c6p,F32c6x,F32c6k,F32c6u
export C25_WAIT_NOSWAP=36 C25_WAIT_NOBIG=16 C25_POLL=10 C25_GAP=20
unset C25_AFTER C25_POST C25_PROF C25_PASSK
cd /Users/kamen/Leap || exit 9
exec /Users/kamen/Leap/data/week98/c25/frozen/run_c25_smoke.sh "2026-11-17 11:10:07" "2026-11-17 15:00:00"
