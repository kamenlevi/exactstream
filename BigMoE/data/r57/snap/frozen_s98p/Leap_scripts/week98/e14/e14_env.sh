# e14_env.sh (Leap, Sun 15 Nov 2026 18:3x): Monday night's E1-E4 env = Breakthrough's c55_engine_env.sh (BigMoE's G32T
# runner env: the exact-45 loop, C 160, MAXK 320, build_g32, v32 + V32_MOE1, SB_FIRST + PLE_LATE) minus the C55 knobs,
# plus E14's: the tag, the rounds, and the arm list (E14_ARMS; default all 8). Sourced by the E14 runner and its head test.
export R57_NAX=1 REF2_KEEP_CACHES=1 R57_WIRED=1 BIGMOE_PAUSE_EVERY_MS=50 R57_S0CARRY=1 REF2_SB_FIRST=1 REF2_PLE_LATE=1
export REF2_COMMIT_SUB=9 R57_DQ1=1 MTPLX_QSA_PREFILL=0 R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG=e14
export R57_KEYS=humaneval/HumanEval/1 BIGMOE_MAXK=320 BIGMOE_EXT_BUILD=build_g32 V32_MOE1=1
export R57_G32TEST=dd__mc_27921,dd__mc_2753,dd__mc_28056 R57_G32TIME=${E14_ROUNDS:-16} R57_G32TIME_WARM=2
export E14_ARMS=${E14_ARMS:-A16,C32,C32one,C32oneB,A16b,A16h,A16c,C32c,A16m,C32onem}
unset REF2_SEAM_HI
