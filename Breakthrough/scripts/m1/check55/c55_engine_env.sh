# c55_engine_env.sh: the engine profile's env = BigMoE's G32T runner (its scratchpad g32time_run.sh, Sun 03:xx) + C55 knobs.
# Sourced by run_c55_engine.sh and the head test.
export R57_NAX=1 REF2_KEEP_CACHES=1 R57_WIRED=1 BIGMOE_PAUSE_EVERY_MS=50 R57_S0CARRY=1 REF2_SB_FIRST=1 REF2_PLE_LATE=1
export REF2_COMMIT_SUB=9 R57_DQ1=1 MTPLX_QSA_PREFILL=0 R57_PRESET=45 R57_REPS=1 R57_WARM=1 R57_TAG=c55eng
export R57_KEYS=humaneval/HumanEval/1 BIGMOE_MAXK=320 BIGMOE_EXT_BUILD=build_g32 V32_MOE1=1
export R57_G32TEST=dd__mc_27921,dd__mc_2753,dd__mc_28056 R57_G32TIME=12 R57_G32TIME_WARM=2
export C55_FRESH=dd__mc_27623,dd__mc_27904,dd__mc_27743,dd__mc_27516,dd__mc_27703,dd__mc_27360,dd__mc_28205,dd__mc_27384,dd__mc_27440,dd__mc_27933,dlog__mbpp13,dd__mc_27173
export C55_PASSK=1 C55_KROUNDS=3 C55_SPIN_MS=2.0
unset REF2_SEAM_HI
