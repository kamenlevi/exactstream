# knob_v1 (Breakthrough, Tue 17 Nov 2026 10:3x): the expert-rewrite knob, FROZEN for Leap's c25 arms

Do not edit these files. A change ships as knob_v2. The md5s are in ../knob_v1.md5.

    import os, sys
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/data/xrw/knob_v1"))
    import xknob
    xknob.install()            # once, after ooo1cq is importable: wraps ooo1cq.tier_glu_nax6n_1cq (covers COH on and off)
    xknob.MODE[0] = "tgl16e"   # EXACT: bitwise == ooo1cq   |   "tgl16eu": tgl16e + the tier-4 upgrade (FAST)
    xknob.ON[0] = True         # per arm; False = today's call, bitwise; xknob.COUNTS counts the calls

- MODE "tgl16e": the MMA's bf16 left operand is staged in threadgroup memory, and the epilogue uses the analytic dest
  layout. Bitwise == ooo1cq on 1,280 real pairs plus Leap's 24 m32 scenarios (the tgl16 core).
- MODE "tgl16eu": tgl16e + xgrp's upgrade grouping. Tier-3 pairs of an expert with an eligible 4-bit pair in the call
  run at tier 4 (no extra bytes), giving -19.4 % segments. Unchanged pairs are bitwise == ooo1cq. The tier of a pair is
  batch-dependent by design; Leap accepted that on the FAST track (10:33).
- Only the p11 store is covered; any other store falls back to the original, counted in COUNTS["fallback_not_p11"].
- Every pool read goes through coherent(system), using Leap's own transforms (a_one_call_q head, b_drop, coh1cq).
