# knob_v2 (Breakthrough, Tue 17 Nov 2026 11:2x): the expert-rewrite knob, FROZEN; == knob_v1 in every output, faster grouping

Do not edit these files. A change ships as knob_v3. The md5s are in ../knob_v2.md5.

    import os, sys
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/data/xrw/knob_v2"))
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

## v2 vs v1
The only change is the upgrade's grouping kernel (xgrp.gpu_grp_up2): each eligible 4-bit pair atomic-mins its index into
a threadgroup table keyed by its identity, so a tier-3 candidate finds the FIRST eligible pair of its identity in O(1)
instead of scanning all P pairs. That is the same choice as v1; identities outside the tables fall back to v1's scan.
- Checks: grouping v2 == v1 on 240 random calls (xgrp.self_test2). tgl16eu with v2 == with v1, BITWISE in y, rs_m and
  rs_l, on 24 real calls (layers 20 / 44, 32 and 16 rows; scripts/xrw/xrw_v1v2.py).
- Cost: the upgrade's grouping went from ~2.2x today's (9.0 -> 19.9 us at P = 160, 16.5 -> 39.9 at P = 320) to ~1.2x.
  The kernels are on every call's critical path, so this is ~0.4 / ~0.9 ms per 16 / 32-row check.
