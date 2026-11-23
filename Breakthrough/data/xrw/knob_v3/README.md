# knob_v3 (Breakthrough, Tue 17 Nov 2026 21:2x): knob_v2 + MODE "tgl16ef" (EXACT), FROZEN

Do not edit these files. A change ships as knob_v4. The md5s are in ../knob_v3.md5.

    import os, sys
    sys.path.insert(0, os.path.expanduser("~/Breakthrough/data/xrw/knob_v3"))
    import xknob
    xknob.install()            # once, after ooo1cq is importable: wraps ooo1cq.tier_glu_nax6n_1cq (covers COH on and off)
    xknob.MODE[0] = "tgl16ef"  # EXACT: tgl16e with its front in one dispatch; bitwise == tgl16e == ooo1cq
    xknob.ON[0] = True         # per arm; False = today's call, bitwise; xknob.COUNTS / xfront.COUNTS count the calls

- MODE "tgl16ef" calls knob_v2's tgl16e unchanged, passing it grp and xs from ONE dispatch (xfront.gpu_front):
  threadgroup 0 groups the pairs with ooo1cq's own key code (spliced in as text). The order is gpu_grp_1cq's rank,
  (key, pair index), with keys broadcast 32 at a time by simd_shuffle and the segment index from a SIMD prefix count.
  Threadgroups 1.. compute rowsum64(x) with tierglu_nax's loop.
- Fallbacks to plain tgl16e (today's two dispatches, still exact), counted in xfront.COUNTS:
  - plain_big_p: P > 1024 (gpu_grp_1cq asserts there today);
  - plain_src: ooo1cq's grouping or tierglu_nax's rowsum text is not the one checked (md5 in xfront.SRC_MD5).
- Every other mode and line is knob_v2's: tgl16.py, tgl16v.py, w4a8.py and xgrp.py are byte-identical to it.
  xknob.py adds the tgl16ef mapping and two docstring lines.

## Checks (docs/MOEBLOCK.md "FRONT FOLD, STACK FORM", pre-registered 1b71659)
- G1 (scripts/xrw/xfront_check.py, data/xrw/xfront_check.txt): pidx, nseg, seg[0..nseg] and xs bitwise ==
  gpu_grp_1cq + rowsum64. Covered: 2,000 random calls (M 1..102, -1 / -2 slots and sidecars, key runs > 16) and 48
  real-routing calls (L20 / L44, 16 / 32 rows). A negative control detects 18 / 20 changed inputs.
- G2 (scripts/xrw/knob3_check.py, data/xrw/knob3_check.txt): through this frozen dir on the real pools of L20 + L44,
  tgl16ef == tgl16e == the original one-call in y, rs_m and rs_l, and two runs identical. 38 cases, all folded:
  knob_check's marked blocks, the stack's form (raw ids, 16 / 32 rows) and xrw_robust's grid (M 1-32, QOM drops).
- G3: the four kernel files' md5 == knob_v2's; xknob.py's diff is only the tgl16ef lines.

## Cost (scripts/xrw/xfront_time.py, quiet lock, data/xrw/xfront_time.txt)
GPU-busy per call, min over 12 real blocks x 8 rounds:
- 32 rows: 16.7 -> 8.0 us (L20), 16.8 -> 8.2 us (L44); 16 rows (L44): 9.1 -> 5.1 us.
- x 48 layers: ~-0.41 ms per 32-row check and ~-0.19 ms per 16-row check [P]. The engine's value is Leap's slot #4.
- The saving is the faster exact grouping. rowsum64 already overlaps the grouping inside today's command buffer, and
  the verbatim fold (variant v) saves only ~0.4 us.
