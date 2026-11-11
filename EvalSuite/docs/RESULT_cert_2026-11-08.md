# RESULT: CERT, the release certificate for the 45 GB FAST config (Sun 2026-11-08 04:5x) — PASS; chat is NOT exact

Pre-registered in docs/PREREG_cert_2026-11-08.md + amendments 1-3. Amendment 3 (23:28, before any row) made IFEval a
scored set: CERT = 10 lines.

**The config:** BigMoE's data/stack/cert_env.txt, v1' + wired: stackb_s2 + KEEP + R57_WIRED at C 256, MEM 47 GB.
- The snapshot is d8eb2e4: screen data/r57/snap/s5/r57_screen_s5.py (fef4d99d91a0), frozen_s5. BigMoE HEAD at the start
  was 58a9dc3.
- Every process's 35 outside modules equal 58a9dc3, and its manifest equals the first process's (37 / 37).

**Process notes:**
- Runner 55 (tools/day_runner55.sh) started at the 01:30 slot.
- It stopped at 02:22 on my process cap (30 per order; CERT's 464 keys need 47), after 300 / 464 forward rows. All 30
  processes had passed their checks.
- It resumed at 02:23 (1ffb7d8, CERT_RESUME=1) with the same reference manifest. Finished rows were kept; no row ran
  twice.

## The lines (vs NAX6-N: results/ref2tnax core + ref2tnaxn.r3 humanevalx / mathx + ref2tnaxn.r2 MBPP)
| Line | Result | Verdict |
|---|---|---|
| HumanEval+ forward (n 164) | 90.2 vs 92.7 %, -2.44 pt, McNemar p 0.125 (0 / 4) | PASS |
| HumanEval+ reverse | 92.7 vs 92.7 %, +0.00, p 1.0 (2 / 2) | PASS |
| math forward (n 100) | 86.0 vs 86.0 %, +0.00, p 1.0 (2 / 2) | PASS |
| math reverse | 84.0 vs 86.0 %, -2.00, p 0.625 (1 / 3) | PASS |
| MBPP forward (n 100) | 76.0 vs 79.0 %, -3.00, p 0.375 (1 / 4) | PASS |
| MBPP reverse | 81.0 vs 79.0 %, +2.00, p 0.727 (5 / 3) | PASS |
| IFEval forward (n 100; amendment 3) | 86.0 vs 87.0 %, -1.00, p 1.0 (3 / 4) | PASS |
| IFEval reverse | 85.0 vs 87.0 %, -2.00, p 0.754 (4 / 6) | PASS |
| Pooled, 3 sets x 2 orders (728 pairs) | -0.96 pt, p 0.265 (11 / 18); bar > -2 pt AND p >= 0.05 | PASS |
| Long code (41 humanevalt rows, tfwin) | 55,966 tokens, blunders 0 = 0.00 per 10K, deviations 1.70 % | PASS |
| **CERT** | | **PASS** |

## The chat check (amendment 2): CHAT-EXACT: NO
- Only 1 of 12 IFEval rows is token-identical to NAX6's AR. The 12 ran interleaved with code prompts, so the profile
  swaps at request boundaries.
- First divergences: ifeval/1000 @0, /1154 @1, /1147 @7, /1219 @7, /1021 @9, /1137 @12, /1051 @27, /1127 @39.
- So chat in the shipped config is lossy. The paper says "chat: lossy, same scores": IFEval passes decision 4 in both
  orders.
- BigMoE's C4 bisect named two causes: REF2_OOO_1C + NAX_1CQ, which also breaks chat exactness and is nondeterministic
  run to run, and REF2_SMALL_NATIVE.

## Which sets run FAST kernels (information, for the paper)
- **Rows token-identical to NAX6-N (forward / reverse):**
  - HumanEval+ 62 / 56 of 164;
  - math 17 / 14 of 100;
  - MBPP 34 / 32 of 100;
  - IFEval 11 / 11 of 100.
- So every set runs non-exact kernels in this config, chat and math included.
- **Rows identical between the two orders:** HumanEval+ 76 / 164, math 25 / 100, MBPP 40 / 100, IFEval 24 / 100.
- **Cap hits (arm forward / reverse vs reference):** HumanEval+ 9 / 5 vs 4; math 13 / 15 vs 14; MBPP 8 / 6 vs 6;
  IFEval 7 / 8 vs 5.
- **Pooled with IFEval (928 pairs):** -1.08 pt, p 0.184 (18 / 28).
- **vs REF2-T** (the core rows only):
  - HumanEval+ forward 86.7 vs 95.0 % (-8.33, p 0.0625: FAIL as information);
  - math forward 86.0 vs 88.0 % (PASS).
  - REF2-T's HumanEval+ is 3.33 above NAX6's own, so the reference choice matters here. CERT's line is NAX6-N, as
    pre-registered.
