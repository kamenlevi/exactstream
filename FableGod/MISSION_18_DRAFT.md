# MISSION 18: THE NEURAL ENGINE DRAFTER. A bigger drafter that costs no check time.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~17:2x. This replaces the earlier mission 18 drafts.)

**Kamen's brief.** Think about something real: the drafter is the bottleneck, and today the Neural Engine opened a door.

**What changed today**
- **ANE-1** (Leap; ~/Leap/data/ane_probe.txt, TABLE_350 "ANE-1"). DFlash2 L1-20 as CoreML int8 runs entirely on the Neural Engine.
  - Speed: ~25 ms per 16-row block natively (body 16 ms at 30-32 GB/s, head 8.5 ms at 75-79 GB/s, CPU 0.4 ms).
  - Cost to the GPU: its expert work slows x1.070 with the Neural Engine drafting back to back, x1.013 at 12 drafts/s.
  - Kept: unchanged (code -0.3 %).
  - The drafter's residual stream overflows fp16 (~196k); the exact fix scales it by 1/32.
  - A stateful cache should bring a block to ~17-19 ms. A 1B-class body is ~40 ms per candidate today.
- **The draft is on the critical path.** It takes 6.3-12 ms of today's 100-166 ms check, and would be 10-25 % of the cycle at the floor engine (~45-60 ms per check).
- **Bigger drafters** (TABLE_350 SD-cap) gave ~+6 % tokens per 1.6x layers (DFlash's Table 6). They were killed only because the drafting cost grew with the layers, which is exactly the cost the Neural Engine moves off the GPU. DRAFTER_98's design A (DFlash3-XL, 2-4B) was blocked by draft time and training cost.
- **Side note, today's lesson.** MATCH's real walk gave about half your profile model's gain (x1.038 vs x1.071 at 16 rows; x1.204 vs x1.375 at 32). Treat every profile-model ratio as an upper bound, anchor estimates to real walks, and say which numbers still need one.

**Deliver**
1. **The pipeline.**
   - How the Neural Engine drafts the next block while the GPU checks the current one, and from which predicted cut points (70 % of code checks end at the block end or at one of the two lowest-confidence positions).
   - What the GPU does when the guess misses.
   - The net cycle at today's engine and at the floor, counting the GPU slowdown (x1.01-1.07 by draft rate).
2. **The size.** For 1x, 2x and 4x today's drafter:
   - Neural Engine ms per block, with the stateful cache
   - how many candidates fit in one check, today and at the floor
   - expected kept per check (SD-cap's +6 % per 1.6x layers, corrected by the side note)
   - RAM

   Then name the best size for the 350 engine.
3. **Training.**
   - Can a 2x drafter train on the laptop in one night? Use D-09's recipe and the existing on-policy data: ~65K code tokens (L1-20's + E2's), plus the math and chat sets.
   - What data does a 4x drafter need, and how long does capturing it take on the Mac?
   - Does free cloud compute change the answer? Never paid GPUs.
4. **The first decisive test** (one night or less) and its kill line, then the ranked plan with owners:
   - Leap: the Neural Engine path and the stateful cache
   - MTP Head: training
   - BigMoE: the engine hook
5. **Short:** what the gated copy override, DUALDRAFT and the Neural Engine drafter add up to on code. Use real-walk numbers where they exist.

**Rules**
- The full 125B (all 512 experts); Kamen's MacBook only.
- Exact mode stays exact: drafters only propose, and the check decides.
- CPU-only analysis on existing files is fine. GPU and Neural Engine runs are proposed for their owners.
- Downloads need Kamen's OK.
- Check ~/TRIED/TRIED.md (SD-cap, MEND-b, Mirror-SD, the 4-8x drafter, D-09, DS / DS2, E2). Revisit a killed idea only when the new design removes the reason it died, and say which reason.
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/Leap/data/ane_probe.txt, ~/Leap/TABLE_350.md (ANE-1, SD-cap, MEND-b, C0, B3), ~/Leap/scripts/ane_probe.py
- ~/TrainLab/REPORT.md (MATCH RESULT, E2, D-09, DS / DS2, DUALDRAFT) and data (match_read.txt, the walks)
- ~/FableGod/DRAFTER_WALL.md, DRAFTER_98.md, RELAX_350.md, wall_cutpoints.out, DAY9_2026-11-02.md
- ~/TRIED/TRIED.md
