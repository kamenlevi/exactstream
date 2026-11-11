# MISSION 22: THE NEXT DRAFTER. Turn today's pieces into one drafter we can train and run.

(Draft by the orchestrator for Kamen, Mon 2 Nov 2026 ~18:3x.)

**Why.** Your mission 21 closed the single-pass self-certificate, so the 45 GB gate stays the drafter's cf gate and the 12 GB lever stays the two-pass check. What is left on every line is the drafter: tokens kept per check. 350 at 45 GB needs ~13 kept per 16 rows (or ~20+ per 32); 120 at 12 GB needs 12-13. Today produced the pieces, each measured or about to be:
- **The Neural Engine path works natively** (Leap ANE-2, TABLE_350 'ANE-2', commit 60c97a5):
  - one batched 3-candidate call of the 1x drafter takes 12.4 ms on the real weights (line 14), hidden cos 0.9988 against MLX fp32
  - it needs a 32K FR-Spec head + the session's own tokens: kept code -0.89 %, math -1.14 %, agent -4.07 %
  - GPU cost at the floor ~x1.02
- **EARLYTAP** (MTP Head, tonight): can DFlash2 draft from taps <= 35 (or <= 25)? That decides whether the Neural Engine can draft during the check.
- **The copy override:** MATCH's real walk gave x1.038 (16 rows) / x1.204 (32). The COPYGATE gate (~/FableGod/copygate/GATE.md) estimates x1.03-1.05 / x1.20-1.22 on code, MBPP ~x1.00, agent x1.03-1.04. Its walk runs tonight.
- **Two drafters:** the confidence pick replays 92.7 % vs 90.9 % alone (DUALDRAFT, live Tuesday). Their errors overlap 61-66 %; a boosted member 2 with post-miss anchor weighting is queued (DIVERSE2, CLOUD_ANSWER_COMMITTEE.md).
- **Limits:**
  - more data is slow (E2: +0.22 kept per doubling)
  - bigger drafters gain ~+6 % per 1.6x layers (SD-cap) and only fit the Neural Engine window of today's long checks
  - modelled ratios ran ~2x high against real walks (MATCH)

**Step 0 (a minute).** When `~/TrainLab/data/taps_raw/` exists (EARLYTAP's extraction keeps the 109 code windows' raw taps with the 125B's top-32), run your mission-21 screen on code: `selfcert_probe.py code`. Report one line. That closes or reopens the self-certificate on code.

**Deliver**
1. **The drafter, as one buildable object:**
   - taps: all five, or <= 35 / <= 25 by EARLYTAP's verdict
   - size: 1x, or 2x only where it fits the window
   - the vocabulary head: 32K + session, or 64K + session
   - training: post-miss anchor weighting
   - its cf calibration for the gate: a better-calibrated cf certifies more rows at 0 flips
2. **The committee:** member 2 (DIVERSE2-style), the selector, and where each runs (both on the Neural Engine in one call, or member 2 on the GPU), with the time budget at today's cycle and at the floor.
3. **The host-side stack:** COPYGATE's override plus the copy-driven block length, and how it combines with the committee's pick.
4. **Expected kept per check** on code / math / chat / agent at 16 and 32 rows, calibrated. Scale any modelled ratio by the MATCH correction, show the arithmetic, and mark which numbers still need a walk. Does this drafter reach 13 kept per 16 rows on code (or 20+ per 32)? If not, how close, and what is missing.
5. **The training plan on the laptop:**
   - data (the ~65K on-policy code tokens + the math and chat sets)
   - steps, time and memory per variant
   - what trains tomorrow by day, and what needs a night
6. **The decision tree:** which variant MTP Head builds first for each outcome of tonight's EARLYTAP (h4 >= 0.95 / 0.93-0.95 / < 0.93), tonight's COPYGATE walk (PASS / BETWEEN / KILL), and Tuesday's DUALDRAFT (>= +8 % or not).

**Rules**
- The full 125B as the target; drafters only propose, and exact mode stays exact.
- CPU-only analysis is fine. Training and walks are MTP Head's, the Neural Engine path Leap's, the engine hook BigMoE's.
- No downloads. Check ~/TRIED/TRIED.md (SD-cap, D-09 / DS / DS2, E2, HPD-1, K-HEAD, L1-06, D-19, MEND-b, FR-Spec R-07, MATCH).
- Mark every number as measured / arithmetic / judgement.

**Inputs**
- ~/FableGod/SELFCERT.md, ANE_DRAFTER.md, DRAFTER_WALL.md, DRAFTER_98.md, RELAX_350.md, CACHE.md, CLOUD_ANSWER_COMMITTEE.md, copygate/GATE.md, DAY9_2026-11-02.md
- ~/Leap/TABLE_350.md (ANE-1, ANE-2), data/ane_v2.txt, frspec_walk.txt
- ~/TrainLab/REPORT.md (MATCH, EARLYTAP, DUALDRAFT, DIVERSE2, E2, D-09) and data (match_read.txt, the walks, taps_raw/)
- ~/TRIED/TRIED.md
