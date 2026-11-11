# CLOUD ANSWER, task 17: the clean-sheet engine (cloud Fable 1, pasted by Kamen Mon 2 Nov 2026 ~22:4x)

(The cloud session's ANSWER.md lives at /home/user/ANSWER.md in its container. This is its final message as Kamen pasted it, followed by the orchestrator's check.)

Floors are pinned (two ceilings printed as 0 from a units slip; the right values are 307 tok/s EXACT and 483 FAST at the byte floor). 3 of 8 Python runs used, no web; ANSWER.md 1,032 words.

**The clean sheet is built around two numbers, not around launches.**
- Per 16-row check, RAM has to move 10.8 GB (trunk plus expert union): 40 ms. That is a 307 tok/s EXACT ceiling at kept 12.3 with everything else free.
- The real expert work is 2.7 ms on the matrix unit; NAX6 spends 43 because its 16-row tiles hold 2.3 real rows.
- So the design attacks bytes and padding:
  - one row-invariant SIMD path for every expert group
  - four fused kernels per layer
  - a streaming loader fed by pre-routing on the partial sum
  - keepalives
  - a seeded head
  - the ANE drafting five candidates from layer-24 taps

**What it keeps and throws away:**
- Keeps: 16-row checks (32 rows gives 190 against 206), the Q1 gate as FAST's core, OOO, and the drafter stack, since the knowledge wall caps kept at 12-13 regardless.
- Throws away: the two-pass at 45 GB, the padded matrix-unit tiling, the full-vocabulary head, the stall-and-wait loader, trees (break-even on code), and rank-32 sketch-on-miss.

**Where it beats the ladder:**
- About +10 % EXACT (206 vs 188 at target) and +4 % FAST (303-320 vs 292-308).
- Chat gains most (46 EXACT, ~76 FAST at 8-row checks), because dense and SSD floors dominate a 2.4-kept check.
- The one assumption it drops is "EXACT means bitwise to today's kernels". The contract only asks for bitwise agreement with the engine's own greedy output, so a clean engine's own row-invariant kernels are the reference. Then the hybrid, the 8-row tiles, the fused dense pass and the seeded head all become EXACT.

**The honest line does not move:**
- EXACT 350 is below this Mac's byte floor; the clean sheet lands ~200.
- FAST 350 sits at the design's edge, needing 37 ms per check with the draft fully hidden and experts at 18 ms; ~300 is the halved number.

**Radical option: integer-exact arithmetic.** Make EXACT mean bitwise to an engine that accumulates in integers. Integer sums are associative, so reduction order stops mattering, and every kernel, tile shape and device becomes bitwise-identical by construction. It turns a kernel-certification problem into one quality test: does A8 keep the scores. Odds ~50 %; the payoff is every future kernel for free, with EXACT and FAST converging on one engine.

---

## Orchestrator's check (Mon 2 Nov 23:1x)
- **Arithmetic holds:**
  - trunk 3.44 + expert union 7.38 GB (Leap's k62p at S 16) = 10.8 GB per check, 40 ms at ~270 GB/s, 307 tok/s at kept 12.3
  - real expert MACs: 7,680 pairs x 4.9 M MACs = 37.7 G, i.e. 2.7 ms at ~14 T MAC/s
- **Weak spot:** the EXACT +10 % needs a row-invariant all-SIMD expert path at ~2x the best ALU rate measured here (3c 7.08 us per pair; MoonShots' microbench 6.0), or the integer reference. Under float sums a hybrid is not row-invariant: a row's bits depend on which path its group takes.
- **The radical option was tested within the hour (Leap, TABLE_350 M27-C and NAX-C):**
  - int8 x int8 -> int32 exists, at 1.55-1.73x bf16's rate.
  - But the int8 kernel gained -0.3 % at S 16: the expert kernel is latency-bound per simdgroup, not throughput-bound.
  - Integer sums give tile-invariance, not speed, for now. Quality: 1.07e-2 |err|/RMS (NAX6 3.6e-3; 3c 1.25e-2).
- **Missed by both cloud answers, found by Fable C's audit:** on code the SSD at 45 GB is bandwidth-bound (11.7 GB/s demand bursts).
  - C 320 needs 37.9 MB per kept token: ~40 ms per check, not 2 ms.
  - 350 needs <= ~33 MB per kept token (fablec/AUDIT.md c).
