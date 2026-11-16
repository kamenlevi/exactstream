# THINK_1116b screens (NewIdeas, Mon 16 Nov evening; PUSH800: 775-800 tok/s lossy by Sat)

Frame (corrected 17:5x, real piece sizes): bounds at 100 % of every peak 27.7 (C5) / 25.2 (C6 + int8) / 24.0
(C6 + V4 + int8) ms. At today's kept 16.91, 100 % with free drafting gives 610 / 671 / 705. 775 / 800 needs kept
x1.43-1.62 at 85-90 % with a 3 ms draft, or x1.63-1.82 at the QAT head's projected 7.4 ms. Each line below gets
<= 10 min: a desk check of its number against our measured records, plus its CPU screen if it has one.

## Fable B (dense): STATS-FORWARD, the HC read in one launch. Desk PASS; goes to Leap's module split
- Claim: the previous layer's HC writer emits K1C's per-tile partials + K2B's coefficients, so each HC read is K3C alone.
  96 reads per 32-row check go from 3 dependent launches to 1. -2.5..-4.5 ms per check [P], -192 launches, no quality
  risk.
- Checked:
  - The 8.14 ms base is Leap's measured hc8r32 'chain / read32' (check25/README.md:142), = 96 x 85 us.
  - Fable B's own CPU screen (think1116b/hc_statsfwd_check.py) shows the partials bitwise equal at the real shapes.
  - Not on the kill list: megakernel grid barriers were killed, and this is a launch fusion inside an existing kernel.
- GPU test (owner Leap's bench): PASS <= 5.0 ms per check for the 96 reads, KILL > 7.0.
- Weight in PUSH800: ~6-8 % of today's 56.4 ms check. A real engine multiplier, but the engine needs x2.
- Side claim, CHECKED and acted on: int8 dense is SLOWER in the engine (Leap run2b, F32if - F32df +1.13 ms over 36
  paired rounds; the kernel bench said -0.3; open). At the bound it saves only 0.9 ms (dense 7.6 vs 8.5).
  - 17:5x: proposed to EvalSuite + BigMoE to DROP int8 dense from G1 (gate C6 +- RES alone; it also cuts G1's memory
    gate ~45 -> ~43). This is conditional on BigMoE re-building the envs before tonight's key test.
  - int8 returns in G2 only if Leap's engine shows it <= 0 ms.
