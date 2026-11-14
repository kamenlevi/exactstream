# PREREG: DEEPDRAFT CAPQ, a ~22-min clean capacity retest (Sat 14 Nov 2026)

Written before the run. The orchestrator's design (15:1x), after t30x4 = HURTS and TAPSQ's amendment 1.
- TTT30C's capacity arms trained a zero-init dense branch at 10x the head's lr. Under Adam that moves every row's output at
  once, and the early spike grew with the width:
  - t30x2: +203 % KL over steps 1-20, +95 % over 21-100;
  - t30x4: +1,052 % / +234 %.
  - Both ended below t30b.
- So "capacity doesn't help" isn't settled. This retest trains the same x2 branch at the head's own lr.

## Setup
- **Arm t30x2c:** t30x2's run with exactly one change, the branch's lr multiplier 10 -> 1.
  - From b5, the 10,240-wide zero-init SwiGLU branch (147.7M trainable), t30b's pinned split, pass seed 1113. So step i
    reads the same slice as t30b's step i.
  - Depth 30, wdecay 0.97, accept-until-fail. Adam at lr 1e-5 with t30b's full-pass schedule: a 20-step warm-up, then
    cosine over 4,206 steps.
  - It is stopped with a SIGTERM at step 200 (the trainer saves the per-step losses).
- **Control:** t30b's saved per-step losses (~/datasets/deepdraft/ckpt/dd_t30b.json), steps 1-200. The same slices and the
  same schedule, so the comparison is paired by step. No arm needs to train for the control.
- Time: ~200 x 6-7 s alone, ~22 min. Peak ~25 GB, so it runs alone after TAPSQ and the information-only eval.

## Lines (result -> action)
- **Measure:** the mean per-step KL of t30x2c over t30b's, steps 101-200, and the share of those steps where t30x2c is lower.
- **Spike check first:** steps 1-20 within +20 % of t30b's. If not, the lr fix didn't remove the spike and the retest is
  INVALID. Then one more ~20-min try with the branch's output scaled down 10x at init time (not run unless needed).

| Result | Line (steps 101-200) | Action |
|---|---|---|
| REOPENS | t30x2c <= 0.95 x t30b, lower on >= 60 % of the steps | Capacity reopens: pre-register a full x2 arm at lr 1e-5 with a stop line at 10-25 % of its steps vs t30b's per-step KL; x4 after it only if x2 pays. |
| CLOSED | anything else (within 5 %, or above) | Capacity is closed for the week. The week's levers are what the drafter sees (taps, fresher features) and the ceiling's verdict. |
