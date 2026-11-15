# PLAN4D re-aimed at 97.8 per band (NewIdeas, Goal A lead)

> (a quote from the project owner, kept private)
good enough. dont push for 98."

All numbers are dev, CPU-only joins on existing dumps, and every script self-checks against published values first:
- research/week98/tree_audit.{py,txt,json}:
  - chain-32 13.769 / 6.924;
  - EAGLE-2-16 11.263 / 7.487 (S9c);
  - MoonShots' audit-chain-32 14.456 / 15.399 / 7.427 / 7.713, all exact.
- research/week98/lab_truestate.{py,txt,json}.
- research/week98/chooser_headside.{py,txt,json}.

## The metric decides the answer
- **CERTIFY98's gate is walk-strict per band** (amendment 4): P(right at d | right at 0..d-1) at the decode walk's
  check starts. The marginal bands (all cells) are information only.
- **Kamen's 97.8 is the true-state head's MARGINAL figure** (97.7-98.0 flat).
- **In walk-strict the profile flips:**
  - The deep bands are already high, because surviving to depth 15+ selects easy stretches.
  - The shallow bands are low, because check starts sit right after a miss, on the 125B's own near-ties.
  - **Even the true-state head is only 93.2 / 95.3 / 96.4 at bands 0 / 1-2 / 3-6.**

## HE+ 24 dev (bands 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30)

| arm (exact unless noted) | walk-strict (THE GATE) | kept/32 | marginal (information) |
|---|---|---|---|
| t30b chain-32 | 92.0 91.5 92.4 94.3 96.6 96.7 97.7 | 13.77 | 98.0 96.3 94.4 93.1 92.2 90.9 89.7 |
| + 2B audit P = 1 | 92.7 90.6 92.8 95.2 96.6 **97.6 98.2** | 14.46 | 97.9 96.4 95.6 94.8 94.7 94.5 94.3 (per cell) |
| + 2B audit P = 2 | 92.7 91.0 93.2 94.8 97.1 **98.3 98.3** | 15.00 | - |
| + 4B audit P = 1 | 93.0 92.1 94.0 95.1 96.7 **97.5 98.2** | 15.40 | 98.0 96.5 96.1 95.8 95.7 95.5 95.3 (per cell) |
| + 2B, head-side chooser P = 1 (out of sample) | 93.6 91.0 93.4 95.2 96.9 97.8 98.3 | 15.11 | 98.0 96.6 95.8 95.4 95.2 95.0 94.8 |
| tree-16 EAGLE-2 (coverage, 16-row) | **98.6** 96.4 93.7 95.0 96.2 - - | 11.26 / 16 | - |
| LAB: true state (not deployable) | 93.2 95.3 96.4 97.8 98.8 99.2 99.3 | 20.77 | **98.0 97.9 97.9 97.9 97.8 97.8 97.7** |
| ceiling: head OR 2B (any number of repairs) | 94.9 94.5 95.2 97.1 98.3 98.6 99.2 | 19.09 | 98.7 97.9 97.2 96.8 96.7 96.5 96.3 |
| ceiling: head OR 4B (any number of repairs) | 94.0 96.4 95.6 97.5 97.8 99.3 99.1 | 20.24 | 98.8 98.2 97.7 97.4 97.3 97.2 97.1 |

## How close each lever gets to 97.8
- **Walk-strict (the gate):**
  - **Bands 15-30: at the line with the audit.** 2B P = 2 gives 98.3 / 98.3, P = 1 97.6 / 98.2, the 4B 97.5 / 98.2.
    t30b alone gives 96.7 / 97.7.
  - **Band 0: the tree, 98.6.**
  - **Bands 1-14 fall short for every deployable lever, at 90.6-97.1.** Even the true state gives 95.3 / 96.4 / 97.8 at
    1-2 / 3-6 / 7-10. No lever we have reaches 97.8 there by Saturday.
- **Marginal (information):**
  - Only the LAB true state reaches 97.8, flat.
  - The best deployable is the 4B audit with the chooser: 95.5 at 23-30. The 2B gives 94.3-94.8.
  - The distilled auditor (item 2) would add +0.5-1.5 if it passes.
  - t30c adds +0.5-1.5 unaudited, to ~90-91.

## Proposed lines (EvalSuite amendment by Wed ~18:00; HE+ 140 fresh)
1. **LAB, the knowledge (not deployable; it needs the 125B's state at every depth):**
   - Metric: the true-state head's MARGINAL accuracy per band.
   - No build: the guess at depth d from anchor a IS the depth-0 guess of anchor a + d, so it comes from the capture's
     depth-0 cells.
   - Cross-check: the shortcut reproduces BigMoE's REPLACE oracle to -0.04 kept/32 on MBPP and -0.13 on HE+ (20.77 vs
     20.90). The marginal bands match "97.76 flat". BigMoE's REPLACE on the fresh cells can confirm it.
   - **Line:** every band's point >= 97.5 and 95 % LB >= 97.0, with each band's point quoted against 97.8. A point line
     of 97.8 per band would be a coin flip, since the truth sits at 97.7-98.0.
2. **C2, t30b + 2B audit (exact).** This REPLACES my 21:3x line (+2.0 pt in every band 3-30), which already FAILS on dev:
   band 3-6 is +1.2 marginal and +0.4 walk-strict.
   - **New line:** walk-strict bands 15-22 and 23-30 point >= 97.8 and LB >= 97.0.
   - Co-primary: kept/32 paired-LB above t30b's.
   - The headline is P = 2 (dev 98.3 / 98.3; kept +8.9 %; 2.12 GB of 2B reads per check). P = 1 is reported (dev
     97.6 / 98.2; +5.0 %).
   - Bands 0-14 are reported, not gated.
3. **C1, the tree (already pinned):** band 0 walk-strict coverage point >= 97.8 and LB >= 97.0 (dev 98.6).
4. **The Saturday statement, if the lines pass on fresh data:**
   - "Exact. The first guess after a check is right ~98.6 % of the time (tree). Deep guesses (15-30) are right ~98 %
     (2B audit).
   - Guesses 1-14 after a check are right 91-97 %, short of 97.8. So is the true-state head (95-96 % at 1-6): these
     sit on the 125B's own near-ties.
   - The head's knowledge, given the true state, is ~97.8 % flat, certified on unseen data."

## Q3: a 4B or chooser route to ~96-97 in the (marginal) deep bands by Wednesday?
- **No.**
  - The 4B audit gives 95.3 at 23-30, and 95.5 with a chooser.
  - The perfect-chooser ceiling is 97.1 for the 4B and 96.3 for the 2B, but the gap can't be learned from draft-time
    features.
- **CHOOSER CLOSED, now with head-side features too:**
  - MoonShots' chooser used 2B-side features only, and its docstring says head features weren't in the dumps.
  - Mine adds the head's top-1 / top-2 probability and its probability of the auditor's token (TOPK dump), on the line
    written before the run.
  - It recovers only +0.2-0.5 pt at 23-30 out of sample (2B: 94.80 / 94.53 vs 94.32; 4B: 95.51 / 95.49 vs 95.29).
- **Side result:** with the 2B the chooser adds +3.5-4.5 % kept/32 over the frozen rule at P = 1 (15.11 / 14.96 vs
  14.46). That is a speed note for the audit, not a band lever.
- In the gate's walk-strict metric, the deep bands are already ~98 with any audit. The shortfall is bands 1-14.

## Item 3 result (tree at 0-2 + audited chain): KILL on HE+
- Budget-correct branch design on HE+, 2B: x0.991-1.016 vs tree-16 (line +2 %).
- Upper bound, budget-free and auditing every tree node: x1.093.
- The 4B lands BETWEEN: OPT x1.033, PES x1.011.
- MBPP PASSES (2B: +2.2 % PES to +6.0 % OPT; 4B +3.7-7.4 %). Information only: the line was HE+.
