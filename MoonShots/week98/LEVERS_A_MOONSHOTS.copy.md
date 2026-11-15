# LEVERS_A_MOONSHOTS: outside-the-literature levers for GOAL A (drift gap 89.7 -> 97.8, HE+ 23-30)

MoonShots, Sun 15 Nov 2026. v0 00:53: the top 3 were posted. v1 01:3x (final): MoonShots ran four screens on the
125B's BANK (A-D below), which re-rank the list; the audit rule is confirmed on HE+. Each lever below has its mechanism, a quick prior-art check (web search only, 5 min each:
"untested" means not found, not proven new), the drafting cost per token, and a 15-min screen with its
result -> action line written first.

## Tonight's four measured results, A-D (MoonShots; scripts and outputs in ~/MoonShots/week98/)
**A. Lever 1, the 2B as a STATE composer: KILL by its line.** A ridge from Qwen3.5-2B's hidden states to the 125B's wd,
scored exactly like bank_proxy.py. The sanity check reproduces the proxy's tap-45 point: R2 0.8285, direction error
0.3398.

| 2B state -> wd (ridge; 31,805 train / 9,984 val rows) | R2 | direction error | predicted MBPP 23-30 |
|---|---|---|---|
| after layer 6 / 12 / 18 | 0.419 / 0.451 / 0.549 | 0.629 / 0.610 / 0.550 | 49.6 / 53.5 / 63.7 |
| final (post-norm) | 0.567 | 0.538 | 65.3 |
| all four layers | 0.588 | 0.525 | **67.1** |
| for scale: the 125B's taps 5+15+25 / tap 45 | 0.497 / 0.829 | 0.580 / 0.340 | 59.7 / 91.9 |

Line: PASS >= 85, KILL < 78. 67.1 is below the head's own chain (72.1). A different model's state, mapped
linearly, doesn't carry the 125B's state: model mismatch, as the honest-ceiling section feared. Still untested: a
fine-tuned composer and fusion with the carry (lever 1b, below).

**B. The 2B as a TOKEN co-drafter: flat with depth, and better than t30b deep.** Qwen3.5-2B's own greedy token,
teacher-forced on the true tokens (what it sees on a correct chain, so the same at every depth), against the 125B's
runtime top-1 (the bank's tk[:, 0]), every answer row:

| set | 2B == 125B, all rows | on rows the 125B is sure of (margin >= 4 nats) | t30b depth 0 | t30b depth 23-30 |
|---|---|---|---|---|
| MBPP 100 | **85.67 %** (15,641 rows) | 97.30 % (71 % of rows) | 94.01 | 72.13 |
| HE+ 24 windows | **94.42 %** (6,375 rows) | 98.83 % (90 % of rows) | 97.90 | 89.65 |
| n1112 val 40 files | 87.29 % (9,984 rows) | 97.23 % (76 % of rows) | - | - |

- The head wins at shallow depth and the 2B wins deep: +13.5 pt on MBPP and +4.8 pt on HE+ at 23-30, with no training.
- CEILING put 75.8 % of the deep misses on rows the 125B is sure of. The 2B gets 97-99 % of those rows right.
- If the two models' errors were independent, a perfect chooser would reach 1 - 0.279 x 0.143 = 96.0 % on MBPP 23-30 and
  1 - 0.103 x 0.056 = 99.4 % on HE+. That is an upper bound only. The real gain needs the joint rows (lever 1, below).
- Files for that join: ~/MoonShots/week98/rows2b_{mbpp,heplus,val}.npz. Each holds, per answer row: file, row, the 2B's
  top-1 and its own margin, and the 125B's top-1 and margin. Lever 7's line (>= 90 % on sure rows) PASSES.

**C. The MBPP join (orchestrator 01:1x; week98/join_mbpp.py, CPU): audit and repair on t30b's own cells.**
- Inputs: BigMoE's CEILING cells (cells_t30b_mbpp100.npz, read-only) and the 2B's per-row picks.
- Alignment: all 438,850 cells matched to their rows exactly, by the stored margin. Depths 0-13 hold every row; from
  depth 14 the eval had dropped 1-24 cells per depth.
- The head's bands reproduce WEEK98: 94.01 at depth 0, 72.13 at 23-30.
- Rule: where the 2B disagrees with the head and its own margin is >= tau, the 2B's token stands. The head's drafted
  token isn't needed: if the head is right, an override is wrong; if the head is wrong, an override is right exactly
  when the 2B is right.
- "Mean accepted" is per anchor (every answer row an anchor), with P = the 2B's audit passes per check (one repair per
  pass).

| MBPP 100 | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | accepted P = 1 / 2 / all |
|---|---|---|---|---|---|---|---|---|
| head (t30b chain) | 94.01 | 89.14 | 83.45 | 79.61 | 77.06 | 74.38 | 72.13 | 7.38 |
| tau 2 | 93.18 | 89.70 | 86.48 | 84.77 | 83.91 | 83.07 | **82.47** | 8.28 / 8.70 / 9.06 |
| tau 3 | 93.84 | 89.69 | 85.61 | 83.42 | 82.28 | 81.13 | 80.33 | 8.23 / 8.56 / 8.83 |
| tau 0.5 | 88.75 | 87.55 | 86.45 | 85.85 | 85.59 | 85.40 | 85.14 | 7.60 / 8.11 / 8.66 |
| schedule tau 4 / 2 / 1 / 0.5 (depth 0 / 1-2 / 3-6 / 7+), chosen after seeing the rows above | **93.96** | 89.70 | 86.90 | 85.85 | 85.59 | 85.40 | **85.14** | 8.26 / 8.78 / 9.34 |

Line (PASS if 23-30 >= 82 with depth 0 within 0.3 pt):
- No single tau passes: tau 2 loses 0.83 pt at depth 0, and tau 3 reaches only 80.33. So it is BETWEEN, and its
  action is a depth-aware chooser.
- The schedule passes (93.96 and 85.14), but it was picked on these same rows. **HE+ confirms or kills it** (MTP Head's
  cells_t30b_heplus.npz, tonight).

What it means:
- The audit makes the drafter flat from depth ~3 at the 2B's own level: +13.0 pt at 23-30 on MBPP, with no training.
- It can't pass the auditor's own agreement: 85.7 % on MBPP, 94.4 % on HE+. So it is not the 98 % method by itself.
- The way up is to raise the auditor's flat level: lever 1c, distil the 2B toward the 125B.

Net speed, in bytes (the check is bandwidth-bound):
- P = 1 is +12 % accepted for 1.06 GB per block, ~+7 % of an exact 49 ms check, so ~+5 % net. At today's ~100 ms check,
  ~+8 %.
- P = 2 is +19 % for 2.12 GB, ~+4 % net at 49 ms.
- A 0.8B auditor (~0.4 GB per pass, within the 0.5 GB spec) would cost a third of the bytes, at a lower flat level.

**FROZEN before the HE+ join (Sun 15 Nov 01:3x, the orchestrator's instruction): the rule and its line.**
- Rule: the 2B's token replaces the head's where the two differ and the 2B's own top-1 margin (nats) is >= tau(depth):
  - tau = 4 at depth 0, 2 at depths 1-2, 1 at depths 3-6, 0.5 at depths 7-30.
  - Audit passes P = 1 and P = 2 are both reported.
- HE+ confirmation LINE (cells_t30b_heplus.npz, every cell; week98/join_mbpp.py's method, nothing re-tuned):
  - CONFIRMED if HE+ depth 0 is within 0.3 pt of t30b, AND 23-30 is >= t30b + 3.0 pt, AND mean accepted at P = 1 is
    >= t30b's x 1.05;
  - FAILS otherwise. No other schedule may be tried on HE+.

**D. HE+ CONFIRMATION of the frozen rule: CONFIRMED** (week98/join_heplus.py; MTP Head's cells_t30b_heplus.npz, md5
86f57b23; 186,465 cells; 125B margin fingerprint mismatches: 0).

| HE+ 24 windows | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | mean accepted P = 1 / 2 / all |
|---|---|---|---|---|---|---|---|---|
| head (t30b chain) | 97.95 | 96.30 | 94.41 | 93.06 | 92.16 | 90.94 | 89.65 | 19.33 |
| frozen schedule (tau 4 / 2 / 1 / 0.5) | 97.90 | 96.40 | 95.56 | 94.76 | 94.67 | 94.52 | **94.32** | 21.07 / 21.42 / 21.83 |

Line: depth 0 -0.05 (within 0.3), 23-30 +4.67 (>= +3.0), accepted at P = 1 x1.09 (>= 1.05). **CONFIRMED.**
- The audited drafter is flat at ~94.5 % from depth 7 to 30. That is the 2B's own agreement on HE+ (94.42 %):
  result C's law holds on a second set.
- 98 % needs an auditor at ~98 % agreement (lever 1c, running now).

**E. LEARNED CHOOSER: pre-registered 01:3x, before any code ran (orchestrator 01:2x).**
- **Question:** how much of the perfect-chooser bound at 23-30 can a cheap chooser recover beyond the frozen tau rule
  (HE+ 94.32, MBPP 85.14)?
- **Bound:** the oracle chooser is right wherever the head OR the 2B is right, computed per cell from the joins. It
  replaces the independence estimate (99.4 / 96.0).
- **Features, only what is known while drafting:**
  - depth (bins 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30, one-hot);
  - the 2B's own top-1 margin at the row, and at the previous row;
  - whether the 2B's top-1 at the previous 1 and 2 rows equalled the drafted token (on a correct prefix, the 2B was
    right there);
  - the class of the 2B's top-1 token: newline / indent whitespace, other whitespace, punctuation / operator,
    identifier / word, number, other;
  - depth bin x margin interactions.
- **Excluded, with reasons:**
  - The cells' m, rank and dlp are the 125B's own log-probs at the target row. They are known only after the check, so
    using them would leak the answer.
  - The head's own margin and the 2B's rank of the head's token aren't in either dump: the cells hold no drafted token
    or head distribution. They need a richer dump of t30b's chain (the drafted token and the head's top-k per cell;
    MTP Head / BigMoE) and then a rerun.
- **Model:** L2 logistic regression, trained on the decision cells only (exactly one of head / 2B right; y = the 2B is
  right). The 2B overrides where P > 0.5.
- **Split, by problem, never by row:** train on MBPP problems 1-50 (the first 50 files in sorted order); test on MBPP
  51-100 and all 24 HE+ windows.
- **Reported on the test sets:**
  - bands, and accepted per anchor at P = 1 / 2 (the join's walk);
  - the frozen tau rule and the head on the same sets, and the oracle bound;
  - the share of the bound recovered at 23-30 = (chooser - head) / (bound - head).
- **LINE:** the chooser is ALIVE if HE+ 23-30 >= 96.0 and MBPP (51-100) 23-30 >= 88.0, with depth 0 within 0.3 pt of
  the head on both. Otherwise the frozen rule stands.
- **Expectation, written first:** little beyond the frozen rule. With only the 2B's side, the chooser sees about what
  tau(depth, margin) already uses. The head's own confidence is the likely missing piece.

**E, RESULT (01:2x; week98/chooser.py, chooser.json): the frozen rule stands.** Trained on 40,693 decision cells of
MBPP files 0-49; tested on files 50-99 and all of HE+.

| held-out, 23-30 (depth 0) | head | frozen rule | learned chooser | oracle chooser (head OR 2B right) |
|---|---|---|---|---|
| MBPP files 50-99 | 72.06 (93.94) | 85.02 (93.89) | 84.93 (93.91) | **89.96** (96.28) |
| HE+ 24 windows | 89.65 (97.95) | 94.32 (97.90) | 94.34 (97.93) | **96.29** (98.67) |
| mean accepted P = 1, MBPP / HE+ | 7.34 / 19.33 | 8.29 / 21.07 | 8.38 / 21.16 | 9.87 / 22.21 |
| share of the bound at 23-30, MBPP / HE+ | - | 0.724 / 0.703 | 0.719 / 0.706 | 1 |

- Line (HE+ >= 96.0 and MBPP >= 88.0): not met. The draft-time features add nothing beyond tau(depth, margin).
- **The new number is the real bound.** The head's and the 2B's errors are correlated: they miss the same hard rows.
  - So a perfect chooser between them reaches only 96.29 (HE+) and 89.96 (MBPP) at 23-30, not the independence
    estimate's 99.4 / 96.0.
  - The frozen rule already recovers ~70 % of that.
- **For the week:** no chooser over the 2B as it is gets to 98. That needs a better auditor (lever 1c). Head-side
  features (the drafted token and the head's top-k per cell) could still recover part of the remaining ~2-5 pt of
  bound, but only up to the bound.

**F. The COMMON MISSES at 23-30, rows where the head AND the 2B are both wrong (orchestrator 01:3x; week98/common_misses.py,
common_misses.json).** They are the gap between the oracle chooser (HE+ 96.29, MBPP 89.96) and 98 %. Token classes come
from a character scan of the decoded answer. "#" comments are reliable; "string text" also catches prose after an
apostrophe.

| HE+ 23-30 | all cells | common misses (3.71 pt) | fixable by the audit (6.64 pt) |
|---|---|---|---|
| 125B margin < 0.25 / 0.25-1 / 1-4 / >= 4 nats (%) | 0.8 / 3.0 / 6.9 / 89.2 | **9.4 / 33.6 / 44.6 / 12.4** | 2.9 / 9.8 / 20.2 / 67.1 |
| comment text / string text / identifier / newline-indent / punct (%) | 14.2 / 46.7 / 9.7 / 17.4 / 6.0 | **74.2** / 0.9 / 11.0 / 6.7 / 4.7 | 34.9 / 14.0 / 24.2 / 8.0 / 13.0 |
| true token seen in the last 64 / 256 / anywhere; a bigram lookup predicts it (%) | 60.9 / 83.3 / 88.6; 49.4 | 25.5 / 43.2 / 45.1; **1.4** | 48.5 / 64.5 / 67.8; 12.0 |

| MBPP 23-30 | all cells | common misses (9.98 pt) | fixable (17.9 pt) |
|---|---|---|---|
| 125B margin < 0.25 / 0.25-1 / 1-4 / >= 4 (%) | 2.3 / 8.1 / 20.1 / 69.5 | **11.1 / 35.3 / 46.1 / 7.4** | 2.1 / 9.8 / 29.8 / 58.3 |
| comment / string text / identifier / newline-indent / punct (%) | 22.0 / 33.3 / 12.4 / 16.3 / 8.3 | **40.2 / 37.3** / 9.0 / 8.4 / 2.5 | 24.8 / 39.1 / 15.4 / 8.4 / 8.4 |
| seen 64 / 256 / any; bigram lookup (%) | 61.0 / 75.0 / 76.5; 20.5 | 31.9 / 45.5 / 48.4; **4.2** | 50.6 / 65.0 / 66.5; 8.8 |

**What a 98 % corrector must learn:**
- The common misses are the 125B's own uncertain wording, mostly inside code comments (HE+ 74 %) and prose / string
  text (MBPP 78 % with comments).
- They sit on rows the 125B itself wasn't sure of: only 7-12 % have a margin >= 4 nats, against 70-89 % of all deep rows.
- Mid-line (81-83 %); rarely copyable.

**Which lever reaches which share** (HE+, of the 3.71 pt; MBPP, of 9.98):
- **Exactly unfixable: the near-ties** (m < 0.25): ~0.35 / ~1.1 pt. Only close-call acceptance (lossy) takes them.
- **Copy / lookup: at most the bigram-predictable rows**, 1.4 / 4.2 % (~0.05 / ~0.4 pt). Copy isn't the lever, as T1
  found.
- **Distillation (1c) or prefix features: the rest (~90 % of the misses), but it is hard material.** The auditor must
  learn the 125B's own phrasing of comments and prose at 0.25-4 nat margins. The 125B's fresh state already carries
  it: the head with the true state holds 97.76 on HE+ 23-30. A 2B doesn't.
- **For the lossy track (the 1k goal):** 74 % of HE+'s remaining deep misses are inside comments, exactly where a
  quality-neutral relaxed acceptance applies (DRAFTER T16 tried comment rows on HumanEval: kept x1.031, KILL). With the
  audit flattening the code rows, that may now be worth a re-screen behind EvalSuite's gate.
- **HE+ arithmetic:** reaching 98 from the frozen rule's 94.32 needs 3.68 pt. Better choosing can add at most 1.97 (up
  to the bound); the other >= 1.71 must come from these common misses, i.e. about half of them.

**G. SIZE SCALING: Qwen3.5-4B as the auditor** (on disk, no download; week98/agree_model.py, join_auditor.py; peak
3.87 GB).
- Teacher-forced agreement with the 125B (all rows; by the 125B's margin < 0.25 / 0.25-1 / 1-4 / >= 4):

| | HE+ 2B | HE+ 4B | MBPP 2B | MBPP 4B |
|---|---|---|---|---|
| all rows | 94.42 | **95.48** | 85.67 | **88.02** |
| by margin | 43.8 / 45.4 / 58.8 / 98.83 | 45.8 / 43.7 / 67.2 / 99.46 | 35.9 / 43.8 / 65.0 / 97.30 | 37.4 / 46.9 / 70.8 / 98.64 |

- The frozen rule with the 4B (not re-tuned), at 23-30 (depth 0): HE+ **95.29** (97.98), oracle bound 97.09; MBPP
  **87.17** (93.98), bound 91.91.
- Mean accepted at P = 1 / 2: HE+ 21.49 / 21.99 (2B 21.07 / 21.42; head 19.33); MBPP 8.50 / 9.15 (2B 8.26 / 8.78).
- Bytes per pass: **2.37 GB** (trunk 2.01 + tied lm_head 0.36), 2.2x the 2B and 4.7x the 0.5 GB spec.
- **Reading:**
  - The flat-at-the-auditor's-level law holds again (95.29 against the 4B's 95.48 agreement).
  - Doubling the size buys ~+1 pt (HE+) / ~+2 pt (MBPP), all of it on rows the 125B is fairly sure of. It buys ~0 on
    its uncertain picks (m < 1), where the common misses live (F).
  - So size alone doesn't reach 98 within any bytes budget. The auditor has to learn the 125B's own choices: lever 1c,
    distillation on the 125B's outputs.

**H. LOSSY: relaxed acceptance in comments / docstrings on top of the frozen 2B audit (T16-HE's rule).**
**Pre-registered 01:4x, before any code ran** (orchestrator 01:4x):
- **Rule R(k), k = 2 / 3 / 5:** after the frozen audit, a wrong final token counts as accepted if
  - (i) the prefix ends inside a `#` comment (not on its closing newline) or inside a triple-quoted docstring, by a
    character scan of the true prefix; ordinary string literals are excluded, because they change program output;
  - and (ii) the final token's rank in the 125B's top-32 at that row is <= k. Rank 1 is a hit; R(k) adds ranks 2..k.
- **Final token's rank:** for the 2B's token (an override), its place in the bank's tk at that row. For the head's
  token, the MBPP cells' rank field.
- **HE+ caveat:** MTP Head's HE+ cells hold no rank on misses (no drafted token in the dump). So HE+ gets bounds:
  - lower = the head's missed tokens are never relaxed;
  - upper = all of them are relaxed when in span.
  - Asked: a richer HE+ dump (the drafted token per cell) for the exact number.
- **Reported:**
  - bands 0 / 7-10 / 15-22 / 23-30, and mean accepted per anchor (every row audited; P = 1 is lower by the earlier
    gap, ~0.8 on HE+);
  - the share of the frozen rule's 23-30 misses fixed;
  - exposure: relaxed acceptances per 100 accepted tokens, and the share of in-span rows at 23-30 whose output differs
    from the 125B's top-1.
- **LINE:** a lossy GOAL A candidate for EvalSuite's code gate (CODE-ONLY) if HE+ 23-30 >= 97.5 with depth 0 within
  0.3 pt of the head, on the lower bound (if only the upper bound reaches it: OPEN until the richer dump).
- **Expectation, written first:**
  - The frozen rule's HE+ misses at 23-30 are 5.68 pt, and ~half or more sit in comments (F: 74 % of the common
    misses).
  - If most of those drafts rank <= 3, R(3) adds ~2-3 pt, to ~96.5-97.5: borderline for 97.5.
  - MBPP carries real quality risk: T16 scored BETWEEN there (one flipped task in 12; MAIN 10.3).

**H, RESULT (01:5x; week98/relax.py, relax.json).** At 23-30, 63.5 % (MBPP) and 71.6 % (HE+) of the rows lie inside a
comment or docstring. Bands at depth 0 / 7-10 / 15-22 / 23-30; mean accepted per anchor with every row audited:

| | MBPP (exact ranks) | HE+ lower bound | HE+ upper bound |
|---|---|---|---|
| head | 94.01 / 79.61 / 74.38 / 72.13; 7.38 | 97.95 / 93.06 / 90.94 / 89.65; 19.33 | same |
| frozen 2B audit | 93.96 / 85.85 / 85.40 / 85.14; 9.34 | 97.90 / 94.76 / 94.52 / 94.32; 21.83 | same |
| R(2) | 97.60 / 91.79 / 91.61 / 91.32; 12.79 | 97.95 / 96.07 / 95.86 / 95.70; 22.34 | 99.53 / 97.06 / 96.96 / 96.84; 23.51 |
| R(3) | 98.47 / 93.73 / 93.74 / 93.48; 14.72 | 97.95 / 96.54 / 96.35 / 96.20; 22.59 | 99.53 / 97.53 / 97.45 / 97.35; 23.96 |
| R(5) | 98.87 / 95.38 / 95.49 / **95.30**; 17.23 | 97.96 / 96.98 / 96.81 / **96.67**; 22.73 | 99.55 / 97.97 / 97.90 / **97.82**; 24.52 |

- **Share of the frozen rule's 23-30 misses fixed:** MBPP R(2) / R(3) / R(5) 0.42 / 0.56 / 0.68; HE+ 0.24-0.44 /
  0.33-0.53 / 0.41-0.62.
- **Quality exposure, at 23-30:**
  - relaxed acceptances per 100 cells: MBPP 1.5 / 2.0 / 2.4; HE+ 0.3-0.6 / 0.5-0.8 / 0.6-0.9;
  - share of in-span rows whose output differs from the 125B's top-1: MBPP 9.7 / 13.1 / **16.0 %**; HE+ 1.9-3.5 /
    2.6-4.2 / 3.3-4.9 %.
  - MBPP's exposure is 3-5x HE+'s. That is where T16 (rank <= 3) flipped one task in 12 (MAIN 10.3). HE+'s docstrings
    are mostly copies of the prompt, so they expose little.
- **Line** (HE+ 23-30 >= 97.5 with depth 0 held, on the lower bound):
  - R(2) and R(3) fail even on the upper bound (96.84 / 97.35).
  - **R(5) is OPEN**: lower 96.67, upper 97.82. An exact HE+ number needs the drafted token per HE+ cell (MTP Head's
    dump has no rank on misses); asked for it.
  - If it passes on the exact dump, it is a lossy GOAL A candidate for EvalSuite's CODE-ONLY gate. MBPP's 16 %
    exposure is the risk that gate must test.
- **Check-side cost:** T16's +1.9 ms per cycle was BigMoE's attribution, "InertScan host work likely" (a Python span scan
  per cycle on the host; not profiled; BigMoE HANDOVER, SAT 14:37).
  - It isn't inherent. A span state kept incrementally per token costs microseconds.
  - The rank test reads the logits the check already computes: one top-5 per row on the GPU, ~0.1-0.3 ms for 32 rows.
  - Expected ~0.2-0.3 ms per cycle if built that way. Profile before relying on it.
- **Approximation:** after a relaxed token, the later cells keep their teacher-forced outcomes. In the real loop both the
  drafter and the 125B then condition on the relaxed token.

**I. The incremental comment / docstring span tracker (for H's rule without T16's 1.9 ms).**
**Pre-registered 01:5x, before any code ran** (orchestrator 01:5x):
- **Tracker:** a state machine updated once per accepted token, from the token's decoded text (cached per id), carried
  across checks. It knows:
  - markdown ``` fences (prose outside a code block is never in span);
  - `#` comments, ended by a newline;
  - single- and triple-quoted strings, with backslash escapes and quote runs split across tokens;
  - docstrings. A triple-quoted string is a docstring only if it opens at the start of a line (only indentation before
    it) and the previous logical line ended with ':', or it is the first statement of the block. Other triple-quoted
    literals are strings, so they are not in span, because relaxing them changes program output.
- **In span after a prefix:** the state after the prefix's last character is inside a comment, or inside a docstring (after
  its third opening quote and before any closing quote).
- **Reference:** Python's `tokenize` (COMMENT tokens) and `ast` (docstring nodes of Module / FunctionDef / ClassDef) on each
  fenced block of every HE+ (24) and MBPP (100) answer in the BANK, read at each answer token's start, with the same
  definition. Blocks that fail ast / tokenize are excluded and counted.
- **LINES:** recall >= 99.0 % of the reference in-span rows; **false in-span <= 0.1 % of reference code rows** (inside
  blocks, outside comments / docstrings: the dangerous error), with prose and ordinary string literals reported apart;
  speed <= 10 us per token in Python (<= 0.3 ms per 32-row check).

**I, RESULT (01:5x; week98/span_tracker.py, span_tracker.json): all three lines PASS.**
- **Coverage:** 124 fenced blocks in 124 answers; 22,016 tokens. 8 blocks didn't parse (mostly truncated answers) and
  are excluded (3,858 rows, counted).

| reference class (rows) | tracker says in span |
|---|---|
| comment (3,871) | 100.0 % |
| docstring (5,876) | 97.69 % (misses are safe: no relaxation) |
| code (7,691) | **0.0 %** |
| ordinary string (115) | 0.0 % |
| prose outside blocks (481) | 0.0 % |

- **Totals:** recall **99.32 %** (>= 99), precision 100 %, false in-span **0 rows** (<= 0.1 % of code rows), **0.87 us
  per token** in Python (<= 10). That is ~30 us per 32-token check, against T16's 1.9 ms.
- **Engine spec for BigMoE:** ~/MoonShots/week98/ENGINE_SPEC_relax.md. It covers:
  - where the state lives: per request, checkpointed with the conversation cache;
  - the rank test on the check's own logits (one compare-and-count, ~0.1-0.3 ms);
  - the counters for EvalSuite (relaxed, relaxed_by_rank, span_rows, relaxed_pos);
  - a must-have guard: a relaxed token has to start AND end inside the span. H's numbers didn't include that guard,
    so they are a slight overestimate.

**H with the guard (MBPP, exact; week98/relax_guard.py, 01:5x).** The span test is now the validated tracker, plus the
start-AND-end guard. The drafted token's text is known for every candidate: it is the 125B's top-32 entry at the stored
rank, or the 2B's top-1.

| MBPP 23-30 (depth 0); accepted; fixed share; in-span rows deviating | no guard | with the guard |
|---|---|---|
| R(2) | 91.30 (97.58); 12.78; 0.415; 9.75 % | **91.07** (97.47); 12.64; 0.399; 9.39 % |
| R(3) | 93.44 (98.45); 14.70; 0.559; 13.13 % | **93.10** (98.31); 14.44; 0.536; 12.59 % |
| R(5) | 95.26 (98.85); 17.15; 0.681; 16.00 % | **94.85** (98.69); 16.57; 0.654; 15.37 % |

The guard costs 0.2-0.4 pt at 23-30. HE+ still waits for the 06:00 cells.

**J. Does H's approximation hold? Pre-registered 01:5x, before any code ran.**
- **(A) Tonight, the BRANCH test on the 2B** (the 2B as the target):
  - Take the first 50 MBPP answers of the 2B's own greedy (Sandbox98 data/mbpp).
  - At up to 3 in-span rows per answer, pick the relaxation event: the 2B's rank-2 token, if it passes the guard.
  - Decode the 2B greedily for 64 tokens from prefix + that token, and compare with the original continuation.
  - Measure: once the comment / docstring ends, is the next code (the first two code lines after the span) the same?
  - **LINE:** >= 90 % unchanged -> H's "later cells keep their outcomes" holds for the code rows, and the exposure is
    the comment text itself. < 70 % -> H's lossy numbers aren't trustworthy, and the code gate will likely see flips.
    In between: the gate decides.
- **(B) The full real-decoding loop, specified for Sandbox98:** draft 30 with s30, verify with the 2B, accept exact vs
  R(5) with the guard. Compare the real mean accepted per anchor and the 23-30 band with the teacher-forced prediction
  on the same windows.
  - LINE: within ~1 pt -> H's numbers stand.
  - It needs an incremental head context and rolling the 2B's Gated-DeltaNet state back to the accepted length. That is
    several hours; it goes to Sandbox98, which owns the 2B head harness.

**J(A), RESULT (02:0x; week98/branch_2b.py, branch_2b.json): NOT TRUSTWORTHY by its line. A relaxed comment changes the
code that follows about half the time.**
- **Setup:** the 2B as the target, the first 50 MBPP answers, 144 relaxation events (the 2B's rank-2 token in a
  comment / docstring span, passing the start-AND-end guard). The 2B decodes greedily for 64 tokens from each.
- **Result:** of 102 decidable events, the first two code lines after the span (code characters only; later comments /
  docstrings skipped) are **unchanged in 53, changed in 49: 52.0 %** unchanged (line: >= 90 % holds, < 70 % not
  trustworthy). 42 more were undecidable within 64 tokens.
- **The changes are real code**, steered by the comment's wording:
  - " strip" -> " filter" in a comment turned `[word.strip() for word in string.split()]` into
    `[word for word in string.split() if word]`;
  - " Note" continuing a docstring gave a loop instead of `return all(...)`;
  - "Calculate" in a docstring gave a `def gcd` helper instead of the edge-case checks.
- **Caveats:**
  - The relaxed token was the 2B's rank 2: a stress choice, though R(5) admits such tokens.
  - The target is the 2B, not the 125B.
  - A code change isn't necessarily a failed test.
- **What it means for H:**
  - The exposure is not confined to comment text, so the code gate will likely see flips (as T16 flipped an MBPP task
    and CD16 failed MBPP).
  - H's teacher-forced kept gains assume the original continuation, so they're not reliable.
  - **H is DOWNGRADED:** not a GOAL A candidate unless EvalSuite's CODE-ONLY gate passes it in real decoding.
- The full loop, J(B), stays specified for Sandbox98, but J(A) already answers the decision question.
- Noted from Sandbox98's status (not my run): 1c (LoRA on 360 files) = KILL, it overfits (best MBPP +0.52). The
  distillation route needs far more data (n1112) or another form.

## K. The hard first guess at check starts (orchestrator 03:1x; pre-registered 03:2x, before any code ran)
**The facts** (WEEK98 top; EvalSuite 9480ba4):
- kept/32 is earned in guesses 0-14.
- At the decode walk's check starts (right after a miss), t30b's depth 0 is 91.6 % on HE+ and 89.5 % on MBPP, against
  97.9 % at random anchors, although depth 0 has the 125B's true state.
- The misses are far from the 125B's top-1 (median -4.5 nats), so the drafter lacks information.

**Hypothesis (from this file's results):** at a check start, the newest token is the bonus token: the 125B's
correction, a surprise to the drafter. The head reads the 125B's state from before that token, plus the token's
embedding, through ONE layer. A same-family LM reads the actual tokens, the bonus token included, through 24-32 layers
(B: flat 94.4 % on HE+). An audit applied at check starts targets exactly this gap.

**Screen K** (CPU; the cells + rows2b / rows4b; my walk scorer = certify98's kept_walk logic, from row 1, 32-row
checks, P audit passes per check, at most one repair per pass):
- K0: t30b alone, the control. It must reproduce certify98.py's kept/32 for the same cells.
- K1: the frozen 2B audit (tau 4 / 2 / 1 / 0.5), P = 1 and 2.
- K2: K1 with the depth-0 threshold tau0 in {2, 1, 0.5}, chosen on MBPP files 0-49 only, then tested unchanged on MBPP
  50-99 and HE+. In the walk, depth 0 is scored only at check starts.
- K3: the 4B as the auditor (frozen schedule, and K2's tau0).
- **Metrics:** kept/32 under the walk; walk-strict at depths 0 / 1 / 2 (strict accuracy at the walk's check starts).
  Cost: bytes per check (2B 1.06 GB, 4B 2.37 GB per pass).
- **LINE:** LIVE if HE+ walk-strict depth 0 goes from 91.6 to >= 95.0, AND kept/32 at P = 1 rises >= +5 % on HE+ and
  on held-out MBPP. Otherwise the check-start gap isn't the 2B's to close.

**Screen L** (CPU; the cells' ranks recover the drafted token on misses): a correction cache within the answer.
- After each miss, remember (previous token, drafted token) -> the 125B's token. At later cells of the same window,
  replace a matching draft with the cached correction.
- **LINE:** LIVE if it fixes >= 10 % of the check-start depth-0 misses while breaking <= 0.5 % of the hits it touches.

**Other sources, not screenable on existing data tonight (designs, with cost):**
- **The 125B's early layers on the bonus token:** a 1-row partial trunk pass, ~0.6 ms per layer, so L5 ~3 ms per check.
  It needs head training to use the new input (LAYERS: early taps map poorly).
- **The check's own rows past the miss:** the row after the miss saw the WRONG token, but its argmax may still
  predict the token after the bonus. It is free (already computed) but needs a re-run that saves the verify's rows past
  the first miss.
- **A 2-way tree at depth 0 after a miss:** a check-side change: +1 row per check, plus tree attention.
- **Online head adaptation on this answer's corrections:** lever 6. The Sandbox98 analog runs in ~15 min.

**K, RESULT (03:3x; week98/checkstart.py, checkstart.json): NOT LIVE by its line.** My walk reproduces certify98's t30b
kept/32 exactly (MBPP 6.924, HE+ 13.769).

| decode walk, 32-row checks, P = 1 | MBPP held-out 50-99 (all 100) | HE+ 24 | HE+ walk-strict d0 / d1 / d2 |
|---|---|---|---|
| t30b alone | 6.879 (6.924) | 13.769 | 91.58 / 92.22 / 90.28 |
| 2B audit, frozen rule | 7.459 (+8.4 %) | 14.456 (+5.0 %) | **92.74** / 90.71 / 89.76 |
| 4B audit, frozen rule | 7.702 (+12.0 %) | 15.399 (+11.8 %) | **93.00** / 91.95 / 91.53 |

- K2 (lower depth-0 threshold, chosen on MBPP 0-49): every lower tau0 is worse, so the frozen tau0 = 4 stays. At
  tau0 = 0.5 (about the 2B everywhere at depth 0), walk-strict d0 drops to 81.9 (MBPP) / 87.3 (HE+). The 2B is WORSE than
  the head at check starts.
- P = 2 (two audit passes, twice the bytes): 2B MBPP-test 7.739 / HE+ 15.000; 4B MBPP-test 8.172 / HE+ 15.819.
  Walk-strict d0 is unchanged (92.7 / 93.3 on HE+).
- Line: HE+ d0 >= 95.0 is not met (92.7 / 93.0). kept/32 rises +5-12 %, but from the deeper guesses.
- **Reading:** the hard first guess after a miss is hard for every cheap model we have: head 91.6, 2B ~87, 4B ~89 at
  check-start depth 0 on HE+. The information there is specific to the 125B, right after it overrode the drafter. A
  same-family LM doesn't carry it.

**L, RESULT (03:3x; week98/checkstart_L.py): KILL.** The in-answer correction cache broke every hit it touched (MBPP
893 / 893, HE+ 82 / 82). It fixed only 3 / 274 (MBPP) and 2 / 42 (HE+) check-start misses, and kept/32 fell (MBPP
6.92 -> 5.34, HE+ 13.77 -> 12.50). A (previous token, draft) pair recurs mostly where the draft is right. Corrections don't
transfer within an answer at this granularity.

**What's left for the hard first guess (all 125B-side, so not screenable on existing data):**
- the check's rows past the miss: free, needs a verify dump;
- a 2-way tree at depth 0 after a miss: +1 row per check;
- a partial trunk pass on the bonus token: ~0.6 ms per layer, plus head training.
The cheapest first test: a verify dump that saves the 125B's argmax at the row after the miss (computed on the wrong
token). Then count how often it equals the token after the bonus.

**K-next: the VERIFY-DUMP spec (orchestrator 03:3x; sent to BigMoE 03:4x).**
- **Question:** at a check that ends at a miss (first miss at position s+j: the drafted d_j != the 125B's argmax = the
  bonus b), the verify's row s+j has already seen d_j and computed the 125B's logits for position s+j+1.
  - On the WRONG prefix (... d_j), does its argmax equal y, the true token after b at s+j+1 (the next check's depth-0
    target)?
  - If so, depth 0 after a miss gets a free candidate informed by the 125B.
- **What each check saves (the full form):** the absolute rows; the drafted d_1..d_n; the miss index j; the bonus b;
  and for the rows s+j .. s+j+2 past the first miss, the top-8 ids and log-probs. Only row s+j answers the question.
  The next two rows show how fast the wrong prefix drifts.
- **The smallest run that answers it, on the BANK's streamed runtime path, not a 45 GB run:**
  - The decode walk's check-end misses are known from the cells.
  - d_j is recoverable from the cells' rank on ~92 % of misses: tk[row][rank]; rank 99 means outside the top-32 and is
    skipped.
  - So per miss, one 125B row is enough: the true prefix up to s+j-1, then d_j, and keep that row's top-8.
  - Batched as the walk does it: ~2,260 check ends on MBPP 100 and ~460 on HE+ 24, i.e. one 1-row branch per miss,
    with the cache snapshot / restore that convcache already does bitwise.
  - Or run the real loop with t30b drafting and keep the rows past the miss. That gives the same rows, with real drafts.
- **Statistics, at check starts:**
  - M1 = P(argmax(row s+j) == y | t30b's depth 0 is WRONG at the next check). This is the lever's headroom.
  - M0 = the same given t30b is right, and P(argmax == y | it disagrees with t30b's depth 0). These two are the rule's
    precision.
  - By margin bin: the 125B's margin at the miss row (about b) and at the target row (about y).
  - Top-1/2/3/5 containment of y in row s+j, for a 2-way tree option.
- **LINE:** a 125B-side depth-0 lever if M1 >= 50 % (MBPP; HE+ reported, with a small n of ~40 misses) AND
  P(argmax == y | disagrees with t30b) >= 60 %.
  - Cost if live: ~0 ms. The verify already computed row s+j; the engine keeps its argmax instead of dropping every row
    past the miss.
  - If M1 < 25 %: the wrong token derails the 125B's next prediction, so only a fresh pass on b would carry the
    information (a partial trunk pass, priced at ~0.6 ms per layer).

**K-next addendum, pre-registered 03:4x before computing** (seen in the dump's first record: the drafter's own d1 = y):
the DRAFTER's own continuation past the miss as a free depth-0 candidate. d1 is t30b's depth k+1 draft, made after its
wrong d_k and already in the block. Same statistics, same lines:
- N1 = P(d1 == y | t30b's fresh depth 0 wrong at the next check);
- P(d1 == y | d1 != t30b's fresh depth 0). The fresh depth-0 token is recovered from the cells: = y if right, else
  tk[t][rank].
- LIVE if N1 >= 50 % (MBPP) AND that precision >= 60 %.

**K-next, FINAL (04:0x): FAILS its pre-registered line on MBPP, and on HE+.** Data: BigMoE's kdump_mbppbank.jsonl
(2,107 branches) and kdump_heplus24.jsonl (326), on the bank's runtime path; week98/kdump_score.py reproduces BigMoE's
kdump_stats.
- MBPP, the pre-registered set:
  - the 125B's row past the miss: **M1 6.6 %** (n 228), M0 17.8 %, P(== y | disagrees with t30b's fresh d0) **0.9 %**
    (n 1,758); y in its top-1 / 2 / 3 / 5 / 8: 16.6 / 22.4 / 25.6 / 30.5 / 35.4 %;
  - by target margin: 8.0 / 10.9 / 17.0 / 19.3 % at 0-0.5 / 0.5-2 / 2-5 / >= 5 nats;
  - the drafter's own d1: P(d1 == y) 76.0 %, **N1 20.5 %** (n 224), P(== y | d1 != fresh d0) **10.6 %** (n 433);
  - recovery: row 1 predicts y2 59.2 % of the time, given d1 == y.
- Both candidates fail on both sets. HE+, as below:

| | overall | where t30b's fresh depth 0 is wrong (n 35) | where t30b's fresh depth 0 is right | when it disagrees with t30b's fresh d0 |
|---|---|---|---|---|
| the 125B's row past the miss: argmax == y | 18.4 % (y in top-2 / 3 / 5 / 8: 25.8 / 31.3 / 35.0 / 38.7) | **M1 2.9 %** | 20.3 % | 0.4 % (n 263) |
| the drafter's own d1 == y | 78.9 % (n 323) | **N1 17.1 %** | 86.5 % | 10.5 % (n 57) |

- **The 125B row DERAILS** (M1 < 25 %), in every margin bin: 20.0 / 3.5 / 16.9 / 23.4 % at target margins
  0-0.5 / 0.5-2 / 2-5 / >= 5 nats. On the wrong token, the 125B predicts the wrong token's continuation.
- **The drafter's d1 is NOT LIVE.** It is right mostly where the fresh guess is right anyway. Where the fresh guess
  fails, it fails too.
- **Recovery one token later:** given d1 == y, the 125B's next row predicts y2 70.2 % of the time (n 255).
- **So:** the information for the hard first guess isn't in any row the check already computes. It needs the 125B's own
  pass on the bonus token, i.e. the next check itself. A partial pass would cost ~0.6 ms per layer, plus head training.

## J(CC). Close-call acceptance end to end on the 2B: a quality pre-screen (orchestrator 09:0x; pre-registered 09:0x, before any code ran)

**Question:** does R1(delta), the engine's close-call accept rule, keep pass@1 when the drafter's near-tie picks are
committed in real decoding?
- EvalSuite gates delta 1 on the 125B Monday night (ccx45cc1; PREREG_lossygate amendment 3).
- Delta 2 lifts teacher-forced kept/32 by +32 % (HE+) and +46 % (MBPP) on the 125B (Breakthrough's
  CLOSECALL_ACCEPT_PRICE.md), but it has no quality data.

**Setup:**
- Target Qwen3.5-2B (mlx-community 4-bit, Sandbox98's). Drafter s30 (Sandbox98's native MTP head trained 30 deep).
  sb/core.py is imported read-only.
- 32-row checks: 31 drafts plus the bonus.
  - Drafts are made lazily, one per committed token. A draft at depth k reads only tokens already committed, so the
    committed text equals block drafting's.
  - The target runs one pass per committed token (batch 1, mlx_lm's cache), so the delta-0 arm is the 2B's own greedy.
- **Rule** = relax_delta.relaxed_k, the engine's code in ccx45cc1:
  - a draft is kept while it equals its row's argmax, or logit(d) - max logit >= -delta;
  - every row may relax;
  - the first row not kept commits its argmax, and after 31 kept drafts the bonus row's argmax is committed.
- **Arms:** delta 0 (exact), 1 and 2 nats.
- **Panel:** EvalSuite's code panel, 264 keys in tools/lossy_panel.json's `code` order.
  - HumanEval+ 164 (humaneval 60 + humanevalx 104; task ids disjoint, checked) and MBPP 100.
  - Prompts: the items' `user` text in the 2B's chat template, thinking off (as Sandbox98's capture, sb/gen.chat_ids).
  - Caps: the items' max_tokens (768 / 1024). An answer stops at an end token.
- **Scoring:** EvalSuite's own scorers, imported read-only (`python -I -B`, nothing written there):
  - HE+: evalsuite/decode_report.score_humaneval (evalplus base + plus tests; humanevalx scored as HumanEval+);
  - MBPP: tools/mbpp_score.py (the 3 asserts, an isolated child, 10 s).

**Lines** (per delta, against the exact arm, paired by key):
- **HOLDS:** the pooled (264) drop is <= 1.0 pt AND the exact McNemar p > 0.2, AND neither set loses with p < 0.05.
- **FAILS:** the arm is lower with p < 0.05 pooled, OR the pooled drop is > 3 pt (EvalSuite's code screening line).
- **BETWEEN:** otherwise.

**Reported, with no line:**
- Changed tokens (R1-only commits) per 1K generated tokens, by where they land: code / comment / docstring / other
  string / prose outside the fence.
  - The class is the span tracker's state at the token's first non-blank character. Whitespace tokens in code count as
    code.
- Answers whose text differs from exact, the length ratio, and answers cut at the cap.
- Real kept per 32-row check and its lift over exact, beside Breakthrough's teacher-forced lifts on the 125B.
- The discordant keys, with the class of the first changed token.

**Dose-matched reading** (the read-across to the 125B, because the 2B is less sure):
- The 125B's teacher-forced code-token exposure is HE+ 4.1 / 5.8 and MBPP 8.1 / 12.8 per 1K kept at delta 1 / 2.
- The 2B arm whose code exposure lands nearest those is read as that 125B delta's proxy.
- If the 2B's delta 1 already exceeds the 125B's delta-2 exposure, a 2B delta 1 that HOLDS supports delta 2 on the 125B.

**Self-checks** (otherwise the loop is wrong):
- (a) The delta-0 arm's kept/32 lands within +-15 % of s30's teacher-forced chain read-out on these sets (Sandbox98
  gate.json: MBPP 4.42, HE 6.43 on its 60 keys). The HE keys differ, and real vs teacher-forced numerics differ.
- (b) On 3 MBPP + 3 HE answers of the delta-0 arm, the loop's drafts equal the teacher-forced chain's (C.chain on the
  full-forward W of the same answer) at >= 99 % of drafted positions. Differences can only be bf16 / fp32 near-ties.

**Caveats, stated before the run:**
- The 2B is a harsh proxy. It is less sure than the 125B, so R1 fires more often per 1K. A HOLD here is strong news for
  the 125B, and a FAIL is weak evidence against it.
- Each arm is one deterministic greedy draw.
- Power: with ~10-15 % discordant keys, a true -3 pt loss reaches p < 0.05 only part of the time. The line is a screen;
  EvalSuite's gate decides.

**Budget:**
- One process per arm. A capped smoke run measures the peak first, and the share lock is sized to it.
- Total <= 5 GB while Sandbox98's 6 GB QAT job runs, a NO-SWAP kill (swap +1 GB over the start), and the MLX cache is
  cleared between keys.
- ~70K tokens per arm, estimated at 20-40 min.

**Smoke (09:1x):**
- Self-check (b) PASS: 3,431 / 3,464 = 99.05 % of drafts equal the teacher-forced chain's.
- Peak footprint 2.1-2.9 GB per process, ~39-52 tok/s, ~435 tokens per key (~115K per arm).
- So the run is ~50 min per arm. Exact and delta 2 started 09:14; delta 1 waits for Sandbox98's 6 GB job to end.

**Amendment 1 (09:2x, before any arm finished; seen so far: the first 9 keys of delta 2 vs exact): a numerics control
arm.**
- **Why:** a diagnostic found that the batch-1 exact arm differs from Sandbox98's own capture of the same 2B greedy
  (mlx_lm BatchGenerator, same prompts) in 8 of 9 answers. The first difference falls 5-262 tokens in: near-tie
  numerics alone move the 2B's greedy.
- **The arm:** exact greedy of the same 2B through BatchGenerator (Sandbox98's sb/gen.greedy_batch, batch 8 per set),
  with the same prompts, chat template and caps.
- **Use:** its pass@1 delta and discordance against the batch-1 exact arm are the 2B's paired noise floor under a
  numerics-only change. They are reported beside each delta. The registered lines don't change.

**J(CC), RESULT (10:1x; week98/jcc/jcc_result.json, score_out.txt): delta 1 HOLDS; delta 2 FAILS, on HE+.**

| arm vs exact (264 keys, paired) | HE+ (164) | MBPP (100) | pooled | arm-only / exact-only | McNemar p | line |
|---|---|---|---|---|---|---|
| exact (batch 1) | 41.46 | 29.00 | 36.74 | - | - | - |
| **delta 1** | 42.07 (+0.61, p 1.0) | 30.00 (+1.0, p 1.0) | 37.50 (**+0.76**) | 23 / 21 | 0.88 | **HOLDS** |
| **delta 2** | 32.32 (**-9.15**, 11 / 26, p 0.020) | 28.00 (-1.0, p 1.0) | 30.68 (**-6.06**) | 22 / 38 | 0.052 | **FAILS** (drop > 3 pt; HE+ lower at p < 0.05) |
| control B (exact, batched numerics) | 40.24 (-1.22) | 33.00 (+4.0) | 37.50 (+0.76) | 18 / 16 | 0.86 | (noise floor) |

**Behaviour** (R1-only commits = committed non-top-1 drafts; per 1K generated tokens):

| arm | R1 per 1K | code | comment | docstring | string | kept per check (lift) | lift without capped answers | length ratio | cap hits (exact 26) | answers changed |
|---|---|---|---|---|---|---|---|---|---|---|
| delta 1, HE+ | 27.3 | 7.5 | 15.0 | 4.5 | 0.3 | 7.46 (+19.9 %) | +9.1 % | 1.07 | 21 (14) | 141 / 164 |
| delta 1, MBPP | 42.5 | 7.4 | 11.0 | 22.8 | 1.2 | 6.27 (+20.9 %) | +19.2 % | 0.95 | 10 (12) | 100 / 100 |
| delta 2, HE+ | 41.4 | 12.5 | 21.6 | 6.4 | 0.8 | 9.88 (+59.0 %) | +23.7 % | 1.16 | 33 (14) | 151 / 164 |
| delta 2, MBPP | 63.3 | 12.9 | 18.5 | 30.3 | 1.7 | 9.56 (+84.3 %) | +45.4 % | 1.13 | 17 (12) | 100 / 100 |
| control B | 0 | - | - | - | - | - | - | 1.00 | 28 | 207 / 264 |

- Prose outside the fence takes ~0 R1 commits.
- The 125B's teacher-forced kept/32 lifts (Breakthrough): delta 1 HE+ +21.8 / MBPP +23.1 %; delta 2 +32.3 / +46.5 %.

**Self-checks:**
- (b) PASS: 99.05 % (smoke).
- Divergence check PASS: in all 264 keys of each delta arm, the first token that differs from exact is the arm's first
  R1 commit.
- (a): HE PASS (6.22 vs the read-out's 6.43, ratio 0.97). MBPP **FAILS its pre-registered +-15 %** (5.19 vs 4.42,
  ratio 1.17).
  - Like for like (the read-out's own window: checks ending by token 512, the capture's cap), MBPP is 4.53 vs 4.42.
  - So the excess comes from answers past 512 tokens (loops, which are easy to draft), not from the loop logic.
  - Answer identity with Sandbox98's capture (batched greedy) in the first 512 tokens: HE 18 / 60, MBPP 8 / 100. This
    is the numerics sensitivity that amendment 1 controls for.

**Reading:**
- **Delta 1 is inside the 2B's noise.** It has +0.76 pooled with 44 discordant keys; the numerics-only control has
  +0.76 with 34.
  - Its real kept-per-check lift (+21 %) equals the 125B's teacher-forced lift (+22-23 %). On HE+ about half of it
    comes from answers that run to the cap (+9 % without them).
- **Delta 2 fails on HE+ by a mechanism, not by noise:**
  - 23 HE+ answers newly run to the 768 cap (4 stop running to it), and 8 of the 26 losses end at the cap.
  - The first R1 commit in 15 of the 26 losses is a docstring token: the 2B rewrites the docstring, and the code
    follows the rewrite (J(A)'s steering).
  - MBPP holds at delta 2 (-1.0, p 1.0).
- **Dose-matched to the 125B** (code-token R1 per 1K): the 125B's teacher-forced exposure is HE+ 4.1 / 5.8 and MBPP
  8.1 / 12.8 at delta 1 / 2.
  - On HE+, the 2B's delta 1 (code 7.5, all 27.3) already exceeds the 125B's delta 2 (5.8, 22.3), and it HOLDS. By the
    pre-registered reading, that supports delta 2 on the 125B for HE+.
  - On MBPP, the 2B's delta 2 (12.9, 63.3) matches the 125B's delta 2 (12.8, 61.9), and MBPP holds there.
  - The 2B's HE+ failure sits at ~2x the 125B's delta-2 code dose.
  - The 125B's exposures are teacher-forced and the 2B's are real decoding, so the match is approximate.
- **So, for Monday:**
  - A delta-2 arm on the 125B is plausible, not established. The 2B can't stand in for it at matched dose on both
    sets with one delta.
  - The failure mode to gate is runaway length. Watch cap hits and loops (decode_report's truncation and loop rows)
    beside pass@1.
  - Docstring rewrites are the main lossy entry point. A docstring / comment guard would cut it.
- **Caveats:**
  - One deterministic draw per arm on a weak model (pass@1 HE+ 41 / MBPP 29).
  - The 2B's exact answers already change in 78 % of keys under numerics alone.
  - The "without capped answers" lifts and the loss mechanism are exploratory, not pre-registered.
- **Files:** week98/jcc_decode.py, jcc_batch.py, jcc_score.py, jcc_prescore.py, jcc_guard.sh;
  jcc/arm{0,1,2,B}.jsonl, scores.json, jcc_result.json.

## Ranked (final, 01:3x)
1. **The 2B audits and repairs the head's deep tokens: CONFIRMED on HE+** (D). On MBPP, the frozen schedule was chosen
   on those rows (C). Training-free.
   - The drafter goes flat at the auditor's own agreement: HE+ 89.65 -> 94.32 and MBPP 72.13 -> 85.14 at 23-30, with
     depth 0 kept.
   - Cost: 1.06 GB per pass, GPU-serial (the ANE can't hide it at the floor; see the pricing below).
2. **Lever 1c: an auditor near ~98 % agreement with the 125B.** Distil the 2B (Sandbox98 owns the screen;
   week98/lever1c_distill.py is ready), and a ~0.4 GB 0.8B for the bytes budget (needs a download with Kamen's OK).
   This is what turns rank 1 into the 98 % method.
3. **kNN memory of true states** (lever 3). Untested, training-free, ~free per token.
4. **The code-structure tracker** (lever 4), diagnostic first.
5. **Per-answer online adaptation of the head** (lever 6).
6. **Keyframe refresh** (lever 2). Sandbox98's k-sweep; it needs a near-oracle STATE source, and result A ruled out the
   2B's state.
7. **The 2B as a STATE composer, fine-tuned or fused** (lever 1b). The REPLACE form was killed tonight (A).
8. **CPU / ANE concurrency** (lever 5). Priced: it can't hide the audit at the floor, because there's no block to audit
   in the check's shadow.
9-12. **Low or in the literature:** prefix KV (KVShot), recurrent memory (ROLLOUT, DEPTHN), block drafting (NewIdeas'
   list), post-norm (not ours).

## Lever 1 (v1): the 2B as a flat co-drafter: audit and repair
**Mechanism.**
- The head drafts the block (cheap).
- The 2B runs ONE block pass over the drafted tokens, teacher-forced. At each row it predicts the next token, exactly
  what it would on a correct chain.
- At the first row where the 2B disagrees with the head, and the 2B is confident (its margin >= tau, perhaps also the
  head's prob < tau_h), its token replaces the head's.
- The head re-drafts the rest of the block from there, and the 2B re-audits the new rows. Repeat up to R times
  (R ~ 2).

Expected repairs per 30-token block: about the head's expected errors on a correct chain (HE+: ~1-2), so 2-3 block
passes per check.

**Prior art:** Staged Speculative Decoding (Spector & Re 2023) and Cascade Speculative Drafting (Chen et al. 2023): a
mid-size model checks or extends a small drafter. **Published mechanism.** Ours: a same-family 2B with the target's
exact tokenizer is flat with depth and beats our trained head at depth 23-30 on both sets, measured tonight.

**Cost in bytes per 32-token block** (the orchestrator, 01:1x: the check is bandwidth-bound, and concurrent reads add in
full, even on the ANE; the WEEK98 drafter spec is <= ~0.5 GB per block for 400 exact). Measured from the 2B's
safetensors headers (week98/bytes_2b.py):
- A 2B text pass reads 1.059 GB: trunk 0.773 GB + the tied embedding as its lm_head 0.286 GB. Vision / MTP weights
  (0.663 GB) aren't read.
- With P audit passes per check: P = 1 -> 1.06 GB per block (2.1x the spec); P = 2 -> 2.12 GB (4.2x).
- Trunk-only (re-ranking the head's own top candidates, reading only their embedding rows): 0.77 GB per pass (1.5x).
- A smaller auditor of the same family fits: Qwen3.5-0.8B exists (mlx-community/Qwen3.5-0.8B-MLX-4bit, ~0.6 GB on disk,
  Apache-2.0), probably ~0.35-0.45 GB per text pass. Its agreement needs a download, i.e. the orchestrator and Kamen's OK.
- In time: 1.06 GB at ~300 GB/s is ~3.5 ms per pass, ~7 % of a 49 ms exact check. So P = 1 with the 2B pays if mean
  accepted per block rises >= ~7 %.
- CPU / ANE concurrency (lever 5) does not hide bytes. It only hides compute, which isn't the limit here.
- Engineering note: the 2B is a Gated-DeltaNet hybrid, so rolling its state back to the accepted length after a check
  needs per-row state checkpoints during the audit pass, or a re-run from the last accepted row (as convcache does for
  the 125B).

**Screen (CPU, ~10 min, needs t30b's per-row chain predictions on the bank: BigMoE's harness):**
- For each (anchor s, depth j) on MBPP 100 and HE+, take the head's top-1 and prob, and the 2B's top-1 and margin at
  the row whose next token the head predicts (rows2b_*.npz).
- Combined pick: the 2B's token where (2B top-1 != head top-1 and margin2b >= tau), else the head's.
- Sweep tau in {1, 2, 3, 4} nats. Report the per-depth bands and kept/32 against t30b.
- **LINE:** PASS if, at the best tau, HE+ 23-30 >= 94 and MBPP 23-30 >= 82, with depth 0 within 0.3 pt of t30b:
  -> build the audit loop (BigMoE / Leap), and time the 2B block pass (k = 30, with lm_head) under a quiet lock.
  BETWEEN (+2 pt at 23-30): try a learned chooser (a logistic gate on depth, head prob, 2B margin).
  KILL if < +2 pt at 23-30.

### ANE pricing of the 2B audit (paper, orchestrator 01:2x; inputs: Leap ANE-1 data/ane_probe.txt, FableGod/ANE_DRAFTER.md)
**Bytes per audit pass**, from the measured headers (trunk 1.374B params, lm_head 0.509B):
- int8: 1.374 + 0.509 = 1.88 GB;
- int4 (4.5 bits with scales): 0.773 + 0.286 = 1.06 GB;
- int4 with a 16K-token lm_head shortlist: ~0.79 GB.

**ms per 32-row pass on the ANE**, at ANE-1's rates (body 32 GB/s, head 79 GB/s; weight-bound, as ANE-1's 16-row blocks
were):
- int8: 42.9 + 6.4 = ~49 ms;
- int4: 24.2 + 3.6 = ~28 ms (int4 streaming on the ANE is unmeasured);
- int4 with the shortlist: ~25 ms.
- Against a 52-55 ms exact check: one int4 pass fits in the shadow, int8 barely, two passes don't.
- The GPU slows ~x1.0017 per ANE call per second (ANE-1). One call per ~58 ms cycle (~17 / s) costs ~x1.03, ~1.6 ms.

**The pipelining bet decides it.** The audit needs the drafted block before it can run:
- t30b drafts block n+1 from the 125B's fresh state at the cut point (the accepted row's wd), and that exists only when
  check n ends. So block n+1 can't be drafted, let alone audited, in check n's shadow.
- (a) Draft n+1 speculatively from the head's carry at block n's end, betting on full acceptance.
  - That drafts at depths 32-62 from the last fresh state (stale), and the bet needs all 31 rows of block n accepted.
  - For t30b on MBPP that is a few % at most: the product of the per-depth rates, 94 -> 72. On the shipped 16-row loop
    the block end is the cut point on 47 % of code checks (ANE_DRAFTER).
  - The audit would help most there (the 2B is flat), but only on that small share of cycles.
- (b) An early-tap drafter (ANE_DRAFTER's variant iii) drafts its candidates during the check.
  - At the floor its ANE call (17-20 ms) already fills the window (taps <= 35: 14-16 ms; <= 25: 20-26 ms), leaving no
    room for a ~25 ms audit.
  - At today's long check (the <= 25 window is ~75 ms) both fit.
- (c) Serial on the ANE: +25-49 ms per cycle, worse than the GPU.
  - On the GPU, 1.06 GB at ~284 GB/s is ~3.7 ms, plus ~0.2 ms per layer of graph build (Leap's figure for the 125B),
    i.e. ~4-9 ms per pass until it is timed.

**Verdict:** at the exact floor, the Neural Engine can't hide the 2B audit behind the check of the block it audits, so
"~+5 % -> ~+12 % net" isn't reachable that way.
- The cost stays GPU-serial: 1.06 GB and ~4-9 ms per pass, ~7-16 % of a 52-55 ms check. For P = 1 on MBPP (+12 %
  accepted) that nets ~-4 % to +5 %. A ~2 min GPU timing settles it.
- It pays clearly only:
  - where the check is long (today's ~100-160 ms);
  - with a ~0.4 GB 0.8B auditor (~2-4 ms);
  - or with a drafter that drafts block n+1 before check n ends (early taps) at a non-floor check.
- Against the line (400 exact = a cycle <= ~58 ms at 23.3 kept): 52-55 ms + a 4-9 ms audit is 56-64 ms. So any
  auditor must also lift kept toward the 98 % regime, which is lever 1c again: an auditor that agrees ~98 % with the
  125B, at ~0.4 GB.

### Lever 1b: the 2B as a STATE composer, fine-tuned or fused (REPLACE killed tonight)
Two training variants stay open. Both are judged by the drafting eval, not the proxy:
- **Fine-tuned composer:** LoRA on the 2B's top layers plus a linear head, regressing the 125B's wd on n1112's captures.
  A 15-min screen can train on the bank's 360 train files and re-score with the proxy.
  - LINE: PASS if direction error <= 0.40 (predicted >= 88).
- **Fusion:** the head reads [its carry, mapped 2B state] through a zero-init gate.

### Lever 1c: raise the auditor's flat level: distil the 2B (or the 0.8B) toward the 125B
**Why.** Result C shows the audit makes the drafter flat at the auditor's own agreement with the 125B. 98 % deep needs an
auditor near the fresh-state oracle's level (HE+ 97.8, MBPP 93.6). Today it is at 94.4 / 85.7, untrained.

**Mechanism.** Token-level distillation:
- LoRA on the 2B's linear layers (or full tuning of its top layers), with a KL loss to the 125B's top-32 log-probs.
- Data: the bank's train files, or n1112's captures (every answer row carries the 125B's tk / lp; ~600K rows).
- The 2B keeps its pretrained composition, which a fresh head can't learn (DEPTHN), and learns the 125B's choices.

**Prior art:** DistillSpec (Zhou et al. 2023): distilling the draft model to the target raises acceptance. Online
Speculative Decoding does the same online. **Published**; our use is a flat auditor beside a feature head.

**Cost:** the same bytes per pass as the auditor (LoRA merged). Training: minutes to an hour on this Mac.

**Screen (MTP Head or NewIdeas, ~15 min of training):**
- LoRA rank 32 on the 2B, KL to the 125B's top-32, on the bank's 360 train files.
- Then re-run week98/lever7_agree.py (MBPP and HE+, held out) and week98/join_mbpp.py with the new picks.
- **LINE:** PASS if MBPP agreement >= 89.0 % (+3.3) and HE+ isn't lower -> the join's 23-30 and accepted, then a longer
  run on n1112; KILL if < +1 pt (the 2B can't move toward the 125B in this budget).
- The same screen on the 0.8B decides whether a within-budget auditor (<= 0.5 GB per block) gets high enough. That needs
  a download: mlx-community/Qwen3.5-0.8B-MLX-4bit, ~0.6 GB, Kamen's OK.

## What the levers have to supply (from WEEK98's log; numbers as recorded there)
- The head reads a fresh state well: given the 125B's true state at every drafted row, t30b holds 97.76 % at HE+ 23-30,
  flat (bank, runtime path). On its own carried state it gets 89.65 (HE+), and 72.13 vs 93.63 on MBPP 23-30.
- T2 (Sandbox98, results/T2_TRUNCATION.md): the 2B run over only the drafted tokens beats the head's carried state by
  +14.2 pt at MBPP 23-30 (69.62 vs 55.42), and with 64 / 256 tokens of prior context it reaches 80.61 / 86.25 (the oracle
  87.71). So the gap is deep pretrained composition over a local window, not far retrieval.
- Ruled out: context-free steppers (ROLLOUT), lookups / copy (at most +2.6 pt), few directions (T3), a zero-start
  extra layer, width, and fresh-state adapters in short budgets. Early-tap maps are worse than the chain (LAYERS).
- The head tolerates unbiased noise far better than map error at the same R2 (Sandbox98 gate: true W + noise at R2 ~0.51
  gives 82.66; a ridge map at R2 0.44 gives 59.19).

Two consequences drive the ranking:
1. The composition has to come from something already pretrained. Nothing we train from zero in a short budget learns it.
2. A state computed from the actual drafted tokens doesn't drift. Its error is the same at depth 30 as at depth 1. Only
   the head's carried state compounds.

## The v0 top 3, as posted at 00:53 (lever 1 below is result A's REPLACE screen, now killed)

### 1. A pretrained small model of the same family as the composer ("borrowed depth")
**Mechanism.** Qwen3.5-2B and Qwen3.5-4B are on disk (mlx-community 4-bit). They share the 125B's tokenizer exactly:
the same 248,077 token ids (checked 00:4x, MoonShots). The small model runs over the verified prefix and the drafted
tokens with its own KV cache. Its hidden state at the last drafted row, mapped into the head's input space (wd) by a
ridge or a small MLP, replaces or joins the carried stand-in. Its 24 / 32 pretrained layers supply the deep
composition that T2 shows the head lacks. It reads the actual tokens, so its error doesn't grow with draft depth. On
MBPP, only the tap-45 ridge (91.61, R2 0.82) has been this flat so far, and it needs the 125B's own trunk.

**Prior art (quick search):** small draft models that draft tokens themselves are classic (Leviathan 2023, Chen 2023).
Drafters fed the target's own features: EAGLE, FastEagle (2509.20416), DFlash, SPD (2605.30852). No paper found that
maps a separate pretrained small model's hidden states into the target's drafter input. **Untested.**

**Cost per drafted token.** Measure it: the composer's pass without its lm_head (that is ~1.2 ms of the 2B's 2.54 ms
head step in results/timedraft.json, and the composer doesn't need it). Estimate: a 4-bit 2B pass reads ~1.1 GB, so
~4-6 ms per pass at any k <= ~32 rows (weights read once), and ~2-3 ms using only its first 12 layers. Per token:
- every token: 4-6 ms, too much (the check is 50-100 ms per 30 tokens);
- with lever 2's keyframes every 8 tokens: (30 / 8 + 1 catch-up) x pass ~= 5 passes, 15-30 ms per check, i.e. 0.5-1 ms
  per drafted token;
- run on the ANE or CPU while the GPU verifies (Mirror-SD, Apple, 2510.13161): it could hide most of that.
Break-even: with a 60 ms check and 7 ms of drafting, +20 ms of composer needs HE+ kept/32 >= ~17.9 (today 13.8;
the fresh-state oracle 20.9).

**Screen (on the 125B's BANK, no 125B run; BigMoE or whoever holds the BANK, ~15 min):**
- Run the 2B (and then the 4B) over the BANK's proxy files' token ids: prompt + answer, one batched forward per file.
  Keep the hidden state after layers {6, 12, 18, 24} (2B) at every row.
- Fit a ridge, composer state -> wd, on the proxy's train rows (31,805). Score the val rows (9,984) with bank_proxy.py's
  direction error and its predicted MBPP 23-30.
- Time one composer pass (k = 1, 8, 32 rows; layers 12 / 24; no lm_head) on the GPU.
- **LINE:** PASS if the best layer's predicted MBPP 23-30 is >= 85 (chain 72.1; tap-45 ridge 91.6):
  -> REPLACE eval with t30b at every drafted row (composer states from the true tokens), then lever 2 on the 125B, then
  the cost table above with the measured pass time.
  BETWEEN 78-85: try a 2-layer MLP map and the 4B before deciding.
  KILL if < 78: a different model's states don't carry the 125B's state.
- Sandbox note: the sandbox's target is the 2B itself, so it can't stand in for "smaller composer, bigger target".
  This screen belongs on the 125B's bank.

### 2. Keyframe refresh: fresh state every k tokens, the head carries in between ("multirate drafting")
**Mechanism.** From numerical integration (multirate methods) and video coding (an expensive I-frame every k frames,
cheap predicted frames between). A costly state source runs only every k drafted tokens:
- after the head drafts k tokens, they are all known, so one block pass of the composer covers them at about the cost
  of one row;
- the head's chain then restarts from the refreshed stand-in;
- the head carries at most k-1 steps, where its accuracy is still high: the 2B chain's band 3-6 is 68 and band 1-2 is
  78, against 55 at 23-30.
This turns lever 1's per-token cost into one pass per k tokens. It also tells the 125B how often a fresh state is
needed.

**Prior art (quick search):** no drafter found that refreshes its state mid-draft from a cheaper model. SPD
(2605.30852) bounds drift with the target's own pipeline stages, and our Leap pricing closed refresh from the 125B's
trunk (a refresh at L45 is a verify). KVShot (2604.26412) found long-range decay persists even with TTT. **Untested.**

**Cost per drafted token:** one composer block pass per k tokens: ~0.5-0.7 ms per token at k = 8 with a 2B composer,
~0.3 ms with 12 layers.

**Screen (Sandbox98, eval-only, s30 unchanged, ~15 min):** reuse sb/trunc.py's batched K-window forward (T2's code),
with K = 64 only. Replace the stand-in only at depths j = k, 2k, 3k ... with the K = 64 state at row s + j - 1, and let
the head's chain carry in between from the replaced stand-in. Arms: k = 4, 8, 16, plus k = 1 (T2's every-row number,
80.61) and the chain (55.42), on the same 50 MBPP answers, every 3rd anchor (3,857 anchors). A second set uses the true
state (full prefix) as the refresh source: the k-curve with a perfect composer.
- **LINE:** PASS if k = 8 keeps >= 70 % of the every-row gain at 23-30 (>= 55.42 + 0.7 x 25.19 = 73.1):
  -> lever 1 is costed at one pass per 8 tokens; repeat on the 125B with lever 1's composer.
  If only k = 4 passes, cost lever 1 at 1 / 4.
  KILL if k = 4 < 65: the composer must run every token, and lever 1 then needs the ANE or CPU route.

### 3. Retrieved true states: a memory of real 125B states keyed by the recent tokens ("kNN state memory")
**Mechanism.** From kNN-LM / example-based learning, but storing STATES, not next tokens. Index every captured
training row's true state (wd for the 125B, W for the 2B) by a hash of its last n tokens, for n = 16, 12, 8, 6, 4
(longest match first). At a drafted row, if the last n drafted tokens match, the stand-in becomes a retrieved true
state; otherwise the carried one stays.
- Why it might work where lookups failed: T1 counted copyable TOKENS. A retrieved state carries the 125B's deep
  composition of a similar local context, and the head still gets this context's specifics (identifiers) through its
  attention over the true prefix.
- A retrieved state is a real state from the model's own manifold, so its error should behave like noise, which the head
  tolerates (82.66 at R2 ~0.51), not like a ridge map's bias (59.19 at R2 0.44). The screen tests that directly: one
  retrieved neighbour (a sample) against the mean of several (a blur).

**Prior art (quick search):** NEST (2405.19325) keys a datastore by hidden states and retrieves next tokens. REST
(2311.08252) matches token suffixes and retrieves continuations; DReSD (2502.15572) and SENSE (2606.00021) do dense
token-level retrieval. No paper found that retrieves states as the drafter's input. **Untested.**

**Cost per drafted token:** a suffix-hash lookup on the CPU (microseconds) and reading 1-8 stored states (20 KB each
for the 125B's wd at fp16, ~2 KB at PCA-1024 int8): ~0.02-0.1 ms. Close to free. Storage: n1112's ~600K answer rows at
PCA-1024 fp16 ~1.2 GB (T3: 1,024 PCs keep 86.9 of the full 93.6 on MBPP 23-30).

**Screen (Sandbox98, eval-only lever `levers/knnstate.py`, `./screen.sh knnstate knnstate "LINE" 0`, ~15 min):**
- `setup`: build the index from the Magicoder captures (data/mc, data/mc2; true W + ids). No MBPP / HumanEval rows: they
  are the held-out sets.
- `standin(j, h, anc)`: at row anc + j - 1, longest suffix match with n >= 6 over the drafted tokens (= the true tokens
  on the scored chain) -> the retrieved W, else h.
- Arms: (a) one neighbour, the most recent; (b) the mean of up to 8; (c) a match only when n >= 12.
- Also log the hit rate per band, and accuracy on hit rows against the chain on the same rows.
- **LINE:** PASS if MBPP-50 23-30 >= +2 pt (p < 0.05), or if hit-row accuracy is >= chain + 10 pt at a >= 25 % hit
  rate:
  -> scale the memory to n1112 + bank on the 125B (PCA-1024), REPLACE eval with t30b.
  If (a) beats (b) by >= 3 pt, the "sample, don't average" reading holds, and every learned stand-in should be judged
  on that basis.
  KILL if hit-row accuracy <= the chain's: retrieved states from other files mislead the head.

## 4. Exact code-structure state: an indent / bracket / scope tracker as input that never drifts
**Mechanism.** From compilers: an incremental lexer over the verified prefix and the drafted tokens (detokenized as they
come) yields, at every drafted row, the exact syntactic position:
- indent level, the bracket stack (depth and top type), and in-string / in-comment flags;
- the enclosing statement keyword (def / for / if / return / call) and whether the row starts a line.
Small zero-init embedding tables add these to the stand-in. They are exact at every depth, so they can't drift.
- Why it might help: CEILING found 75.8 % of the 23-30 misses on rows the 125B was sure of (m >= 1 nat). On very sure
  rows (m >= 4) the first guess misses 0.1 % and the deep guesses 16.1 %. That is "losing the thread", and the
  syntactic part of the thread is free to compute exactly.

**Prior art (quick search):** DOMINO (2403.06988) predicts tokens from parser state by counts. Speculative decoding
for Verilog (2503.14153) uses AST boundaries to train Medusa heads. LinguaSpec (ACL Findings 2026) adds linguistic
priors. None feeds an exact structure state into a deep drafter's state to stop depth drift. **Adapted / untested.**

**Cost:** an incremental lexer step on the CPU, ~2-10 us per token, plus one table lookup. Code only: chat and agent
text would need a different tracker (lists, quotes, markdown).

**Screens:**
- (a) **Diagnostic, 2 min, no training:** on MBPP 23-30, class the FIXED misses (chain misses, oracle hits) by token:
  newline / indent, bracket / colon / comma / operator, keyword, identifier, literal. Any per-row eval dump with
  chain-hit and oracle-hit flags will do.
  - LINE: if >= 40 % are structural (newline / indent / bracket / punctuation) -> run (b); else the tracker can buy at
    most that share, so park it.
- (b) **Training screen**, Sandbox98 fast format (`./screen_fast.sh struct "structemb" "LINE" STEPS 12`, the calibrated
  STEPS):
  - `levers/structemb.py` adds E_indent[min(level, 15)] + E_br[depth, top] + E_str[flag] (zero-init, lr x 10) to the
    stand-in at every depth;
  - features from Python's `tokenize` run on the decoded text up to each row.
  - LINE: PASS if MBPP-50 7-14 >= +1.5 pt (p < 0.05) -> the depth-30 long screen; else KILL.

## 5. Run the composer beside the check: CPU or ANE while the GPU verifies (cost only, for levers 1-2)
**Mechanism.** Pipelining across compute units:
- block N+1's composer passes run while the GPU verifies block N, on the CPU (MLX's CPU stream or Accelerate) or on the
  ANE (Core ML);
- they read the drafted tokens, which exist before the check starts; on a rejection they are discarded, as the drafts
  are;
- with a 98 %-class drafter, rejections early in a block are rare.
The check is memory-bound (exact floor 49.1 ms at 45 GB), so the open question is bandwidth contention, not compute.

**Prior art:** Mirror Speculative Decoding (Apple, 2510.13161; NPU draft beside a GPU target, 2.8-5.8x reported).
**Published** for drafting; using it for a state composer is the adaptation.

**Cost:** ~0 visible per token if contention stays low; otherwise the check slows by the contention.

**Screen (Leap or BigMoE, under a quiet lock because it is a timing, ~15 min):** time the exact-45 16-row check loop
alone, then with the 2B composer looping block passes (k = 8, no lm_head) on (a) the MLX CPU stream and (b) the ANE if a
Core ML conversion exists (else skip (b)).
- LINE: PASS if the check slows <= 5 % and a CPU composer pass is <= check time / (30 / k + 1) -> levers 1 + 2 cost ~0
  per token; else cost them serially (lever 1's table).

## 6. Per-answer online adaptation of the head ("dynamic evaluation")
**Mechanism.** From language modelling (Krause et al. 2018, dynamic evaluation; test-time training for LMs):
- after each check, the verified prefix's true states and tokens are known;
- a few SGD steps on the head (or a low-rank adapter) with the chain loss, on anchors inside this answer's verified
  part, fit it to this file's identifiers, style and repeated structures before the next draft.

**Prior art:** Online Speculative Decoding (2310.07177) distils the drafter online from the target's corrections, across
requests; TIDE and OnlineSPEC adapt the drafter over a deployment. Adapting within one answer, against deep drift:
**adapted / untested**.

**Cost:** a few backward passes of a one-layer head over 32-256 rows per check, ~2-6 ms per check (0.1-0.2 ms per
drafted token), or asynchronous on the CPU.

**Expected:** small to moderate. T1 says copy-like information is worth at most ~2.6 pt. It helps most on long answers
(HE+ windows), least on short MBPP ones.

**Screen (Sandbox98, ~15 min):** for each MBPP-50 / HE-30 answer, copy s30 and adapt it on anchors in the answer's
first half (10 steps, lr 1e-5, Adam reset). Read out 23-30 on the second half's anchors against unadapted s30 on the
same anchors.
- LINE: PASS if >= +1.5 pt at 23-30 on the second halves (p < 0.05) -> the 125B's HE+ windows; else KILL.

## 7. The composer's own logits audit the draft (a mid-tier check; with lever 1)
**Mechanism.** Lever 1's block pass over k drafted tokens also gives the 2B's next-token prediction at each of those
rows, for ~1 ms of its lm_head. Where the 2B confidently disagrees with a drafted token (its top-1 differs, with a margin
>= m), cut the draft there (a shorter check), or take the 2B's token and re-draft from it. Errors die before they cost a
125B check.

**Prior art:** Staged Speculative Decoding (Spector & Re 2023) and Cascade Speculative Drafting (Chen et al. 2023).
**Published**; here it comes almost free with lever 1.

**Screen (CPU on the BANK, ~10 min):** the 2B's teacher-forced top-1 against the 125B's greedy token (the bank's tk[:, 0])
on the val answer rows: agreement overall, and on the rows where t30b's chain misses at 23-30 (needs BigMoE's per-row
chain hits).
- LINE: PASS if the 2B is right on >= 50 % of the chain's deep misses while agreeing with the 125B on >= 95 % of the
  chain's hits -> price the audit rule; else KILL.

## 8. The target's prefix KV for the head (GLIDE / KVShot): low
**Mechanism.** The head cross-attends to the 125B's own K / V of the verified prefix at a deep full-attention layer, so
it reads retrieval-ready keys instead of its own.

**Prior art:** GLIDE with a CaPE (2402.02082). KVShot (2604.26412, Qwen3-8B) found that KV reuse improves long-range
acceptance, but end-to-end gains are marginal: shallow drafters estimate the target's queries poorly. **Published.**

**Why low:** T2 says far retrieval isn't the gap (K = 256 is within 1.5 pt of the full prefix), and T1 caps lookups at
+2.6 pt. Also, the 125B has KV only in its few full-attention layers; its Gated-DeltaNet layers keep recurrent states.

**Screen (only if levers 1-4 all fail):** a zero-init cross-attention to the target's deep-layer prefix KV,
Sandbox98 fast format.
- LINE: PASS if >= +1.5 pt at 7-14.

## 9. Recurrent / state-space memory in the drafter: low
**Mechanism.** Give the head a recurrent (Mamba / Gated-DeltaNet) layer over the drafted tokens, so its carry is a
learned recurrent state.

**Prior art:** Mamba Drafters (2506.01206), about drafting speed and memory, not depth drift. **Published.**

**Why low:** ROLLOUT showed context-free steppers fail, and a zero-start extra layer of any kind did nothing in short
budgets (DEPTHN). A pretrained recurrent stack does the job, and that is lever 1: Qwen3.5-2B is itself mostly
Gated-DeltaNet. No separate screen.

## 10. Block / hierarchical drafting: in the literature (NewIdeas' list)
Fewer sequential carries per block: block diffusion, semi-autoregressive and dependent-block drafters (DFlash / DFlash2
in our stack; DSpark 2607.05147, SpecBlock 2605.07243, ParallelSpec 2410.05589, DBLAST 2608.05448, Bastion 2605.29727).
The outside variant is a MegaByte-style split: one global state update per patch of P tokens, plus a small local
decoder. It cuts the number of carried updates by P, but needs a new architecture, so no 15-min screen. Left to
NewIdeas' prior-art pass.

## 11. Post-norm between chain steps (Attention Drift, 2605.09992; EAGLE 3.1): checked on paper, not ours
That paper's failure is an unnormalized residual carry whose RMS grows with depth. Our heads re-normalize the carried
stand-in at every step before fc: Sandbox98 sb/core.py:174 `hn = D.pre_fc_norm_hidden(Win)`, the native MTP layout the
125B's head shares. So that failure mode doesn't apply, and no screen is needed.

## The combined design (v1, after tonight's results)
- **The head drafts the block** from the check's true anchor state, as today.
- **One auditor pass per check (P = 1)**, at ~1.06 GB for the 2B, or ~0.4 GB for a 0.8B within the spec. It runs over
  the accepted tail of block N plus block N+1's drafted tokens: the catch-up and the audit in one pass.
  - Where the auditor disagrees with a depth-scheduled margin (result C), its token replaces the head's, and the head
    re-drafts from there.
  - A second pass (P = 2) adds another ~7 points of accepted on MBPP, at twice the bytes.
- **kNN true states (lever 3)** can replace the head's carry at rows with a long suffix hit: training-free, ~free.
- **The auditor gets distilled toward the 125B (lever 1c).** Its agreement is the ceiling of the deep bands.
- **Keyframes (lever 2)** need a near-oracle STATE source, and the 2B's state isn't one (result A). They stay useful
  only if a fine-tuned composer (1b) reaches direction error <= ~0.40.

Bytes per 32-token block decide the cost (the check is bandwidth-bound; concurrent reads add in full):

| design | bytes per block | MBPP mean accepted per anchor (head 7.38) | net at a 49 ms exact check |
|---|---|---|---|
| 2B, P = 1, depth schedule | 1.06 GB | 8.26 (+12 %) | ~+5 % |
| 2B, P = 2 | 2.12 GB | 8.78 (+19 %) | ~+4 % |
| 2B, every row audited (upper bound) | ~3+ GB | 9.34 (+27 %) | ~0 |
| 0.8B, P = 1 | ~0.4 GB (within spec) | to measure (download) | to measure |

## Honest ceiling (v1)
- The audit makes the drafter flat at the auditor's own agreement with the 125B: MBPP 85.7, HE+ 94.4 today. So 98 % at
  depth 30 needs an auditor at about the fresh-state oracle's level (HE+ 97.8, MBPP 93.6).
- On HE+ the 2B is already 3.4 pt from that; on MBPP, 7.9 pt. Distillation (1c) is the named way to close it. Whether a
  0.5 GB-class auditor can get there is the open question for the week.
- The only near-oracle STATE source is still the 125B's own trunk, which costs a verify. A different model's state,
  mapped linearly, carries model mismatch (result A: proxy 67.1). So the TOKEN route, a flat auditor, is the one that
  worked tonight.

## Sources (quick searches, Sun 15 Nov 00:4x-01:1x)
- Speculative Pipeline Decoding, https://arxiv.org/abs/2605.30852
- KVShot / When Hidden States Drift, https://arxiv.org/abs/2604.26412
- Attention Drift, https://arxiv.org/abs/2605.09992
- Mirror Speculative Decoding (Apple), https://arxiv.org/abs/2510.13161
- GLIDE with a CaPE, https://arxiv.org/abs/2402.02082
- Mamba Drafters, https://arxiv.org/abs/2506.01206
- NEST, https://arxiv.org/abs/2405.19325; REST, https://arxiv.org/abs/2311.08252; DReSD, https://arxiv.org/abs/2502.15572;
  SENSE, https://arxiv.org/abs/2606.00021
- DOMINO, https://arxiv.org/abs/2403.06988; Verilog SD, https://arxiv.org/abs/2503.14153
- Online Speculative Decoding, https://arxiv.org/abs/2310.07177
- FastEagle, https://arxiv.org/abs/2509.20416
- Block drafting: DSpark https://arxiv.org/abs/2607.05147, SpecBlock https://arxiv.org/abs/2605.07243, ParallelSpec
  https://arxiv.org/abs/2410.05589, DBLAST https://arxiv.org/abs/2608.05448, Bastion https://arxiv.org/abs/2605.29727
- DistillSpec, https://arxiv.org/abs/2310.08461; Staged Speculative Decoding, https://arxiv.org/abs/2308.04623;
  Cascade Speculative Drafting, https://arxiv.org/abs/2312.11462 (cited from memory, not searched tonight)
- Qwen3.5-0.8B (mlx-community 4-bit), https://huggingface.co/mlx-community/Qwen3.5-0.8B-MLX-4bit
