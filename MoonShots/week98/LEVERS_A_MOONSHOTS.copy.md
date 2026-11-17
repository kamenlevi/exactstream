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

**K, correction and full bands (Sun 22:0x; PLAN4D item 1, re-derived; week98/walkstrict_bands.py, walkstrict_bands.json;
TRUE-STATE (lab) numbers):**
- **Correction:** checkstart.py's walk-strict counted a no-target cell (the walk hitting an answer's end) as reached,
  i.e. as a miss. Excluding it (MTP Head's K98 definition, and NewIdeas' tree_audit.py) gives:
  - HE+ t30b d0 92.0 (was 91.58; 2 cells);
  - MBPP 89.7 (was 89.46; 6 cells).
  - kept/32 is unaffected, and the reading above stands.
- **Full bands under the decode walk (32-row checks).** This independently re-derives NewIdeas' tree_audit.py
  (PLAN4D_REAIM978.md, 8df555d) to the printed digit.

| HE+ 24, walk-strict | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 [LB] | 23-30 [LB] | kept/32 |
|---|---|---|---|---|---|---|---|---|
| t30b | 92.0 | 91.5 | 92.4 | 94.3 | 96.6 | 96.7 [95.9] | 97.7 [96.7] | 13.769 |
| 2B audit P = 1 | 92.7 | 90.6 | 92.8 | 95.2 | 96.6 | 97.6 [96.6] | 98.2 [97.4] | 14.456 |
| 2B audit P = 2 | 92.7 | 91.0 | 93.2 | 94.8 | 97.1 | **98.3** [97.5] | **98.3** [97.5] | 15.000 |
| 2B audit P = inf | 92.4 | 90.8 | 93.3 | 95.0 | 96.7 | 98.7 [98.0] | 99.4 [98.7] | 15.179 |
| 4B audit P = 1 | 93.0 | 92.1 | 94.0 | 95.1 | 96.7 | 97.5 [96.5] | 98.2 [97.3] | 15.399 |

- **MBPP 100** (same columns, no LB):
  - t30b 89.7 / 86.0 / 85.6 / 84.0 / 86.1 / 87.3 / 86.8 (6.924);
  - 2B P = 1 88.5 / 86.7 / 87.4 / 86.1 / 86.4 / 88.9 / 87.9 (7.427);
  - 2B P = 2 88.1 / 86.2 / 87.2 / 87.5 / 88.2 / 91.5 / 89.9 (7.671);
  - 2B P = inf 87.9 / 86.1 / 87.2 / 87.9 / 89.6 / 93.2 / 93.6 (7.911);
  - 4B P = 1 88.9 / 88.0 / 87.9 / 85.9 / 87.2 / 88.8 / 89.0 (7.713).
- LB = the window-clustered 95 % lower bound (2,000 window resamples, seed 1115).
- **Reading:**
  - Walk-strict deep bands are conditioned on reaching depth 15+ (only easy stretches survive 15 straight guesses), so
    even t30b alone sits at 96.7 / 97.7.
  - Against NewIdeas' re-aimed C2 line (15-22 and 23-30 point >= 97.8 AND LB >= 97.0):
    - 2B P = 2 passes on dev, 0.5 pt over 97.8 on the point and 0.5 over 97.0 on the LB, on both bands;
    - P = 1 and the 4B fail at 15-22.

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

## TREE-E2E. The exact tree check in real decoding on the 2B (orchestrator 14:4x; pre-registered 14:5x, before any code ran)

**Question:** does s30's best-first EAGLE-2 tree, verified by the 2B in real decoding, keep more tokens per check than
the 16-row chain, stay exact, and keep pass@1?
- Kamen chose the exact drafter (Sun morning), so trees are the exact lead. MTP Head's S9 priced them teacher-forced
  with an off-path proxy: an off-path node takes the true path's top-3 at its depth (TrainLab REPORT "WEEK98 S9
  TREE-DYN", sb98/tree_price.py eagle2()).
  - s30 exact, 16 rows: MBPP-50 4.280 -> 5.128 (x1.198 [1.176, 1.219]); HE-30 5.971 -> 6.802 (x1.139 [1.104, 1.175]).
  - S9c on the 125B (real t30b trees): x1.136 / x1.060.
- Nobody has decoded with a tree yet.

**Setup:**
- Target, drafter, panel, prompts and caps are J(CC)'s:
  - Qwen3.5-2B 4-bit and s30, with sb/core.py read-only;
  - 264 keys (HE+ 164, MBPP 100);
  - each item's max_tokens; an answer stops at an end token.
- **Rows:** 16 per check = the bonus row + 15 draft nodes, as the sim's chain-16 / tree-16 (N - 1 = 15 nodes). The
  ask's "16 nodes" are these 16 rows.
- **Arms:**
  - **C16:** s30's greedy chain of 15 drafts (depth 1, then 14 deeper steps).
  - **T16:** best-first EAGLE-2, the sim's eagle2() run on the head's real distributions:
    - a heap of nodes ordered by path score (the product of the head's softmax probabilities along the path), top-3
      children per node; pop until 15 nodes;
    - a popped node's children come from the head run on that node's own path: deeper with the node's token, attending
      to the true pairs <= the anchor plus the node's own ancestors' chain entries;
    - no proxy anywhere; slot 0 = the argmax on ties; heap ties broken by the rank sequence, as the sim.
  - **T32 (optional):** 31 nodes, beside the sim's x1.368 / x1.332. It runs only if the main arms are done by ~19:00,
    the GPU is free, and the batch rule below makes it comparable token for token.
- **Verify** (one code path for every arm; Leap's route (a)): each root-to-leaf path is its own chain in a batch of rows.
  1. The 2B's cache at the check start is forked to B rows (attention KV plus the GDN conv and recurrent states).
  2. The 2B steps all rows one token per forward ([B, 1]): offset 0 feeds the bonus; offset j feeds each path's j-th
     token.
  3. Accept the longest path prefix that matches the 2B's argmax at every offset. Commit those drafts plus the 2B's
     argmax after them (the new bonus), cut at an end token or the cap.
  4. The accepted row's cache becomes the new cache. Stepping stops at the offset where the last matching row ends, so
     nothing is rolled back.
- **Batch shape (rule fixed now; a smoke probe picks the branch):**
  - The probe compares a [B, 1] step's row 0 with a [1, 1] step from the same cache, bitwise (logits and the post-norm
    hidden), for B in {2, 4, 8, 15, 16, 31, 32}, at contexts of ~300, ~800 and ~1,250 tokens.
  - **If every row is bitwise equal:** B = each check's path count, with no padding. Every arm must then also equal
    J(CC)'s batch-1 exact arm (jcc/arm0.jsonl) token for token.
  - **Otherwise:** one fixed B for every target step of every arm, 16 rows (paths padded by repeating path 0). C16 and T16
    then share their numerics exactly. T32 is skipped. The arm0 comparison is reported as the batch-shape noise floor.
- **Why stepped, not one pass:** on the 2B with MLX's stock kernels, a one-pass [B, L] verify runs other kernels than the
  one-token decode (SDPA, the GDN chunk path). Its numerics then depend on where each check starts, so a tree and a chain
  could never match token for token, and near-tie flips would look like tree bugs.
  - Stepping makes each committed token's numerics independent of the tree and of the check boundaries.
  - This tests the tree logic, the per-path state fork and the commit. The one-pass bitwise kernel at 125B scale is Leap's
    CHECK55. Speed is not measured here: a check costs m + 1 target steps.

**Reported:**
1. kept/check per set: committed tokens per check = accepted + 1, over every check including the last one cut by the
   stop, as the sim's walk counts it.
   - The T16 / C16 ratio (sums per set) with a paired answer bootstrap (10,000 draws, fixed seed), beside the sim's.
   - Like for like with the sim: its keys (MBPP-50 / HE-30) and checks that start within the first 512 answer tokens
     (the capture's cap).
2. Exactness:
   - T16 vs C16, token ids per answer;
   - both vs jcc/arm0.jsonl (batch-1 greedy).
   - The first differing position of any mismatch, with the 2B's top-2 gap there.
3. pass@1 per arm, with EvalSuite's scorers read-only (as J(CC)), and McNemar vs C16.

Also reported, with no line:
- tree shape: depth, leaves (paths) per check;
- the share of checks whose accepted path leaves the #1 path (a sibling rescue);
- the accepted-depth histogram;
- target steps and head calls per check;
- wall time.

**Lines:**
- **EXACT:** T16 equals C16 token for token on 264 / 264. Any mismatch is a FAIL (a tree-code bug), reported with its
  first position and the top-2 gap there.
- **GAIN SURVIVES:** T16 / C16 kept/check >= 1.03 on both sets, with the 95 % CI above 1 (S9's SAME-COST WIN line).
- **MATCHES SIM:** on the sim's keys within 512 tokens, the sim's ratio lies inside the real ratio's 95 % CI or below
  it. **BELOW SIM:** the CI lies entirely below the sim's ratio.
- **pass@1:** identical by construction when EXACT holds. A difference is a FAIL.

**Self-checks:**
- (a) C16's real kept/check on the sim's keys within 512 tokens lands within +-10 % of the sim's chain-16 (MBPP 4.280,
  HE 5.971). The texts differ (BatchGenerator greedy vs one-token steps), so they cannot be exact.
- (b) My deeper-with-a-token equals sb.core.deeper bitwise on the same inputs. In smoke checks the tree's all-#1 path
  equals the chain's drafts at the same anchor (bitwise), as deep as the tree goes.
- (c) In smoke checks, the path-batched verify's accepted count equals the longest prefix of each path that matches the
  next committed tokens of a lazy batch-1 reference (J(CC)'s loop) at the same check start.

**Caveats, stated before the run:**
- This is a correctness and kept/check test, not a speed test. Pricing a one-pass tree verify is Leap's.
- The sim read Sandbox98's capture: BatchGenerator greedy, cap 512, MBPP-50 / HE-30. This run decodes 264 keys with
  one-token steps and the items' caps, so ratios can move with the text. The like-for-like row addresses that.
- s30 is the 2B's head. Its tree gain is not the 125B's (S9c: x1.136 / x1.060).
- Each arm is one deterministic greedy draw.

**Budget:**
- One process per arm. A full-length smoke run measures the peak first, and the share lock is sized to it.
- jcc_guard.sh adds a footprint and NO-SWAP kill. The MLX cache is cleared between keys.
- Expected < 4 GB, so it stays under the by-day one-job-over-5-GB line.
- ~77K tokens per arm (J(CC)'s exact arm: 2,643 s), so ~45-70 min per arm.
- Scripts: ~/MoonShots/week98/tree_decode.py and tree_score.py. Outputs: week98/tree/.

**Batch-shape probe (14:5x; tree/probe.json, probe.log, probe_v1.*):**
- One-token stepping through the fork code (B = 1) reproduces J(CC)'s exact arm token for token on its longest answer
  (mbpp/215, 1,024 tokens, teacher-forced). A B = 1 fork equals the original cache bitwise.
- **Rows are NOT batch-invariant:**
  - for every B >= 2, a [B, 1] step's row 0 differs from the [1, 1] step;
  - max |logit diff| is 0.22-0.47 at contexts 300 / 800 / 1,203;
  - the diffs are identical within kernel groups: B 2-8, B 15-16 and B 31-32.
- **Within one shape, rows are exact copies:**
  - every row equals row 0 for the same input (24 / 24);
  - row 0 does not change with the other rows' tokens (21 / 21).
- **So, by the rule:**
  - B = 16 is fixed for C16 and T16, with paths padded by copies of path 0;
  - T32 is skipped;
  - C16 vs arm0 is the batch-shape noise floor;
  - EXACT stays a clean bug test.
- ms per step, beside BigMoE's 80 % GPU job (so loose): B 1 ~6-12, B 16 ~22-24, B 32 ~28-30.

**Smoke (15:0x; 7 keys, both arms, --selfcheck; tree/smoke_*.log, tree/*_smoke.jsonl):**
- Self-check (b) PASS:
  - my deeper-with-a-token equals sb.core.deeper bitwise at 5,908 / 5,908 depths;
  - the tree's all-#1 path equals the chain drafted at the same anchor in 422 / 422 checks.
- Self-check (c) PASS on the smoke: 0 bad checks of 491 (C16) and 422 (T16).
- The padded copy rows equal row 0 at every step (2,520 / 2,520 per arm).
- EXACT on the smoke: T16 equals C16 on 7 / 7. Both differ from arm0 on the same 5 keys, at the same first positions:
  that is the batch-shape floor.
- Peak footprint 3.5-3.6 GB (MLX peak 2.53 GB); 16-30 tok/s beside BigMoE's GPU job.
- T16 per check: 4.5 paths, depth 8.3, 15 head calls, ~6 target steps; 36 % of checks accept off the #1 path.
- Smoke kept/check (7 keys, not judged): T16 / C16 x1.15 (HE) and x1.20 (MBPP).

**Amendment 1 (15:0x, logistics only, before the full run):** one process runs C16 and T16 key by key, instead of one
process per arm. If a run stops early (heat pause, the 23:00 app quit), the two arms still cover the same keys. No line
changes. Full run started 15:08 (tree/run.log).

**TREE-E2E, RESULT (16:4x; week98/tree/tree_result.json, score.log, gaps_c16_vs_arm0.json): EXACT 264 / 264; GAIN
SURVIVES on both sets; MATCHES SIM on both; pass@1 identical.**
- The run finished at 16:40: 264 keys per arm, 78,985 tokens per arm, 5,625 s in one process, peak footprint 3.8 GB,
  no pause.

| set | C16 kept/check | T16 kept/check | T16 / C16 [95 % CI] | like for like: the sim's keys, checks within 512 | S9 sim | lines |
|---|---|---|---|---|---|---|
| HE+ (164) | 6.041 (7,660 checks) | 6.933 (6,675) | **x1.148** [1.135, 1.160] | 5.745 -> 6.698, **x1.166** [1.142, 1.188] (30 keys) | 5.971 -> 6.802, x1.139 [1.104, 1.175] | GAIN SURVIVES; MATCHES SIM (the sim's ratio sits below the CI) |
| MBPP (100) | 4.935 (6,575) | 5.864 (5,533) | **x1.188** [1.175, 1.201] | 4.259 -> 5.177, **x1.216** [1.195, 1.236] (50 keys) | 4.280 -> 5.128, x1.198 [1.176, 1.219] | GAIN SURVIVES; MATCHES SIM (inside the CI) |

- Both arms commit the same 78,985 tokens. T16 needs 12,208 checks and C16 14,235 (x1.166 fewer, pooled).
- Bookkeeping (added 16:5x, after the orchestrator's check): the checks' kept totals sum to 78,721 (46,275 + 32,446).
  The other 264 tokens are one per key: each answer's first token is the 2B's argmax at the end of the prefill,
  committed before the first check. On every key of both arms, n = 1 + the sum of kept.

**Exactness:**
- **EXACT:** T16 equals C16 token for token on 264 / 264 answers.
- **The batch-shape noise floor:** both arms differ from J(CC)'s batch-1 exact arm (arm0) on the same 142 / 264 keys
  (HE+ 68 / 164, MBPP 74 / 100), at the same first positions (median 119, range 5-726).
  - At those 142 first differences, the 2B's top-2 gap (read by a third numerics, one [1, S] forward) is at most 0.375
    logits (median 0.125; 56 are exact bf16 ties).
  - The two arms' tokens are the 2B's top 2 in 136 / 142 cases, and within 0.25 of the top-1 in the other 6.
  - So every divergence from arm0 is a near-tie flip caused by the batch shape. None is a logic difference.

| pass@1 (264 keys, paired) | HE+ (164) | MBPP (100) | pooled | vs C16 | vs arm0 |
|---|---|---|---|---|---|
| C16 (B = 16) | 43.29 | 29.00 | 37.88 | - | +1.14 (7 / 4, p 0.55) |
| T16 (B = 16) | 43.29 | 29.00 | 37.88 | **0.00 (0 / 0, p 1.0): identical** | +1.14 (7 / 4, p 0.55) |
| arm0 (J(CC) exact, B = 1) | 41.46 | 29.00 | 36.74 | | - |

- The +1.14 against arm0 is batch-shape numerics only (J(CC)'s numerics control: +0.76).

**Self-checks:**
- (a) PASS on both sets. C16 like for like: HE+ 5.745 vs the sim's chain-16 5.971 (0.962); MBPP 4.259 vs 4.280
  (0.995). The smoke's HE+ 0.865 came from 4 keys.
- (b) PASS (smoke, as pre-registered): see the smoke block.
- (c) PASS on every check of the full run: 0 bad of 14,235 (C16) and 12,208 (T16).
  - **Deviation, stated:** as run, (c) checks each check's paths against the committed text at the same numerics
    (B = 16), not against a lazy batch-1 reference.
  - Under the fixed-shape branch, a batch-1 reference has other numerics. It would flag the 142 near-tie flips above,
    not logic. EXACT 264 / 264 covers the same-numerics greedy reference.

**Shape (no line):**

| arm | paths per check | depth | checks accepting 0 drafts | all 15 | off the #1 path | target steps per check | head calls per check | wall time |
|---|---|---|---|---|---|---|---|---|
| C16 | 1 | 15 | 13.5 % (HE+ 11.6, MBPP 15.7) | 6.9 % | - | 5.54 | 15 | 2,834 s |
| T16 | 4.16 (HE+ 3.89, MBPP 4.49) | 8.8 (9.3 / 8.2) | 4.3 % (4.0 / 4.6) | 5.7 % | 36.6 % (33.8 / 40.0), 5.66 kept on them | 6.46 | 15 | 2,758 s |

- Mean accepted drafts per check: C16 4.56, T16 5.48.
- Wall time is not a speed result. Both arms step one token per target forward at B = 16 (~28 tok/s beside BigMoE's GPU
  job), so a check costs m + 1 steps in either arm.

**Reading:**
- **The exact tree check works end to end on the 2B.** A best-first EAGLE-2 tree of 15 nodes, verified path by path,
  commits exactly the chain's text. It keeps x1.148 (HE+) / x1.188 (MBPP) more tokens per check, and pass@1 is
  identical.
- **The sim priced it right.** Like for like, the real gain is at the sim's (MBPP, inside the CI) or a little above it
  (HE+, overlapping CIs). The off-path proxy did not overprice trees on the 2B.
- **Where the gain comes from:**
  - the root's top 3 cut the checks that accept nothing from 13.5 % to 4.3 %;
  - 36.6 % of checks accept a path off the #1 path.
- **For Saturday's certificate:** "exact" has to mean equal to the greedy decode at the same numerics.
  - On the 2B with MLX's stock kernels, the batch shape alone flips 142 / 264 answers at near-ties (<= 0.375 logits).
  - So the certificate needs one shape for every target step, or a batch-invariant kernel (Leap's CHECK55 bitwise
    one-pass).
- **Not shown here:**
  - speed (the one-pass tree verify is Leap's CHECK55);
  - the 125B's gain (S9c on real t30b trees: x1.136 / x1.060);
  - T32, skipped by the batch rule. A comparable T32 needs C16 rerun at B = 32 too (~2 h for both arms).
- **Novelty (for the papers):** the method is published (EAGLE-2 trees; tree verification is lossless by construction).
  Ours is engineering: an exactness test that tells tree-logic bugs from batch-shape flips, plus the real-vs-sim check.
- **Files:**
  - scripts: week98/tree_decode.py, tree_score.py, tree_gaps.py, jcc_guard.sh;
  - outputs: tree/c16_fixed.jsonl, t16_fixed.jsonl, scores.json, tree_result.json, score.log, gaps_c16_vs_arm0.json,
    run.log, probe.json.

## TREE-E2E T32. Tree-32 vs chain-16 at one 32-row shape (orchestrator 16:5x: "T32: YES, this evening"; pre-registered 16:5x, before any T32 code ran)

**Question:** does s30's best-first EAGLE-2 tree with 31 nodes keep as many more tokens per check over the 15-draft
chain as the 2B's own sim priced?
- The sim (S9, TrainLab data/sb98/tree_price.txt, s30 exact, off-path proxy, 32 rows = the bonus + 31 nodes):
  - eagle2-32 vs chain-16, MBPP-50: 4.280 -> 5.856 = x1.368 [1.346, 1.390];
  - HE-30: 5.971 -> 7.951 = x1.332 [1.300, 1.357].
  - For scale: the sim's chain-32 is only x1.009 / x1.045 over chain-16.

**Setup:** TREE-E2E's target, drafter, panel, prompts, caps, verify code and scorer, except:
- **One shape: B = 32 rows for every target step of both arms.** The probe put B 31-32 in another kernel group than
  B 15-16, so C16 is rerun at B = 32. The B = 16 arms are not reused for the line.
- **Arms:**
  - **C16@32:** s30's 15-draft chain, padded to 32 rows by copies;
  - **T32@32:** best-first EAGLE-2 with 31 nodes, top-3 per node, the same heap and tie rules as T16. At most 31
    paths, padded to 32 by copies of path 0.
- One process runs both arms key by key, in panel order. The order interleaves the sets (every prefix holds HE+ and
  MBPP at ~62 / 38), so an early stop leaves a balanced subset.
- **Stop rule:** all 264 keys if time allows, with a hard stop at 22:15 (finish by 22:30; tonight's quiet chain starts
  at 23:00). After a stop, the keys both arms hold are scored. That subset counts only if each set's T32 / C16 95 % CI
  half-width is <= 3 % (the orchestrator's rule); otherwise the result is INCOMPLETE.

**Lines:**
- **MATCHES SIM (the line), per set:** on the sim's keys (MBPP-50 / HE-30) and checks that start within the first 512
  answer tokens, the sim's ratio lies inside the real T32 / C16 ratio's 95 % CI or below it.
  - **BELOW SIM:** the CI lies entirely below the sim's ratio.
  - The CI comes from the paired answer bootstrap (10,000 draws, seed 1115), as for T16.
- **EXACT:** T32@32 equals C16@32 token for token on every key run. Any mismatch is a FAIL (a tree-code bug), reported
  with its first position and the 2B's top-2 gap there.
- **pass@1:** identical by construction when EXACT holds. A difference is a FAIL.

**Also reported, with no line:**
- the full-panel T32 / C16 ratio with its CI, and S9's SAME-COST WIN test on it (>= 1.03 with the CI above 1);
- T32 vs T16 kept/check, as context for whether 32 rows pay on the 2B. T16 ran at B = 16, so the texts can differ at
  near-ties; kept/check is still paired by key;
- the 16-vs-32 batch-shape floor: C16@32 vs C16@16, answers changed and the top-2 gaps at the first differences;
- shape (paths, depth, sibling rescues, the accepted-depth histogram), target steps and head calls per check, wall time.

**Self-checks:**
- (a) C16@32 like for like within +-10 % of the sim's chain-16 (MBPP 4.280, HE 5.971).
- (b) Smoke at B = 32:
  - deeper-with-a-token equals sb.core.deeper bitwise;
  - the tree's all-#1 path equals the chain at the same anchor;
  - the padded copy rows equal row 0 at every step.
- (c) Offline, every check: no path matches the committed text (same numerics) further than the accepted count.

**Guards and budget:**
- A full-length smoke (mbpp/215's 1,024 tokens, both arms, --selfcheck) measures the peak first. The share lock carries
  that MEM; target <= 4 GB footprint. If the smoke's peak is over 4 GB, the MLX cache limit is lowered before the run.
- One process, jcc_guard.sh footprint and NO-SWAP kill.
- Yield: before each key the runner checks for another session's quiet lock or a queued quiet ticket. If either
  exists, it waits (peer wait) until the Mac is clear. Heat pauses go through Sandbox98's PAUSE file as before.
- Time: B = 32 steps cost ~28-30 ms vs ~22-24 at B = 16 (probe, beside BigMoE). T32 needs ~31 head calls per check.
  Expected ~2 h for both arms on 264 keys, plus pauses.

**Caveats:** TREE-E2E's, plus: B = 32 numerics differ from B = 16, so C16@32 is not C16@16 token for token (the floor
above). T32 needs twice the head calls of T16 per check; this run measures kept/check, not cost.

**T32 smoke and amendment 1 (17:0x, memory only, before the full run; tree/smoke32*.log, v2check16.log):**
- First smoke (mbpp/265): at 32 rows its answer ends at 211 tokens (at 16 rows it ran to the cap), so it was not full
  length. The second smoke used two keys that run to the cap in arm0 and C16@16 alike: HumanEval/126 (768) and
  mbpp/228 (1,024).
- Self-checks at B = 32 PASS:
  - deeper-with-a-token bitwise at 4,920 / 4,920 depths;
  - the all-#1 path equals the chain in 164 / 164 checks;
  - the padded copy rows equal row 0 at every step (both arms);
  - T32 equals C16 on both keys.
- **Memory:** the peak footprint was 4.27-4.29 GB, over the 4 GB budget.
  - Cause: select() kept the next row as slices of the R-row buffers. MLX slices are views, so the whole R-row KV and
    GDN buffers stayed alive through the next check. On top of that, the first step of each check grew the forked KV
    buffer by a copy.
  - **Fix (tree_decode.py fork / select v2):**
    - select() gathers the kept row into its own buffers (mx.take) and evaluates it at once;
    - fork() builds the R-row KV buffer at offset + 256 positions, the buffer that first step would have built (same
      values, same strides).
  - **Check:** v2 reproduces v1's outputs exactly (answer ids and every per-check record) on both smoke keys at
    B = 32 (both arms) and on HumanEval/1 at B = 16 (C16 and T16, 768 tokens each).
  - Peak footprint with v2: 3.17-3.25 GB at B = 32 (MLX peak 2.84 GB); 2.39 GB at B = 16.
  - The MLX cache limit is set to 0.25 GB (SB_CACHE_GB; no effect on numerics). The guard kills at 4 GB.
- No line changes. Full run started 17:03 (tree/run32.log), with the stop rule at 22:15 and quiet-ticket yielding on.

**TREE-E2E T32, RESULT (21:4x; week98/tree/tree_result_t32_fixed_b32.json, score32.log, gaps_c16_16_vs_32.json): MATCHES
SIM on both sets; EXACT 264 / 264; pass@1 identical.**
- **The run:**
  - 17:03-21:44 in one process; all 264 keys in both arms, so the 22:15 stop rule never fired;
  - 76,619 tokens per arm, peak footprint 3.26 GB (guard exit 0, no swap growth);
  - it yielded 10 times to other sessions' quiet windows (~1 min each);
  - the orchestrator's HOLD (18:52, Kamen closing the laptop) parked it in Sandbox98's PAUSE file at a key boundary
    (180 / 264) until the partial lift (20:55).
  - No line changes.

| set | C16@32 kept/check | T32@32 kept/check | T32 / C16 [95 % CI] | like for like: the sim's keys, checks within 512 | S9 sim (eagle2-32 / chain-16) | line |
|---|---|---|---|---|---|---|
| HE+ (164) | 5.945 (7,723 checks) | 7.985 (5,750) | **x1.343** [1.329, 1.358] | 5.784 -> 7.864, **x1.360** [1.328, 1.390] (30 keys) | 5.971 -> 7.951, x1.332 [1.300, 1.357] | MATCHES SIM (inside the CI) |
| MBPP (100) | 4.984 (6,108) | 6.812 (4,469) | **x1.367** [1.345, 1.387] | 4.394 -> 5.970, **x1.359** [1.333, 1.384] (50 keys) | 4.280 -> 5.856, x1.368 [1.346, 1.390] | MATCHES SIM (inside the CI) |

- Both arms commit the same 76,619 tokens. T32 needs 10,219 checks and C16@32 needs 13,831 (x1.353 fewer, pooled).
  - The kept totals sum to 76,355 (45,911 + 30,444). The other 264 tokens are the prefill tokens, one per key, as in
    T16.
- **Full panel, S9's SAME-COST WIN test** (>= 1.03 with the CI above 1, both arms at 32 rows): passes on both sets.
  - The CI half-widths are 1.1 % (HE+) / 1.5 % (MBPP). The whole panel ran, so the subset rule wasn't needed.
- **T32 vs T16** (kept/check, paired by key; T16 ran at B = 16, so the texts can differ at near-ties):
  - HE+ 6.933 -> 7.985, x1.152 [1.121, 1.181];
  - MBPP 5.864 -> 6.812, x1.162 [1.079, 1.245].

**Exactness:**
- **EXACT:** T32@32 equals C16@32 token for token on 264 / 264 answers.
- **The 16-vs-32 batch-shape floor:** C16@32 differs from C16@16 on 120 / 264 answers (144 identical).
  - At the 120 first differences, the 2B's top-2 gap (read by a third numerics, one [1, S] forward; tree_gaps.py,
    1.9 GB) is at most 0.375 logits (median 0.125; 57 are exact bf16 ties).
  - The two runs' tokens are the 2B's top 2 in 117 / 120.
  - So every difference is a near-tie flip caused by the batch shape, like T16's floor against arm0 (142 / 264, max
    0.375).
  - Against J(CC)'s batch-1 arm0, both T32 arms are identical on 118 / 264.

| pass@1 (264 keys, paired) | HE+ (164) | MBPP (100) | pooled | vs C16@32 |
|---|---|---|---|---|
| C16@32 | 43.90 | 25.00 | 36.74 | - |
| T32@32 | 43.90 | 25.00 | 36.74 | **0.00 (0 / 0, p 1.0): identical** |
| C16@16 (TREE-E2E, for the floor) | 43.29 | 29.00 | 37.88 | |

- C16@32 vs C16@16 is -1.14 pooled (5 / 8 discordant, exact McNemar p 0.58; MBPP -4.0, 2 / 6, p 0.29). That's batch-shape
  numerics only: against arm0, C16@32 is 0.00 (8 / 8, p 1.0).

**Self-checks:**
- (a) PASS on both sets. C16@32 like for like: HE+ 5.784 vs the sim's chain-16 5.971 (0.969); MBPP 4.394 vs 4.280
  (1.027).
- (b) PASS (smoke at B = 32, as pre-registered): see amendment 1.
- (c) PASS on every check of the full run: 0 bad of 13,831 (C16@32) and 10,219 (T32@32). T16's stated deviation applies:
  the reference is the same-numerics committed text.

**Shape (no line):**

| arm | paths per check | depth | checks accepting 0 drafts | all drafts (15 / 31) | off the #1 path | target steps per check | head calls per check | decode time |
|---|---|---|---|---|---|---|---|---|
| C16@32 | 1 | 15 | 13.1 % (HE+ 11.5, MBPP 15.2) | 7.0 % | - | 5.53 | 15 | 3,954 s |
| T32@32 | 8.64 (8.18 / 9.23) | 12.07 (12.66 / 11.31) | 3.9 % (3.7 / 4.3) | 0.46 % | 49.3 % (49.1 / 49.5), 7.42 kept on them | 7.49 | 31 | 4,316 s |

- Mean accepted drafts per check: C16@32 4.52, T32 6.47 (HE+ 4.97 -> 7.03, MBPP 4.01 -> 5.85).
- Decode time is not a speed result: the run held a share lock beside other jobs, through 10 yields, and both arms step
  one token per target forward at B = 32.

**Reading:**
- **The sim prices the 32-node tree right on the 2B too.** Like for like, the real gain sits inside the sim's band on both
  sets (x1.360 vs x1.332; x1.359 vs x1.368). With T16, both tree sizes the sim priced now match it in real decoding.
- **At the same 32-row shape, tree-32 keeps x1.34-1.37 over the chain.** The sim's chain-32 gains only x1.01-1.05 over
  chain-16, so the extra rows pay as breadth, not depth:
  - about half of T32's checks (49 %) accept a path off the #1 path;
  - checks that accept nothing fall from 13.1 % to 3.9 %.
- **Exact and quality-neutral:** 264 / 264 identical, and pass@1 identical. The only text changes come from the batch
  shape (16 vs 32 rows), and all of them are near-ties.
- **Against T16:** x1.15-1.16 more kept per check, for twice the head calls (31 vs 15) and a 32-row check. So 32 rows beat
  16 only where that costs less than ~1.15x a 16-row check. The cost side is Leap's K-row pricing; this run makes no
  speed claim.
- **Not shown here:** speed; the 125B (S9c's real t30b trees: EAGLE-2-32 x1.371 / x1.531 vs chain-16).
- **Novelty (for the papers):** no new method (EAGLE-2 trees; tree verification is lossless by construction).
  Engineering: the same-shape exactness protocol and the real-vs-sim check at two tree sizes.
- **Files:**
  - scripts: week98/tree_decode.py (fork / select v2, --rows, --stop-at, quiet-ticket yield), tree_score.py,
    tree_gaps.py, jcc_guard.sh;
  - outputs: tree/c16_fixed_b32.jsonl, t32_fixed_b32.jsonl, tree_result_t32_fixed_b32.json, scores.json, score32.log,
    gaps_c16_16_vs_32.json, gaps32.log, run32.log.

## PILLAR 3. Drafting off the critical path (NewIdeas 22:3x, the orchestrator's assignment; pre-registered Sun 15 Nov 22:5x, before any code ran)
**Goal:** tree-32's drafting costs <= 4 ms per block on the critical path, with kept within 1 % of today's head.
- Either in total, or hidden beside the check at <= +4 ms of check slowdown.
- HE+ drafts ~20.8 levels per tree-32 block, MBPP 14.6 (NewIdeas, BIGGER_TREES.md).

**The records, and the arithmetic they force:**
- A level costs ~1.15 ms standalone (Leap P2) and ~2.2 ms in the engine, so ~24 ms per HE+ tree-32 block today.
- Leap P2's fit: a step is ~0.54 ms fixed (per-op launches and syncs) plus bytes at ~290 GB/s.
  - The best cut is 95.1 MB / 0.864 ms (4-bit Linears + an 8K head); attention + MoE (65 MB) were left untouched.
  - So 32 steps at any bytes take >= ~17 ms.
- t30b's layer (dd_t30b.safetensors header, fp32, 263.8 MB) and the 125B's MoE config (BigMoE cfg: 512 experts, 10 per
  token, 640 wide; hidden 2560; vocab 248,320). At 4 bits (~4.5 bits per param with scales), per level and row:
  - q_proj [12288, 2560] 17.7 MB (the largest), o_proj [2560, 6144] 8.8, fc_embedding + fc_hidden 7.4, k + v 1.5,
    shared expert 2.8, router 0.7: ~39 MB dense;
  - the routed experts, 10 x ~2.76 MB = ~27.6 MB (from the 125B's weights; not in the file);
  - the screened head: 8K ids 11.8 MB, 32K ids 47 MB.
  - About 78 MB at 4-bit + 8K, against Leap P2's measured 95.1 MB, which also counts the attention's context reads and
    other stored formats.
- **4 ms / 20.8 levels = 0.19 ms per level.** At 290 GB/s that allows <= ~55 MB per level even at zero fixed cost. So
  the target needs BOTH of these; neither alone is enough:
  - (i) the fixed cost per level down to <= ~0.05 ms (today ~0.54);
  - (ii) the bytes per level down to <= ~40-45 MB (today >= ~78-95).
- **Concurrency doesn't remove bytes.**
  - On a bandwidth-bound check, a concurrent draft's reads add in full: ANE-1 slowed the check x1.070 beside the ANE
    drafter.
  - Block n+1 also depends on check n's outcome (the accept point, the bonus, the 125B's fresh state at the cut). So
    drafting it during check n is speculation (section K's pricing).

**Candidates, ranked (mine):**
1. **(a') A byte-cut, sync-free level step: the critical-path route, attacking both terms.**
   - **Sync-free:** draft the level-synchronous tree as ONE lazy MLX graph per block, with no host sync until the block
     ends:
     - a fixed frontier of F in {1, 2, 3} nodes per level, chosen on the GPU (top-F by path score over the candidates);
     - token ids gathered by index, and chain entries appended.
     - This tests whether the fixed cost falls from ~0.54 ms toward the per-kernel encode cost.
     - Then try mx.compile. A hand-fused Metal dispatch (mx.fast.metal_kernel) for the dense GEMVs comes only if those
       two leave more than 0.1 ms fixed per level.
   - **Byte cuts of a draft-only copy** (the verify path is untouched, so exactness holds by construction):
     - routed experts top-10 -> 4 or 2 at draft time (-16.6 / -22 MB);
     - frontier-shared routing (the F rows of a level use one expert set, ranked by their summed router scores), so the
       experts don't grow with F;
     - q / o / fc at 3 or 2 bits (-6 to -15 MB);
     - head 8K -> 4K, or session ids only (-6 to -9 MB).
     - Budgets: {95, 70, 55, 40} MB per level.
   - A fixed-F level-synchronous tree isn't best-first tree-32, so its kept is measured, not assumed.
2. **(b) Speculative next-block drafting during the check, priced on paper first (desk, from the dumps).**
   - **The guess:** the check accepts the #1 path (the highest path_p root-to-leaf path in the 31-node tree) up to its
     leaf, and the bonus is the leaf's top-1 child. Variant: the 2 highest-path_p leaves.
   - **Hit rate:** from BigMoE's real tree dumps (tree_t30b_{heplus24,mbppbank}.npz: 32 pops per anchor, parent,
     path_p, on_true, ynext_rank), under tree-32's own walk (certify98's rule from row 1). A hit means:
     - the accept node is the guessed leaf;
     - and ynext_rank == 0 there.
   - **Value per block** = hit x (the critical-path draft time) minus two costs:
     - the contention, paid on every block (the speculative draft's bytes / BW);
     - the stale-state loss on hits (block n+1 is drafted from the head's carry, not the 125B's fresh state).
   - **Paper KILL line:** at today's bytes (~2 GB per block, ~6.8 ms of contention) against ~24 ms of drafting, KILL if
     hit < ~28 % (one guess) or < ~57 % (two guesses at twice the bytes). The stale-state loss is not counted, so the
     line is generous to (b).
3. **(c) The ANE beside the check.**
   - It inherits (b)'s dependency: it can only draft block n+1 speculatively.
   - It adds ANE latency (~1+ ms per call x ~20.8 levels; WHOLE-CHIP-2's handoff is 78-147 us per layer) and ANE-1's
     x1.07 check slowdown.
   - Priced on paper with (b). Built only if (b) passes on paper AND (a') fails.

**Measurements (Monday by day):**
- **Budget:** <= 4 GB, one process, share lock with MEM. Timings under a share lock are indicative: interleaved arms,
  medians. The timing verdict comes in the first window the orchestrator allows.
- **B0 (desk, first; CPU, no lock):** (b)'s hit rate on HE+ 24 and MBPP bank, one guess and two, with the full-accept
  share and the accept-depth histogram.
- **M0:** today's t30b level step in MLX at F = 1 / 2 / 3 (4-bit + 8K head, synthetic states as in Leap P2). It must
  reproduce Leap P2's 0.86-1.15 ms per level within 15 %; otherwise stop and reconcile.
- **M1:** the sync-free block (one eval per 21 levels) at M0's bytes. It gives ms per level, so the fixed cost f.
- **M2:** the byte budgets {95, 70, 55, 40} MB, sync-free. It gives ms per level, fitted to f + B / BW.
- **M3 (kept; TRUE-STATE, teacher-forced with the 125B's true state):** for each cut, on the bank's HE+ 24 and MBPP 100
  windows:
  - the per-depth agreement of the cut head's top-3 with the full head's;
  - tree-32 kept/32 under its own walk vs the full head's;
  - it runs on t30b's cells harness (MTP Head's or BigMoE's code, imported read-only, with a weights hook), or by its
    owner.
  - The screened-head pick must equal the full head's at the same ids (subset logits bitwise, as MTP-STEP-1).

**Lines:**
- **PASS:** a measured sync-free block <= 4.0 ms per 20.8-level tree-32 block (<= 0.19 ms per level), at a cut whose M3
  kept/32 is within 1 % of the full head's on both sets.
- **BETWEEN:** <= 8 ms at a cut within 1 % kept. Report it as the route, with the gap.
- **KILL (route a'):** > 8 ms at every cut within 1 % kept.
- **(b) / (c):** the paper line above.

**Expectation, written first:**
- The fixed cost falls a lot once there's no sync (to ~0.1-0.2 ms per level), but bytes bind.
- At ~78-95 MB, a level costs >= ~0.27-0.33 ms, i.e. ~6-7 ms per block.
- Getting to <= ~45 MB needs 2-bit attention plus top-2 experts, which I expect to cost more than 1 % kept.
- So BETWEEN (~6-8 ms) is my prior. PASS needs a byte cut that keeps kept.
- For (b), I expect a hit rate of ~10-25 %, so a paper KILL.

**Not in scope:**
- fewer levels per block (a head that drafts 2 tokens per level: a drafter change);
- a smaller distilled draft head;
- the check side (Leap).

**B0, RESULT (22:3x; week98/p3_b0_spec.py, p3_b0_spec.json; TRUE-STATE, from BigMoE's real t30b tree-32 dumps under
tree-32's own walk): (b) KILL on paper by its line. HE+ is marginal, and it hangs on M1.**
- **Self-check PASS:** the walk reproduces S9c's real EAGLE-2-32 kept/32 exactly (HE+ 16.263, MBPP 9.031). That needed
  tree_real125's extra end-of-window slot (kept 1, no tree), which isn't counted as a speculation check.

| tree-32, own walk | checks | stop at a leaf | stop at the root | G1 node / hit | G2 node / hit | accepted p25 / p50 / p75 / p90 |
|---|---|---|---|---|---|---|
| HE+ 24 | 391 | 50.4 % | 1.5 % | 35.0 / **27.1 %** | 46.0 / **34.5 %** | 5 / 12 / 30 / 31 |
| MBPP bank | 1,727 | 42.1 % | 1.4 % | 11.7 / **4.2 %** | 21.8 / **8.7 %** | 4 / 7 / 11 / 15 |

- Guess and hit, as defined above: the node must be right, and so must the bonus (the stop node's best child outside
  the tree).
- **Line** (KILL if hit < ~28 % for one guess, < ~57 % for two): HE+ 27.1 / 34.5 is a KILL, marginal for one guess.
  MBPP 4.2 / 8.7 is a clear KILL.
- **Sensitivity, stated after the result.** The break-even hit equals the bytes' share of a draft level's time, because
  only the bytes contend with the check:
  - 53 % on Leap P2 as stored (1.147 ms, 177.6 MB);
  - 38 % at its best cut (0.864 ms, 95.1 MB);
  - 15 % at today's in-engine level (~2.2 ms, mostly overhead that would have to hide fully behind the check).
  - So HE+ (27 %) clears break-even only in a world where a level's overhead stays large and overlaps the check
    completely. That is exactly the world (a') removes. The stale-state loss on hits (block n+1 drafted from the head's
    carry) is not counted, which favours (b).
- **Action:** (b) and (c) are parked behind (a'). Revisit (b) on HE+ only if M1 shows a level's sync-free fixed cost
  stays above ~1 ms; it would then also need the stale-state kept measured.
- **Also from the dumps:** half the HE+ checks (50.4 %) stop at a leaf, and a quarter accept >= 30 of 31. So tree-32's
  depth binds on HE+, the K* story.

**M0-M2, RESULT (Mon 16 Nov 08:35-08:38; week98/p3_m012.py, p3_m012_{stock,fused}.json / .log, p3_timing.log. TIMING
under a quiet lock in NewIdeas' slot: Kamen away 8 h, GPU 0 % at the start. Synthetic states as P2, so no kept here.)
Route (a') in stock MLX is a KILL by its line, on timing alone.**
- The sync-free block cuts the fixed cost per level from 0.55 to 0.31 ms. But 0.31 ms x 20.8 levels = 6.5 ms per HE+
  tree-32 block before any bytes.
- The best measured cut is 9.3 ms per block (K8: 39.8 MB, compiled, F = 1), above the 8 ms BETWEEN line.
- **M0 PASS:** P2's own chain reproduces: as stored 1.126 ms (x0.981 of P2's 1.147); 4-bit + head 8K 0.857 ms (x0.992 of
  0.864).
- **Self-checks PASS:**
  - sync / free / async give the same tokens at F = 1 and 3, shared routing included;
  - the F = 1 tree level equals P2's chain token for token;
  - frontier-shared routing equals the stock MoE at one row (max diff 0.0);
  - compiled sub-blocks equal eager (share 1.0: K1 / K3 / K8 / EK1).
- **M2 fits** (F = 1; 9 eager cuts from 177.6 to 39.8 MB; residuals <= 0.015 ms):

| level step | f, ms per level | BW, GB/s |
|---|---|---|
| one host sync per level (P2's design) | 0.549 | 260 |
| sync-free, one eval per block | 0.452 | 288 |
| sync-free, async_eval per level | **0.311** | 293 |
| async + mx.compile of the pure sub-blocks (3 cuts) | 0.307 | 300 |

- **Where the 0.31 ms goes.** One level is 237 MLX primitives, 144 of them not views.
  - The async overlap hides the CPU graph build: 0.112-0.135 ms per level eager, 0.077-0.106 compiled.
  - What's left is the GPU's chain of ~144 dependent dispatches, ~2.1 us each. TRIED's M5 record: a dependent dispatch
    costs 1.3-1.6 us, and an in-kernel grid barrier is no cheaper (1.97-2.47 us; the megakernel row, KILL).
  - Neither mx.compile (hc reads, MoE, tail, prepare) nor the engine's MTPLX fusions (gate|up, MoE kernel, QSA qkv)
    cut it. Per level: C1 0.645 / K1 0.632 / E1 0.676 ms; C3 0.542 / K3 0.529 / E3 0.545.
  - Side fact: MTPLX_FUSED_HC=1 on this head's bf16 mixes runs 5.26 ms per level, x8 slower.
- **Per block** (ms; async; F = 2 / 3 with frontier-shared routing). Best-first trees need ~1.5 nodes per level on HE+
  tree-32 and ~2.1 on MBPP, so F = 1 is a floor:

| cut | MB per level | ms per level F1 / F2 / F3 | HE+ tree-32 (20.8) F1 / F2 | MBPP tree-32 (14.6) F2 / F3 | HE+ tree-16 (12.6) F1 / F2 | MBPP tree-16 (10.3) F1 / F2 |
|---|---|---|---|---|---|---|
| C1 4-bit + head 8K | 95.1 | 0.645 / 0.734 / 0.879 | 13.4 / 15.3 | 10.7 / 12.8 | 8.1 / 9.2 | 6.6 / 7.6 |
| C3 + top-4, head 4K | 70.8 | 0.542 / 0.617 / 0.731 | 11.3 / 12.8 | 9.0 / 10.7 | 6.8 / 7.8 | 5.6 / 6.4 |
| C6 + q/o 2-bit, top-2 | 52.9 | 0.477 / 0.553 / 0.676 | 9.9 / 11.5 | 8.1 / 9.9 | 6.0 / 7.0 | 4.9 / 5.7 |
| C8 all 2-bit, top-2, head 2K | 39.8 | 0.460 / 0.514 / 0.640 | 9.6 / 10.7 | 7.5 / 9.3 | 5.8 / 6.5 | 4.7 / 5.3 |
| K8 (C8 compiled) | 39.8 | 0.446 / 0.506 / 0.646 | 9.3 / 10.5 | 7.4 / 9.4 | 5.6 / 6.4 | 4.6 / 5.2 |

- **The lines:**
  - **PASS** (<= 4.0 ms = 0.192 ms per level) is unreachable as built: f alone is 0.31.
  - **BETWEEN** (<= 8 ms = 0.385 ms per level) needs <= 21.7 MB per level at f = 0.311. The leanest cut reads 39.8 MB.
  - **KILL (a')** (> 8 ms at every cut): yes, on timing alone. M3's kept can't rescue it.
  - **(b)** stays parked: the sync-free fixed cost (0.31 ms) is below its ~1 ms revisit threshold.
- **What a hand-fused level would need** (the pre-registered next step, since compile leaves > 0.1 ms; bytes at 293 GB/s):
  - **BETWEEN:** f <= 0.060 ms at C1 (95.1 MB), 0.143 at C3, 0.204 at C6, 0.249 at C8.
  - **PASS:** f <= 0.056 at C8, and impossible at C3 or above.
  - At ~1.5 us per dependent step, f <= 0.056 means <= ~35 dependent steps per level, against ~144 today.
  - By my count, the level's critical path has ~23 global dependencies (norms, GEMVs, the attention, top-k, from fc to
    the head's frontier pick). So PASS sits at the floor even with perfect fusion, and needs C8's bytes within 1 % kept.
  - BETWEEN at C3 needs ~90 steps or fewer: plausible with real op fusion, but it's a multi-day kernel build.
- **Useful now, short of the line** (output exact by construction: the verify is untouched; kept is M3's):
  - The sync-free level at C1's formats costs 0.645 ms, against 0.901 with a host sync per level (-28 %).
  - HE+ tree-32 drafting comes to 13.4-15.3 ms per block, vs the brief's ~24 ms in the engine today.
  - Tree-16 halves it again (HE+ 8.1 ms at C1, 6.8 at C3).
- **My expectation was wrong on f.** I wrote "falls to ~0.1-0.2 ms per level once there's no sync"; it fell to 0.31,
  because most of it is the GPU's dependent-dispatch chain, not the host sync. My BETWEEN prior (~6-8 ms) came out worse:
  9.3-13.4 ms.
- **Next:**
  - M3 (kept, TRUE-STATE) on C1, C3, C8, plus Q4 / H8 to split C1's cost. Harness: week98/p3_dd_tree.py, a copy of
    BigMoE's dd_tree.py with cut hooks. It runs in the day's big-job slot (~11-12 GB; after t30c or Tuesday), behind
    its bitwise gate at --cut none.
  - The hand-fused level is a separate build, priced above, and needs NewIdeas' / the orchestrator's go.

**M3, RESULT (Mon 16 Nov 08:45-10:17; week98/p3_dd_tree.py, p3_m3.sh, p3_score.py; p3/m3_run.log, p3/m3_scores.{txt,json},
p3/tree_t30b_<set>_<cut>.json, dumps p3/*.npz kept local. TRUE-STATE (lab): teacher-forced, the 125B's true state at
every anchor; kept under certify98's walk (NewIdeas' treek_final.py, read-only) at rows 16 and 32. Share lock MEM 12 GB
beside other GPU jobs: kept only, no timing.)
Route (a') is a KILL on kept as well: no cut stays within 1 % of the full head on either set, at either depth. With
M3, Pillar 3's lines are out of reach for the level-by-level drafter on this machine, hand-fused level included.**
- **Gate PASS:** --cut none at --run t30b reproduces BigMoE's tree_t30b_{heplus24,mbppbank}.npz bitwise, every array
  (21 / 21 on both sets). The walk gives S9c's numbers: HE+ 11.263 / 16.263, MBPP 7.487 / 9.031 at rows 16 / 32.
- **Kept per cut** (cut / full head; 95 % CI over 2,000 window resamples, seed 1115; HE+ 24 windows, 6,375 anchors;
  MBPP 100 windows, 15,641 anchors). MB per level = the config as the engine would store it, F = 1:

| cut | MB per level | HE+ rows 16 | HE+ rows 32 | MBPP rows 16 | MBPP rows 32 |
|---|---|---|---|---|---|
| full head (the gate) | 488.0 | 11.263 | 16.263 | 7.487 | 9.031 |
| H8: screen 8K + session tokens only | 142.2 | x0.976 [0.962, 0.990] | x0.963 [0.946, 0.978] | x0.990 [0.981, 0.998] | x0.985 [0.975, 0.993] |
| Q4: trained Linears + hc mixes 4-bit only | 440.9 | x0.963 [0.940, 0.982] | x0.922 [0.893, 0.948] | x0.938 [0.922, 0.953] | x0.899 [0.882, 0.913] |
| C1 = Q4 + H8 | 95.1 | x0.935 [0.909, 0.959] | x0.875 [0.849, 0.901] | x0.925 [0.909, 0.940] | x0.889 [0.870, 0.906] |
| C3: + experts top-4, screen 4K | 70.8 | x0.863 [0.835, 0.888] | x0.767 [0.738, 0.793] | x0.836 [0.820, 0.849] | x0.791 [0.774, 0.806] |
| C8: 2-bit, top-2, screen 2K | 39.8 | x0.255 [0.230, 0.282] | x0.186 [0.158, 0.220] | x0.372 [0.358, 0.385] | x0.327 [0.312, 0.341] |

- **The line** (pre-registered: tree-32 kept within 1 % on both sets): **FAIL at every cut**, and at both halves of C1
  alone. H8's MBPP CIs reach 0.99, but its HE+ CI at rows 32 stops at 0.978.
- **Per-depth agreement with the full head** (pre-registered; true-path states both trees share; j = 0 is the root,
  j = depth + 1 for a node). Top-1 same / top-3 set same:
  - H8 is flat in depth: top-1 >= 99.3 %, top-3 set 71-77 % (HE+) / 81-84 % (MBPP).
  - The weight cuts drift with depth through the head's carried state. Q4: top-1 98.6 -> 96.8 % (HE+) and 96.4 -> 88.7 %
    (MBPP); top-3 set 55 -> 23 % and 63 -> 30 % (j 0 -> j >= 16). C1 tracks Q4 a little lower; C8's top-1 is 72-75 %
    at the root and 27-28 % by j 4-7.
  - The gate's own agreement is 1.000 / 1.000 at every depth.
- **Why tree-32 loses 1.3-2x more than tree-16:** top-1 mostly holds, the lower ranks don't, and tree-32's extra nodes
  ride on those ranks. Its gain over tree-16 shrinks from 5.00 to 4.67 (H8) / 4.16 (Q4) / 3.69 (C1) kept on HE+, and
  from 1.54 to 1.48 / 1.10 / 1.10 on MBPP.
- **Screened kernel** (Leap's ScreenedHead, gathered 4-bit rows, on each window's roots: one call of 125-400 rows per
  HE+ window, 18-520 per MBPP window):
  - Its logits equal the full head's subset bitwise on 24 / 24 HE+ windows (C1, C3, H8; 23 / 24 at C8).
  - On MBPP, 79 / 100 windows at 8K (C1, H8), 62 / 100 at 4K (C3), 40 / 100 at 2K (C8).
  - The kept above doesn't depend on it: the dumps mask the full head's logits, exact by construction.
  - The pre-registered condition ("subset logits bitwise, as MTP-STEP-1") fails at these batched shapes. MTP-STEP-1's
    1-row claim isn't tested here. Not characterized further; a shape-dependent matmul path is my guess.
- **Pick change** (the head's top-1 outside the screened set, all node picks): H8 1.50 % HE+ (3,162 of 210,375) / 1.95 %
  MBPP (10,078 of 516,153); C1 1.56 / 1.84 %; C3 2.35 / 3.35 %; C8 6.25 / 5.47 %.
- **What it means for the hand-fused level** (proposed for next week; it cuts f, not bytes):
  - The kept line leaves no tested byte cut. At P2's base (C0: bf16 dense, 32K screen, 177.6 MB; its own kept is not
    measured here), the bytes alone take 0.607 ms per level at 293 GB/s. That is 12.6 ms per HE+ tree-32 block at f = 0
    (MBPP tree-32 8.9; tree-16 7.6 / 6.3).
  - Even at the critical-path floor (f ~ 0.035 ms: ~23 dependent steps x ~1.5 us), BETWEEN needs <= ~102 MB per level
    and PASS <= ~46 MB. The only tested configs that small are C1 and below, which keep x0.875-0.889 down to x0.19-0.33
    at tree-32.
  - So no fixed-cost work brings (a') to either line within 1 % kept on this machine.
  - The hand-fused level is still an exact engine gain if it reaches f = 0.056. At C0, async 0.921 ms per level
    (measured) would become ~0.66, so HE+ tree-32 drafting goes 19.1 -> ~13.8 ms per block (-28 %; -47 % vs the synced
    level's 1.246 ms). Whether that is worth a multi-day build is an engine question, not Pillar 3's.
- **Useful now:**
  - Keep the draft head's trained Linears and hc mixes at bf16, as stored. 4-bit g64 costs 3.7-6.2 % kept at tree-16
    and 7.8-10.1 % at tree-32.
  - A screened head for tree drafting is not free. 8K + session tokens costs 1.0-2.4 % at tree-16 and 1.5-3.7 % at
    tree-32. MTPSCREEN's chain-pick PASS for the 32K set doesn't carry to tree-32 by itself: the 32K set's tree cost is
    unmeasured.
  - The sync-free level (M0-M2) stays the one exact gain: 1.246 -> 0.921 ms per level at C0 (-26 %).
- **Expectation check:** I wrote "Getting to <= ~45 MB ... I expect to cost more than 1 % kept". It cost 67-81 % (C8).
  But my BETWEEN prior assumed C1-class bytes (78-95 MB) would keep kept: the 4-bit cut of the trained parts alone
  misses the line by 8-10x, and the 8K screen alone by 1.5-3.7x.
- **Next:** none from me on Pillar 3. (a') is closed on both legs; (b) and (c) stay parked (NewIdeas 08:4x). The
  hand-fused level is re-scoped above as an engine item, for NewIdeas / the orchestrator to weigh.

## BET 7 (PLAN900). The fused level kernel (NewIdeas 12:2x; pre-registered Mon 16 Nov 12:3x, before any code ran)
**Assignment (NewIdeas 12:2x):**
- (1) the dependent-step plan: <= ~35 dependent steps per level, against ~144 today;
- (2) a 1-level prototype, bitwise against the stock level on the full head;
- (3) measured fixed cost per level, and ms per block at tree-16 / tree-32 depths, at the full head and at a ~26 MB
  byte budget.

**Definitions.**
- **Stock level:** P2's C0 build, as timed in M0-M2 (p3_m012.py):
  - the shipped MTP head as stored (mtp.safetensors header):
    - bf16: fc_embedding, fc_hidden, the three hyper-connection mixes;
    - 4-bit g64: q / k / v / o and the shared expert;
    - 8-bit g64: the router and the indexer's qk projection;
    - 4-bit g32: the routed experts (top-10 of 512);
  - Leap's ScreenedHead (FR-Spec 32K screen, 4-bit g64);
  - M0's tree tail: logsumexp, top-K = 8 per row, frontier top-F by path score;
  - MLX 0.32.2 eager, engine flags off; 177.6 MB per level at F = 1 [K p3_m012_stock.log].
- **"Full head" for the bitwise check:** both the 32K screen and the full 248,320-row head.
- **Bitwise:** every array a level hands on, bit for bit:
  - the widened state hh;
  - the K / V cache rows, and the indexer's raw-key and pooled-bank rows;
  - the top-K ids and log-probs, the path scores, the parents and the tokens.
  - Checked over 21 chained levels x 3 contexts (T = 400 / 1,000 / 2,000 keys) x 3 seeds, on P2's synthetic states.
- **Scope of the prototype:**
  - F = 1 (one row per level) first; F = 2-3 are a stretch.
  - Dense attention only: T <= 2,048 keys (the indexer's budget, where QSA selects the full causal set). Above that
    the prototype hands the level back to the stock path.
- **f:** async ms per level (M1's protocol: sync-free, one sync per block) minus MB / BW, with BW = 292.7 GB/s (M2's
  async fit) [K p3_m012_stock.log]. If the fused level reaches a different BW, its own fit over the byte points it
  supports is reported beside it.
- **The ~26 MB point is a projection [P]:** levels x (f + 26 MB / BW).
  - No head that small exists or can be built from today's formats: M2's most aggressive cut, C8, is 39.8 MB.
  - "4-bit Linears + 8K screen" is C1 = 95.1 MB per level, measured [K M0-M2], not ~26.
  - FAR's 26 MB treated ~115 MB as the vocabulary side. The 32K screen is 47.2 MB; the rest is the routed experts
    (30.7 MB) and the parts already stored at 4 / 8 bits (~37 MB).

**The line.**
- **PASS (NewIdeas'):** bitwise AND f <= 0.056 ms per level at the stock head, in a quiet slot (medians over >= 20
  rounds).
- **BETWEEN (mine):** bitwise AND 0.056 < f <= 0.10. At f = 0.10, a <= 26 MB head would still draft a tree-32 block in
  3.9 ms, inside Kamen's 3-5 ms window [P: 20.8 x (0.10 + 26 / 292.7)].
- **FAIL:** not bitwise, or f > 0.10.
- Reported beside the verdict: ms per block at both depths, stock vs fused, at the full head (measured) and at 26 MB
  [P]. Levels per block: tree-16 12.6, tree-32 20.8 [K BIGGER_TREES.md].

**What the line buys [P, f + MB / BW per level]:**

| f (ms per level) | full head (177.6 MB): tree-16 / tree-32 | C1 (95.1 MB) | ~26 MB |
|---|---|---|---|
| 0.311 (stock, measured) | 11.6 / 19.1 | 8.0 / 13.2 | 5.0 / 8.3 |
| 0.10 (BETWEEN) | 8.9 / 14.7 | 5.4 / 8.8 | 2.4 / 3.9 |
| 0.056 (PASS) | 8.4 / 13.8 | 4.8 / 7.9 | 1.8 / 3.0 |

- So a <= 3 ms tree-32 block needs BOTH: f <= ~0.056 AND <= ~26 MB per level. With C1's bytes, the fused level gives
  7.9 ms.

**Sub-measurement M7a, the dispatch floor (information):**
- A chain of N dependent minimal kernels per level (N = 8 to 144), as stock MLX ops and as custom kernels, timed
  async per level as in M1.
- Reported: ms per dependent step (slope), the intercept, and the CPU-side cost per call.
- Under a share lock beside EvalSuite's gate it is INDICATIVE; the verdict uses a quiet slot.

**Expectation [G]:**
- The plan below needs ~15-17 dependent steps per level. At ~1.5-2.1 us per dependent dispatch, the GPU-side floor is
  ~0.025-0.035 ms, so PASS looks possible.
- Main risks:
  - (i) The CPU side: Python plus MLX's encode cost per custom-kernel call may bind when bytes are small.
  - (ii) Bitwise copies of MLX kernels whose sources aren't shipped: RMSNorm, and the host-side tile choices for gemv
    and the M = 4 GEMM (fc_hidden).
  - (iii) A multi-day build. Today: the plan, the dispatch floor and the first exact building blocks. The whole level
    comes later this week.

**Deliverable (1): the dependent-step plan (F = 1; desk, from qwen4_exp.py's decode path).**
- Today: 237 primitives per level, 144 that aren't reshape-type views [K p3_m012_stock.log, C1 graph]. That 144 is
  the "~144 dependent steps" in PLAN900; some of it (slices, transposes) still runs without a kernel, so the stock
  dispatch count is somewhat lower.
- Fused: 15 dependent kernels per level (16 if the stock SDPA takes its 2-pass form at that T).
- Each step is one custom kernel. Each threadgroup recomputes the small prologue it needs (a norm over <= 10,240
  values, the router's 512-way softmax and top-10) instead of waiting for a separate kernel. Independent matrices
  that read the same input go into one dispatch.

| step | one kernel | stock ops it absorbs |
|---|---|---|
| S1 | prep | embedding-row gather + dequant, both pre-fc RMSNorms, fc_embedding (M = 1), fc_hidden (M = 4), the add |
| S2 | attention HC, down | grouped RMSNorm x weight, down (320) and inject (4) bf16 GEMVs, silu(/4), 2 sigmoid(/4) |
| S3 | attention HC, up | up GEMV (10,240), sigmoid, x normed, the mean over 4 streams -> mixed |
| S4 | projections | q (12,288 incl. gate), k, v (512 each) at 4 bits, indexer qk (640) at 8 bits |
| S5 | attention | q / k RMSNorms, partial RoPE, K / V cache writes, the indexer's q prep, raw-key and pooled-bank writes, SDPA over T keys |
| S6 | o_proj | sigmoid output gate, 4-bit GEMV (2,560), the residual write h + out x inject |
| S7 | MLP HC, down | as S2, on the written stream |
| S8 | MLP HC, up | as S3 -> mixed2 |
| S9 | router + shared | router (512, 8-bit), shared gate / up (640, 4-bit) with silu x up, shared-expert gate (1) |
| S10 | routed gate / up | precise softmax + top-10 + renorm (per threadgroup), 10-expert gather GEMVs (4-bit g32), silu x up; shared down beside it |
| S11 | routed down | 10-expert gather GEMV, the expert-weighted sum, + shared x sigmoid(gate), the residual write -> hh |
| S12 | mixer, down | as S2, without inject |
| S13 | mixer, up | as S3 -> x |
| S14 | head | 4-bit GEMV over the 32K screen (or 248,320 rows) |
| S15 | tail | logsumexp, top-K in the sort's order, path scores, top-F, token ids, parents (one threadgroup) |

- **Exactness method:**
  - MLX's own device code where the headers ship it, inlined into the kernels: quantized.h (qmv, gather_qmv),
    gemv.h, sdpa_vector.h, softmax.h, logsumexp.h, steel GEMM.
  - RMSNorm isn't shipped: rebuilt in the stock order and bit-tested.
  - Elementwise ops use MLX's own functors at the stock dtypes, rounding to bf16 at every op boundary the stock graph
    has (compiled and eager bf16 sigmoid differ at some inputs, per mtplx's attention_math notes; the stock here is
    eager).
  - Every step is bit-tested alone against the stock ops it replaces before it joins the level.
- **Build order (riskiest first):**
  1. the exact RMSNorm;
  2. S2 / S3 (the HC read, used three times per level);
  3. the qmv steps (S4, S6, S9);
  4. the MoE (S10, S11);
  5. attention (S5);
  6. S1 (the M = 4 GEMM);
  7. head and tail (S14, S15);
  8. the whole level.
- **Reuse:** none of mtplx's kernels as-is. Its exact ones are fixed M = 4 (verify widths); its M = 1 HC read is "not
  bit-identical" by its own docstring. Its notes on stock order are read as references.

**M7a, RESULT: the dispatch floor (quiet slot 12:46, NewIdeas'; information)** [K ~/MoonShots/week98/bet7/
fl_floor_quiet.log / .json; GPU busy 0 % at the grant; 30 rounds, 2 dropped]
- Chains of N = 8-144 dependent tiny kernels per level, async per level, cost per step on the wall clock:
  - stock MLX op 1.31 us;
  - custom Metal kernel 2.04 us (2.07 at 64 threadgroups);
  - the custom chain under mx.compile 1.38 us;
  - intercepts 0.012-0.015 ms per level.
- In every arm, CPU time (graph building + async_eval) tracks the wall within 2 %. The chains are CPU-bound, so:
  - a tiny dependent dispatch costs <= 1.3 us on the GPU (an upper bound);
  - a custom-kernel call costs ~2.0 us on the CPU, ~0.7 us more than a stock op (compile takes it back to 1.4).
- **The stock C0 level at F = 1: 0.954 ms per level** (M1: 0.921), 0.348 ms above its 177.6 MB at 292.7 GB/s.
  - It is GPU-bound. C8 (the same ops on fewer bytes) ran at 0.46 ms in M2, so the level's CPU work is <= ~0.46 ms.
    The 0.72 ms spent inside async_eval at C0 is waiting on the GPU.
- **So today's ~0.31-0.35 ms isn't all launch overhead.**
  - Dispatches account for <= 144 x 1.3 us = <= 0.19 ms.
  - The rest (>= ~0.16 ms) is work off the bandwidth line: the sort-based top-K over the 32K logits, single-threadgroup
    reductions, GEMVs below full bandwidth.
- **For the fused level:**
  - 15 calls are ~0.03-0.06 ms of CPU per level, hidden behind the GPU at the full head;
  - <= 15 x 1.3 us = <= 0.02 ms of dispatches on the GPU.
  - PASS therefore turns on the kernels' own efficiency (GEMVs at full bandwidth, a fast tail) as much as on the step
    count. That is the design constraint now, beside bitwise.

**Deliverable (2), RESULT: the fused level is BITWISE equal to the stock level (Mon 13:1x, 4646311)**
[K ~/MoonShots/week98/bet7/fl_bittest_level.json, fl_bittest_steps.json; code fl_fused.py, fl_hc.py, fl_moe.py,
fl_attn.py, fl_level.py]
- **The whole level:** 21 chained levels x 2 start tokens, at 400, 1,010 and 2,000 cached keys. The 1,010 run crosses
  the stock SDPA's switch from 1 pass to 2 passes at 1,024 keys.
  - Bit-equal: every array a level hands on (hh, token, path score, top-K ids and log-probs, candidates, frontier
    pick) and every cache row the levels wrote (KV keys and values, the indexer's raw keys and pooled blocks).
- **17 dependent kernels per level** (18 at >= 1,024 keys), against 144 stock primitives:
  - S1 prep; S2-S3 attention HC read; S4 projections; S5 attention (+1 kernel at >= 1,024 keys); S6 o_proj;
  - S7-S8 MLP HC read; S9 / S9b / S10 / S11 MoE; S12-S13 mixer; S14 head; S15 tail.
  - The cache maintenance stays the stock's own code, run on the fused outputs (off the critical path).
- **Each step was bit-tested alone first:**
  - HC read: all three modules, every output;
  - MoE: 40 trials (router logits, top-10 ids in order, output);
  - attention: 5 cache lengths across both SDPA forms (output after o_proj, the new key row);
  - S1 / S4 / S6 / S14 / S15: 12 trials each.
- **The stock's kernel choices, pinned by bit-matching candidates built from MLX's shipped headers** (MLX's dispatch
  code isn't shipped):
  - bf16 GEMVs: HC down / inject = gemv BN8 SN32 TN4; fc_embedding / HC up = BN1 SN32 TN4; fc_hidden (M = 4) =
    gemv_wide k_lanes 32;
  - quantized: qmv_fast for q / k / v / o, the router, the shared gate / up and the head; plain qmv for the shared
    down (K = 640) and the shared-expert gate (N = 1);
  - SDPA: 1 pass below 1,024 keys; 2 passes with 64 blocks at exactly 1,024 and 128 above (the gqa form is unused);
  - reductions: the HC mean and the expert combine = col_reduce with 8 threads along the rows (rows j, j + 8 per
    thread, partials folded in order); the score sum is a bf16 running sum; top-k = argpartition's stable merge
    sort, i.e. the k largest by (value, index) in ascending order;
  - elementwise: standalone sigmoid = precise exp / divide; compiled silu / swiglu = the JIT functor (fast exp); the
    two differ on 1 of 65,280 bf16 inputs.
- **Sensitivity:** the wrong mean orders for the HC read fail on ~30 % of elements. Ambiguous GEMV matches resolved
  only with cancellation inputs (random +-H column pairs); plain random inputs let 17-72 candidates through.
- **Negative control on the whole level** [K fl_bittest_level_control.json]: the HC reads' 4-stream mean in float
  order instead of the stock bf16 order (one rounding step) FAILS at once.
  - hh, path scores, log-probs and candidates differ from level 0, the top-K ids by level 0-1, the tokens by level
    3-5, and every cache row differs.
  - So the 21-level comparison would have caught any order slip. It also shows how fast a last-bit difference reaches
    the head's picks: within 3-5 levels.
- **Efficiency reworks, each re-verified bitwise (710151d):** S15's top-K merge (2 barriers, not 24), the router's
  top-10 as packed keys, and Fused2 (the next level's input norms folded into S15, so S1 is pure GEMVs).

**Deliverable (3), RESULT: FAIL. Bitwise, but the best fused level's fixed cost is f = 0.269 ms per level (stock
0.353), against PASS <= 0.056 and BETWEEN <= 0.10 (quiet slot Mon 14:26, in EvalSuite's idle pause 2)**
[K ~/MoonShots/week98/bet7/fl_time_quiet.log / .json, fl_prof_quiet.log / .json; GPU busy 0 % at the grant; 30
rounds, 2 dropped, L = 21, async per level; the fused tokens equal the stock tokens in every timed block at both cache
lengths]

| arm (F = 1) | ms per level, 400 keys, full head (177.6 MB) [p10-p90] | f = ms - MB / 292.7 | 8K screen (142.2 MB) | own two-point fit | 1,500 keys, full head (f) |
|---|---|---|---|---|---|
| stock (144 primitives) | 0.960 [0.935-0.978] | 0.353 | 0.795 | 215 GB/s, f 0.134 | 0.996 (0.390) |
| fused (17 kernels) | 1.052 [1.043-1.070] | 0.445 | 0.882 | 207 GB/s, f 0.196 | 1.134 (0.527) |
| fused2 (S1 as pure GEMVs) | 0.876 [0.870-0.898] | 0.269 | 0.735 | 251 GB/s, f 0.168 | 0.956 (0.349) |

- **Verdict on fused2, the best arm:** bitwise yes; f 0.269 > 0.10, so FAIL. Its own fit's intercept (0.168) fails
  too.
- It is still an exact engine gain: -8.7 % per level at 400 keys, -4.0 % at 1,500.
- **ms per block at 400 keys** (levels x ms per level; full head measured per level; C1 and ~26 MB [P] = levels x
  (f + MB / 292.7)):

| arm | full head: tree-16 / tree-32 | C1 (95.1 MB) [P] | ~26 MB [P] |
|---|---|---|---|
| stock | 12.1 / 20.0 | 8.5 / 14.1 | 5.6 / 9.2 |
| fused2 | 11.0 / 18.2 | 7.5 / 12.4 | 4.5 / 7.4 |

  - At f = 0.27, a tree-32 block's fixed cost alone is 20.8 x 0.27 = 5.6 ms, above the 3-5 ms window at any byte
    budget. Tree-16 at ~26 MB [P] (4.5 ms) is the only cell inside it.
  - This replaces M3's "~13.8 ms" for the hand-fused level at tree-32 (a projection at f = 0.056): measured 18.2 ms.
- **Where f goes (the profile: each step alone, 30 back-to-back calls on the same inputs; weights are cache-warm and
  calls overlap, so its GB/s are upper bounds):**
  - S1 with its norm prologue: 0.257 ms for 26.2 MB (102 GB/s). That is why plain fused lost to stock. Fused2's
    pure-GEMV S1G takes 0.064 ms; the folded norms add ~0.001 ms to the tail.
  - Latency-bound, no weight bytes: S5 attention 0.036 ms at 400 keys; the tail (S15P) 0.031 ms; S0 0.026 ms (a
    block's first level only).
  - Below the line even warm: the MoE, 0.124 ms for 34.9 MB (281 GB/s). At or above it warm: S6 295, the HC reads
    321-340, S14 head 361, S1G 411, S4 433 GB/s.
  - Attention at 1,500 keys: fused2 rises by 0.080 ms per level from 400 keys, the stock by 0.037. The fused 2-pass
    attention is ~0.04 ms slower than the stock's there.
  - **The chain:** the isolated steps sum to 0.584 ms (fused2) and 0.776 ms (fused), against 0.876 and 1.052 ms
    chained. So ~0.28-0.29 ms per level is lost between steps: each of the 17-18 whole-GPU kernels starts only once the
    previous one has drained, then reads its weights cold. That is ~16 us per boundary, ten times M7a's <= 1.3 us
    per tiny dispatch [G: the split between drain and cold weights isn't measured].
- **So the step count wasn't the lever.** 144 -> 17 kernels took f from 0.353 to 0.269 (-0.084 ms). What's left is
  ~0.07 ms of latency-bound steps and ~0.2 ms of chain and below-line GEMVs [G].
- **What could reach 0.10 [G]:**
  - (a) A persistent kernel per level that loads each step's weights before its input is ready (the weights don't
    depend on the data). Prior art on CUDA: megakernels (Hazy Research's Llama-1B, Mirage's MPK). Metal has no
    grid-wide sync, so this needs co-resident threadgroups and spin barriers: a risky multi-day build.
  - (b) Fewer dependent levels per block: F > 1 rows per level share each weight read.
  - A cheap check before (a), ~1 min quiet: the same 17 kernels with their data dependencies cut (each step fed a
    precomputed input, so the encoder overlaps them). If the level drops toward ~0.6 ms, the loss is the boundaries
    and prefetch can recover it; if it stays near 0.88, it is the kernels' cold-weight rates.

**NewIdeas' call (14:3x):**
- (c) Bank fused2 as an exact engine item (-8.7 % per level at 400 keys, -4.0 % at 1,500) for the engine owners'
  list, with a600df5.
- (b) F > 1 is a head-architecture change with an accuracy cost: parked for MTP Head's list (after S13 / S14).
- (a) The check runs now, in EvalSuite's idle pause 2.

**(a)-CHECK, pre-registered (NewIdeas' line, 14:3x, before the run):**
- The test arm is `cut` (fl_overlap.py): the same steps in the same order, each fed the reference pass's inputs. No
  data dependency runs between steps; within a step (HC down -> up, the MoE's four) the kernels stay dependent.
  T0 = 400, full head, F = 1, L = 21, async per level, 30 rounds (2 dropped), interleaved with three more arms:
  - `stock` (anchor 0.960 ms);
  - `fused2` (anchor 0.876 ms, with the cache maintenance);
  - `chain` (fused2's kernels chained, without the maintenance; T fixed).
- **PASS** (prefetch / overlap can recover the loss): `cut` <= 0.65 ms per level (>= ~80 % of the 0.29 ms
  recovered). **KILL:** >= 0.80 ms. **BETWEEN** otherwise. Only a PASS justifies the multi-day persistent-kernel
  build.
- Reported beside it (my additions, not the line):
  - the maintenance's cost (fused2 - chain), and `cut` plus that cost, since a persistent kernel would have to absorb
    or hide it;
  - chain - cut, the dependency cost.
  - `cut` is optimistic for prefetch: overlap also hides the latency-bound steps (attention, tail), which prefetch
    can't.
- Checks: chain's level 0 equals fused2's level 0 bitwise; cut's outputs equal the reference pass's.

**(a)-CHECK, RESULT: BETWEEN. With no dependencies between steps, a level takes 0.699 ms (PASS <= 0.65, KILL >=
0.80); quiet slot Mon 14:34**
[K ~/MoonShots/week98/bet7/fl_overlap_quiet.log / .json; GPU busy 0 % at the grant; both checks True]

| arm (400 keys, full head, F = 1) | ms per level [p10-p90] | f = ms - 177.6 MB / 292.7 |
|---|---|---|
| stock (anchor 0.960) | 0.949 [0.929-0.966] | 0.342 |
| fused2 (anchor 0.876) | 0.887 [0.874-0.908] | 0.280 |
| chain (no cache maintenance) | 0.870 [0.858-0.886] | 0.263 |
| cut (no dependency between steps) | 0.699 [0.680-0.717] | 0.093 |

- **The split:** the cache maintenance costs 0.017 ms per level (fused2 - chain); the dependencies between steps cost
  0.171 ms (chain - cut). Fused2 -> cut recovers 0.188 ms, ~65 % of the ~0.29 ms chain loss (the line asked for
  >= ~80 %).
- **What it bounds:** even fully overlapped, this level's steps reach 0.70 ms, an effective 254 GB/s over its
  177.6 MB.
  - So cross-step prefetch in a persistent kernel can at best take fused2 from f 0.28 to ~0.09-0.11, i.e. BET 7's
    BETWEEN line (0.10), not PASS (0.056) [G].
  - `cut` is optimistic: it also hides the latency-bound attention and tail behind the GEMVs. It is pessimistic in
    one way only: the within-step chains (three HC down -> up pairs, the MoE's four kernels) stay dependent.
- **What f ~0.09 would buy [P, levels x (f + MB / 292.7)]:**
  - tree-32 at the full head 14.5 ms;
  - at ~26 MB, 3.8 ms (tree-16 2.3), inside the 3-5 ms window. Only with a <= ~26 MB head, which doesn't exist yet.

**NewIdeas' call (14:4x):** the persistent kernel is NOT NOW. Its best case is BETWEEN and it only pays with a
<= ~26 MB head; revisit after MTP Head's S13 QAT verdict. Optional and cheap: (a') prefetch without a persistent
kernel.

**(a') PREFETCH BY TOUCH KERNELS, pre-registered (NewIdeas' line, 14:4x, before the timing run):**
- **Build (fl_touch.py):** fused2 plus a touch kernel per step that reads the next step's weights (one 4-byte load
  per 64 bytes) while the current step runs.
  - Ordering without a barrier: the touch takes the overlapped step's input as an unused anchor input, and
    mx.depends ties the touch into the next step's input, so the next step's barrier waits for both. MLX 0.32's
    encoder inserts a barrier only before a kernel that reads a buffer written since the last barrier.
  - 9 touch sets per level: attention HC (during S1G), S4 (during the HC read), o_proj (during attention), MLP HC
    (during o_proj), router + shared expert (during the MLP HC read), the 10 routed experts' down weights (during S10,
    from S9b's ids), mixer HC (during S11), head (during the mixer HC read), the next level's S1G (during S15P).
  - Sets measured: S1G 26.2, HC 13.1-13.2, S4 20.9, o_proj 8.85, router + shared 4.16, head 47.2 MB; the experts'
    down ~10 MB. The other routed weights can't be known in time.
- **Arms** (interleaved, 30 rounds, 2 dropped, L = 21, T0 = 400, full head, F = 1):
  - `stock`, `fused2` (anchors);
  - `pre` (whole sets);
  - `cap` (each touch capped at 4 MB, a prefix: about the first wave of threadgroups);
  - `late`, the control: the same 9 sets, each touched while its own consumer runs (the same extra reads, no
    prefetch benefit).
- **The line, on the better of `pre` and `cap` (both reported):**
  - PASS <= 0.76 ms per level (>= 2/3 of cut's 0.188 ms recovered);
  - KILL >= 0.85;
  - BETWEEN otherwise.
- **Risk (NewIdeas):** double DRAM reads if the cache can't hold a step's weights until the step reads them. That
  matters at the full head (~10-47 MB per touch); `cap` and `late` show it.
- **Checks:** smoke PASS at 14:45 (`pre`, `cap` and `late` bit-equal to fused2 over 21 levels: hh, top-K ids,
  log-probs, tokens) [K fl_touch_smoke.log]; in the timing, every arm's tokens equal stock's in every block.

**(a') RESULT: KILL. Touch prefetch makes the level slower: best arm `cap` 0.935 ms per level, against fused2's
0.860 in the same slot (quiet, Mon 14:46)**
[K ~/MoonShots/week98/bet7/fl_touch_quiet.log / .json; GPU busy 0 % at the grant; tokens equal stock's in every block,
every arm]

| arm (400 keys, full head, F = 1) | ms per level [p10-p90] | vs fused2 |
|---|---|---|
| stock | 0.922 [0.914-0.929] | +0.062 |
| fused2 | 0.860 [0.856-0.868] | |
| pre (9 whole sets, ~157 MB touched per level) | 1.312 [1.308-1.321] | +0.452 |
| cap (4 MB per touch, ~36 MB per level) | 0.935 [0.931-0.950] | +0.075 |
| late (control: the same 9 sets, each during its own step) | 1.165 [1.159-1.176] | +0.305 |

- **The touched lines don't survive until the next step reads them.**
  - `pre` costs +0.45 ms against the ~0.53 ms its ~157 MB of extra reads take at 292.7 GB/s, so nearly all of it is
    extra DRAM traffic.
  - `pre` is even 0.15 ms worse than the control (`late`), whose touches share some lines with the step reading them
    at the same moment.
  - So on this chip the system cache doesn't hold a step's weights (4-47 MB) ahead of its consumer. The extra reads
    compete for the DRAM bandwidth the chain already lacks.
- **The first-wave prefetch (`cap`) doesn't pay either:** +0.075 ms for ~36 MB of touches.
- **So the chain loss can't be recovered through the cache.** Only a kernel that holds the next step's first tiles
  on-chip (the persistent kernel, NOT NOW) could still try. Its ceiling remains the (a)-check's 0.699 ms.
- **Implementation note:** the barrier-free ordering (an unused anchor input + mx.depends) is reusable. Every arm was
  bit-equal to fused2, and the touches did run: the arms' costs differ by placement.

## MODULE 2 (Leap's split, 11-16). The lossy expert one-call: where the time goes (diagnostic, quiet, Mon 22:43)
Leap's line (exp32_bench default, per 32-row call on ~105 MB, quiet): PASS <= 400 us, BETWEEN 400-480, KILL > 480; today
571-614. Bits: == ooo1cq on ooo1c_m32_check's 24 scenarios and exp32's 24 real blocks. This window is a diagnostic
(NewIdeas' quiet ticket 22:43-22:44, GPU 0 % at the grant, away 1,521 s); Leap's verdict bench is Wed.
[K ~/MoonShots/week98/m2/m2_prof_w1.{txt,json}, m2_abl_w1.*, m2_bits_w1*.*; layer 20, all 512 resident, exp32's blocks]

**Per stage (24 blocks, each stage alone in a command buffer, GPU us):** grp 19.3, rowsum64(x) 3.3, kernel A 388.3,
rowsum64(h) 4.3, kernel B 197.9, combine 9.2; whole call in one command buffer 615.7. exp32's way: 582.6 us per call wall,
534.3 GPU busy; 8.0 dispatches, 2.7 barriers, 2.88 command buffers per call. Bytes per call: 138.2 MB read by the
segments (A 92.1, B 46.1), 105.6 MB distinct (62.1 segments, ~5.2 pairs each).

**Fixed cost or rate? Neither: the kernels are compute-bound.** (8-block ablations, GPU us, A / B)

| variant | A | B |
|---|---|---|
| base (== ooo1cq) | 351.9 | 180.9 |
| no code loads (decode + MMA kept) | 331.0 | 168.9 |
| no activation loads | 340.6 | 173.4 |
| no decode | 318.5 | 166.3 |
| no MMA (activation loads die with it) | 247.1 | 125.1 |
| pure read of the same bytes | 234.0 | 114.8 |
| every threadgroup returns at once | 11.2 | 13.7 |

- The pure reads stream the segments' bytes at 370 / 378 GB/s, i.e. ~285-300 GB/s on distinct bytes: DRAM peak, with
  the tier re-reads served from cache. That is the memory floor: ~349 us for A + B on these blocks.
- Without its code loads A still takes 331 us: the memory adds only ~21 us on top of the compute. The MMA (+ the
  activation loads) costs ~105 / 56 us, the decode ~33 / 15. The MMA runs 16-row tiles on ~5.2 pairs per segment
  (~2/3 padding).
- The fit fixed + MB / rate across blocks is uninformative (R2 0.08-0.15: the blocks span too few MB).
- Fixed parts on the critical path: grp 19 + rowsum64(h) 4 + combine 9 us; empty threadgroups (grid y = P = 320 rows)
  ~10 (A) + ~3.5 (B) us.

**Every bit-exact form tried is SLOWER than ooo1cq** (exp32's way, us per call, two interleaved passes; all == ooo1cq bit
for bit on both gates, negative controls fail): ref 584.0 / 580.0; C3B (B in tile rounds) 585.6 / 585.5; C1 (register
prefetch) 598.1 / 597.0; C3 (A: 2 tiles per threadgroup + writes rowsum64(h)) 611.0 / 610.4; C2 (grid capped,
segment loop) 611.2 / 611.9; C1d2 635.8 / 629.2; C1C3C3BC2 640.2 / 645.6; C1x (activation prefetch) 781.6 / 781.9.
C4 (the combine folded into B by counters) on the graph way: 573.1 vs ref 569.2. Round / cap variants (TPB 2 / 8,
TPT 4, GY 128) don't change the picture. Reading: in a compute-bound kernel, extra registers (prefetch), loops (C2) or
fewer threadgroups (C3, C3B) cost occupancy; that outweighs what they save.

**What this means for the line:** at today's bytes the call's floor is ~364 us for A + B (105.6 MB at ~290 GB/s) +
~32 us of grp / rowsum64(h) / combine on the critical path + exp32's ~48 us of wall over GPU busy = ~444 us (exp32's
way), even with the compute fully hidden under the memory. So kernel work alone can reach BETWEEN (~430-450), not PASS;
PASS needs the wall overhead gone (module 1) or fewer bytes as well. Next levers, all bit-exact in principle: smaller MMA
tiles for <= 8-pair segments (the padding), a cheaper decode, a faster grouping kernel (19 us for one threadgroup).
- **Building block for the first lever (22:5x, m2_mma8.py, 256 random tiles with cancellation columns):** MPP's
  matmul2d at M = 8 (rows 0-7, then 8-15) gives every row's 32 outputs bit-identical to M = 16 (0 of 4,096 rows
  differ; both within 1.1e-5 of fp64 at scale 268), with the kernels' operand layouts and two K halves accumulated.
  So a <= 8-pair segment can run an 8-row tile and keep ooo1cq's bits; its speed is the open question.
  [K ~/MoonShots/week98/m2/m2_mma8.log]
- **Timing screen (Leap 22:5x: time first; quiet, 22:51 inside NewIdeas' slot, away 2,012 s): 8-row tiles run
  PADDED. M8T (the MMA at M = 8, timing only) gives kernel A 396.2 vs 388.8 us, exp32's way 599.8 / 599.4 vs
  584.3 / 581.3 us per call. The accelerator's cost doesn't fall below a 16-row tile, so lever (1) is dead.
  [K ~/MoonShots/week98/m2/m2_bits_m8t.txt]
  - Reading: the MMA's ~94 / 48 us is not overlapped with the streaming (base ~ memory floor + MMA).
    The bit-exact levers left are small: the decode, at most ~33 + 15 us per call. The big one, overlapping the
    MMA with the loads, means restructuring the kernels' threadgroup memory: the epilogue reuses the double
    code buffer for the chunk partials, so occupancy can't rise without a rewrite. Or a FAST (not bit-exact)
    small-segment path.

### MODULE 2, FAST small-segment path: paper cost (Leap 00:5x, desk only; Tue 07:5x) = KILL on paper
- **Real histogram** [K m2_seghist.py, CPU replica of _GRP1CQ on exp32's 99 blocks; matches the GPU profile: 60.9
  segments per block, 5.25 pairs each]: 1 pair 8 %, 2 pairs 15 %. Segments with <= 2 pairs are 23 % of segments and
  7.2 % of pairs; <= 4 pairs 55 %; <= 8 pairs 84 %.
- **MMA path:** ~2.43 us per segment, A + B [K: no-MMA saves 94 + 48 us on 58.5 segments, x loads excluded]. The cost is
  fixed per 16-row tile (the 8-row screen).
- **SIMD-FMA path:** 4.92 M FMAs per pair (A 2 x 640 x 2560 + B 2560 x 640). At ~4 T FP32 FMA/s [G: 20 cores x 128 lanes
  x ~1.6 GHz] that is ~1.2 us per pair, plus converting each segment's nibbles to float (~1.3 pair-equivalents per
  segment).
- **Break-even:** FMA wins only for 1-2-pair segments, and only if the conversion is free: <= ~6 us per call (~1 %),
  ~0.3 ms per 48-layer check; likely <= 0 with the conversion [P].
- **Overlap upside:** up to ~34 us per call (~6 %) only if ALU and MMA work run fully concurrently, which the
  ablations argue against (decode and MMA costs add) [G].
- **Verdict:** KILL on paper; no screen. It is also under the orchestrator's 07:3x "no +5-50 % levers" rule.

## Q8. Is post-hoc 8-bit free on the 125B? (NewIdeas 21:1x; pre-registered Mon 16 Nov 21:2x, before any Q8 run)
- **Why (NewIdeas):**
  - MTP Head's sandbox S17 (LAB, s30) found post-hoc 8-bit free: tree-16 / 32 MBPP x0.997 / x1.001, HE x0.995 / x1.000.
  - 4-bit fails even with QAT (S16).
  - The 125B's 4-bit loss (M3's Q4: x0.922 HE+ / x0.899 MBPP at tree-32) matched the sandbox's, so 8-bit is likely free
    on the 125B too. Saturday's numbers still need it measured.
  - Under PUSH800 the head plan is 8-bit Linears + the 8K screen (~42 MB per level).
- **Run:** M3's harness (p3_dd_tree.py; gate PASS Mon 08:4x: --cut none == BigMoE's tree_t30b dumps bitwise) with a new
  cut Q8 = Q4 with every 4 -> 8.
  - Fake-quantized bf16 -> affine g64 8-bit -> dequantized: t30b's trained Linears (q / k / v / o, the shared expert,
    fc_hidden / fc_embedding) and the head's bf16 HC mixes. Router at 8, as in Q4.
  - Full head, experts top-10.
  - Both sets: heplus24 (24 windows) and mbppbank (100 windows).
  - TRUE-STATE (lab): teacher-forced, the 125B's true state at every anchor.
- **Score:** `p3_score.py --cuts none,Q4,Q8 --tag q8`.
  - Kept per check at rows 16 and 32 under certify98's walk, against the full head (BigMoE's tree_t30b dumps).
  - Window-clustered 95 % CI (2,000 resamples, seed 1115).
  - Q4 is re-scored beside it as the anchor: it must reproduce M3's x0.9626 / x0.9224 HE+ and x0.9376 / x0.8993 MBPP
    (rows 16 / 32).
- **Line (NewIdeas):** PASS = Q8 >= 0.99x at tree-16 AND tree-32 on both sets (point estimates; CIs reported beside
  them). FAIL otherwise, naming the failing cells.
- **Baseline note:** the reference is t30b's fp32 checkpoint as BigMoE dumped it (M3's "none"), not a bf16 copy. Q8's
  fake-quantizer casts to bf16 first, so bf16 rounding is inside Q8's delta.
- **Slot:** Wed 18 Nov 06:30 (NewIdeas: the released QAT slot, under the away rule).
  - It must finish before the band-screen fallback (latest start 07:15), so no dump starts after 07:10.
  - ~12 GB peak, ~30 min, under a share lock "MEM 12 GB".
  - Earlier only in a >= 12 GB window that NewIdeas or the orchestrator opens.
  - Runner week98/p3_q8.sh. Outputs: p3/q8_run.log, p3/dump_Q8_*.log, p3/tree_t30b_*_Q8.{npz,json} and
    p3/q8_scores.{txt,json}.
- **Context, not part of the line:** the 8K screen alone (M3's H8) costs x0.963 HE+ / x0.985 MBPP at tree-32. If Q8
  passes, the plan's 8-bit + 8K head is bounded by the screen, not by the bits.
- **Amendment (NewIdeas 22:3x, before any run): the screen's size, beside Q8.**
  - Why (NewIdeas): MTP Head's sandbox S19 (static Magicoder rank, no session extras) gave 8K x0.968-0.983,
    16K x0.982-0.989 (FAIL), 32K x0.995-0.999 (PASS). Training under the screen doesn't help (S18), so the screen's size
    sets its cost. 8K vs 32K is near-neutral in tok/s on NewIdeas' arithmetic (+31 MB per level), so the 125B's own
    H8 vs H32 (+ Q8) decides the head plan.
  - New cuts (fp32 weights unless stacked): H32 = FR-Spec top-32K + the session extras (screen_masks, cap 4096, as H8);
    H16 = the same at top-16K; Q8H32 = Q8's weights with the 32K screen.
  - Order (NewIdeas' priority): Q8 > H32 > H16, then Q8H32 if time is left; heplus24 then mbppbank per cut; no dump
    starts after 07:10. A cut cut short reports the sets it has.
  - Line, per cut (as Q8): PASS = >= 0.99x at tree-16 AND tree-32 on both sets vs the full head (point estimates, CIs
    beside). M3's H8 is re-scored beside them (must reproduce x0.9759 / x0.9631 HE+, x0.9896 / x0.9847 MBPP).
  - Start: 06:30, or earlier if the Mac is free (no other session's quiet lock held or queued, no other GPU job over
    4 GB, >= 20 GB available, away >= 900 s), per NewIdeas' "or earlier in any >= 12 GB window".

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
