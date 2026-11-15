# LOSSY_CHECK_PLAN: a 32-row check under 25 ms at 45 GB (WEEK98 goal B, lossy)

Breakthrough, Sun 15 Nov 2026 00:4x-01:0x. CPU only, from data on disk.
- Pre-registered: Breakthrough docs/LOSSY_CHECK_PRICE.md (f4d546b).
- Script and data: Breakthrough scripts/m1/lossy_check_price.py, data/m1/lossy_check_price.out.
- Labels: [K] = measured, [P] = priced / predicted.

## Bottom line

0. **Pre-registered verdict (all three domains pooled): KILL.** No lever reaches <= 25 ms with predicted flips at or
   within 1.5x the levels on code, IFEval and multi together. Three parts of the reading below are post-hoc and labelled
   so:
   - the code-scoped reading;
   - the recalibrated low-bit error (the pre-registered t3_all check failed);
   - the non-tie view.
1. **On HumanEval-like code, < 25 ms is reachable at the bandwidth floor within the map's quality levels**, with one
   combined lever: acceptance-weighted bit-plane tiers (AcceptMoE-style weighting + HOBBIT-style low bits + drops).
   - Today's drafter: 6.43 GB, 23.0 ms at the floor.
   - 98 % drafter: 6.88 GB, 24.6 ms at the floor.
   - Predicted code flips: 1.21-1.30 % against the 1.34 % level; non-tie flips 0.59-0.66 % against 0.87 % [P].
2. **At a realistic kernel efficiency (84 % of 280 GB/s) the same lever gives 27-29 ms.** 25 ms needs kernels at
   ~92-98 % of bandwidth, or about 0.7-1.0 GB less dense reading. The dense part (3.44 GB = 12.3 ms at the floor) is
   half the budget, and the lossy expert levers can't touch it.
3. **With chat in the gate, the limit is ~32 ms at the floor, ~38.5 ms at 84 %.** IFEval and multi flip 5-10x more than
   code per unit of missed energy, so a pooled same-scores gate binds there. The 25 ms target holds only with a
   code-scoped gate, or with the lossy check switched on for code only.
4. **Rank by price:**
   1. acceptance-weighted tiers;
   2. tiers alone;
   3. union caps;
   4. close-call acceptance, a small multiplier on tokens per check that shares the quality budget.
5. **The low-bit error must be calibrated, not assumed.** The map's t3_all (MSB-3 on every expert) flips 3-6x more than
   rd_sim's 0.02 error predicts. Net of the bit-plane path's own noise (t4_all), MSB-3 behaves like an error of ~0.08
   per w^2 [K-derived]. This halves the value of the bit-plane tiers compared with a naive HOBBIT reading.

## The byte budget [K: Leap's model, data/week98/cheapcheck.md]

| | GB per 32-row check | ms at the floor (280 GB/s) | ms at 84 % |
|---|---|---|---|
| exact (dense 3.44 + 99.9 distinct experts per layer x 48 x 2.135 MB) | 13.68 | 48.9 | 58.2 |
| the 25 ms target | 7.0 | 25.0 | 29.8 |
| 25 ms at 84 % | 5.88 | 21.0 | 25.0 |

- Dense stays at 3.44 GB: BigMoE's D2 / D3 screens (Wed) killed a 2- or 3-bit trunk.
- So the experts must fit 3.56 GB at the floor or 2.44 GB at 84 %: 34.7 or 23.8 four-bit-equivalent experts per layer,
  against 99.9 exact.
- The exact union reproduces Leap's 100.5 within 1 % (36 map windows tiled into 299 full 32-row blocks).

## Quality model [K-calibrated, then P]

- Per row, the missed energy is E = sum over layers and ranks of err x w^2 / 48 (w: top-10-normalised routing
  weights, the map's xcap capture), with err 1 for a drop and 0.08 for MSB-3.
- Flips = a E^b per domain, fitted on 7 measured drop arms (topk2 / 5 / 8, q1u_cf09 / cf099, qom_0_live_c51 / c147).
  Log-residual rms: 0.10-0.26 on all rows, 0.12-0.30 on non-tie rows.
- Out-of-sample: t3_all is predicted within ~10 % on IFEval and multi once its path noise is added.
- Lines: the map's verdict levels. All rows: code 1.34 %, IFEval 5.42 %, multi 3.69 %. Non-tie (p1 - p2 > 0.1): 0.87
  / 2.22 / 1.63 %. Tie rows (5.7 %) flip under any numeric change; they are the close calls.

## Ranked levers (a 32-row check; tok/s on HumanEval+ incl. ~7 ms drafting, kept 13.8 today / 23.3 for a 98 % drafter)

| rank | lever | 4-bit-eq experts / layer | GB | ms floor / 84 % | tok/s at 84 % (today / 98 %) | predicted flips: code all / non-tie | IFEval / multi all | quality risk | gate needed |
|---|---|---|---|---|---|---|---|---|---|
| 1a | acceptance-weighted tiers, today's drafter (s 0.935, lambda 1.2e-2) | 29.2 | 6.43 | **23.0 / 27.3** | 402 / 679 | 1.30 / 0.66 | 10.4 / 8.8 | code at its levels; chat over | code-scoped |
| 1b | the same, 98 % drafter (s 0.98, lambda 1.8e-2) | 33.5 | 6.88 | **24.6 / 29.2** | 381 / 643 | 1.21 / 0.59 | 9.6 / 8.1 | code at its levels; chat over | code-scoped |
| 1c | the same, three tiers (4-bit / MSB-3 / drop), s 0.98, lambda 1.6e-2 | 36.4 | 7.17 | 25.6 / 30.5 | 368 / 622 | 1.08 / 0.50 | 8.7 / 7.4 | as 1b, simpler | code-scoped |
| 1d | the same, pooled-safe (s 0.98, lambda 8e-3) | 54.8 | 9.05 | 32.3 / 38.5 | 303 / 512 | 0.51 / 0.15 | 4.9 / 3.9 (non-tie 1.5 / 1.3) | within non-tie; all-row <= 1.5x | pooled (EvalSuite tier 1) |
| 2 | bit-plane tiers alone, no acceptance weight (lambda 1.6e-2 code; 8e-3 pooled) | 44.1 / 63.9 | 7.96 / 9.99 | 28.4 / 33.8; 35.7 / 42.5 | 338 / 571; 279 / 471 | 0.81 / 0.32; 0.33 / 0.08 | 6.9 / 5.8; 3.7 / 2.9 | low | as 1 |
| 3 | union caps alone (cap 40 code; cap 60 pooled) | 39.9 / 58.8 | 7.53 / 9.47 | 26.9 / 32.0; 33.8 / 40.3 | 354 / 597; 292 / 493 | 1.20 / 0.58; 0.67 / 0.24 | 9.6 / 6.6; 5.3 / 3.4 | drops only: the steepest | as 1 |
| 4 | close-call acceptance (Thu's price) on top of any lever | (no bytes) | - | - | x1.01-1.03 at delta 0.05; x1.02-1.18 at rho 0.25 (code, 32-row) | adds ~-1 pt IFEval (delta 0.05) to -7 to -9 (rho 0.25) | - | shares the budget | same gate, one budget |

- **(a) AcceptMoE, as priced here:** row d of a check is used only if reached, with probability s^d. An expert's
  importance is the sum of s^d w^2 over the rows, and quality is the reach-weighted E. I did not read arXiv 2608.02989
  itself; this is the generic acceptance-weighted rule [G].
  - Alone it adds little over plain caps.
  - Combined with tiers it is the strongest lever, more so with today's drafter, whose deep rows are rarely reached.
- **(b) HOBBIT, as priced here:** weak experts at MSB-3 (the records3 MSB plane, no extra RAM) instead of dropping them.
  - With the calibrated 0.08 error, MSB-3 carries 20-33 of the experts per layer in the best mixes.
  - MSB-2 is almost never chosen, so three tiers suffice.
- **(c) Union caps alone** are the bluntest (drops only); predicted non-tie flips rise fastest.
- **(d) Close-call acceptance** changes tokens per check, not ms per check. It composes with any of the above, but its
  deviations and the lossy check's flips share the same quality gate.

**Uncertainty (leave-one-arm-out refits of the calibration, 7 per domain)** [P]:

| lever | code all-row flips % (level 1.34) | code non-tie (level 0.87) | multi all-row (level 3.69) |
|---|---|---|---|
| 1a | 1.30 [1.10, 1.38] | 0.66 [0.56, 0.71] | |
| 1b | 1.21 [1.06, 1.28] | 0.59 [0.52, 0.63] | |
| 1c | 1.08 [0.99, 1.15] | 0.50 [0.46, 0.53] | |
| 1d (pooled) | | | 3.94 [3.65, 4.13] |

- 1a's range crosses the code line, so it is borderline.
- 1b and 1c stay inside the code levels in every refit. **Screen 1c first** (the simplest build, 25.6 ms at the floor),
  then 1b.
- 1d sits on the multi all-row line (IFEval 4.85 [4.70, 4.98] is within; non-tie is within).

## How to build the rank-1 lever (per check, per layer; the router runs first)

1. From the 32 rows' router outputs, compute I_e = sum over rows d of s^d w(d, e)^2 for every expert e in the union
   (w normalised over each row's top-10).
2. Choose a tier per expert. With three tiers (4-bit, MSB-3 at error 0.08 and size 0.778, drop):
   - 4-bit if I_e > 2.775 lambda;
   - MSB-3 (read only the records3 MSB plane) if 0.846 lambda < I_e <= 2.775 lambda;
   - drop otherwise, with no renorm (the map's drop semantics).
3. s = the drafter's per-depth survival (0.935 today on HumanEval+, 0.98 for a 98 % drafter). lambda = 1.2e-2 to
   1.8e-2 for code, 8e-3 pooled. Example at lambda 1.6e-2: 4-bit if I > 0.044, MSB-3 if 0.0135 < I <= 0.044.
4. Byte reads become: dense (unchanged) + 4-bit experts + MSB planes. The check's kernel must read only the chosen
   planes at near-floor bandwidth.
- Today a 32-row check is compute-bound (~90-97 ms; Leap screen 1: ~30 + 2.47 ms per row), and there is no exact
  single-pass 32-row check yet (the gate carries <= 185 ids). These byte levers pay only once the 32-row kernels run
  near the floor (Leap / BigMoE kernel work), so they are the second step after that.

## Gate plan (EvalSuite LOSSY_GATE.md tiers)

1. **Map screen first (cheapest):** the lever as a decision-map ablation (Breakthrough's map; ~4 min per arm at 45 GB,
   share lock ~20 GB).
   - It is the replay pre-screen EvalSuite asks for: the exact rows teacher-forced through the variant, flips by margin
     class (tie / mid / sure from floor_rows.json).
   - Score code all-row and non-tie against 1.34 / 0.87 %; chat against its levels if the gate is pooled.
   - Acceptance-weighted arms must be scored reach-weighted (by s^d within each 32-row block), else deep rows count
     as if always used.
2. **EvalSuite tier 0** (15-min kill screen), then **tier 1** (364 items, ~45 min at 45 GB, pooled -3 pt line), then
   **tier 2** (card gate).
3. **Per variant, the engine needs:**
   - an env flag on a committed screen snapshot;
   - a runs.json counter per prompt (experts per check by tier: 4-bit / MSB-3 / dropped) that proves the lossy path
     engaged.
4. **Order:** 1b or 1c (code-scoped) first, then 1d (pooled), then 2 and 3 only as controls.

## Caveats [P]

- The calibration rests on 36 map windows (12 per domain, ~3.4K rows each) and the energy law (flips depend on the mean
  missed energy). Its fits are tight for drops; the low-bit error comes from one arm pair (t3_all vs t4_all), and MSB-2's
  error is assumed (4x MSB-3).
- The byte floors assume perfect kernels; the 84 % line is the realistic one. KV reads (long contexts) are not counted.
- The predicted flips are map-level. EvalSuite's gate decides adoption, and Kamen's call adopts.
- Not priced (no data on disk): a verify-time vocabulary shortlist for lm_head (~0.3 GB), dense-side bit planes, and
  kernel efficiency above 84 %. These are where the last 2-4 ms to 25 ms at 84 % would have to come from.

## Addendum (Sun 01:1x): the dense part does not have to stay

The orchestrator 01:0x: "challenge that". Breakthrough scripts/m1/dense_cut_price.py, data/m1/dense_cut_price.out.
CPU only. No GPU arm: the GPU was at 99 % and 26 GB was free, below a map arm's ~20 GB peak plus 8 GB.

**What the dense 3.437 GB is** [K: the weight-file headers; = Leap's 3.44 GB]:

| part | GB | share | stored as |
|---|---|---|---|
| HC mixers | 1.281 | 37 % | bf16 |
| GDN projections | 1.173 | 34 % | 4-bit |
| lm_head | 0.358 | 10 % | 4-bit |
| attention projections | 0.336 | 10 % | 4-bit |
| shared expert | 0.133 | 4 % | 4-bit |
| router | 0.067 | 2 % | 8-bit |
| PLE | 0.066 | 2 % | bf16 |
| indexer | 0.021 | 1 % | 8-bit |

**The dense levers** (per 32-row check; code flips all-row / non-tie on the map, against levels 1.34 / 0.87):

| lever | GB saved | ms saved floor / 84 % | code flips | verdict |
|---|---|---|---|---|
| HC mixers + PLE, bf16 -> 8-bit | 0.631 | 2.3 / 2.7 | hc8w: 0.33 / 0.06 [K], the same as the numeric-noise reference t4_all (0.33 / 0.06) | near-free; screen first. It is the 12 GB line's adopted HC8 |
| HC mixers + PLE -> 4-bit | 0.968 | 3.5 / 4.1 | +0.42 / +0.12 (bm_d4 - bm_res; confounded by resident-only experts) | small; needs a clean screen |
| lm_head shortlist, 16K rows from code outputs | 0.334 | 1.2 / 1.4 | misses 0.03 % of code rows [K, upper bound] | near-free on code |
| GDN + attention, 4 -> 3-bit (RTN) | 0.335 | 1.2 / 1.4 | unmeasured alone; the whole trunk at 3-bit (bm_d3) adds +7.4 non-tie points [K] | no, not without a better quantizer and a screen |
| shared expert at MSB-3 | 0.029 | 0.1 | - | not worth it |
| KV / GDN-state precision, verify only | <= 0.1-0.2 | <= 0.7 | unmeasured; state errors carry forward | not worth it |
| router / indexer | 0.09 in all | - | routing risk | no |

- Why the shortlist is cheap on code: BigMoE's bank of code answers uses only 8,775 distinct top-1 tokens in 261K rows.
- Chat needs its own list: the code list misses 32 % of IFEval rows and 76 % of multi rows.
- The KV lever is small because sparse attention caps the KV read at the indexer's 2,048 positions, and the GDN state is
  113 MB in fp32.

So the dense read can drop from 3.44 GB to 2.47 GB (mixers at 8-bit + the shortlist) or 2.14 GB (mixers at 4-bit +
the shortlist): -28 to -38 %, almost all of it from the bf16 mixers.

**The combined code plan:** three acceptance-weighted expert tiers (s 0.98) plus the dense cuts. Code flips are the
experts' prediction [leave-one-arm-out range] plus the dense levers' measured standalone flips, added. Adding is
conservative on all rows, because tie rows flip under any change and get counted twice.

| experts (lambda, 4-bit-eq) | dense | GB | ms floor / 84 % | tok/s HumanEval+ at 84 % (today / 98 %) | code all-row (1.34) | code non-tie (0.87) |
|---|---|---|---|---|---|---|
| 1.2e-2 (43.7) | mixers 8-bit + shortlist | 6.95 | 24.8 / 29.6 | 378 / 637 | 1.17 [1.08, 1.23] | 0.41 [0.37, 0.43] |
| 1.2e-2 (43.7) | mixers 4-bit + shortlist | 6.61 | **23.6** / 28.1 | 393 / 663 | 1.26 [1.17, 1.32] | 0.47 [0.43, 0.49] |
| 1.6e-2 (36.4, = 1c) | mixers 8-bit + shortlist | 6.20 | 22.1 / 26.4 | 414 / 698 | 1.44 [1.34, 1.51] | 0.59 [0.55, 0.62] |
| 1.6e-2 (36.4, = 1c) | mixers 4-bit + shortlist | 5.86 | 20.9 / **24.9** | 432 / 730 | 1.53 [1.43, 1.60] (tie-aware ~1.23 [P]) | 0.65 [0.61, 0.68] |
| 1.8e-2 (33.6) | mixers 4-bit + shortlist | 5.57 | 19.9 / 23.7 | 450 / 759 | 1.65 [1.50, 1.73] | 0.74 [0.67, 0.78] |

**Reading:**
- **The dense part is not fixed.** Two cheap levers cut ~1 GB of it with little measured quality cost: the bf16 mixers
  to 8 or 4 bits, and a code shortlist for lm_head.
  - With them, the code plan is under 25 ms at the floor within both code lines in every refit: 23.6 ms (lambda 1.2e-2,
    4-bit mixers).
  - It reaches **24.9 ms at a realistic 84 %** with 1c's experts, with non-tie flips within the line. All rows are over
    only under the double-counting accounting; tie-aware, ~1.23 [P].
- **Quickest win:** mixers at 8-bit (measured at noise level) + the shortlist (0.03 % misses). Together -0.97 GB, i.e.
  3.5 ms at the floor / 4.1 ms at 84 %, for almost no quality cost. They add to any expert lever.
- **Next map screens (GPU, when free; ~4 min each):**
  1. mixers at 4-bit with exact experts (unconfounded);
  2. GDN + attention at 3-bit alone (the D3 confound);
  3. the combined arm, 1c + 4-bit mixers + shortlist, which settles the all-row tie question directly.
- **Toward ~1k lossy on code:** at 24.9 ms the speed rule gives HumanEval+ ~430 tok/s with today's drafter and ~730 with
  a 98 % drafter. 1k needs a ~16-17 ms check. The remaining 2.1 GB of dense is 7.6 ms at the floor, so that also needs
  the kernels at the floor and the dense side's own quantizer work (GDN / attention below 4 bits, done well).
- **How to build:**
  - Mixers: the QuantizedLinear swap of the hc8w recipe. GatedResidual's fused read falls back to the eager path, so
    this needs a kernel for speed.
  - Shortlist: gather K rows (a static per-domain list + the drafted token + the drafter's top candidates), take logits
    over them and the argmax over the shortlist.
  - Both are lossy, so the map screen comes first, then EvalSuite's tiers.

## Map screens, MEASURED (Sun 01:19-01:35; Breakthrough d9b2db7)

Code windows (12), flips against the exact baseline, with 95 % window-bootstrap ranges:

| arm | code all-row % (level 1.34) | code non-tie % (level 0.87) | verdict (pre-registered) |
|---|---|---|---|
| **1c + 4-bit mixers + 16K shortlist** (the 24.9 ms-at-84 % plan) | **1.25** [0.45, 2.30] | **0.60** [0.19, 1.06] | **both lines met: the code plan** -> BigMoE's integration (LOSSY_CHECK_SPEC.md) + EvalSuite's code gate |
| 1c experts alone | 1.13 [0.36, 2.28] | 0.48 [0.11, 0.97] | the expert calibration holds (predicted 1.08 / 0.50) |
| 4-bit mixers alone | 1.16 [0.41, 2.12] | 0.48 [0.17, 0.84] | over its standalone line, but only +0.09 on top of 1c |
| GDN + attention at 3-bit alone | 4.43 [2.15, 7.37] | 3.62 [1.90, 5.78] | closed until there is a better quantizer |

- **Flips are sub-additive.** The dense lever adds +0.09 on top of the expert tiers, because both perturbations flip
  mostly the same fragile rows. The tie-aware estimate (~1.23) was right; the additive one (1.53) was too conservative.
- So the code plan measures at **24.9 ms at 84 % (20.9 at the floor) within both code lines** at the point estimate:
  5.86 GB per 32-row check, 4-bit-equivalent experts 36.4 per layer, dense 2.14 GB.
- With 12 windows the ranges are wide (the upper ends cross the lines). EvalSuite's code gate (tier 0 -> 1) decides,
  against the lossy-45 base.

## The combined plan on all 24 code windows (Sun 02:19; Breakthrough f755d70)

| code windows | all-row % (level 1.34) | non-tie % (level 0.87) |
|---|---|---|
| **all 24** (6,327 rows) | **1.09** [0.57, 1.73] | **0.62** [0.32, 0.96] |
| 1-12 (the first screen) | 1.25 (reproduced exactly) | 0.60 |
| 13-24 (new) | 0.91 | 0.65 |

- Both code lines are met at the point estimate, with the ranges about half as wide as on 12 windows. The upper ends
  still cross, so EvalSuite's code gate decides. The pre-registered action is unchanged: integrate (LOSSY_CHECK_SPEC.md).
- The AWQ 3-bit dense screen is ready but deferred for BigMoE's G32 memory slot (~14 GB, ~12 min, once memory allows).

## The 3-bit dense wall, measured (Sun 04:10; Breakthrough 43ef47c)

GDN + attention at 3 bits with activation-aware scaling (AWQ-style; per-linear alpha by a diagonal-Hessian error,
calibrated on held-out IFEval):
- code flips 3.86 / 2.99 % (all-row / non-tie), against plain RTN's 4.43 / 3.62 and the code lines 1.34 / 0.87;
- the alpha-0 control reproduces RTN on 100 % of rows, so the arm is valid.

Closed for scaling-only quantizers. The next tries are a full-Hessian GPTQ or a trellis code; until then GDN and
attention stay at 4 bits, and the code plan stays at 5.86 GB (20.9 ms floor / 24.9 ms at 84 %).
