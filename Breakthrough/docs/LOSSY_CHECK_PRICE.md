# A lossy 32-row check under 25 ms at 45 GB: price the levers (WEEK98 goal B; Sun 15 Nov 2026 00:5x)

> (a quote from the project owner, kept private)
~/FableGod/deepdraft/LOSSY_CHECK_PLAN.md. CPU only.

## Pre-registration (written before the computation)

**Bytes model** (Leap, data/week98/cheapcheck.md): a check reads dense 3.44 GB plus 2.135 MB per expert per layer for
the distinct experts of its rows' union, at 280 GB/s.
- Exact 32 rows: 100.5 distinct experts per layer, 13.74 GB, 49.1 ms.
- Targets: 25 ms is <= 7.0 GB at 280 GB/s (the floor) and <= 5.88 GB at 84 % of it (dense decode reaches 77-89 %).
- The dense 3.44 GB stays (D2 / D3: a 2- or 3-bit trunk is KILL / heavy), so experts must fit 3.56 GB (floor) or
  2.44 GB (84 %): the equivalent of 34.7 or 23.8 distinct 4-bit experts per layer.

**Routing data:** the decision map's xcap capture. For each of the 36 windows' answer rows (12 code, 12 IFEval,
12 multi; 10,316 rows) it holds the top-10 expert ids and their top-10-normalised weights w per layer, rank-sorted,
summing to 1. The answer rows are tiled into consecutive 32-row blocks; the last partial block is kept and its
distinct count is reported. Self-check: exact distinct per layer at 32 rows within +-10 % of Leap's 100.5 (BigMoE's
84-file capture: mean 107-110).

**Error model:** the energy law's missed energy. Per row and layer, E is the sum of err x w^2 over the row's top-10
(drop: err 1; MSB-3: 0.02; MSB-2: 0.10, rd_sim's figures).

**Quality calibration:** fit flips = a E^b per domain over the map arms whose dropped set I can rebuild. Flips are
all-row, against baseline.json's top-1, pooled per domain (my d2_crosscheck reading). The arms:
- topk2, topk5, top_k8 (rank >= k dropped);
- q1u_cf09, q1u_cf099, qom_0_live_c51, qom_0_live_c147 (their stored keep masks);
- t3_all (MSB-3 on every expert) as an out-of-sample check of the low-bit error model.

The fit is reported with its residuals. The prediction for each lever is the fitted flips at its E, against the map's
verdict levels (code 1.34 %, IFEval 5.42 %, multi 3.69 %: gate_top2_cf099's own all-row levels).

**Levers priced per 32-row block:**
- (c) union cap N per layer (100 / 80 / 60 / 40 / 30 / 20): keep the N experts with the largest sum of w^2 over the
  block's rows; the rest are dropped (the map's drop semantics: no renorm).
- (b) HOBBIT-style bit-plane tiers: per expert in the union, one of 4-bit (size 1), MSB-3 (0.778, err 0.02), MSB-2
  (0.556, err 0.10) or drop (0, err 1), RD-optimal under a lambda sweep. These are the same record's planes, so they
  need no extra RAM.
- (a) acceptance weighting (AcceptMoE-style): row d of a check (d = 0..31) matters only if it is reached, with
  probability s^d.
  - s = 0.935 (today's code drafter, per-depth survival), 0.60 (chat) or 0.98 (the 98 % drafter).
  - The importance used by (b) and (c) becomes the sum of s^d w^2; quality is the reach-weighted E.
- (d) close-call acceptance: a tokens-per-check multiplier from CLOSECALL_ACCEPT_PRICE (Thu); no new computation.
- (e) combinations.

**Speed per lever:** ms at the floor and at 84 % (bytes / bandwidth), with tok/s from the charter's speed rule (drafting
~7 ms per check; kept 13.8 today, 23.3 for a 98 % drafter on HumanEval+).

**Pre-registered lines** (map-level screens; EvalSuite's same-scores gate decides adoption):
- PASS: some lever reaches <= 25 ms at 84 % with predicted flips at or under the levels on all three domains.
- BETWEEN: <= 25 ms only at the floor (100 %), or within 1.5x the levels.
- KILL: neither.

**Files:** scripts/m1/lossy_check_price.py, data/m1/lossy_check_price.out.

## Result (Sun 01:0x): pre-registered verdict KILL (pooled); code-scoped reading post-hoc

The full ranked plan is in ~/FableGod/deepdraft/LOSSY_CHECK_PLAN.md.

**Self-check: PASS.** The exact 32-row union is 99.9 distinct experts per layer, against Leap's 100.5 (299 full
blocks): 13.68 GB, 48.9 ms at the floor.

**Calibration:** the drop fits are tight (log rms 0.10-0.26). The pre-registered out-of-sample check FAILED for the
low-bit model: t3_all flips 3-6x more than MSB-3 at error 0.02 predicts.
- Recalibrated, net of t4_all (the bit-plane path at 4 bits, 0.33 / 2.15 / 1.58 % flips of its own): MSB-3 = 0.08
  (0.065-0.09); MSB-2 = 0.32 (4x, assumed). With these, t3_all is predicted within ~10 % on IFEval / multi.
- Post-hoc addition: a non-tie calibration (floor_rows.json; non-tie levels 0.87 / 2.22 / 1.63 %).

**Pre-registered lines (all three domains): KILL.** Nothing reaches <= 25 ms with flips at or within 1.5x the levels
on code, IFEval and multi together. The best within the pooled levels is acceptance-weighted tiers at s 0.98, lambda
8e-3: 32.3 ms at the floor, 38.5 at 84 % (non-tie within; all-row <= 1.5x).

**Code-scoped reading (post-hoc):**
- Acceptance-weighted tiers reach 23.0 ms (today's drafter, s 0.935) and 24.6 ms (98 %, s 0.98) at the floor, within
  the code levels (1.21-1.30 % all-row, 0.59-0.66 % non-tie). At 84 % that is 27.3 / 29.2 ms.
- MSB-2 is barely chosen, so three tiers suffice.
- Union caps alone, at the code levels: cap 40, 26.9 / 32.0 ms. Tiers alone: 28.4 / 33.8 ms.

**Files:** scripts/m1/lossy_check_price.py, data/m1/lossy_check_price.out.
- **Uncertainty** (leave-one-arm-out, 7 refits per domain; post-hoc). Code all-row flips:
  - 1a (s 0.935, lambda 1.2e-2): 1.30 [1.10, 1.38], crossing the 1.34 line;
  - 1b (s 0.98, lambda 1.8e-2): 1.21 [1.06, 1.28], inside in every refit;
  - 1c (3 tiers, lambda 1.6e-2): 1.08 [0.99, 1.15], inside in every refit.
  - Pooled 1d: multi all-row 3.94 [3.65, 4.13] against 3.69, i.e. on the line.
  - Recommended screen order: 1c, then 1b.

## Addendum, dense cuts (Sun 01:1x; the orchestrator 01:0x; not separately pre-registered: a pricing, not a screen)

scripts/m1/dense_cut_price.py, data/m1/dense_cut_price.out.

**Dense 3.437 GB by module** (weight headers): HC mixers 1.281 GB (bf16), GDN 1.173, lm_head 0.358, attention 0.336,
shared expert 0.133, router 0.067, PLE 0.066, indexer 0.021.

**Cuts:**
- Mixers + PLE to 8-bit: -0.631 GB. hc8w measured 0.33 / 0.06 % code flips, equal to t4_all's numeric noise.
- Mixers + PLE to 4-bit: -0.968 GB, +0.42 / +0.12 % (confounded).
- lm_head 16K-row code shortlist: -0.334 GB, 0.03 % code misses. The bank's code answers use 8,775 distinct top-1
  tokens.
- GDN + attention at 3-bit: -0.335 GB, unmeasured alone; the whole-trunk 3-bit run adds +7.4 non-tie points.
- KV, shared expert, router: each <= 0.2 GB.

**Combined with the 1c-family experts** (s 0.98):
- lambda 1.2e-2 + 4-bit mixers + shortlist: 6.61 GB, 23.6 / 28.1 ms, code 1.26 [1.17, 1.32] / 0.47. Within both lines.
- 1c + 4-bit mixers + shortlist: 5.86 GB, 20.9 / 24.9 ms, code 1.53 additive (~1.23 tie-aware) / 0.65 non-tie. 25 ms at
  84 % is reached; non-tie is within.

## Map screens (Sun 01:2x; the orchestrator 01:2x). Pre-registered result -> action, written before the runs

Setup:
- Arms: bt_comb, bt_mix4, bt_ga3, bt_1c, on the 12 code windows (DM_SETS=code, DM_C 51), in one process under a share
  lock, after the gate.
- Masks: scripts/m1/build_tiers_masks.py (self-check: 36.40 4-bit-eq per layer = the pricing's 1c).
- Code: ~/FableGod/decision_map/dm_bt.py and the 'tiers' gate in dm_common.py (backups .pre_bt).
- Flips: all-row / non-tie vs baseline.json, code windows. The 16K lm_head shortlist is applied offline to bt_comb's
  recorded top-8 (if the top-1 is outside the list, take the first listed token in the top-8).
- The masks come from the exact routing. With the mixers changed, the run-time ranks shift slightly (noted, not
  corrected).

| arm | prediction | result -> action |
|---|---|---|
| (a) bt_comb (1c + 4-bit mixers + shortlist) | all-row 1.53 additive / ~1.23 tie-aware; non-tie 0.65 [0.61, 0.68] | **both lines met** (<= 1.34 / <= 0.87): this is the code plan (24.9 ms at 84 %), on to BigMoE's integration + EvalSuite's code gate. **non-tie met, all-row over**: the tie-aware reading fails; fall back to lambda 1.2e-2 (23.6 / 28.1 ms) and screen that combination. **non-tie over**: additivity fails; fall back to 8-bit mixers + lambda 1.2e-2, re-screen |
| (b) bt_mix4 (exact experts) | +0.42 / +0.12 (confounded) | **non-tie <= 0.20 and all-row <= 0.60**: 4-bit mixers adopted in the plan. **else**: 8-bit mixers (hc8w, measured at noise level) |
| (c) bt_ga3 (exact experts) | none (D3 bound: up to +7.4 non-tie) | **non-tie <= 0.50**: GDN + attention at 3-bit (-0.335 GB) enters the plan; read chat next. **> 1.0**: closed until there is a better quantizer. **between**: BETWEEN |
| (d) bt_1c (experts only) | 1.08 [0.99, 1.15] / 0.50 [0.46, 0.53] | **within +-50 % of the prediction**: the expert calibration holds. bt_comb - bt_1c = the dense increment, the additivity check |

### Map screen results (Sun 01:19-01:35; code, 12 windows, DM_C 24, share lock; scripts/m1/bt_screen_read.py)

| arm | all-row % (level 1.34) [95 % window bootstrap] | non-tie % (level 0.87) | prediction | pre-registered action |
|---|---|---|---|---|
| bt_comb + 16K shortlist | **1.25** [0.45, 2.30] | **0.60** [0.19, 1.06] | 1.53 additive / ~1.23 tie-aware; 0.65 | **both lines met: the code plan** (24.9 ms at 84 %) -> BigMoE integration + EvalSuite code gate |
| bt_comb (no shortlist) | 1.22 | 0.57 | - | the shortlist adds +0.03 (predicted 0.03) |
| bt_1c (experts only) | 1.13 [0.36, 2.28] | 0.48 [0.11, 0.97] | 1.08 / 0.50 | **the expert calibration holds** |
| bt_mix4 (exact experts) | 1.16 [0.41, 2.12] | 0.48 [0.17, 0.84] | +0.42 / +0.12 | over its line (<= 0.20 non-tie): no 4-bit mixers as a standalone lever; HC8 (8-bit, shipped) stays |
| bt_ga3 (exact experts) | 4.43 [2.15, 7.37] | 3.62 [1.90, 5.78] | - | **closed** until there is a better quantizer |
| references: hc8w / t4_all | 0.33 / 0.33 | 0.06 / 0.06 | | numeric noise |

**Reading:**
- **Flips are sub-additive.** bt_comb - bt_1c = +0.09 / +0.09, while 4-bit mixers alone flip 1.16 / 0.48: the two
  perturbations flip mostly the same fragile rows. So the additive accounting was very conservative, and the tie-aware
  estimate (~1.23) was right.
- 4-bit mixers are costly alone but nearly free on top of the expert tiers. That is why (a) holds while (b) fails its
  standalone line.
- The intervals are wide with 12 windows: the point estimates meet both lines, the upper ends don't. EvalSuite's code
  gate decides.

**Data:** data/m1/bt_screens/ (the four arms' json, gzipped; the dm_common 'tiers' diff; the hook diff; a copy of
dm_bt.py); data/m1/bt_screen_read.out.

## Next screens (Sun 01:4x; the orchestrator 01:3x). Pre-registered result -> action

**(1) The combined plan on all 24 code windows.**
- Setup: DM_OUT=out_code24 (the map's own out/ is untouched); 'baseline scapture', then the masks
  (scripts/m1/code24_comb.py masks; windows 1-12 must equal the screen's masks), then bt_comb; ties from the new
  baseline's p1 - p2 <= 0.1.
- Result -> action:
  - point estimates within both lines and the 95 % upper ends within too: integrate (firm);
  - points within, upper ends crossing: integrate, EvalSuite decides (as now);
  - a point over a line: fall back to lambda 1.2e-2 + 4-bit mixers + shortlist (23.6 / 28.1 ms) and screen that before
    integrating.
- Determinism check: windows 1-12 must reproduce 1.25 / 0.60.

**(2) A better 3-bit quantizer for GDN + attention** (activation-aware per-input-channel scaling, AWQ-style, with a
diagonal-Hessian objective).
- Calibration: E[x^2] per input channel of each of the 228 linears on the 12 IFEval map windows (a held-out domain).
- Rule: per linear, the alpha in {0, 0.25, 0.5, 0.75, 1} that minimises the sum over j of E[x_j^2] * ||W_:,j - Q3(W s)_:,j / s_j||^2,
  with s = RMS(x)^alpha (group 64). At inference, y = Q3(W s)(x / s).
- (a) bt_ga3awq alone, exact experts, code 12. Result -> action:
  - non-tie <= 0.50: enters the plan, then screen (b);
  - 0.5-1.0: BETWEEN;
  - > 1.0: closed for scaling-only quantizers (a full-Hessian GPTQ or trellis is the next try).
  - Reference: bt_ga3 (RTN) 4.43 / 3.62.
- (b) only if (a) enters: bt_comb + AWQ dense, code 12. Both lines met = dense -0.335 GB, i.e. 5.53 GB, 19.7 / 23.5 ms.
- Validity: alpha 0 for every linear must reproduce bt_ga3 (RTN) bit for bit. If not, a projection bypasses the
  module call and the arm is invalid.

### (1) result: the combined plan on all 24 code windows (Sun 01:41-02:19; code24_comb.py; data/m1/code24_comb.out)

| code windows | all-row % (level 1.34) [95 % window bootstrap] | non-tie % (level 0.87) |
|---|---|---|
| **all 24** (6,327 rows) | **1.09** [0.57, 1.73] | **0.62** [0.32, 0.96] |
| 1-12 (the screen's) | 1.25 [0.44, 2.27] (= the screen: determinism holds) | 0.60 [0.19, 1.07] |
| 13-24 (new) | 0.91 [0.35, 1.53] | 0.65 [0.23, 1.13] |

- Pre-registered action: the points are within both lines and the upper ends cross, so integrate; EvalSuite's code gate
  decides.
- The ranges are about half as wide as on 12 windows, and the all-24 point (1.09) is further inside the all-row line.
- The masks for windows 1-12 were identical to the screen's (PASS).

**(2) the AWQ screen:** deferred. The 24-window run ended at 02:19, after the 02:12 cut-off kept clear of BigMoE's G32
memory slot (the orchestrator 01:56: no new job above 5 GB from 02:20 until G32 has loaded). Ready to run:
dm_bt.py v2, scratchpad dm_tiers/run_awq.sh, ~12 min, ~14 GB.

## lm_head shortlist with context (Sun 02:5x; the orchestrator 03:4x after Sandbox98's 69c93b2). Pre-registered, CPU, before computing

**Why no GPU arm is needed:** the shortlist acts only on the final logits, and the map recorded each row's top-8.
- The restricted argmax is exact from the top-8 when an in-list token is there.
- If none of the top-8 is in the list, the row counts as a flip (conservative).

**Lists:**
- S_K: the K most frequent top-1 tokens of the deepdraft bank (code; disjoint from the map windows), K = 4096, 8192,
  16384.
- C_K: S_K plus every token of the window before the row (the prompt and the answer so far). A real check gathers the
  context once per check and always scores the drafted tokens; under teacher forcing this equals the tokens before the
  row.

**Measured** (code, IFEval, multi):
- misses = the baseline top-1 outside the list, i.e. the flips the shortlist adds to the exact model;
- for code, the combined arm with the list applied (bt_comb on the 12 code windows and on all 24), all-row / non-tie
  flips against the code lines;
- lm_head rows read per check (K plus the mean unique context tokens).

**Lines:**
- **code-safe:** code misses <= 0.05 % (the 16K static's level) and the combined arm within both code lines;
- **chat-safe:** IFEval and multi misses <= 0.5 % each (<= ~10 % of the chat levels 5.42 / 3.69).

**Files:** scripts/m1/lmhead_ctx_shortlist.py, data/m1/lmhead_ctx_shortlist.out.

### Result (Sun 03:46; data/m1/lmhead_ctx_shortlist.out)

| list | code misses % | IFEval % | multi % | combined code arm, 24 windows (all-row / non-tie) |
|---|---|---|---|---|
| S_4096 | 4.25 | 39.5 | 78.5 | 4.76 / 4.31 |
| C_4096 | 1.04 | 32.5 | 52.3 | 1.80 / 1.33 |
| S_8192 | 1.49 | 33.0 | 76.6 | 2.31 / 1.84 |
| C_8192 | 0.48 | 27.3 | 51.2 | 1.39 / 0.91 |
| **S_16384 (the plan)** | **0.03** | 32.0 | 76.3 | **1.09 / 0.62** |
| C_16384 | 0.03 | 26.4 | 51.0 | 1.09 / 0.62 |

Context adds 122 / 192 / 201 unique tokens on code / IFEval / multi windows.

**Verdict (pre-registered lines):**
- **code-safe:** only S_16384 and C_16384 qualify. Context helps at 4-8K (code misses 4.2 -> 1.0 % at 4K, 1.5 -> 0.48 % at
  8K) but C_8192's combined arm is just over both lines (1.39 / 0.91). At 16K context adds nothing: the remaining 0.03 %
  are tokens absent from the bank. Bytes don't favour the small lists either: 4K / 8K save <= 18 MB (~0.06 ms) against
  16K.
- **chat-safe: no list qualifies.** Context cuts the code list's chat misses (IFEval 32 -> 26 %, multi 76 -> 51 %) but
  is nowhere near 0.5 %. A chat shortlist needs its own static vocabulary (a chat-derived list), which this screen did
  not test. Sandbox98's 2B result (most loss removed) came from a setup where the static list matched the domain.
- **Action:** keep the 16K static code list; the lm_head shortlist stays code-scoped (R57_LMSL_SETS).

### (2) result: AWQ-style 3-bit GDN + attention (Sun 04:06-04:10, after G32; code 12 windows; share lock)

| arm | all-row % (level 1.34) [95 % window bootstrap] | non-tie % (level 0.87) |
|---|---|---|
| bt_ga3 (RTN 3-bit) | 4.43 [2.19, 7.31] | 3.62 [1.92, 5.72] |
| bt_ga3w0 (alpha 0, the wrapper path) | 4.43 | 3.62 |
| **bt_ga3awq** (activation-aware scaling) | **3.86** [1.61, 6.82] | **2.99** [1.36, 5.03] |

- **Validity: PASS.** bt_ga3w0's top-1 equals bt_ga3's on 100.00 % of 3,366 rows, so no projection bypasses the
  wrapper.
- Calibration: E[x^2] of all 228 linears from 6 IFEval windows (2,347 rows each). Chosen alpha: 0.25 on 167 linears,
  0.5 on 55, 0 on 2, 0.75 on 3, 1.0 on 1.
- **Pre-registered action: CLOSED for scaling-only quantizers** (non-tie 2.99 > 1.0). Activation-aware scaling cuts
  RTN's flips by only 13-17 %. The 3-bit dense wall stands; a full-Hessian GPTQ (column error compensation) or a trellis
  code is the next try, otherwise GDN and attention stay at 4 bits.
- Data: data/m1/bt_screens/bt_ga3w0.json.gz, bt_ga3awq.json.gz, awq_stats.npz.gz, dm_bt_v2.py.copy;
  data/m1/awq_screen_read.out.

## Erratum check: per-domain bytes (Mon 16 Nov 08:3x; Fit12 08:29). Pre-registered, CPU, before computing

**Claim (Fit12, ~/Fit12/PLAN_12GB_90_fit12.md "LOSSY32 RESULT", self-check (c)):** the plan's exact union (99.9 per
layer) and 1c's 4-bit-equivalent count (36.4) are means over all 36 map windows (12 code + 12 IFEval + 12 multi), while
the lever is code-scoped. Fit12's recount: code 114.2 / 39.2, IFEval 98.1 / 38.4, multi 87.7 / 31.8.

**Cause, read from the code:** lossy_check_price.py's evaluate() returns keq = size.sum() / NBL, where NBL counts the
block-layers of all three domains, and the exact line uses distinct.mean() over all blocks. The quality side was per
domain (code rows' E); the bytes were pooled. The map screens' measured flips are not affected: the 1c rule is applied
per block, so the code windows' masks were the code blocks' own tiers.

**Check:** scripts/m1/lossy_check_domain_recount.py rebuilds the same 32-row blocks and the same tier rule. It prints, per
domain and pooled:
- the exact union per layer;
- the tier counts and the 4-bit-eq per layer for 1a / 1b / 1c / 1d, and for 1c at lambda 1.2e-2 .. 2.2e-2;
- GB and ms with the plan's dense parts (exact 3.44 GB; mixers at 4-bit + the shortlist 2.14 GB).
- Self-check: the pooled values reproduce the plan (union 99.9; 1a 29.2, 1b 33.5, 1c 36.4, 1d 54.8) to 0.1. Otherwise
  the recount is invalid and nothing is restated.

**Result -> action:**
- Code 1c within 1 % of Fit12's 39.2: the claim holds. LOSSY_CHECK_PLAN.md gets a CORRECTED note at the top and at
  each place the code plan's GB / ms / tok/s appear (old numbers kept, marked), restated with the code blocks' counts.
- Off by more than 1 %: restate with my count, and send both counts to Fit12.
- If the restated code plan is above 25 ms at 84 % (expected ~26 ms): tell the orchestrator that "24.9 ms at 84 %" no
  longer holds on code. Give the bandwidth efficiency 25 ms needs, and the lambda that reaches it at 84 % with its
  predicted flips from data/m1/lossy_check_price.out.
- Either way the plan gets the engine context: the 84 % line is a kernel target, not today's engine (the CHECK55
  engine profile a082f8c: the exact 32-row check takes 102.1 ms, GPU-bound). Fit12's realistic model (49.8-53.3 ms
  for the plan on today's engine) is cited as theirs.

### Result (Mon 16 Nov 08:36-08:39; data/m1/lossy_check_domain_recount.out): the claim HOLDS; the code plan is 26.1 ms at 84 %, not 24.9

- **Self-check PASS.** The pooled values reproduce the plan: union 99.90, 1a 29.17, 1b 33.53, 1c 36.40, 1d 54.77.
- **Fit12's code counts confirmed:** union 114.24 (+0.04 %), 1c 39.18 (-0.06 %).

| per 32-row block, by domain | code | IFEval | multi | all (as published) |
|---|---|---|---|---|
| exact union / layer | **114.2** | 98.1 | 87.7 | 99.9 |
| 1a 4-bit-eq (s 0.935, lambda 1.2e-2) | 30.9 | 31.1 | 25.7 | 29.2 |
| 1b 4-bit-eq (s 0.98, lambda 1.8e-2) | 35.7 | 35.6 | 29.4 | 33.5 |
| 1c 4-bit-eq (3 tiers, s 0.98, lambda 1.6e-2) | **39.2** | 38.4 | 31.8 | 36.4 |
| 1d 4-bit-eq (s 0.98, lambda 8e-3) | 62.3 | 55.5 | 46.8 | 54.8 |

**The code plan restated** (1c + 4-bit mixers + 16K shortlist; dense 2.135 GB as in dense_cut_price.py):

| | GB | ms floor | ms at 84 % | 25 ms needs | tok/s HE+ at 84 % (today / 98 %) |
|---|---|---|---|---|---|
| as published (pooled bytes) | 5.86-5.87 | 20.9 | 24.9 | 83.8 % of 280 GB/s | 432 / 730 |
| **code blocks** | **6.15** | **22.0** | **26.1** | **87.9 %** | **416 / 703** |

- Fit12's engine-semantics count is 6.12 GB / 21.9 ms: there, 1c's "4-bit" tier keeps REF2-T's base planes.
- **Other levers on code blocks** (dense 3.44; GB, ms at the floor / 84 %):
  - 1a: 6.61 GB, 23.6 / 28.1 ms;
  - 1b: 7.10 GB, 25.3 / 30.2 ms (over 25 ms even at the floor);
  - 1c: 7.45 GB, 26.6 / 31.7 ms;
  - 1d (code only): 9.82 GB, 35.1 ms at the floor.
- **25 ms at 84 % on code** needs lambda ~1.8e-2: 35.7 eq, 5.80 GB, 24.6 ms. Its predicted code flips are ~1.33 / ~0.71
  [P, post hoc]: the experts alone 1.21 [1.06, 1.28] / 0.59 [0.52, 0.63] (data/m1/dense_cut_price.out), plus the dense
  cuts' measured +0.12 / +0.12 on top of 1c (bt_comb + shortlist 1.25 / 0.60 vs bt_1c 1.13 / 0.48, 12 windows). That
  is right at the all-row line 1.34. Not screened; a map arm would settle it (~4 min, share lock ~20 GB).
- **The measured flips stand** (1.09 / 0.62 on 24 code windows): the tier rule is applied per block, so the code windows
  were always scored with their own tiers. Only the bytes were pooled.
- **Actions taken:**
  - ~/FableGod/deepdraft/LOSSY_CHECK_PLAN.md: a CORRECTED section at the top and a marker at each old number (old
    numbers kept). LOSSY_CHECK_SPEC.md: V2's "36.4" is marked.
  - The orchestrator already carries the correction in WEEK98.md (Fit12's LOSSY32 entry, Mon 08:2x, number-checked by
    NewIdeas). So my note to it is a one-line confirmation, not a new finding.
- **Context:** 84 % is a kernel target. Today's full engine runs the exact 32-row check in 102.1 ms (90.5 with v32one),
  GPU-bound (a082f8c). Fit12's realistic model of the plan on today's engine is 49.8-53.3 ms.
