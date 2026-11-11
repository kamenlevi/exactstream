# 350 tok/s on the 125B: the assumptions behind today's caps, and the bets that break them (Leap, Sat 2026-10-24 13:45)
Kamen (13:1x, via the orchestrator): 350 must be on the 125B, through a genuine breakthrough. The earlier table's caps
(~260 bit-exact, ~305 with v3 tiers, ~430 with an expert budget) are not verdicts. Each rests on an assumption of
today's methods. This page lists those assumptions, the bets that would break each one, and their prior art. The old
numbers are kept at the end as the baseline every bet must move.

## The arithmetic every bet must move
**tok/s = tokens accepted per pass / pass time**, and **pass time >= bytes read per pass / 280 GB/s / kernel efficiency**.
- Code today: ~65 tok/s exact on the 125B (v3 tiers, half the experts resident, MTP). At the bytes floor, with every
  method we have, it caps at 196-305.
- **350 needs <= ~5.5 GB per pass** at today's drafter: 9.9 tokens per 16-row pass is <= 28.2 ms per pass. Take off
  3.5 ms of drafting, and at 80 % kernel efficiency that leaves 19.8 ms at the floor.
  - Today's design reads 13.2 GB per 16-row pass at 4 bits: 3.44 GB dense + 73 experts / layer x 48 x 2.77 MB.
  - So 350 needs ~2.4x fewer bytes per pass, or ~2.4x more tokens per pass, or a mix (e.g. 1.6x each). All of it must
    also be resident.

## Assumptions -> bets
Tags: **published** = shown elsewhere, untested here; **here** = measured on this Mac; **new** = no prior art found.

| # | assumption today | what it costs | bets that would break it | tag / prior art | owner |
|---|---|---|---|---|---|
| A1 | Experts are stored at ~4 bits per weight (68 GB), so the 125B cannot be resident under the 45 GB rule and experts stream from the SSD | stalls; 2.77 MB per expert read per use | (a) ~2-2.25-bit data-aware expert codes (GPTQ / VQ / trellis) | published: GPTQ, AQLM, QuIP#; MoE: MoQE (3-bit plain, 2-bit with QAT), BitsMoE (2-bit Qwen3-30B), TriS-MoE (sub-2-bit with binarized residuals). Here: C1 / Q1, scalar 2-bit weight error 4.4-5.1x MSB-3's | Fit12 (Q2, 13:2x) |
| | | | (b) **structure shared across a layer's 512 experts**: a shared base + small per-expert parts, a shared input / output subspace + low-bit coefficients | published at 16 bits and mild ratios: MoE-SVD (ICML 2025: one V shared by all experts, 60 % compression, minimal loss on Mixtral / DeepSeek / Qwen2 MoE), MoLAE (shared latent projection), D2-MoE (shared base + SVD deltas), MoBE (shared basis, -24-30 % at 1-2 points on Qwen3-235B / DeepSeek-V3 / Kimi-K2), MC-SMoE / HC-SMoE (merging). **Untested anywhere we found: sharing combined with low-bit codes at <= 1.5 bits/param** | **Leap (L1-16, below)** |
| | | | (c) pruned experts (fewer experts, same routing quality) | published: REAP (ICLR 2026): 50 % of experts pruned, near-lossless code on Qwen3-Coder-480B / Kimi-K2 | open |
| | | | (d) sub-1-bit entropy-coded experts with an in-kernel decoder | published: QMoE, 0.8 bits/param (SwitchTransformer, an older, more redundant MoE) | open |
| A2 | Every routed expert's FULL weights are read for every pass that touches it | 9.7 GB per 16-row pass at 4 bits | (a) A1's codes (fewer bytes per weight) | as A1 | |
| | | | (b) read only the expert neurons a row needs (activation sparsity inside experts) | published for dense models: Deja Vu, TEAL (training-free 40-50 % activation sparsity), CATS; untested on MoE experts | open |
| | | | (c) shared expert parts read once per pass, not per expert (follows from A1-b) | new | Leap (with L1-16) |
| A3 | Each verify accepts 6.9 (block 8) to 9.9 (block 16, fine-tuned) tokens on code | the tokens-per-pass term | (a) a better drafter on the 125B's own outputs | here: L5 block-16 FT x1.44 under a budget; L4 (agentic) killed | |
| | | | (b) lossless verification of more paths (trees, block verification) | published: SpecInfer, EAGLE-2 trees, block verification (+5-10 %); here: trees +5 %, parked | |
| | | | (c) cost-aware drafts on MoE (verify the cheap prefix) | published: EVICT, ECOSPEC (2026); here: L1-14 truncation +8.4 % on the 35B | port to the 125B |
| | | | (d) near-tie acceptance (non-exact; Kamen's call) | published: lossy / typical acceptance | |
| A4 | Every verify row runs all 48 layers | no shortcut per row | early exit / layer skipping | here: L1-09 killed (layer 40 agrees 0.057) | closed |
| A5 | The dense part (3.44 GB: hyper-connections 1.28, GDN 1.18, LM head 0.36, attention 0.36) is read in full every pass | ~12 ms of the floor per pass | dense diet (2.1 GB, Chat1000, non-exact); screened-argmax LM head; A1-style codes for dense layers | here / proposed (MoonShots) | open |
| A6 | Kernels run at ~40-50 % of the bytes floor at 8-16 rows | ~2x pass time | kernel programs | here: GDN16 (35B pass 44 -> 37 ms, loop +12.9 %); MMA-tile experts (L1-07) | Fit12 (GDN port), MoonShots |
| A7 | Drafting, rollback and the pass run serially | 4-8 ms per pass | overlap; submit the rollback on its own | here: the 35B loop +12.9 % once the GDN rollback replay was submitted on its own | port to the 125B |
| A8 | Union of experts grows as ~r^0.72 with rows | wide passes amortize weakly (32 rows: 121 experts / layer) | routing-aware drafts (prefer drafts whose experts are already in the union) | published-adjacent: ECOSPEC (cost-aware drafting) | open |

## Update 15:40: what the bytes bets measured today (Fit12 Q2 / Q3, Leap L1-16 / L1-17; same inputs and metric)
- **Expert codes are near their rate-distortion bound.** Every code tried reaches MSB-3's output error only at ~3.1-3.2
  bits per weight:
  - per-expert GPTQ-2 at 2.5 b: 3.65x MSB-3's error;
  - GPTQ-3 at 3.5 b: 0.61x;
  - transform coding in the input eigenbasis with eigenvalue-sorted bits: 1.52x at 3.0 b;
  - shared input subspace: 1.75x at 2.7 b;
  - structure shared across experts: none.
  So codes alone buy ~10-15 % bytes at equal quality.
- **Fewer experts per token is the other byte lever.** Top-8 costs less error than MSB-3 (0.78x) and reads 16 % fewer
  experts; adaptive p 0.8 reads 24 % fewer at 1.43x. Both are non-exact and need an Eval-suite verdict.
- **Stacked estimate (all non-exact, quality unverified), code, 16-row pass:**
  - ~3-3.5-bit experts everywhere (the 125B resident, ~44-52 GB) + adaptive p 0.8 + the dense diet = ~7-8 GB per pass;
  - that is ~240-280 tok/s at 80 % kernel efficiency with today's 9.9 tokens per pass.
- **So 350 also needs ~1.3-1.45x more tokens per pass on code** (A3: ~13-14 per pass). That makes the drafter / verify
  side the binding lever now.

## Which bet is strongest, and why
- **Bytes per expert (A1 / A2) is the only lever that is both necessary and close to sufficient.**
  - It is the precondition for everything else: the 125B has to be resident, and only fewer bytes per weight get it under
    45 GB at ~4-bit quality.
  - Arithmetic: experts at ~1.3 bits/param effective (0.80 MB each) plus the dense diet make a 16-row pass ~4.9 GB. With
    today's drafter that is ~390 tok/s at 80 % kernel efficiency (3.5 ms drafting included), and the whole model is
    ~26 GB.
  - At ~2.25 bits incl. scales (Fit12's line) the pass is ~7.0 GB, ~285 tok/s: needed, but not enough alone.
- Fit12 is testing per-expert ~2-bit codes. **I take the part below 2 bits: structure shared across experts (A1-b),
  which can stack with Fit12's codes.**
- **Tokens per pass (A3) is the second lever.** It is multiplicative, but the published methods give +5-15 %, not 2x.

## L1-16 (Leap, pre-registered 13:45): is there enough structure shared across a layer's experts to go below 2 bits?
- **Hypothesis:** a layer's 512 experts share a large part of their weights' useful content, judged on real inputs.
  Candidates:
  - a common base, the activation-weighted mean expert;
  - a common input subspace: all experts read the same hidden state, whose covariance is strongly anisotropic;
  - a common output subspace.
  If so, the shared parts are stored once per layer and each expert keeps only small low-bit coefficients.
- **Cheapest decisive test (no training; on the Mac).** On layers 12 / 24 / 36:
  1. Weight space: the share of each expert's energy captured by (i) the mean expert and (ii) the top-r shared input /
     output singular subspaces of the stacked experts, vs one expert's own top-r.
  2. Output error on real routed inputs, at equal bytes per expert (shared parts amortized over 512), relative to the
     4-bit expert. Methods compared:
     - MSB-3 (the 3-bit tier that keeps quality);
     - per-expert RTN-2 g64;
     - shared input subspace at rank r + 4-bit per-expert coefficients;
     - shared base + 2-bit residual.
     Inputs: Fit12's Q2 capture (layers 2 / 12 / 24 / 36 / 46) once it saves, or my own.
- **Kill:** at <= 1.5 bits/param effective, no shared representation beats per-expert RTN-2 g64 (2.5 bits incl. bf16
  scale + bias) on output error. Same metric, layers and experts as Fit12's Q2. That would mean no exploitable cross-expert structure; the bet folds into
  Fit12's per-expert codes.
- **Pass (go to learned shared codes + layer-wise distillation, free cloud compute if needed):** at <= 1.5 bits/param, the
  output error is <= MSB-3's.
- **Step 1 result (14:10, data/wspace16.txt; layers 12 / 24 / 36): no shared structure in WEIGHT space.**
  - The mean expert holds 0.20-0.36 % of the energy (1/512 = 0.20 %), so the experts are not aligned copies and a
    shared base is useless.
  - Shared input subspace (gate / up), share of energy captured: top-512 of 2560 directions 0.28-0.30, top-1024 0.50-0.51.
    An isotropic spread would give 0.20 and 0.40. The shared output subspace of down: the same.
  - One expert alone: r 128 captures 0.39-0.42, r 512 0.93. The spectra are flat.
  - Not yet a kill: the line is on OUTPUT error, where the input covariance H weights the directions (step 2).
- **Step 2 result (15:06, data/act16.txt): KILLED by its lines.**
  - Setup: Fit12's Q2 metric and inputs; 5 layers x 8 experts; output error relative to the 4-bit expert.
  - Results:

    | format (bits / param) | gate | up | down | vs MSB-3 |
    |---|---|---|---|---|
    | MSB-3 (3.5) | 0.017 | 0.022 | 0.026 | 1.0x |
    | RTN-2 g64 (2.5) | 0.130 | 0.171 | 0.203 | 7.7x |
    | shared subspace r 768, 4-bit coefficients (1.35) | 0.084 | 0.112 | 0.539 | 10.1x |
    | shared subspace r 1024 (1.80) | 0.059 | 0.078 | 0.439 | 7.9x |
    | shared subspace r 1536 (2.70) | 0.029 | 0.038 | 0.268 | 4.5x |
    | per-expert activation-aware low rank 256 (2.25) | 0.105 | 0.133 | 0.188 | 6.4x |

  - At <= 1.5 bits no shared form beats RTN-2 over the whole expert. **Down has no usable shared output subspace**
    (0.44-0.65), and the pass line (MSB-3's error) is far away.
  - **Partial positive, the input side:** gate / up projected onto the layer's activation-weighted input subspace beat
    RTN-2 at about half the bits (r 768 at 1.35 b: 0.084 / 0.112 vs 0.130 / 0.171).
    - The layer inputs are anisotropic: H's top-512 of 2560 directions hold 78-88 % of the energy, top-1024 90-95 %.
    - This is a component for Fit12's codes: project, then GPTQ / VQ the coefficients. It is not a sub-2-bit expert by
      itself.
- **Prediction (before data):** a small shared part. Modern MoE experts are trained to specialize, and published sharing
  saves 24-30 %, not 4x. I expect the weight-space check to show the mean expert carries < 10 % of energy, but the shared
  INPUT subspace (activation-weighted) carries much more. So the input side may compress ~2x, the output side little.
  Roughly 50 / 50 on the pass line.

## L1-17 (Leap, pre-registered 15:10): fewer experts per token (top-k below 10, or an adaptive k)
- **Why:** bytes per 16-row pass scale with the union of experts it touches (73 / layer at top-10).
  - With Fit12's ~2-bit codes, a ~30 % smaller union gives a ~5.4 GB pass, the ~350 zone.
  - Non-exact: adopting it is Kamen's call. Testing is free.
- **Prior art (published, untested here):**
  - dynamic expert skipping (Lu et al. 2024, Mixtral);
  - top-p routing (Huang et al. 2024, "Harder tasks need more experts");
  - Ada-K routing (2024).
  - For fine-grained MoEs (hundreds of experts, top-8 / top-10) the router weights are known to be flat, so the risk is
    real.
- **Cheapest decisive test (no training, ~3 min GPU):**
  - Setup: Fit12's captured MoE inputs (5 layers x 8,336 tokens) through the real router and the real 4-bit experts.
  - Measure: the MoE-block output error (shared expert included) of
    - top-8 / 6 / 4 (renormalized, as the model does);
    - adaptive k: the smallest set holding >= p of the top-10 weight, p = 0.8 / 0.9;
    against MSB-3 on all routed experts, plus the 16-token union for each.
- **Kill:** top-8's output error > 3x MSB-3's. Then the routing is too flat to drop even 20 % of experts cheaply.
- **Pass (to an Eval-suite run):** some setting cuts the 16-token union by >= 25 % at <= 1.5x MSB-3's output error.
- **Prediction:** top-8 ~1-2x MSB-3's error (union -15 %), top-6 ~3-5x (union -30 %). The adaptive p 0.9 setting is the
  borderline case.
- **Result (15:31, data/topk17.txt; 5 layers x 8,336 tokens): NOT killed. The pass line is missed by one point.**

  | variant | MoE-block output error | vs MSB-3 on all 10 | 16-token union | vs top-10 |
  |---|---|---|---|---|
  | MSB-3 on all 10 (reference) | 0.0334 | 1.00x | 74.8 | 1.00 |
  | **top-8** | 0.0261 | **0.78x** | 62.4 | **0.84** |
  | adaptive p 0.9 | 0.0127 | 0.38x | 67.1 | 0.90 |
  | **adaptive p 0.8** | 0.0478 | **1.43x** | 57.2 | **0.76** |
  | top-6 | 0.0996 | 2.98x | 49.5 | 0.66 |
  | top-4 | 0.3245 | 9.70x | 35.6 | 0.48 |

  - **Top-8 costs less error than MSB-3 does**, and reads 16 % fewer experts per 16-row pass.
  - Adaptive p 0.8 reads 24 % fewer at 1.43x MSB-3's error. The pass line was -25 % at <= 1.5x.
  - The router is flat: of the top-10 weight, top-1 holds ~21 %, top-4 ~58 %, top-6 ~75 %, top-8 ~88 %.
  - The gated shared expert carries 15-56 % of the block's output energy, so routed-expert cuts cost less than the
    router weights suggest.
  - Next: a full-model quality verdict (Eval suite) on top-8 / adaptive p 0.8. The engine needs a routing flag (BigMoE /
    Breakthrough own the engine), so it goes through the orchestrator.
  - Stacked with ~2.5-bit expert codes (Fit12's Q3, if it lands), a 16-row pass is ~6.2 GB, ~320 tok/s at 80 % kernels.

## 24 GB: what 350 needs at C_m 192 (Leap 20:16; Kamen ~20:00: 350 on the 125B in <= 24 GB, 45 GB a waypoint)
- **Frame:** C_m 192 holds 192 of each layer's 512 experts in RAM at 3.0 bits (Fit12's 24 GB setting, 25.8 GB peak).
  The rest stream from the SSD on a miss, at ~10.6 GB/s.
  - Measured today (Fit12 RES1, exact): code 35.8 tok/s with MTP (4-row checks, ~3.4 tokens). Verify 63-82 ms, of
    which 31-49 ms is stall; 155-283 SSD reads per check.
- **Model** (scripts/calc24.py), one check of R rows:
  - Experts read per layer: U = 73.6 x (R / 16)^0.72.
  - Misses: (155-283) x (R / 4)^x, with x = 0.72 (like the union) or 1.0 (tail experts grow per row).
  - RAM time = (dense 2.1 GB + hits) / (280 GB/s x kernel efficiency).
  - SSD time = misses x 1.845 MB / 10.6 GB/s. Today it is exposed in full (Fit12's stall is about its read time). With
    perfect prefetch it would hide behind compute.
  - Plus drafting, 3.5 ms.
  - Calibration: on today's 4-row MTP frame the model gives 40-53 tok/s at 45 % kernels, against 35.8 measured, so it is
    15-50 % optimistic.
- **16-row checks at 24 GB** (misses from 155 / 283 per 4-row check, x 0.72; kernels 95 %):

  | tokens per check | prefetch | tok/s | SSD-bytes cut needed for 350 |
  |---|---|---|---|
  | 9.5 (today's best drafter, L1-20) | none / perfect | 89 / 58 ; 124 / 69 | impossible: the RAM side alone caps at ~290 |
  | 12.4 | perfect | 162 / 90 | impossible: RAM cap 345 |
  | **15 (Kamen)** | none | 141 / 91 | x10-18 |
  | **15** | **perfect** | 196 / 109 | **x1.9-3.4** (x2.7-5.0 if misses grow per row) |
  | 20 at 32 rows (block 32, oracle-picker level) | perfect | 161 / 89 | x2.2-4.1 |

- **So 350 at 24 GB needs all three at once:**
  1. **Tokens per check >= 12.6 at 16 rows, even with zero SSD reads.** That is the RAM floor at 95 % kernels, 3.0-bit,
     top-10 (11.3 with top-8; 14.7 at 80 % kernels).
     - Today's best is 9.5-10.2 (L1-20), and a picker adds +1 % (L1-21).
     - A perfect pick within the drafter's top-64 gives 14.1. Kamen's 15 is 96 % of the block-16 cap (15.6), so it
       means block 32 and its extra rows.
  2. **SSD bytes per check cut ~2-3.4x**, to <= ~0.42 GB per 16-row check (<= ~16 misses per generated token, against
     28-81 today), **and** reads fully overlapped with compute. Without the overlap, the cut is 10-18x.
  3. **Kernels at ~95 %** of 280 GB/s (today ~45 %).
- **SSD-side levers** (they multiply; each is uncertain):

  | lever | effect | status |
  |---|---|---|
  | top-8 routing | union -16 %, misses ~-20 % | passed the Eval suite 5/5 |
  | truncated rows | 16 -> 11 rows at -3 % tokens: misses -24 % | |
  | cold experts at 2.0-2.5 bits, misses only | -17-33 % | published as HOBBIT; needs an Eval verdict |
  | residency per layer by routing skew | ? | untested |
  | faster read path, 10.6 -> ~13 GB/s | -18 % | |
  | prefetch that works at 16-row width | hides the reads | Fit12: JIT prefetch dead at 12 GB |
  | skip / substitute missed experts (non-exact) | removes misses | Fit12's budget: b16 failed the suite; b20 being judged |

  - Product, if all the exact-side ones hold: ~2.5-3x.
- **45 GB waypoint (C_m 378; RES1: 76-135 misses per 4-row check):**
  - 15 tokens per check, 95 % kernels, perfect prefetch: 223-381 tok/s (SSD cut x1.0-1.6).
  - Without prefetch: 154-213 (cut x5-9).
  - At 9.5 tokens: <= 241.
- **L1-21's pass line, re-derived:** 13.1 tokens per 16-row check resident (top-8 stack). At 24 GB, no tokens-per-check
  value reaches 350 alone. It needs >= ~15 together with the SSD cut and the overlap above.
- **A pruned 125B (22:38; orchestrator: ISTA-DASLab GSQ-RCO-Coder keeps 256 of 512 experts per layer, 3.5 bpw;
  scripts/prune26.py on Breakthrough's routing traces, CPU, no download):**
  - RCO stand-in: keep each layer's 256 experts with the most top-10 gate mass.
  - Simulator check: unpruned 512 at C 192 gives 38-51 misses per token, against Fit12's measured ~46-83 (optimistic
    up to ~1.7x).
  - **Code at 16-row checks, misses per token:**

    | setup | misses per token |
    |---|---|
    | unpruned 512 @ C 192 | 46-48 |
    | 256 @ C 192 (75 % resident) | 13.3-19.3 |
    | 256 + top-8 @ C 192 | 10.5-14.1 |
    | 256 @ C 224 (fits 24 GB at ~3.0 b) | 5.7-8.1 |
    | 256 @ C 240 | 2.8-3.9 |

    - **Under the ~16 line with top-8 or C 224.**
  - Quality proxy (kept share of the top-10 gate mass): code 0.86-0.95. With a code-only keep-set, chat 0.40 and
    Bulgarian 0.36 (non-code collapses, as the card says); with a mixed calibration, 0.83-0.89 everywhere.
  - Pruning barely shrinks a check's bytes (distinct experts per layer, 16 rows: 72 -> 63-68; 54-57 with top-8). Its
    value is residency.
  - At 45 GB the whole 256-pool at 3.5 b (~30 GB) is resident: zero misses.
> (a quote from the project owner, kept private)
    experts per layer stay; no pruning or merging. The residency lesson carries over to all-expert methods below.
- **All 512 experts at 24 GB (22:41; scripts/resid27.py; same traces and replay):**
  - Misses per token, code, 16-row checks:

    | routing | C 192 | 224 | 256 | 288 | 320 | 352 | 384 |
    |---|---|---|---|---|---|---|---|
    | top-10 | 49 | 37 | 28 | 22 | 16 | 11 | 8 |
    | top-8 | 34 | 26 | 20 | 15 | 11 | 8 | 5 |

    - Truncated 11-row checks and 4-row MTP checks give the same per-token misses.
  - **<= 16 needs ~320 (top-10) or ~288 (top-8) of 512 resident** (~350 / ~320 if the replay is 1.7x optimistic).
  - What 24 GB holds (dense diet + 1.5 GB runtime): C <= 197 at 3.5 b, 230 at 3.0 b, 276 at 2.5 b, 345 at 2.0 b.
  - Cold experts stored at 2.0 b relax the SSD allowance: 350 allows 24.6 misses per token at 10.6 GB/s, 32.5 at 14
    GB/s (16.4 / 21.7 at 3.0 b).
  - The top-192 experts by usage carry 72-80 % of the routed mass, ranks 192-319 15-19 %, ranks 320-511 5-9 %.
  - Mass-weighted output error relative to MSB-3, using Fit12 Q3's codes:

    | setup (top-8) | misses per token | weighted error |
    |---|---|---|
    | 276 resident at 2.5 b, cold at 2 b | ~17 (29 at 1.7x) | ~4.4x |
    | 192 at 3 b + 57 at 2 b, cold at 2 b | ~20 | ~2.9x |
    | 3 b everywhere, C 230 | ~25 (SSD-bound 180-300 tok/s at 14 GB/s) | 1.5x |
    | B12LL (passed the Eval suite), for reference | - | ~0.5-0.6x |

  - **So exact routing at 24 GB needs 2-2.5-bit codes ~2-3x better than ours. Otherwise it stays at ~25 misses per
    token.**
  - **Non-exact, all 512 kept: frugal wait-top-2** (Breakthrough D1). My replay: 5.5 rank-0-1 misses per token on code
    at C 192 (9.3 at C 128; math 6.6, chat 5.5).
    - **CORRECTED (00:5x): the measurement does not support it.** The Eval suite's b12c192 is D1 on the v3 tiered store
      (4433333333, C_m 192, C_l 77, MTP). Run 1 measured **27.8 SSD reads per generated token (22.9 waited), 40 MB
      per token**, flat over the run, at 30.9 tok/s core.
    - It also **fails quality: 4/5, IFEval 82 vs 90** (-8.0, p 0.01; -5.0 vs B12LL); math 86 = ref; HumanEval+ 90.0.
    - Likely cause of the ~5x gap: my replay had one pool of whole experts. The real store waits for ranks 0-1 at 4
      bits = an MSB-3 base part (pool C_m 192) **plus an LSB-1 sidecar (pool C_l 77)**, and the small sidecar pool
      misses most of the time.
    - **Reconciled (01:40, scripts/b12rec29.py, data/b12rec29.txt):** with both pools (MSB C_m 192 + LSB C_l 77), the
      replay gives code 5.4 MSB + 15.5 LSB = **20.9 reads per token**, chat 21.8, math 23.7 (MTP or 16-row checks
      alike). That matches the measured **22.9 waited** reads per token. The measurement's other ~5 reads are
      unwaited, and its 40 MB per token is above my assumed part sizes (24-28 MB).
      - **The 77-slot sidecar pool causes ~3/4 of the reads.**
      - C_l 192 would give 11-13 reads per token (16-19 MB, +4.3 GB RAM); all sidecars resident, 5.4-6.6 (12-14 MB).
    - Quality: the Eval suite's pooled verdict on D1 at C_m 192 is **IFEval -6.0, fails**.
    - **So the 24 GB SSD wall is NOT solved by frugal.** Its reads are fixable (sidecar residency), its quality is not
      (yet).
  - Eval suite (22:4x): top-8 passes on U1 and U2. Adaptive p 0.8 fails (token-0 floods, likely a NaN).

## One number for Kamen (15:40), reconciling Leap's ~250-290 with Fit12's ~450-500

| setting (code, 16-row pass, 9.88 tokens / pass) | GB / pass | tok/s |
|---|---|---|
| Fit12's frame: 3.0-bit experts only (union 69), no dense, perfect kernels, no drafting | 6.1 | 453 |
| + the dense diet (2.1 GB per pass) | 8.2 | 337 |
| + drafting (3.5 ms per pass) | 8.2 | 301 |
| + 80 % kernel efficiency | 8.2 | 246 |
| **3.0-bit experts + adaptive k p 0.8 (L1-17) + dense diet + drafting + 80 % kernels** | **7.2** | **~280** |
| today's design, resident: 4-bit, top-10, full dense, 80 % | 13.4 | 156 |

- **The one number: ~280 tok/s on code.** Everything non-exact must pass the Eval suite. 350 needs **12.4 tokens per pass**
  (x1.26 over today's 9.88): the A3 bet below.

## L1-18 (Leap, pre-registered 15:42): how much drafting headroom does the 125B leave on code?
- **Hypothesis:** most of DFlash2's misses on code fall where the 125B itself is confident, so a stronger drafter has room.
  A drafter that matched the target wherever it is confident would accept >= 16 tokens per 32-row pass: 1.3x the 12.4
  needed.
  - If instead the misses sit at the target's own near-ties, no drafter can do much better. Exact tokens per pass would
    then be capped by the target's uncertainty, and only non-exact acceptance remains.
- **Prior art:**
  - acceptance-aware drafter training (TV objective, 2026); DistillSpec (on-policy distillation, alternative
    divergences); EAGLE-3 (training-time test); EAGLE 3.1 (up to 2x acceptance on long-context production);
  - acceptance vs drafter-target divergence (theory).
  - No measurement found of how much headroom a target's own uncertainty leaves. That measurement for this target is
    ours.
- **Cheapest decisive test (no training).**
  - Data: the 24 public HumanEval samples already extracted (data/dfl2d_code, 11,119 tokens of the 125B's own code).
  - Step A: one streamed 125B prefill adds its top-32 log-probs per position (dfl2c_extract.py).
  - Step B: replay DFlash2 (incoai weights) at block 16, as the official loop runs. Measure:
    - its agreement by target-confidence bin (top-1 p >= 0.99 / 0.9-0.99 / 0.5-0.9 / < 0.5) and by depth;
    - an oracle drafter's accepted length at block 16 / 32. The oracle is always right where the target's top-1 p >= t
      (t = 0.9, 0.99) and has DFlash2's observed agreement elsewhere.
- **Kill:** the oracle at block 32 (t = 0.9) accepts < 12.4 tokens per pass. A3-exact is then capped by the target's
  uncertainty.
- **Pass (go to a stronger drafter: bigger, on-policy distilled on the 125B's code, block 32):** oracle tau32 >= 16, and
  DFlash2's tau16 <= 0.7 x oracle tau16.
- **Prediction:** the 125B is very confident on most code tokens (p >= 0.9 on ~85-90 %), and DFlash2 misses a fair number
  of those. My guess: oracle tau32 ~15-20, so a pass, with headroom ~1.5x.
- **Result (15:51, data/headroom18.txt; 6,375 completion tokens, 715 blocks): between the lines, with a strong finding.**
  - **The 125B is nearly deterministic on code.** Of its completion tokens, 88.9 % have top-1 p >= 0.99; 4.0 % 0.9-0.99;
    5.6 % 0.5-0.9; 1.5 % < 0.5.
  - **DFlash2 is the bottleneck.** 63 % of its first misses fall on tokens the target is >= 99 % sure of.
    - Its agreement on those confident tokens decays with depth: 1.00 0.99 0.95 0.92 0.88 0.85 0.81 0.73 0.67 0.57 0.43
      0.30 0.22 0.16 0.12. It was trained at block 8.
  - Tokens per pass:

    | drafter | block 16 | block 32 | block 64 |
    |---|---|---|---|
    | DFlash2 (incoai), measured | 8.88 | - | - |
    | oracle: right where target p >= 0.9, DFlash2 elsewhere | 10.86 | 15.01 | 18.07 |
    | oracle, t = 0.99 | 9.86 | 12.84 | 14.71 |
    | perfect drafter (the target's argmax) | 15.59 | 30.50 | - |

  - Lines: the oracle tau32 is 15.01, so NOT killed (>= 12.4) but short of the pass line (16). DFlash2's 8.88 is above
    0.7x the oracle's 10.86.
  - **The headroom is real:** exact greedy on code allows ~30 tokens per 32-row pass with a perfect drafter. The loss is
    the drafter's depth decay on tokens the target finds obvious.
  - **Cheapest next lever, no training:** DFlash2 is a one-step block-diffusion drafter. Try a second, refining step that
    re-masks its low-confidence positions and predicts them again (MaskGIT-style). It costs one more drafter forward
    (~3.5 ms per pass), so it must add > ~10 % to tokens per pass.

## L1-19 (Leap, pre-registered 15:52): a 2-step refining DFlash2 drafter, no training
- **Idea:** DFlash2 drafts a block in one denoising step, and its deep positions are weak (L1-18). Step 2 feeds the block
  again with the confident step-1 tokens kept and the rest masked, and predicts again (MaskGIT-style iterative decoding).
  - The drafter's cache holds only target-context rows, so step 2 is one extra drafter forward over the block: ~3.5 ms per
    pass on the 125B.
- **Prior art:** iterative masked decoding (MaskGIT 2022; diffusion LMs: MDLM, LLaDA) is published. Untested here for
  DFlash2's one-step drafting. If DFlash was trained only on all-masked blocks, step 2 is out of distribution; the test
  shows it.
- **Cheapest decisive test:** a teacher-forced replay of the official loop on the 24 public HumanEval samples (~2 min GPU).
  Variants: re-mask conf < 0.5 / < 0.9 (kept tokens forced); re-mask the tail after the first unsure position; no masks.
- **Kill:** no variant adds >= 10 % tokens per pass at block 16. The extra forward costs ~10 % of a ~35 ms pass.
- **Pass:** >= +15 % (tau16 >= 10.2 from 8.88). Then measure it in the real loop and with the block-16 fine-tune.
- **Prediction:** 40 % chance of a pass. Likely the drafter was trained on all-masked blocks, and kept tokens confuse it.
- **Result (16:06, data/refine19.txt): KILLED.**
  - The reuse path reproduces step 1 exactly, so the test is valid. Every refining variant is worse:

    | variant | tokens / pass, block 16 |
    |---|---|
    | 1-step (as shipped) | 8.88 |
    | re-mask conf < 0.5 | 8.62 |
    | re-mask conf < 0.9 | 7.81 |
    | re-mask the tail | 8.25 |
    | no masks | 1.81 |

  - At block 32 the shipped drafter gets 8.56: its agreement collapses past depth ~10 (0.18 at depth 13).
  - DFlash2 was trained on all-masked blocks only, so filled-in tokens are out of distribution.
  - Any depth or refinement gain needs TRAINING the drafter. That is the next bet: a depth-robust block-32 retrain, with
    random partial masking so refinement becomes possible.

## L1-20 (Leap, pre-registered 16:42): a depth-robust block-32 DFlash2, trained with partial masking
- **Why:** L1-18 found ~2x headroom (perfect drafter 30.5 per 32-row pass), and the loss is DFlash2's depth decay.
  L1-19 showed it cannot refine untrained.
- **Training** (local: the drafter is < 1B parameters; the 1-2 h rule):
  - Starting point: the incoai weights, fine-tuned at block 32 on the 125B's own greedy outputs. Prompts: the 200 public
    L5 prompts (80 HumanEval+ problems the evals never use + 120 GSM8K), ~50K generated tokens.
  - Same loss as L5: on-path soft cross-entropy to the 125B's top-32 + the selector's loss.
  - New: random partial masking. Half the blocks reveal a random 10-60 % of their positions as the true tokens, and
    those positions get no loss. A step-2 refinement then becomes in-distribution.
- **Test:** a teacher-forced replay of the official loop on the 24 held-out HumanEval samples (data/dfl2d_code). Score
  tokens per pass, best of block 16 / 32 x 1-step / 2-step.
- **Kill:** < 10.0, i.e. no real gain over L5's block-16 fine-tune (9.88).
- **Pass:** >= 13.0, i.e. 12.4 with the real loop's 0.93-0.98 factor. Then confirm in the real loop, then scale the data:
  the 125B's self-generated coding problems, on-policy.
- **Prior art:** DFlash's own training; L5 here (block 16, +11 %); MaskGIT / masked-diffusion training; acceptance-aware
  drafter training. New here: block 32 with partial masking for a refining DFlash2.
- **Prediction:** ~11 tokens per pass: alive, not a pass on 50K tokens of data. 25 % pass, 20 % kill.
- **Result (19:13, data/refine19_l120.txt, data/cand21_l120.txt): BETWEEN THE LINES by its letter (10.51); +7 % once
  rows are priced. Its pass line was mis-set: it ignored that a 32-row verify costs more.**
  - Training: 109 windows (80 HumanEval+ + 29 GSM8K prompts; 99 train / 10 val by prompt; 25.2K generated tokens, 22.6K
    in train; corrected Mon 03:33, it said '42K'), 201 steps, 28 min,
    peak 33.7 GB, swap -0.2 GB. Held-out selector agreement at depth 16: 0.22 -> 0.63; depths 1-5 unchanged.
  - Tokens per pass on the 24 held-out HumanEval samples:

    | drafter | block 16, 1 step | block 16, 2 steps (best) | block 32, 1 step | block 32, 2 steps (best) |
    |---|---|---|---|---|
    | incoai (original) | 8.88 | 8.88 (L1-19: all worse) | 8.56 | 8.13 |
    | L5 fine-tune (block 16, Sat 05:47) | 9.88 | - | - | - |
    | **L1-20 (block 32 + partial masking)** | 9.54 | 9.58 | 10.23 | **10.51** (tail 0.9) |

    - Partial masking made step 2 safe (L1-19: -3 to -12 %), but it adds only +0.3 (+3 %).
  - **Priced at the stack** (scripts/cand21.py: pass = (2.1 + 5.06 x (rows / 16)^0.72) GB / 224 GB/s + drafting; rows =
    1 + drafts kept by L1-14's running-confidence truncation):

    | drafter | block 16, best t | block 32, best t |
    |---|---|---|
    | incoai | 292 tok/s (t 0.5, 8.22 at 9.7 rows) | 280 (t 0.5) |
    | L1-20 | 311 (t 0.3, 9.22 at 11.0 rows) | 313 (t 0.3, 9.73 at 12.0 rows) |

    - With top-8 instead of adaptive p 0.8 (Fit12 TK1: p 0.8 collapses acceptance on code; top-8 passed the Eval suite
      5/5), every figure is ~5 % lower: L1-20 best 298 tok/s. 350 then needs 13.1 tokens per 16-row pass.
  - **So block 32 buys nothing once rows are priced**, and the fine-tune is worth +7 %, as published (DBloom 2608.30427:
    block 16 -> 24 post-training, +0.8-1.4 tokens).
  - **The new finding is in the lattice.** The drafter's top-K candidates usually contain the 125B's token. Only its
    first-order selector picks wrong. An oracle that picks the true token whenever it is among each depth's top-K:

    | drafter | selector, block 16 | oracle top-16 | oracle top-64 | selector, block 32 | oracle top-64, 32 rows |
    |---|---|---|---|---|---|
    | incoai | 8.88 | 11.74 | 13.01 | 8.56 | 14.02 |
    | L1-20 | 9.54 | 12.83 (362 tok/s) | **14.11 (398 tok/s)** | 10.23 | **20.29** (401 untruncated) |

    - L1-20's top-64 recall stays >= 0.91 at every depth of block 16 and >= 0.79 to depth 31 of block 32.
    - A picker that conditions on the tokens it has already chosen could close much of that gap. That is L1-21.

## L1-21 (Leap, pre-registered 19:17): an autoregressive picker over the drafter's top-64 lattice
- **Why:** L1-20's lattice oracle. The 125B's token sits in the drafter's top-64 at >= 91 % of depths. A perfect pick
  gives 14.11 tokens per 16-row pass, but DFlash2's first-order selector gets 9.54. The selector scores a candidate
  only against the one token before it. A picker that conditions on every token it has already chosen, and on the
  drafter's hidden state at each depth, could close much of that gap.
- **Prior art:**
  - LiLiCorr (2608.20530, NVIDIA): a lattice scorer co-trained with DFlash, +7-19 % over vanilla DFlash.
  - Kim et al. (2404.09221, NeurIPS 2024): top-k lattice rescoring of blockwise-parallel drafts with n-gram / neural
    LMs, +5-25 %.
  - Sequential draft heads: Hydra, ReDrafter, Clover. Block-iterative drafting: SpecBlock. Causal parallel tree head:
    JetSpec.
  - DFlash2's own first-order selector (predecessor x hidden x successor codebooks).
  - Ours: the headroom measured on a near-deterministic MoE target (+48 % at block 16, +98 % at block 32). Untested
    here.
- **Picker** (~5 M parameters, local training): a 3-layer causal transformer over depths (d 512).
  - Input at depth j: the drafter's hidden state at j, plus the 125B's embedding of the token chosen at j-1.
  - Each of the 64 candidates is scored from the selector's successor codebook, its rank and its drafter logit.
  - Trained with teacher forcing on the 125B's own tokens from L1-20's 109 windows. It chooses greedily (exact
    left-to-right, run as Jacobi iterations).
  - Drafters: L1-20 (main), and incoai as a clean control (it never saw these windows).
- **Test:** the held-out HumanEval replay (24 samples), as L1-20, at blocks 16 and 32. Score tokens per pass, and tok/s at
  the stack with running-confidence truncation (cand21.py's pricing).
- **Kill:** < 10.5 tokens per 16-row pass (< +10 % over the selector's 9.54), and no priced gain >= +10 % at block 32.
- **Pass:** >= 13.1 tokens per 16-row pass, or >= 350 tok/s priced at the top-8 stack. Then measure the picker's cost
  per pass (budget <= 1 ms, as one fused kernel), and train the drafter and picker together on more on-policy data.
- **Prediction:** ~11.0-11.5 at block 16 (captures ~1/3 of the oracle gap): alive, not a pass. 20 % pass, 25 % kill.
- **Prior-art note added 19:22, before the run:** LiLiCorr's own table (Qwen3-8B, greedy, K 8, block 16) puts DFlash2's first-order
  selector level with LiLiCorr on HumanEval (7.08 / 7.08). The per-slot autoregressive correctors Domino and DSpark get 6.86 /
  7.16 (+1 % at best). So a small picker has published evidence against a big gain. I now expect ~10.0-10.5 (a
  kill or near it). The run still measures how much of a 64-wide lattice a picker captures. If it is little, the gap
  needs the target's own knowledge: verify several candidates in the pass (a tree), priced by rows.
- **Result (20:20, data/pick21_ft.txt, pick21_orig.txt): KILLED.** The picker adds +1 %, as the published per-slot
  correctors do.

  | drafter, block | selector | picker | oracle top-64 | best priced, selector -> picker |
  |---|---|---|---|---|
  | L1-20, 16 | 9.54 | 9.65 (+1.2 %) | 14.11 | 311 -> 312 tok/s (t 0.3) |
  | L1-20, 32 | 10.23 | 10.33 (+1.0 %) | 20.29 | 313 -> 315 |
  | incoai, 16 (control) | 8.88 | 9.00 (+1.4 %) | 13.01 | 292 -> 292 |

  - Training (11.3K blocks from L1-20's windows; 12.3 M parameters) plateaued almost at once: the loss went 0.34 -> 0.28
    and stayed there.
  - The true token is in the lattice (top-64 recall 0.98 on held-out windows), but nothing in the drafter's hidden
    states plus the chosen prefix tells a small model which candidate it is.

## L1-22 screen (Leap, pre-registered 19:24): let the target pick, with a token tree over the lattice, priced by rows
- **Why:** if a small picker cannot choose within the lattice (L1-21, and the published per-slot correctors), the
  125B can. It verifies a tree of candidates in one pass, but every extra row costs expert bytes.
- **Tree:** best-first over the selector's 16 candidates per depth, scored by the selector's own conditional scores, so
  the greedy chain is the tree's top path. N nodes + the anchor = N + 1 rows. Blocks 16 (N 15-63) and 32 (N 31-95).
- **Prior art:** token trees are published (SpecInfer, Medusa, EAGLE-2 dynamic trees, Sequoia, JetSpec for parallel
  drafters). Here they were parked at block 8 (L1-11: the block cap). Untested at block 16 / 32 on this MoE with rows
  priced; GDN layers would need tree states (TreeWY-style).
- **Kill:** no tree beats the best truncated chain (311 tok/s at the p0.8 stack, 296 at top-8) by >= +10 %.
- **Pass:** >= 350 tok/s at the top-8 stack. Then: tree-aware GDN verify cost on the real engine.
- **Prediction:** rows^0.72 makes wide trees costly. Best ~ +0-5 % at N ~23-31: a kill.
- **Result (20:20, data/tree22.txt): KILLED by its line.** Trees raise tokens per check at equal rows, but rows^0.72
  eats the gain:

  | tree | tokens per check | rows | tok/s at the stack (top-8) |
  |---|---|---|---|
  | chain, block 16 (selector) | 9.54 | 16 | 269 (254) |
  | chain, block 16, truncated t 0.3 | 9.22 | 11 | **311 (296)** |
  | tree N 15, block 16 | **10.43** (+9 % at the same 16 rows) | 16 | 294 (278) |
  | tree N 23 / 31 / 63, block 16 | 11.01 / 11.30 / 11.67 | 24 / 32 / 64 | 255 / 226 / 157 |
  | tree N 31 / 95, block 32 | 12.14 / 13.09 | 32 / 96 | 240 / 137 |

  - A tree of ~10 nodes (11 rows) was not run. My estimate is ~9.6-9.8 tokens, +4-6 % over the truncated chain: still
    under the +10 % line.
  - Exact tokens per check is stuck near 9.5-10.5 at <= 16 rows across every drafting lever tried today: retraining,
    refinement, a picker, trees. The information is in the drafter's lattice (an oracle reaches 14.1), but neither a
    small model nor an affordable tree extracts it.

## L1-23 (Leap, pre-registered 20:26; MoonShots' L1-01b): bit-exact 16-row kernels for the 125B's dense layers
> (a quote from the project owner, kept private)
  shapes (data/b8mr/denseinv.txt). A bit-exact 16-row check today chunks its rows (12 + 4 for the 4-bit shapes, 5s for
  the bf16 hyper-connection mixers), because MLX's kernels change numerics with the row count. Its dense layers take
  ~57 ms (4-bit ~37.5, bf16 ~17.5, 8-bit ~1.7), against a ~12 ms floor and ~16 ms for 1-row decode. Exactness costs
  ~40 ms per 16-row check, at any RAM budget.
- **Bet:** one batch-invariant kernel per shape, called with 16 rows every time (decode pads), so each row's bits never
  depend on how many rows run.
  - 4-bit: qmm16c (L1-15's NAX code-MMA kernel), or MLX's own 16-row call where it is faster and padding-independent.
  - bf16: MLX's 16-row call, which has a fixed tile configuration, if padding-independent.
- **Prior art:** batch-invariant kernels (Thinking Machines, "Defeating Nondeterminism in LLM Inference", 2025; SGLang
  deterministic mode); "exact under its own numerics" for speculative decoding (here: L1-15, MoonShots L1-01).
  - New here: the 125B's whole dense set at 16 rows on the M5's neural accelerators.
- **Test** (scripts/kern23.py, ~5 min on the GPU): random weights with the 125B's dense shapes and per-check counts
  (3.4 GB, cold cache).
  - Per shape: MLX at 1 row, MLX at 16 rows, MLX chunked as today, qmm16c configurations, and the floor at 280 GB/s.
  - Bitwise: every real row equal across row counts 1-16 and across padding contents.
- **Kill:** the best exact 4-bit set takes > 25 ms per 16-row check (today ~37.5).
- **Pass:** <= 15 ms (floor ~9). Then it goes to Fit12 / BigMoE as an exact verify path, and the bf16 mixers come next.
- **Prediction:** 4-bit ~12-16 ms (qmm16c at ~70-85 % of the floor on the big shapes; launch-bound small shapes);
  bf16 ~8-12 ms. A pass, narrowly.
- **Result (20:58, data/kern23.txt): PASS.**
  - The 4-bit set takes 14.0 ms per 16-row check, against 35.2 today, chunked (floor 7.1).
  - The whole dense part falls from 56.8 to 27.4 ms (floor 11.9; 1-row decode 15.6): **-29 ms per 16-row check**, exact.
  - **MLX's own 16-row call is padding-independent for every shape** (rows 1/2/5/8/12 padded with zeros or random rows
    equal the all-real call, bitwise). So "always call with 16 rows" is exact even without a new kernel.
  - qmm16c wins where MLX is slow at 16 rows, and is as accurate or more (max error vs fp32 0.010 vs 0.023 on o_proj):

    | shape (count per check) | exact today | MLX 16 | qmm16c (ks, nt) | floor |
    |---|---|---|---|---|
    | o_proj + GDN out_proj (48) | 6.87 | 7.51 | **2.39** (2, 2) | 1.52 |
    | in_proj_z (36) | 5.18 | 5.65 | **1.79** (1, 1) | 1.14 |
    | in_proj_qkv (36) | 9.23 | 3.31 | **2.53** (2, 2) | 1.90 |
    | lm_head (1) | 5.69 | 2.21 | **1.53** (4, 1) | 1.28 |
    | q_proj (12) | 3.45 | 1.44 | **1.29** (2, 2) | 0.76 |
    | shared down (48) | 1.41 | 1.39 | **1.12** (1, 2) | 0.16 |
    | small 4-bit: shared gate/up, k/v, in_proj_a/b, expert gate | 3.4 | **3.3** | 3.7 | 0.4 |
    | bf16 HC mixers up / down (97 each) | 9.64 / 6.73 | **2.85 / 5.83** | - | 2.27 / 2.27 |
    | bf16 block_inject 4 x 10240 (96) | 4.07 | 4.06 | - | 0.03 |

  - The new outputs differ from today's engine by rounding (new reference outputs); speculative == plain decode holds
    bitwise when both use these kernels.
  - Left on the table: block_inject (4 ms for an 82 KB matrix, launch-bound), HC mix down (2.6x its floor), the small
    shapes (launch-bound, ~3 ms; fuse gate + up). Then the routed experts, the biggest byte block (~10 GB at 4-bit,
    ~35 ms at the floor per 16-row check).
- **Follow-ups (21:45, data/bf23.txt, data/exp24.txt):**
  - bf16 shapes: other MLX forms are bitwise equal to x @ W.T and also padding-independent. Picking the fastest per
    shape takes bf16 from 13.4 to 9.0 ms. block_inject as (W @ x.T).T: 3.48 -> 0.38 ms. HC mix up with W.T made
    contiguous: 2.66. HC mix down as (W @ x.T).T: 5.32.
  - **The dense part of an exact 16-row check is ~23 ms** (+ ~1.7 ms of 8-bit router / indexer), from 56.8.
  - **Routed experts (L1-24 screen,** real 4-bit weights of 4 layers, synthetic routing with the measured union):

    | rows | distinct experts per layer | floor per check | best path | share of the floor |
    |---|---|---|---|---|
    | 4 | 25 | 11.8 ms | 18.9 ms | 63 % |
    | 8 | 42 | 19.9 ms | 33.4 ms | 60 % |
    | 16 | 72 | 34.1 ms | 60.7 ms | 56 % |

    - The three paths the engine has (unsorted per-pair gather, MLX's sorted tensor-unit gather, BigMoE's row-map
      kernel) are bitwise identical to each other, and rows alone equal rows inside a 16-row call. **The experts are
      already exact at any row count; they need no chunking.**
    - Their inefficiency is probably wasted compute: MLX's sorted kernel tiles 32 / 64 rows per expert for ~2 real
      rows. A few-rows expert kernel (weights read once, only real rows computed) could recover ~20-25 ms per
      16-row check at 4-bit.
  - **A full exact 16-row check today** (resident, 4-bit, top-10) is ~100 ms: dense ~25, experts ~61, attention / GDN
    core / routing / glue ~10-20 (not measured here), drafting ~3.5.
    - Its byte floor is ~46 ms; with 3.0-bit + top-8 + dense diet, ~27 ms.
    - So 350 at ~10 tokens per check needs all three: those byte levers, ~90 % kernels, and more tokens per check
      (~12).

## L1-24 (Leap, pre-registered 21:51): read each distinct expert once, with MLX's own per-row arithmetic
- **Why:** the L1-24 screen above. A 16-row check's routed experts take 60.7 ms (4-bit), against a 34.1 ms floor
  (56 %).
  - MLX's per-pair path streams at near-peak bandwidth but re-reads an expert for each row it serves.
  - The sorted path reads each expert once, but slowly.
  - All of the engine's expert paths are bitwise equal to MLX's per-pair single-row arithmetic.
- **Bet:** scripts/gmr24.py, two kernels per layer.
  - One threadgroup per (distinct expert, 8 output rows). Each 512 / 256-value weight block is loaded once and then
    applied to every row routed to that expert, with MLX 0.32.2's load_vector / qdot (and the qmv tail for K 640).
  - MLX's silu(gate) * up between the two kernels; segments built on the GPU (no host sync).
  - Bit-identical outputs, so no reference change.
- **Prior art:** MoonShots' qmv_mr (L1-01: the same idea for dense layers, bitwise, but slower than MLX's 16-row qmm
  there); Breakthrough's tier_glu (per-pair fused expert GLU, bitwise); grouped / sorted MoE GEMMs (MegaBlocks,
  MLX's own sorted gather). New here: reading each expert once with the single-row arithmetic, for few-rows verify
  checks on the M5.
- **Test** (scripts/test_gmr24.py; real 4-bit weights of 4 layers): bitwise vs the engine's path at 1 / 4 / 8 / 16 rows,
  plus one expert in all 16 rows; ms per layer at 4 / 8 / 16 rows vs the sorted path and the floor.
- **Kill:** not bitwise equal, or >= 1.10 ms per layer at 16 rows (< 13 % faster than the sorted path's 1.264).
- **Pass:** bitwise equal and <= 0.90 ms per layer at 16 rows (>= 79 % of the floor): -17 ms or more per 16-row check.
- **Prediction:** bitwise equal; ~0.85-1.0 ms per layer at 16 rows (70-84 % of the floor).
- **Result (22:30, data/gmr24.txt): KILLED by its line, with the cause found.**
  - **Bitwise equal** to the engine's path at 1 / 4 / 8 / 16 rows, and with one expert in all 16 rows.
  - But only 6 % faster: at 16 rows (58 distinct experts), gmr 1.132 ms per layer vs sorted 1.209 vs a floor of 0.575
    (51 %). At 8 rows 0.651 vs 0.663; at 4 rows 0.398 vs 0.380.
  - **Cause: the expert path is ALU-bound, not bandwidth-bound.** Time grows with routed pairs (~7 us each: 40 / 80 /
    160 pairs = 0.40 / 0.65 / 1.13 ms), not with distinct experts. Exp24's sorted path barely moved between 72 and 58
    distinct experts.
    - MLX's single-row arithmetic (pre-scaled x, masked nibbles, scalar FMAs) costs ~4.9 M MACs per pair on the
      SIMD ALUs. Reading each expert once cannot fix that.
  - **The fix is the tensor units.** Run each expert's rows through a 16-row NAX tile (qmm16c-style). Compute then
    becomes nearly free and the check bandwidth-bound: ~0.6-0.75 ms per layer at 16 rows, i.e. -20 to -25 ms per
    check.
    - Cost: new rounding, a one-time reference change, as with L1-23's qmm16c. Batch-invariant by construction
      (padding-independent tiles), with decode on the same kernel.

## L1-25 (Leap, pre-registered 22:32): the routed experts on the tensor units (one 16-row NAX tile per expert)
- **Why:** L1-24. The expert path is ALU-bound on MLX's single-row arithmetic (~7 us per routed pair). On the matrix
  units the compute is nearly free, so it becomes bandwidth-bound.
- **Bet:** scripts/qmmg25.py = qmm16c's code-MMA kernel with a segment prologue and masked stores.
  - The prologue picks the expert's weights and gathers the expert's token rows; rows past the count copy row 0 and
    are never stored.
  - One threadgroup per (distinct expert, 32 output columns), for gate / up and for down; MLX's silu(gate) * up between.
  - Batch-invariant by construction. New rounding (fp32-exact dequant, as qmm16c), so the reference outputs change
    once, for decode and verify together.
- **Prior art:** grouped / sorted MoE GEMMs on tensor cores (MegaBlocks, vLLM / SGLang fused MoE, MLX's sorted
  tensor-unit gather); batch-invariant kernels. New here: one fixed 16-row tile per expert for exact few-rows verify
  checks on the M5.
- **Test** (scripts/test_qmmg25.py, real 4-bit weights of 4 layers):
  - bitwise row-independence (rows alone vs inside the 16-row call);
  - max error vs an fp32 reference, against MLX's path;
  - ms per layer at 4 / 8 / 16 rows over (ks, nt) configurations, vs MLX's sorted path and the floor.
- **Kill:** not batch-invariant, error worse than MLX's path, or > 0.95 ms per layer at 16 rows (sorted 1.21-1.26).
- **Pass:** <= 0.75 ms per layer at 16 rows (-22 ms or more per 16-row check), invariant, error <= MLX's.
- **Prediction:** invariant, error ~0.5-1x MLX's, ~0.65-0.8 ms per layer at 16 rows.
- **Result (22:38, data/qmmg25.txt): BETWEEN THE LINES**, 3 % short of the pass line.
  - Batch-invariant: rows 1 / 2 / 4 / 7 / 8 / 12 alone equal the same rows inside the 16-row call, bitwise.
  - 2.3x more accurate than MLX's path vs fp32 (max error 0.00061 vs 0.00139).
  - Speed per layer, vs MLX's sorted path:

    | rows (distinct experts) | NAX | MLX sorted | share of the floor (NAX) | per check, NAX vs sorted |
    |---|---|---|---|---|
    | 16 (59) | **0.776 ms** | 1.242 | 75 % | **37.2 vs 59.6 ms (-22 ms)** |
    | 8 (34) | 0.496 | 0.664 | 69 % | 23.8 vs 31.9 |
    | 4 (22) | 0.351 | 0.383 | 63 % | 16.9 vs 18.4 |

    - Best configurations: gate / up ks 4, nt 1; down ks 1-5.
  - **With L1-23:** an exact resident 16-row check's dense + experts go from ~116 to ~60 ms (4-bit). The remaining
    parts (attention, GDN core, routing, glue) are not measured here.
  - Rounding changes once, for decode and verify together.

> (a quote from the project owner, kept private)
- **Why:** L1-24 / L1-25. The 16-row paths are ALU-bound on unpacking.
  - qmm16c's lane reads 8-byte pieces of 4 weight rows per 64-value group.
  - It spends 3 ops (shift, mask, or 0x4300) per 4-bit value to make a bf16 operand.
- **Bet:** scripts/qmm26.py stores the weights in the order the tensor-unit fragments consume them.
  - Per (32-column tile, group, lane): the lane's 8 words are contiguous (the simdgroup reads one 1 KB block, as two
    uint4 per lane).
  - Each word pairs the same 4 codes of the lane's rows fm and fm + 8, so one (shift, mask 0x000F000F, or
    0x43004300) makes two bf16 operands: half the unpacking ops.
  - Scales / biases are stored per (tile, group) as 32 contiguous columns.
  - The MMA gets the same values in the same order, so it should be bit-identical to qmm16c / experts_nax (no new
    reference beyond L1-23 / L1-25's). The store is repacked once at load.
- **Prior art:** Marlin and FasterTransformer (interleaved int4 layouts, multi-value conversion per op); FLUTE
  (offline restructuring to cut unpacking); BitBLAS / Ladder (layout transforms); Apple WWDC26 session 330 (dequant
  into a cooperative tensor fed to matmul2d; native int4 / int8 operands from a macOS 26 update). New here: a
  fragment-native layout for the M5's matmul2d 16x32x16 in exact few-row checks.
- **Test** (scripts/test_qmm26.py): bitwise vs qmm16c on 3 dense shapes and vs experts_nax on real experts (4 / 8 / 16
  rows); ms per check vs qmm16c / experts_nax and the floor.
- **Kill:** not bitwise equal, or < +10 % on the 16-row experts (experts_nax 0.776 ms per layer).
- **Pass:** >= 85 % of the floor on the 16-row experts (~0.68 ms per layer, -4.6 ms per check) and bitwise equal.
- **Prediction:** bitwise equal; 16-row experts ~0.66-0.72 ms per layer (80-88 %); dense shapes +10-25 %.
- **Result (23:31, data/qmm26.txt): KILLED by its letter on the experts; a clear gain on the dense layers.** Bitwise equal
  everywhere.
  - A first run looked 2.5x slower on the experts: a fixed ~1.5 ms per layer. The cause was strided scale / bias arrays
    (numpy moveaxis views), which metal_kernel copies on every call. Made contiguous, then rerun.
  - **Experts** (real weights), native vs experts_nax:

    | rows | native | experts_nax | gain | share of the floor (native) |
    |---|---|---|---|---|
    | 16 | 0.709 ms per layer | 0.762 | +7.5 % | 82 % |
    | 8 | 0.461 | 0.505 | +9.5 % | |
    | 4 | 0.323 | 0.346 | +7 % | |

    - At 16 rows that is 34.0 vs 36.6 ms per check: under the +10 % kill line and the 85 % pass line.
  - **Dense 4-bit shapes, same run:**

    | shape | native | qmm16c | gain | share of the floor (native) |
    |---|---|---|---|---|
    | in_proj_qkv | 2.21 ms | 2.57 | +16 % | 86 % |
    | o_proj + out_proj | 1.92 | 2.82 | +47 % | 79 % |
    | in_proj_z | 1.48 | 2.33 | +57 % | 77 % |

    - qmm16c's own times vary +-15-25 % between runs; the comparison within one run is what counts.
  - So adopt the layout for the dense 4-bit set (L1-23's 14.0 ms -> ~11 ms est.). On the experts it is a small extra
    (+7.5 %).
  - **An exact resident 16-row check, 4-bit, top-10, measured kernels:** dense ~20-21 ms (floor 11.9) + experts ~34-41
    ms (floor 28-34) = ~55-62 ms of matmuls, ~75-80 % of their floor.

## Mend item (c), measured (23:30; scripts/ovh27.py, data/ovh27.txt): is a native per-check loop worth it?
- MLX's CPU-side cost per op, on a chain of small ops (GPU work negligible):
  - eager graph build + eval 3.1-3.4 us per op (building alone 0.4-0.5 us);
  - through mx.compile 0.4-0.7 us per op;
  - a small matmul ([16, 2560] x [2560, 256]) 16-25 us.
- A 125B verify check runs ~2,000-3,000 kernels (48 layers x ~40-60). The overhead per check is ~6-9 ms eager
  (10-15 % of a ~60 ms kernel-optimised check) and ~1-2 ms compiled (1.5-3.5 %).
- Fit12 measured "pass edges" at ~5 % of a check in the engine.
- So a native loop pays only where the verify path still builds eager graphs. First check which parts of BigMoE's
  verify path are compiled (MTPLX_COMPILED_VERIFY / QWEN4EXP_COMPILE); compiling the rest gets most of the gain
  without a rewrite.
- It becomes worth doing once the kernels bring the check under ~30 ms: 2-9 ms would then be 7-30 %.
- Item (b), the drafter on the Neural Engine, is KILLED on arithmetic. Drafting is 3-8 % of a check and has to wait for
  the check's result (prior art: Mirror-SD 2510.13161).

> (a quote from the project owner, kept private)
- **Why:** L1-24 / L1-26. The few-row paths are ALU-bound on unpacking: experts reach 82 % of the floor with the
  native layout, the dense shapes 77-86 %.
  - The installed SDK (26.5; MPPTensorOpsMatMul2d.h) lists **bfloat x uint4b_format -> float** for matmul2d.
  - MLX's affine codes are unsigned 0..15, two per byte, so the packed buffer can be the right operand as is (a
    tensor_inline over the raw pointer). The probe compiled with xcrun metal -std=metal4.0
    (scripts/metal/t27.metal).
- **Bet:** scripts/qmm27.py. Per 64-value group: one matmul2d(16 x 32 x 64) of the x rows (bf16 tensor_inline) with the
  codes (uint4b_format tensor_inline) into a float cooperative tensor; then acc += scale[n] * part + bias[n] *
  rowsum[m] per element (get_multidimensional_index).
  - No software unpacking.
  - New rounding (the tensor unit's own accumulation), batch-invariant by construction (fixed tiles).
- **Prior art:** Apple WWDC26 session 330 ("feed your quantized data straight into TensorOps"); BaseRT (2607.19438,
  Metal 4 tensor kernels on M5, formats not public); Rigel (2606.12765, reverse-engineering the Metal 4.1 tensor path
  on M4 Max). New here: native int4 operands for exact few-row 4-bit checks of a 125B MoE.
- **Test** (scripts/test_qmm27.py):
  - error vs fp32 (vs MLX and qmm16c), and bitwise row-independence;
  - ms on 3 dense shapes and on real experts at 4 / 8 / 16 rows, vs qmm16c / qmm16n (L1-26) / experts_nax / the
    native layout.
- **Kill:** error > MLX's, not batch-invariant, or 16-row experts not faster than the native layout (0.709 ms per
  layer).
- **Pass:** >= 90 % of the floor on the 16-row experts (<= ~0.64 ms per layer) or on the dense shapes.
- **Prediction:** the K-loop of 64-wide ops with per-element coordinate epilogues costs some ALU, so ~85-92 % of the
  floor. 40 % pass, 25 % kill (the API may load the operands less efficiently than hand-fed fragments).
- **Result (23:40, data/qmm27.txt): KILLED. It works but is slower.**
  - The first run failed to compile. The runtime compiler rejects `#pragma unroll full`, and the element mask is
    `is_valid_element(i)`, not the header comment's `get_mask`.
  - Error vs fp32 is identical to qmm16c's (better than MLX's on o_proj / in_proj_z), and batch-invariant (bitwise).
  - **Slower:**

    | shape | int4 operands | qmm16c (same run) | share of the floor (int4) |
    |---|---|---|---|
    | 16-row experts | 0.992 ms per layer | 0.757 (native layout 0.709) | 58 % |
    | in_proj_qkv | 3.05 ms | 2.60 | 62 % |
    | o_proj | 3.41 | 2.31 | 45 % |
    | in_proj_z | 2.26 | 2.07 | 50 % |

  - Letting the tensor op load its operands through tensor_inline (one op per 64-value group) is less efficient than
    hand-fed fragments.
  - **Correction to L1-26:** in this run, at equal (ks, nt), the native layout was slower than qmm16c on the dense
    shapes (2.78 / 2.68 / 2.56 vs 2.60 / 2.31 / 2.07 ms). L1-26's "+16-57 %" compared native's best of 5
    configurations against qmm16c's one, so the selection was partly noise. A fair interleaved A/B is queued
    (scripts/ab28.py). Until it lands, the dense gain is unconfirmed.
- **Fair A/B (23:41-, data/ab28.txt; equal (ks, nt), 12 shuffled interleaved rounds x median of 7, paired ratios): the
  native layout's dense gain is REAL, at its honest size.**

  | shape | qmm16c | native layout | int4 operands |
  |---|---|---|---|
  | in_proj_qkv | 2.571 ms (74 %) | **2.244 (84 %), x1.15** (IQR 1.14-1.15) | 3.114 (61 %), x0.82 |
  | o_proj + out_proj | 2.364 (64 %) | **1.949 (78 %), x1.22** (IQR 1.21-1.23) | 3.484 (44 %), x0.68 |
  | in_proj_z | 1.798 (63 %) | **1.505 (76 %), x1.20** (IQR 1.19-1.20) | 2.074 (55 %), x0.87 |
  | experts, 8 rows (vs experts_nax 2.060 ms, 68 %) | | **1.925 (73 %), x1.06** (IQR 1.06-1.08) | 2.570 (55 %), x0.80 |

  - The single sequential run in L1-27 that showed native slower was the noisy one.
  - L1-26's original +16-57 % (best of 5 configurations) was inflated. The honest gain is +15-22 % on the dense
    shapes and +6-8 % on the experts. Adopt the layout for the dense set.
  - 16-row experts (finished 00:42, after the quiet window): native 2.751 ms (81 %) vs experts_nax 2.990 (74 %),
    **x1.08** (IQR 1.08-1.09); int4 x0.77.
  - **Where an exact resident 16-row check stands (4-bit, top-10):**

    | part | today, chunked | now | floor |
    |---|---|---|---|
    | dense layers | 56.8 ms | ~21 ms | 11.9 |
    | routed experts | ~60 ms (MLX sorted) | ~34 ms | ~28-34 |

    - Not yet measured: attention, the GDN core (Fit12's port), routing and glue.

## L1-28 (Leap, pre-registered 01:43): the native-layout kernel with two 32-column tiles per simdgroup
- **Why:** with the native layout, the 16-row experts are at 81 % of the floor and the dense shapes at 76-84 %.
  Per 64-value group each tile re-loads its 16 x rows (2 KB from cache per 1 KB of weights) and re-sums them.
- **Bet:** scripts/qmm28.py. Two tiles share the x fragments, the row sums and the K loop; each keeps its own B
  fragments, accumulator and epilogue. The arithmetic per output is unchanged, so it should be bit-identical to
  qmm16n.
- **Test** (scripts/ab29.py): fair A/B at equal ks, interleaved 12 rounds, paired ratios; 3 dense shapes and 16-row
  experts (real weights); bitwise check.
- **Kill:** < +5 % on both the 16-row experts and the dense shapes, or not bitwise.
- **Pass:** >= +10 % or >= 88 % of the floor on the 16-row experts.
- **Prediction:** +5-12 %.
- **Result (01:44, data/ab29.txt): KILLED.** Bitwise equal to qmm16n everywhere, but slower at equal ks:
  - 16-row experts x0.94 (IQR 0.94-0.95; 77 % vs 82 % of the floor);
  - in_proj_qkv x0.99, o_proj x0.96, in_proj_z x0.99.
  - Sharing the x loads costs more than it saves; the doubled accumulators and fragments are the likely cause
    (register pressure, lower occupancy).
  - The native layout at 75-85 % of the floor looks near this design's practical limit.
  - The rest of a 16-row check's gap is elsewhere: the bf16 HC mixers (~8.5 vs 4.5 ms), the GDN core (Fit12's 4-row
    map: ~5.4 ms over a ~1 ms floor; more at 16 rows), and check edges / sync (~9-15 ms).

## L1-29 (Leap, pre-registered 01:46): a bf16 16-row tensor-unit matmul for the hyper-connection mixers
- **Why:** the HC mixers take ~8.5 ms per 16-row check, about 2x their floor (mix down 5.32 ms vs 2.27, x97).
- **Bet:** scripts/bmm30.py, qmm16c's MMA skeleton with bf16 weights read directly; K split over KS simdgroups;
  batch-invariant by construction; new rounding.
- **Test** (scripts/test_bmm30.py): config sweep, then a fair interleaved A/B vs MLX's best invariant form;
  invariance; error vs fp32.
- **Kill:** < +15 % on HC mix down.
- **Pass:** >= 80 % of its floor (<= 2.85 ms).
- **Prediction:** ~3.0 ms (x1.75).
- **Result (01:47, data/bmm30.txt): PASS.**
  - HC mix down: **2.59 ms per check (88 % of the floor)** vs MLX's best invariant form 5.11 (44 %), paired **x1.97**
    (IQR 1.97-1.98), at ks 10 / nt 1. Batch-invariant (bitwise); max error vs fp32 = MLX's.
  - HC mix up: no gain (x0.97); keep MLX's x @ contiguous(W.T) at 2.63 (86 %).
  - ple key: x1.18.
  - **The dense part of an exact 16-row check is now ~18 ms** (was ~21; 56.8 before Sat night).

## L1-30 (Leap, pre-registered Sun 02:55): do fewer bits buy time in RAM? 3-bit and 2-bit experts on the tensor units
- **Why:** the ~72-77 ms check assumes the ~3.5-bit tier experts run at the 4-bit kernel's 82 % of floor, and the 350
  recipe needs ~2-bit experts at both budgets. But the few-row expert path is limited by unpacking arithmetic (L1-24 / L1-26):
  4-bit native = 3 ALU ops per two bf16 operands. 2-bit costs the same per operand on half the bytes; 3-bit (a 2-bit plane
  + a 1-bit plane) costs ~6. If unpacking is the limit, fewer bits buy less than their bytes.
- **Prior art:** low-bit GEMM kernels with fast dequant are published (BitBLAS / Ladder, FLUTE, LUT Tensor Core, QuIP# /
  AQLM kernels, MLX qmm at 2-8 bits, llama.cpp Q2_K / Q3_K Metal). Tag: published style, untested on the M5's tensor units
  at 16 rows.
- **Test** (test_low30.py, GPU ~5 min): one native-layout kernel at 4 / 3 / 2 bits (the 4-bit build must be bitwise equal
  to experts_native); real codes from 4 layers (3-bit = q >> 1 as the v3 tier stores it, 2-bit = q >> 2, for timing);
  synthetic routing as in ab28 at 8 / 16 rows; fair interleaved A/B (12 rounds x median of 7) against MLX's sorted
  gather_qmm at 4 / 3 / 2 bits; error vs fp32; row invariance.
- **Ideal speedups from bytes** (g64 codes + bf16 scale and bias): 3-bit x1.29, 2-bit x1.80 vs 4-bit.
- **Prediction:** 3-bit x1.0-1.15, 2-bit x1.4-1.6 at 16 rows.
- **Kill line** (for "fewer bits convert to RAM time"): 2-bit < x1.4 vs 4-bit native at 16 rows. Then the recipe's
  low-bit term needs a new kernel idea. Pass: 2-bit >= x1.6 and 3-bit >= x1.2. Between the two: partial.

- **Result (Sun 03:00): KILLED by its line.** 2-bit is x1.22 faster than 4-bit native at 16 rows (line x1.4; bytes
  would allow x1.80). 3-bit is x1.08 (bytes allow x1.29). All three builds are row-invariant (bitwise), at the same error
  vs fp32 (rel 0.004-0.005), and the 4-bit build is bitwise equal to experts_native. data/low30.txt.

| 4 layers, real codes | 8 rows (35 distinct) | 16 rows (56 distinct) | % of own floor at 16 rows | per expert at 16 rows |
|---|---|---|---|---|
| native 4-bit (L1-26) | 1.857 ms | 2.715 ms | 81 % | 12.1 us |
| native 3-bit (2+1 planes) | 1.785 (x1.04) | 2.497 (x1.08) | 68 % | 11.1 us |
| native 2-bit | 1.562 (x1.19) | 2.217 (x1.22) | 55 % | 9.9 us (floor 5.5) |
| MLX sorted 4 / 3 / 2-bit | 2.62 / 2.44 / 2.82 | 4.75 / 4.53 / 5.36 | 46 / 38 / 23 % | |

- **What it means:** at 16 rows each expert costs ~10 us of work that does not shrink with bits (MMA on a 16-row tile
  that holds ~3 real rows, x loads and row sums, unpacking). 4-bit sits where that cost meets its byte floor (9.9 us);
  below 4 bits the bytes stop mattering. MLX's own path is worse: its 2-bit is slower than its 4-bit.
- **Correction to the check:** the expert row assumed ~3.5-bit at 82 % of floor (~28 ms). Measured 3-bit at 16 rows:
  11.1 us x 62 experts x 48 layers = ~33 ms. The exact resident 16-row check is ~77-82 ms (~115-125 tok/s at 9.5
  tokens), not 72-77.
- **For the 350 recipe:** it needs ~2-bit experts near their byte floor (~5.5 us per expert at 16 rows). Today's
  kernel takes 9.9 us. So the bytes term needs a kernel that halves the per-expert non-byte cost. Next (L1-30b): profile
  that ~10 us (MMA issue vs unpack vs x work). Candidate fixes: pair codes along K so two operands fill one register
  (no half-split moves), row sums computed once per group instead of per tile, and 8-row tiles if the tensor unit runs
  them at half the cost.

## L1-30b (Leap, pre-registered Sun 03:08): where the ~10 us per expert goes, and one fix
- **Profile** (2-bit, 16 rows, same routing as L1-30): the kernel with one part removed at a time: no tensor-unit
  multiply, no unpacking (operands fixed once), no activation work (x loads, row sums), and weight reads only. A
  measurement, no kill line. Prior: the ALU side (unpack, operand moves, row sums) is ~60 % and the multiply ~30 %.
- **Fix candidate:** pair codes along K (the two halves of one unpacked word are adjacent operand slots, k and k + 1 of
  one row) instead of across rows (slots i and i + 4), so the compiler can move both operands as one register. The MMA
  gets the same values in the same slots, so the output must be bitwise equal to L1-30's kernel.
- **Kill line (fix):** < +5 % on 2-bit at 16 rows, or not bitwise -> drop. Pass: >= +15 %.

- **Profile result (Sun 03:08, data/low30b.log), per expert at 16 rows:**

| mode | 2-bit | 4-bit |
|---|---|---|
| full kernel | 9.8 us (56 % of floor) | 11.9 us (83 %) |
| no tensor-unit multiply | 7.3 | 11.4 |
| no unpacking | 8.2 | 11.7 |
| no activation work | 9.2 | 11.7 |
| weight / scale reads only | 7.0 (78 %) | 11.5 (86 %) |

  - 4-bit is memory-bound: the reads alone take 11.5 of 11.9 us.
  - At 2 bits the reads alone already miss the floor (7.0 vs 5.5 us), and multiply + unpack add ~2.8 us that do not
    overlap with memory. So the levers are the read pattern and the overlap, not the arithmetic.
  - The K-pair A/B did not run (a dict-key bug in the harness); it moves to L1-30c.
- **L1-30c (pre-registered 03:12):** two read-side suspects, same kernel, bitwise-equal outputs.
  - Empty workgroups: the grid launches one row per routed pair (160 at 16 rows), but only ~59 rows hold an expert; the
    rest exit at once. Test: launch exactly the distinct count (host-known here; in the engine this needs a bound or a
    loop over segments).
  - One load in flight per lane: prefetch the next group's words during this group's work (register double buffer).
  - Plus the K-pair A/B. All at 4 / 3 / 2 bits, 8 and 16 rows, fair interleaved.
  - Prediction: exact grid +5-10 % and prefetch +5-15 % at 2-bit; K-pair 0-5 %.
  - Kill line (read-side fixes): all combined < +15 % at 2-bit, 16 rows. Pass: >= +30 % (2-bit >= x1.6 vs today's 4-bit).

- **L1-30c result (Sun 03:13, data/low30c.log): KILLED by its letter, the best subset between the lines.** All
  variants are bitwise equal to the base.

| 16 rows, 60 distinct, 4 layers | 4-bit | 3-bit | 2-bit |
|---|---|---|---|
| base (L1-30 kernel) | 2.829 ms (83 %) | 2.659 (69 %) | 2.373 (55 %) |
| K-paired codes | x1.00 | x0.97 | **x1.157** -> 2.051 ms (64 %) |
| launch exactly the distinct count | x1.01 | x1.03 | x1.03 |
| prefetch next group | x0.98 | x0.94 | x0.95 |
| all three | x1.00 | x0.96 | x1.13 |

  - All combined = +13 % at 2-bit, under the +15 % kill line, because prefetch hurts everywhere (register pressure).
    K-pairing alone is +15.7 %: 2-bit is now x1.38 vs 4-bit at 16 rows (x1.34 at 8 rows), 8.5 us per expert vs a
    5.5 us floor. Exact grid sizing adds ~3 % (in the engine it needs a loop over segments, no host sync).
  - The cap is the read pattern itself: at 2 bits, reads alone already take 7.0 us per expert (78 % of floor).
- **Where that leaves experts in a 16-row check** (62 per layer x 48): 4-bit ~35 ms, 3-bit ~33 ms, 2-bit (K-paired)
  ~25 ms. The 350 recipe assumed 2-bit at ~95 % of floor (~16 ms). Kernels on this design top out ~9 ms short of that.

## L1-31 screen (Leap, pre-registered Sun 03:13 EET; DST ended, clocks went 04:00 EEST -> 03:00 EET): the GDN core at 16 rows
- **Why:** the check's "GDN core, attention core, HC elementwise, routing, misc ~12-15 ms" is an estimate scaled from
  Fit12's 4-row map. At 16 rows the engine runs the stock chain for the 36 GDN layers: concat + conv1d + silu + split +
  fp32 l2norm, mlx_lm's recurrence kernel, the sigmoid-gated RMS norm (MTPLX's fused conv+norm only covers 2-6 rows, its
  fused gate 32+ rows), and capture-commit replays the recurrence over the accepted rows.
- **Bytes floor:** ~7.5 MB per layer (mostly the fp32 state read + write, 2 x 3.1 MB) -> ~1.0 ms per check.
- **Test** (gdn31.py, GPU ~2 min): each piece over 36 layers in one eval, random inputs of the real shapes, S = 16
  (and the replay at T = 10), median of 15; MTPLX's fused rows kernel at S = 16 if it runs.
- **Kill line (for "the GDN core is a lever"):** stock GDN core < 4 ms per 16-row check (then <= 5 % of the check).
  Pass: >= 8 ms (a kernel program pays).

- **Result (Sun 03:15 EET, data/gdn31.txt): BETWEEN THE LINES.** Per 16-row check, 36 layers (floor 0.97 ms):

| piece | ms |
|---|---|
| prework: concat, conv1d, silu, split, fp32 l2norm | 1.51 (MTPLX's fused rows kernel runs at S 16: 1.09, not bitwise: max diff 0.0078) |
| recurrence (mlx_lm gated_delta_update), T 16 | 2.58 (T 1: 1.33, so ~37 us per layer is state traffic + launch) |
| gated norm (rms_norm + fp32 sigmoid gate) | 0.61 |
| **chained core** | **4.07** (kill line 4, pass 8) |
| + capture-commit replay of the recurrence, T 10 | 2.10 -> **6.2 ms in all** |

  - The GDN core is ~8 % of the ~77-82 ms check. A fused exact program (prework + recurrence + norm, replay folded in or
    overlapped) could reach ~2-3 ms: ~3-4 ms saved, ~4-5 %. A modest lever, not a breakthrough. Fit12 owns the GDN port.
  - The check's "GDN core, attention core, HC elementwise, routing, misc ~12-15 ms" row: GDN measured 6.2; the rest
    (attention core over the KV cache and indexer, HC elementwise, routing, misc) ~6-9 ms, still estimated.

## L1-32 screen (Leap, Sun 03:17 EET, CPU): intra-request parallelism, k streams of one answer in one check
- **Idea:** one chat decodes k independent parts of its answer at once. Each part gets its own 16-row speculative
  check, and all k share one pass. In a batch-1 MoE the cost is the expert union over all rows, and parts of one answer
  share experts.
- **Prior art:** parallel decoding of one answer is published: Skeleton-of-Thought (2023), APAR (2024), Hogwild!
  inference, PASTA, Multiverse (2025); typical gains 1.2-2x, with the model planning the parts. The expert-union
  economics for this MoE (k streams vs one longer chain or tree) is ours, untested.
- **Pre-registered:** experts read per accepted token at 4 x 16 vs one 16-row chain: kill > 0.75x, pass <= 0.55x.
  Prediction 0.5x.
- **Result (data/par32.txt, Breakthrough routing traces, 200 draws):** **between the lines, near pass.**

| per layer, top-10 (top-8) | rows | distinct experts | expert reads per accepted token |
|---|---|---|---|
| one chain, 16 rows | 16 | 66.1 (54.6) | 6.96 (5.75) at 9.5 tokens |
| one chain, 32 rows (block 32) | 32 | 100.5 | 9.57 at 10.5 tokens |
| one 64-row tree | 64 | 142.6 | 12.22 at 11.67 tokens |
| 2 streams of one answer x 16 | 32 | 105.4 | 5.55 (0.80x) |
| **4 streams of one answer x 16** | 64 | 155.5 (131.5) | **4.09 (3.46): 0.59x (0.60x)** |
| 4 unrelated sequences x 16 | 64 | 208.1 | 5.48 (0.79x) |

- **Rough check arithmetic, resident, 4 streams:**
  - Experts: 2.35x the union, ~78 ms.
  - Dense at 64 rows: about 1.7x, ~30 ms, because the tensor units become the limit.
  - GDN for 4 states: ~16 ms. Attention, edges and drafting: ~20 ms.
  - Total ~145 ms for ~38 tokens, about 2.1x one stream (~260 vs ~120 tok/s).
  - At 24 GB I predicted SSD reads per token would fall by a similar ~0.6x. The replay below says the opposite.
- **Open:** whether the parts are really parallel. How much of an agentic or code answer the 125B can write as
  independent parts at equal quality is a model question. It's not exact (outputs change), so adopting it is Kamen's
  call, and whether a parallel answer counts for "350 for one chat" is his call too.
- **L1-32b replay at 24 GB (Sun 03:19 EET, data/par32b.txt; resid27's LRU pools, warm-started from the other sequences):
  KILLED for the 24 GB budget.** 4 lockstep streams (the 4 quarters of one answer) RAISE misses per generated token:

| misses per token, 1 -> 4 streams | C 192 | C 288 |
|---|---|---|
| top-10, parts of one answer | 1.68-1.94x more (code 44.9 -> 87.0) | 1.05-1.18x more |
| top-8, parts of one answer | 1.25-1.44x more | 1.03-1.05x more |
| top-10, 4 unrelated sequences | 5.2x more | 3.1x more |

  - The SSD regime is set by the working set vs the pool, and 4 streams make the working set 2.35x larger. The pool
    thrashes. One stream's temporal locality is what the pool lives on.
  - So intra-request parallelism is a RESIDENT-only lever: ~0.59x DRAM reads per token, ~2x end to end estimated. With all
    512 experts resident, it fits the 45 GB waypoint only at <= ~2.65 bits per weight on average (2-bit codes + g64
    scales = 2.5 bits: 37.7 GB of experts + dense + runtime ~42.6 GB). Not a lever for the 24 GB goal.
- **If NewIdeas wants the resident branch:** a quality probe (skeleton-style parallel answers on a few Eval-suite tasks,
  scored against sequential) and 2-bit quality are the gates. Both are non-exact, so Kamen's call.

## B1 (NewIdeas' bet order Sun 04:12; Leap, pre-registered 04:16 EET): neuron-sparse expert reads, gate-first
- **Bet:** for a cold (SSD) expert, read its gate rows first (1/3 of the expert), compute silu(x Wg), keep the
  intermediate neurons that carry the output, then read only those neurons' up rows + down columns (stored bundled).
  It breaks "each expert use reads the whole expert". Non-exact (skipped neurons are small, not zero), so Kamen's call.
- **Prior art (Sun 04:15):**
  - FloE (ICML 2025): the same two-phase idea for PCIe-offloaded MoE experts on Mixtral. A cheap ultra-low-bit up
    projection is computed first, then low-magnitude channels are dropped from gate and down. 9.3x fewer bytes per
    expert at 4.4-7.6 % average quality loss.
  - "Uncovering Intra-expert Activation Sparsity" (arXiv 2605.08575, 2026): a fused sparse up-down kernel; notes the
    dense first projection caps the speedup at ~3x.
  - DualSparse-MoE (2508.18376).
  - CATS (2404.08763): SiLU-gate thresholding, 50 % sparsity within 1-2 % on dense 7B.
  - TEAL, Deja Vu, WINA (activation x weight-norm selection).
  - LLM in a flash (row-column bundling of each neuron's up row + down column; flash reads need >= ~32 KB chunks).
  - PowerInfer-2 (neuron clusters on phones; ReLU-sparse models).
  - SmallThinker (natively sparse experts for local SSD inference).
  - Tag: published in pieces. Untested here: the 125B's 640-neuron SwiGLU experts, gate-first selection from the
    4-bit gate, on this Mac's SSD.
- **Physics the kill line leaves out:** one neuron's bundled up row + down column at 4-bit is ~2.9 KB. Small random SSD
  reads are limited by reads per second, not GB/s, so the read pattern matters as much as the bytes. Hence step 2.
- **Test:**
  - Step 1, b1_energy.py (MLX, ~3 min): Fit12's captured MoE inputs (5 layers x 2,048 tokens), the real router + 4-bit
    experts. Retained output energy per (token, expert) vs the fraction of neurons kept, under an oracle ranking
    (|h_j| ||d_j||), a gate-only weight-informed ranking (|silu(g_j)| ||Wu_j|| ||d_j||, usable before the second read)
    and plain gate magnitude (CATS). Also the MoE-block error at f 0.2 / 0.3 / 0.5 vs MSB-3 on all 10.
  - Step 2, b1_ssd.c: random-read GB/s by chunk size (4 KB-2.9 MB) at 8 / 32 / 64 threads, page cache bypassed, plus
    QD1 latency (the extra round trip).
- **Kill line (NewIdeas'):** the top 30 % of neurons carry < 90 % of output energy at the median. I apply it to the
  gate-only ranking (the one a gate-first read can use) and report the oracle beside it.
- **Prediction:** SwiGLU experts aren't sparse. At 30 %: oracle ~0.85-0.92, gate-w ~0.75-0.85 -> a kill on the usable
  ranking. On the SSD, 3-16 KB random reads reach well under the ~11-14 GB/s of whole-expert reads, so neuron-granular
  reads need clustering into >= 32-64 KB chunks to pay at all.
- **Result (Sun 05:44 EET; data/b1_energy.txt, b1_ssd.txt, b1_bundle.txt): KILLED on the line, and the SSD closes
  every predictor variant too.**
  - Output energy kept per (token, expert), median over 102,400 pairs (5 layers x 2,048 tokens x top-10):

    | ranking | 10 % | 20 % | 30 % | 50 % | neurons for 90 / 95 / 99 % |
    |---|---|---|---|---|---|
    | oracle (needs the up projection) | 0.710 | 0.855 | 0.926 | 0.983 | 0.26 / 0.37 / 0.57 |
    | gate-w (gate-first, usable) | 0.519 | 0.664 | **0.769** | 0.911 | 0.48 / 0.59 / 0.77 |
    | gate (CATS) | 0.516 | 0.661 | 0.768 | 0.912 | 0.48 / 0.59 / 0.77 |

    The gate alone can't find the neurons that matter. Weight norms add nothing (gate-w = gate); the up projection
    decides.
  - MoE-block error, every routed expert cut under gate-w, against MSB-3 on all 10 (0.0337, reading ~0.78 of the
    expert's bytes): f 0.2 -> 0.188 (5.6x, reads 0.47); f 0.3 -> 0.145 (4.3x, reads 0.53); f 0.5 -> 0.077 (2.3x, reads
    0.67). On error per byte read, that is worse than simply using fewer bits.
  - SSD, random reads by size (64 threads, page cache bypassed): 4 KB 1.10 GB/s, 8 KB 2.18, 16 KB 4.28, 32 KB 7.10,
    64 KB 10.15, 128 KB 12.72, 256 KB 13.55, 1 MB 13.68, whole expert (2.9 MB) 13.78. Reads of 16 KB or less hit a cap
    of ~265k reads/s. QD1 latency: 16 KB 82 us, 1 MB 217 us, 2.9 MB 331 us. So a dependent second read costs >= ~0.1-0.2
    ms per layer, ~5-10 ms per check unless hidden.
  - With a perfect, free predictor (oracle scores, bundles chosen in-sample, so optimistic), cold-expert read time
    against one whole-expert read:
    - per neuron: 26 % of the neurons, but 3.07x slower (the reads-per-second cap)
    - bundles of 16 neurons (68 KB): best order needs 62 % of bundles -> 0.83x
    - 32 (135 KB): 66 % -> 0.71x
    - 64 (270 KB): 68 % -> 0.69x

    One token's important neurons are spread across most bundles under every order tried (stored, mean score,
    co-activation). Best case: ~1.45x faster at only 90 % of the energy per expert. 2-bit codes alone read 0.556 of
    the bytes (1.8x).
  - Prediction: right. Oracle 0.926 was a little above my 0.85-0.92; gate-w 0.769 was in range; the SSD came out as
    predicted.
  - General lessons:
    1. On this SSD, any sub-expert read scheme needs >= 128 KB units to keep >= 90 % of peak bandwidth.
    2. In these SwiGLU experts, neuron sparsity is real per token (26 % for 90 %) but not clustered, and the gate alone
       does not predict it. FloE's up-first preview hits the same bundle wall.

## B5 (NewIdeas' order; Leap, pre-registered Sun 05:51 EET): no Python per check, for the 16-row check
- **Bet:** in a 16-row check the CPU side leaves the GPU idle for >= 5 ms. The CPU side is the Python graph build
  plus MLX's per-op encode. A compiled or native per-check loop (or simply overlapping the build with the GPU)
  recovers that time. It is exact: same kernels, same graph.
- **Prior art (Sun 05:50):**
  - CUDA Graphs capture of decode / verify steps: standard in vLLM, SGLang, TensorRT-LLM.
  - llama.cpp's Metal backend: native, with a few command buffers per graph, encoded on several threads.
  - Metal indirect command buffers.
  - mx.compile.
  - In this engine (read only), MTPLX compiles runs of GDN layers for S <= 4 only (MTPLX_COMPILED_GDN). Its own note:
    "the qL=1 decode step is CPU-dispatch-bound: ~20.8 ms of Python graph construction per token against <= 14 ms
    of GPU work". It also has a compiled fixed-M4 verify lane (qwen4_fixed_verify.py, graphbank.py).
  - At 16 rows, every layer runs eager and verify widths get no mid-loop eval: the whole 48-layer graph is built in
    Python before the GPU starts.
  - Tag: published (graph capture); untested here: GPU idle per 16-row check.
- **Test (b5_slice.py):** a resident 12-layer slice of the real 125B (layers 0-11 = 9 GDN incl. PLE + 3 QSA; ~18.5
  GB; lazy load, truncated before any weight is read), through MTPLX's own forward under the verify scopes (decode_verify,
  target_verify, exact, capture). 16 real code tokens per check after a ~600-token prefill, 20 checks per variant:
  - A, eager as the engine runs it: build ms (Python, GPU idle) and eval ms.
  - B, the same forward with async_eval every 4 layers (build overlapped). Logits must match A bitwise.
  - C, MTPLX's compiled GDN runs forced at S 16, if they trace.
  - Then a Metal System Trace (xctrace) of A: GPU busy per forward against wall.
  - Per check = 4 x the slice's idle (the head is one op, so its CPU cost is negligible).
- **Kill line (NewIdeas'):** < 5 ms saved per check -> kill, i.e. 4 x (wall - GPU busy) of A < 5 ms.
- **Prediction:** the build is ~0.3-0.45 ms per layer -> ~15-22 ms per check of pure GPU idle in A, plus 1-3 ms of
  exposed encode. B hides most of it (idle ~2-5 ms per check) with no native code. C cuts the build to ~5-8 ms. So a
  PASS, but the cheap overlap may take most of the gain before any native loop is written.

### B5 result (Sun 06:55 EET; quiet lock, 31 s of GPU): PASS by the letter, but the cheap exact overlap takes it
Data: data/b5_slice.txt (timings, 20 checks per variant), data/b5_gpu.txt (trace of A, scripts/b5_gpu.py).

| per 12-layer 16-row check (resident, 4-bit) | build ms | eval ms | wall ms (min) | x4 per 48-layer check |
|---|---|---|---|---|
| A eager (the engine today) | 5.04 | 32.22 | 37.18 (34.20) | 148.7 |
| B async_eval every 4 layers (bitwise = A) | 17.35 | 15.44 | 32.80 (31.01) | 131.2 (-17.5, -11.8 %) |
| C compiled GDN runs at S 16 (bitwise = A) | 1.22 | 32.05 | 33.28 (31.36) | 133.1 (-15.6) |
| Metal trace of A, checks 2-9 | - | - | 38.25 traced; GPU busy 31.94 | idle 25.2 traced, 21.0 untraced |

- The build is 0.42 ms per layer. In A the GPU idles from t0 until the first commit (3.6-9 ms in the trace); once it
  starts, the encoder keeps it fed (idle = first-work delay + ~0.35 ms).
- **Kill line:** 4 x (wall - GPU busy) of A = 21.0 ms untraced (25.2 traced) >= 5 ms, so a PASS. The prediction held on
  every count: idle 15-22 + 1-3 ms; B leaves ~2-5 (measured 3.4 = 4 x (32.80 - 31.94)); C's build 4.9 ms per check.
- **But the native loop itself is not justified on the resident path.** B is a Python-level change: MTPLX already has
  the mid-loop async_eval for prefills of 1,024+ rows; it would extend to verify widths. B is exact and recovers 17.5 of
  the ~21 ms. A native loop's increment over B is ~3.4 ms per check, under the 5 ms line.
  - Adopt-candidate (exact, ~10 lines): mid-loop async_eval every 4 layers in the 16-row verify, -11.8 % per check.
- **Open (not measured):** the SSD-streamed engine. There each layer syncs for its routing before the expert fetch, so
  the 0.42 ms per layer of build sits on the critical path every layer (~20 ms per check) unless BigMoE overlaps it.
  That needs a trace of BigMoE's streamed 16-row check; it comes after B10.
- Caveats:
  - x4 counts the head and embedding four times, so the busy x4 (128 ms) is an upper bound. The idle is a per-layer
    quantity, so its x4 holds.
  - The trace's 1 s marker gap drops the GPU clock: checks 0-1 after it are slow (56 / 74 ms) and are excluded.

## B10(b) (NewIdeas 06:24 + 06:57; Leap, pre-registered Sun 07:02 EET): trellis-decode cost inside the expert GLU
- **Bet (B10, NewIdeas):** keep experts as ~2.1-bit bitshift-trellis codes (Fit12 B3: TRL-2 at 1.15x MSB-3's output
  error, in-sample), 1.29 MB per expert instead of 2.16 MB. Then ~53-65 % of experts fit at 24 GB and all of them at
  45 GB (MoonShots B10-c: -85 % SSD bytes per token at 24 GB).
  - Spec 06:57: ranks 0-1 decode base + a residual trellis (two decodes, summed); ranks 2-9 decode the base only.
  - My part: is the decode cheap enough on this GPU?
- **Format (Fit12's B3 = Breakthrough's D2, read from fit_b3.py / trellis.py):**
  - RHT on the input dimension: per-layer random signs + block Hadamard 512 for gate / up, so x' is computed once per
    token per layer. Per-expert signs + block Hadamard 128 for down, so h' is computed once per (row, expert) pair.
  - Per-row bf16 RMS scale.
  - 16 x 16 tiles, row-major. L = 12 bit state, k = 2 new bits per weight, free 12-bit start state per tile.
  - Reconstruction lut[state]: 4,096 Gaussian quantiles in a fixed random order (seed 0).
  - Weight t of a tile reads the 12-bit window at bit 2t of the tile's stream, so any weight decodes independently.
    With k = 2, tile row r's 16 weights sit in bits [32r, 32r + 42): two aligned 32-bit words.
  - Layout here: tiles padded to 17 words (2.125 b + scales, vs 2.04 tight), interleaved so a simdgroup's loads are
    contiguous. A chained trellis (no per-tile start state) would give 2.0 b with the same decode; that is Fit12's
    call (encoder change).
- **Prior art (Sun 07:00):**
  - QTIP (Tseng et al., NeurIPS 2024): bitshift trellis + RHT, fused CUDA decode-matvec kernels for batch 1. It uses
    computed codes (1MAD / 3INST) or a small 2^9 x 2 LUT (HYB) so that decode stays under the memory bound.
  - ik_llama.cpp's integer trellis quants IQ1_KT..IQ4_KT: CUDA (PR 113), CPU, NEON and a Metal port (PR 475). So a
    Metal trellis decode exists; I found no Metal speed numbers.
  - Not found: a 4,096-entry LUT trellis code inside a routed MoE GLU on Apple GPUs, or a two-stage (base + residual)
    trellis for the top ranks.
  - Tag: published codes and kernels (CUDA, Metal port); untested here: their cost vs 3c inside the routed GLU.
- **Baseline (MoonShots' microbench, 3 layers x 3 traces):** 3c costs ~5.6-6.7 us per (row, rank) pair at 3-bit
  (plan 3333333333) and ~7.7 us per 4-bit pair. It is bound by per-pair work, not bytes (~820 G weights/s).
- **Test (b10_trellis.py, one real layer L20, routing from MoEPrefetch's code trace, S = 4 / 16 natural top-10):**
  - The same timed chain as MoonShots' microbench.
  - 3c arms: plans 3333333333 and 4444444444 (per-pair 3-bit and 4-bit) and 4433333333 (today), on records3 pools.
  - Trellis arms: base on all ranks, base + residual on all ranks, and the spec mix (residual on ranks 0-1). They use
    pools of 512 random-stream experts (decode cost does not depend on the code values), x' / h' rotations and the
    combine included.
  - Correctness: kernel output vs a numpy decode of the same streams (fp32 reference, tolerance check).
- **Kill (NewIdeas'):** base path per pair > 2x 3c's 3-bit, or base + residual per pair > 2x 3c's 4-bit.
- **Prediction:** decode adds ~2-3 ALU ops per weight (window extract + LUT read) to 3c's ~3, and bytes drop 40 %.
  - Base: 6-9 us per pair, 1.0-1.5x 3c-3: passes.
  - Base + residual: 10-16 us, 1.3-2.1x 3c-4: at risk.
  - If the residual path fails, the fallback is a decode-once kernel that serves every row of a distinct expert. At
    S = 16 there are ~2.5 rows per distinct expert; that kernel is not built yet.

### B10(b) result (Sun 10:33 EET; quiet lock, ~40 s of GPU, normal fan mode): the LUT code FAILS, QTIP's computed 3INST code PASSES
- Correctness: the kernels match a numpy decode of the same streams (max error 2e-3 of max |y|, bf16 output), and the
  output is bitwise the same across geometries.
- us per (row, rank) pair, layer 24, natural routing (6 passes, chains of 30). Median of 7 interleaved rounds
  (data/b10_bench2.txt; 3c-3 timed twice, 5.69 / 5.70); data/b10_bench.txt (3 rounds) agrees within 1-2 %.

| arm | S = 4 | S = 16 | vs the kill baseline (S = 4 / 16) |
|---|---|---|---|
| 3c 3-bit (3333333333) | 6.58 | 5.69 | - |
| 3c 4-bit (4444444444) | 8.94 | 7.77 | - |
| 3c today (4433333333) | 7.16 | 6.14 | - |
| trellis, LUT code, base (best of 18 geometries, LUT in device or threadgroup memory alike) | 17.23 | 16.40 | 2.62x / 2.88x 3c-3: **KILL** |
| trellis, LUT code, base + residual | 31.45 | 30.39 | 3.5x / 3.9x 3c-4: **KILL** |
| trellis, 3INST code, base | 7.59 | 6.88 | 1.15x / 1.21x 3c-3: pass |
| trellis, 3INST code, base + residual | 14.10 | 13.28 | 1.58x / 1.71x 3c-4: pass |
| trellis, 3INST code, spec (residual on ranks 0-1) | 10.10 | 9.21 | 1.41x / 1.50x today |

- Read: the 4,096-entry LUT gather caps the decode at ~300 G weights / s, whether the LUT sits in device memory (L1)
  or in threadgroup memory (random 2-byte reads conflict). QTIP's 3INST code computes the value from the state in
  4 ALU ops (multiply-add, and / xor, two fp16 halves summed) and runs 2.4x faster.
- The fix needs no format change. Tabulate 3INST into the encoder's 4,096-entry LUT and re-encode. The CPU check
  (b10_codes.txt, exact Viterbi on synthetic data) puts its NMSE at 1.00x the LUT code at k = 2 on Gaussian data,
  1.04x on Laplace, and at k = 3 1.01x / 1.20x. The real cost needs Fit12's held-out test with the 3INST table.
- Published: QTIP designed its computed codes (1MAD / 3INST / HYB) to avoid exactly this gather.
- Context: Fit12's B10a (10:2x) killed the 2-bit base on held-out quality (2.49x MSB-3; base + residual k = 2 at
  0.66 E4, kill > 0.2). So this speed pass matters only for the remaining option, TRL-3 (3.07 b; down 0.69x MSB-3
  held out). The k = 3 decode is not measured yet. Expected: about k = 2's cost with 3INST (the decode is ALU-bound;
  the stream has 1.5x the bytes), so ~1.2-1.4x 3c-3. Measuring it needs the kernel generalised to k bits plus one
  short lock.
- The sweep's lone 3c arm read 6.73 us per pair among 54 trellis arms. bench and bench2 agree at 5.69-5.71, so that
  value is an outlier from its context. The ratios use bench2.
- Caveats:
  - Random streams (the decode cost does not depend on the values).
  - Per-pair kernels with no decode sharing across rows of the same expert (~2.3 rows per distinct expert at
    S = 16). A grouped decode could cut the decode part by up to 2.3x, and the same holds for 3c.
  - The test tiles are padded to 17 words (2.125 b, 1.313 MB per expert).

## S11 estimate (NewIdeas 10:36; Leap, Sun 23:31 EET, CPU / analysis only): decode once per distinct expert in today's 3c
- **Question:** how many ms per 16-row check would a decode shared by every row of a distinct expert save? The check
  is truncated to 10.3 rows, top-10, 48 layers: P = 103 pairs and D = 53 distinct experts per layer (QUEUE base line),
  so 1.94 pairs per expert. Compared against today's per-pair 3c and L1-30's per-expert kernel (11.1 us at 3-bit).
- **Inputs** (layer 24, natural routing, B10(b) bench2): 3c-3 per pair 6.58 us at S = 4 (27.8 distinct) and 5.69 at
  S = 16 (69.7 distinct).
  - A linear fit gives 4.20 us per pair + 3.42 us per distinct expert, so 0.614 ms per layer at 10.3 rows. Today's
    mix (4433333333) is ~0.66 ms.
  - Memory: at S = 4, 3c reads its distinct bytes (60 MB) at 228 GB / s, already at the floor. At S = 16 it reads them
    at only 165 GB / s, because it is ALU-bound on the pairs.
- **3c's ALU split per weight** (tierglu_v3d.py kernel A): ~4.4 ops per weight per pair. About 3.2 of those are decode
  (load, spread8 of 3-bit codes into nibbles, the qdot mask, the int-to-float conversion), which a group can share.
  About 1.2 are per row (the FMA and the x handling).
- **Grouped kernel, ALU:** 53 x 4.14 + 103 x 1.55 us = 0.38 ms per layer.
- **Grouped kernel, memory floor:** 53 x 2.16 MB = 114.5 MB per layer. At 240 GB / s (L1-30b's measured read rate,
  86 % of 280) that is 0.477 ms. So the grouped kernel is memory-bound at ~0.48-0.52 ms per layer.
- **Estimate per 48-layer check at 10.3 rows:**

| vs | today per layer | grouped per layer | saved per check |
|---|---|---|---|
| per-pair 3c, 3-bit | 0.614 ms | 0.48-0.52 | **4.5-6.4 ms (central ~5)** |
| per-pair 3c, today's mix (+ LSB of ~17 rank-0/1 experts, +13 MB) | ~0.66 | 0.53-0.57 | 4.3-6.2 |
| L1-30 per-expert MMA kernel (11.1-12 us x 53) | 0.59-0.64 | 0.48-0.52 | 3.4-7.7 |
| bounds: L1-30-like 69 % of floor / 100 % of 280 GB / s | | 0.59 / 0.41 | ~1.2 / ~9.8 |

  - At the full 16 rows (62 distinct, 160 pairs) the saving is larger: ~0.2-0.28 ms per layer, ~10-13 ms per check.
  - At S = 4 the saving is ~0, because 3c already reads at the floor there.
- **Verdict:** central ~5 ms per check, at NewIdeas' >= 5 line, so pre-register the kernel test (S11 below). The
  estimate assumes the grouped kernel overlaps its ALU with memory; L1-30's MMA kernel did not at 3-bit (69 % of the
  floor), so this is the risk.

## S11 (Leap, pre-registered Sun 23:31 EET): grouped decode in 3c, bitwise equal to per-pair 3c
- **Bet:** a copy of 3c's kernels A (gate + up + SwiGLU) and B (down + combine). One threadgroup serves one distinct
  expert and all its pairs, up to RMAX rows per chunk.
  - Each lane decodes its 16-value chunk once: integer spread, mask and int-to-float. It then runs 3c's exact per-pair
    float sequence for each pair: qdot on the pre-scaled x, the same accumulation order, simd_sum, scale and bias.
  - The grouping stays on the GPU with no host sync. Pairs are sorted by expert with a stable argsort. P threadgroups
    launch; a threadgroup whose pair is not a chunk leader exits at once.
- **Prior art:** grouped MoE GEMM is published (MegaBlocks; vLLM / SGLang fused_moe; MLX gather_qmm with sorted
  indices; llama.cpp mul_mat_id). Untested here: a SIMD (non-MMA) grouped decode at ~2 rows per expert on Apple GPUs
  that is bitwise equal to the per-pair kernel, so it is exact for spec decode with no reference change. Tag:
  published, untested here.
- **Test** (scripts/s11_grouped.py, one short lock):
  - Bitwise check vs 3c for plans 3333333333, 4444444444 and 4433333333, at S = 4 / 8 / 16.
  - Timing at layer 24, natural routing, S = 4, 8, 10, 11, 16, in interleaved paired rounds vs 3c.
  - Report ms per layer, the read rate on distinct bytes, and saved ms per check = mean(S 10, 11) delta x 48.
- **Kill:** not bitwise, or < 3 ms per check saved at 10.3 rows. **Pass:** >= 5 ms. Between: partial (NewIdeas' call).
- **Prediction:** 3c ~0.61 ms per layer at 10-11 rows, grouped ~0.50, so ~5 ms per check (3.5-6). S = 16:
  ~10-13 ms per check. S = 4: ~0.

### S11 result (Sun 23:37 EET; quiet lock, ~30 s of GPU): KILLED. Bitwise exact, but the grouped decode is SLOWER
- Bitwise: h, the per-rank y and the combined y are equal to per-pair 3c in every case (S = 1 / 4 / 8 / 16, 3 token
  offsets, plans 3333333333 / 4444444444 / 4433333333, gm 2 / 4 / 8, both load paths). data/s11_test.txt.
- Timing, layer 24, natural routing, interleaved rounds (data/s11_bench.txt, s11_diag.txt), ms per layer:

| arm | S = 10 | S = 16 |
|---|---|---|
| 3c fused, 3333333333 | 0.597 | 0.911 |
| 3c per-rank + a separate combine (grouped's tail) | 0.612 | 0.932 |
| the sort kernel alone | 0.015 | 0.017 |
| grouped, gm 1 (all the machinery, no sharing) | 0.673 | 1.023 |
| grouped, gm 2 (best sharing arm) | 0.714 | 1.058 |
| grouped, gm 4 / gm 8 | 0.955 / 1.848 | 1.346 / 2.666 |

- Per 48-layer check at 10.3 rows, the best arm (gm 2) is 5.8 ms SLOWER on 3-bit and 6.5 ms slower on today's
  4433333333. At 16 rows it is 7.4 / 8.2 ms slower. The kill is < 3 ms saved; this is a loss.
- Why the estimate was wrong: it modelled 3c as ALU-bound on the decode (~73 % of 3.2 / 4.4 ops per weight). The
  measurement says otherwise.
  - Sharing the decode (gm 1 -> gm 2) ADDS 0.04 ms per layer, and every larger group costs more. The extra registers
    per thread (one x chunk per member) cut occupancy, and 3c depends on occupancy to hide its code / scale read
    latency. The ALU it saves does not pay for that.
  - The machinery alone (sort, member scan, expert-sorted order, unfused combine) costs another +0.06-0.11 ms per
    layer.
- What it means for the 350 stack: 3c's ~5.7-6.6 us per pair is not reducible by sharing work between rows of one
  expert in this design. Together with L1-30, this caps the expert term for a 16-row check at today's kernels, unless
  a structurally different kernel appears; L1-30's per-expert MMA kernel ties per-pair 3c at 10.3 rows. Drop S11.

## D32 (NewIdeas Sun 23:41; Leap, pre-registered Mon 03:09 EET): is the dense part of an exact check at 32 rows near its 16-row cost?
- **Why:** block 32 at the oracle's 15.0 tokens per check cuts expert reads per token only ~4 % (par32: 100.5 vs 66.1
  distinct experts per layer). Dense per token falls ~37 % IF the dense part at 32 rows costs about what it does at 16.
  NewIdeas: this is the number that decides block 32 for 350.
- **Test** (scripts/d32.py, ~3-5 min of GPU, one quiet lock after the silent night):
  - All 17 dense shapes at their per-check counts (incl. the 8-bit router and indexer); random weights, cold.
  - 16 rows: today's exact paths (qmm16n native layout, bmm16, MLX forms a / b / c).
  - 32 rows:
    - two 16-row calls;
    - one launch with the two row tiles in adjacent threadgroups (weights read from DRAM once, unpacked twice);
    - one simdgroup doing both row tiles of its 32 columns (weights read and unpacked once, two MMAs per step);
    - MLX's own 32-row call.
  - Bitwise: each 32-row path, per row, equals the 16-row path (same K split), and real rows are independent of the
    padding. Paths that fail are timed for information only.
  - Interleaved paired rounds (8 x median of 5), PAUSE-aware.
- **Prior art:** batch-invariant kernels (Thinking Machines 2025; SGLang deterministic mode); multi-row register tiles
  are standard GEMM practice. The measurement for this model's exact check at 32 rows is ours.
- **Prediction:**
  - 16 rows ~19-20 ms (with the 8-bit router / indexer).
  - 32 rows ~25 ms: big 4-bit shapes x1.3-1.5 (one unpack, two MMAs and x loads), bf16 mixers x1.2-1.4, launch-bound
    small shapes x1.0-1.2. So R = 32-row / 16-row ~1.3-1.4.
  - Dense per token at the oracle's 15.0 tokens: -13 to -19 %. At L1-20's trained block-32 drafter (10.51): +16 to +25 %.
- **Lines:** PASS (dense stays near its 16-row cost) R <= 1.25. KILL R >= 1.60 (no dense gain per token even at the
  oracle's 15.0 tokens). Between: alive, and R goes into the stack pricing.
- **Result (Mon 10:02; 41 s of GPU; data/d32.txt, d32.json): PASS by a hair, R = 1.24.** Dense per check: 16 rows 17.96
  ms, 32 rows 22.19 ms (bandwidth floor 12.3; two 16-row calls would be 29.6 ms, R 1.65).
  - Bitwise: both 32-row kernels (one simdgroup doing both row tiles 'mr2', adjacent threadgroups 'mg2') equal the
    16-row path per row at every K split tried, padding-independent. MLX's own 32-row call equals 2 x 16 for bf16 and
    for the big 4-bit shapes, but not for 4-bit with small N (shared expert, k/v, in_proj_a/b) or the 8-bit router /
    indexer: those stay 2 x 16 MLX calls.
  - Where the extra 4.23 ms goes:
    - Big 4-bit projections (qkv, o/out, z, q, lm_head, shared gate/up): +2.84 ms. mr2 wins everywhere, at 56-67 % of
      the floor vs 75-85 % for the 16-row kernels: one unpack, then two MMAs and two per-group epilogues makes it
      ALU-bound.
    - 8-bit router + indexer and the small MLX-only shapes: +1.04 ms (router 0.98 -> 1.66 ms) for want of an exact
      32-row call.
    - bf16 HC mixers, PLE, block_inject: +0.25 ms. They are bandwidth-bound, so the second row tile is almost free
      (x1.01-1.06).
  - Per token: block 16 at 9.38 tokens = 1.91 ms of dense per token. Block 32 breaks even on dense at 11.6 tokens per
    check; at the oracle's 15.0 it is -23 %, at L1-20's 10.51 it is +10 %. Experts break even later: 100.5 vs 66.1
    distinct experts per layer means ~14.3 tokens per check. So block 32 pays only with a drafter at ~12-14+ tokens
    (L1-20: 10.5). That is DS's question.
  - vs prediction: R 1.3-1.4 predicted, 1.24 measured. 16 rows came in at 17.96 ms, not 19-20 (the 8-bit router and
    indexer are cheaper than I guessed). The 4-bit shapes were as predicted (x1.27-1.36). I was wrong on the bf16 mixers
    (predicted x1.2-1.4, got x1.02-1.06).
  - Untested levers that would lower R further:
    - an exact 8-bit MMA kernel for the router + indexer (MLX runs them at 20-24 % of the floor): ~-1.3 ms at 32 rows,
      ~-0.7 ms at 16 rows. The engine's 16-row decode path would have to use it too.
    - a 32-row 4-bit kernel nearer the 16-row kernels' ~80 % of the floor: up to ~-2 ms.
    - Together R ~1.05-1.10.

## DS (NewIdeas Mon 03:27; Leap, pre-registered Mon 03:33 EET): is DFlash2's depth decay a data problem?
- **Why:** NewIdeas' 350-at-45-GB arithmetic. With the non-exact stack (top-8, TRL 2.73 b, dense diet), 350 needs ~14
  tokens per 16-row check or ~17-19 per 32-row check on code (cand21's bytes-only pricing: ~17.5 at 32 rows). L1-20
  gets 10.2-10.5, its lattice oracle 20.3, a perfect drafter 30.5. L1-20's held-out-prompt validation was still rising
  from step 100 to 200 (selector agreement d16 0.59 -> 0.66, d31 0.33 -> 0.38), with the loss still falling.
- **Test** (ds_batch.sh under run_locked.sh; scripts/ds_report.py):
  - Data: a stratified random half of L1-20's 99 train windows (36 / 72 code, 14 / 27 math, seed 50; 11.3K of 22.6K
    train tokens; 201 of 401 groups), the same 10 val windows. data/ds50 holds symlinks, so the trainer is L1-20's code
    unchanged (ada5b1f).
  - Arm B = 50 % at 201 steps (4 epochs; L1-20's step count), ~28 min.
  - Arm A = 50 % at 101 steps (2 epochs; L1-20's epoch count), ~14 min, run only if s_fix < 1.12.
    - It separates data from compute: L1-20 was still improving with steps, so a flat s_fix alone could mean
      compute-limited.
  - Scoring: L1-20's replay on the 24 held-out HumanEval samples (disjoint from the training problems), per sample.
    Modes: block 32 1-step and 2-step tail 0.9 (fixed, not re-tuned), block 16 1-step. L1-20 is re-scored the same
    way.
  - Primary T = the better block-32 mode per drafter.
  - s_fix = T(L1-20) / T(B); s_prop = T(L1-20) / T(A); c50 = T(B) / T(A). 90 % paired bootstrap CIs over the samples
    (sample noise only).
  - Each arm's weights (1.1 GB) are deleted once scored (disk at the floor).
- **Prior art:** drafter data / compute scaling (EAGLE-3's scaling with training data; DFlash's own training; DBloom
  2608.30427 block-size post-training). Ours: the slope for a near-deterministic MoE target at block 32.
- **Lines** (NewIdeas', plus the compute check):
  - GO (4x on-policy windows, ~3 h of 125B time): s_fix >= 1.12.
  - KILL (data-flat): s_fix < 1.05 AND c50 < 1.05.
  - s_fix < 1.05 with c50 >= 1.05: compute-limited, so next is 100 % at 402 steps (~56 min, no new data).
  - Between: one 2x-data point decides.
- **Prediction:** s_fix 1.02-1.06 (B ~9.7-10.0 vs L1-20's 10.23 at block 32 1-step); s_prop 1.04-1.08. So the kill zone
  or just between, unless compute moves. 25 % GO.
- **Result (Mon 10:43-11:26; data/ds_report.txt, ds_l120 / ds_B / ds_A .json): data-flat; compute moves ~5 % per
  doubling. By the lines that is 'compute-limited', so DS2 is next. No new 125B data (GO needed s_fix >= 1.12).**
  - Tokens per pass on the 24 held-out samples (block 32 1-step / 2-step tail 0.9 / block 16 1-step):

    | drafter | data | steps | b32 1-step | b32 tail 0.9 | b16 1-step |
    |---|---|---|---|---|---|
    | L1-20 | 100 % | 201 | 10.23 | 10.51 | 9.54 |
    | B | 50 % | 201 | 10.13 | 10.38 | 9.67 |
    | A | 50 % | 101 | 9.86 | 9.85 | 9.30 |

  - s_fix = 1.013 (90 % CI 0.990-1.035): doubling the data at fixed steps buys ~1 %.
  - c50 = 1.052 (CI 1.025-1.080; per mode 1-step 1.027, tail 1.054): doubling the steps at 50 % data buys ~5 %. This
    is just over the 1.05 line.
  - s_prop = 1.066 (CI 1.036-1.096): double data and steps together buy ~7 %, nearly all of it from the steps.
  - Held-out-prompt val agrees (selector agreement at depth 16 / 31): L1-20 0.63 / 0.38, B 0.61 / 0.38, A 0.62 / 0.35.
  - vs prediction: s_fix 1.02-1.06 predicted, got 1.013 (just below); s_prop 1.04-1.08 predicted, got 1.066.
  - For 350: block 32 needs ~17-19 tokens per check, and more now that D32 put dense at 22.2 ms. That is +65-80 % over
    10.5. More on-policy data is off the table. At ~5 % per doubling of steps, training this recipe harder is out of
    reach (~10 doublings); only a much steeper compute slope at full data would reopen it (DS2).
  - Run 1 (10:14) was stopped at step 85 because its own 10-minute checkpoint took the disk to 99 GB (health ACT). The
    rerun trained from scratch, with weights on a RAM disk.

## DS2 (Leap, pre-registered Mon 11:30 EET; DS's pre-registered next step): the compute slope at 100 % data
- **Why:** DS found the drafter data-flat with a small compute slope at 50 % data (c50 1.052, at the line). Before
  closing 'train the drafter harder' for 350, measure the slope where it would be used: on all the data.
- **Test** (ds2_batch.sh under run_locked.sh; ds_report.py):
  - Arm C = L1-20's recipe on 100 % of its windows (data/l120) at 401 steps (--epochs 4; L1-20 = 201 steps).
  - Same trainer, weights on the RAM disk, scored like DS.
  - c100 = T(C) / T(L1-20), primary modes, 90 % paired bootstrap CI.
  - ~57 min of GPU, peak ~34 GB. Ticket marked [low] so it runs in a gap.
- **Lines:** reaching ~17.5 tokens needs +67 % over 10.5.
  - ALIVE (price a long run; at 1.12 per doubling that is ~4.5 more doublings, ~23x the steps): c100 >= 1.12.
  - KILL (training this drafter harder cannot reach it: >= 7 doublings at < 1.07): c100 < 1.07.
  - Between: one more doubling (801 steps, ~2 h) decides.
- **Prediction:** c100 1.03-1.07 (T ~10.8-11.2). DS showed four passes over the data cost nothing vs two, so full data
  should not steepen the slope much. 70 % KILL, 10 % ALIVE.
- **Result (Mon 11:30-12:25; 52 min of training; data/ds_report.txt, ds_C.json, ds_C.state.json): KILL, c100 = 1.005
  (90 % CI 0.980-1.035).**
  - C (100 %, 401 steps): block 32 1-step 10.53, tail 0.9 10.57, block 16 1-step 9.62. L1-20 (201 steps): 10.23 /
    10.51 / 9.54.
  - Per mode: 1-step 1.030, tail 1.005, block 16 1.009.
  - Held-out-prompt val plateaus after ~200 steps while the training loss keeps falling. Selector agreement d16 / d31:
    step 200 0.62 / 0.38, step 300 0.65 / 0.39, step 400 0.64 / 0.38.
  - vs prediction: 1.03-1.07 predicted, 1.005 measured (below the range).
- **DS + DS2 together:** this drafter recipe (DFlash2 < 1B, block 32, partial masking) saturates at ~10.5 tokens per
  32-row check on code.
  - Twice the on-policy data: +1 %. Twice the steps: +0.5 % at full data, +5 % at half.
  - Neither more data nor more training reaches the ~17-19 that 350 at 45 GB needs. The plateau looks structural (what
    the drafter can see and how it drafts deep positions), not budget.
  - So the drafter crux needs a different drafter or a different verify shape (NewIdeas' NT-1: the target's top-k at
    the first misses; lattice oracle 20.3). The other route is a behavior change (L1-32 parallel streams, Kamen's call).

## CA (NewIdeas Mon 13:46, QUEUE row CA; Leap, pre-registered Mon 14:05 EET): anatomy of one exact resident check
- **Why:** NewIdeas' repricing (QUEUE '! REPRICED', data/price350.txt) puts 'other' (attention, GDN core, routing, glue)
  at an assumed 10-20 ms of a 16-row check. Nobody has measured it. It sets the check's floor and ranks the check levers.
- **Test** (ca_batch.sh -> scripts/ca_anat.py, ca_report.py; one quiet lock, ~10-15 min):
  - A resident 12-layer slice of the 125B (layers 0-11: 9 GDN incl. the one PLE layer + 3 QSA; the model is GGGA x 12),
    as B5, through MTPLX's own forward under the engine's verify scopes, exact mode.
  - Each check does the engine's bookkeeping: a lazy pre-verify snapshot, the verify forward with capture, the greedy
    acceptance read (argmax .tolist(), the engine's sync), and a capture-commit of a partial accept (keep 7 / 10 / 11 at
    8 / 16 / 32 rows, ~ tokens per check). Its lazy replay lands in the next check, as in the engine.
  - Real code tokens after a 1K or 8K-token prefill; 8 / 16 / 32 rows; 10 checks after 3 warm-up per config. The PLE
    n-gram table stays file-backed (no 21.6 GB pre-read), as it would at 45 GB.
  - time: A = eager, as the engine runs 16 rows; B = async_eval every 4 layers (B5). Wall per check. Plus the DFlash2
    1-step draft at block 16 / 32.
  - prof, under a Metal System Trace, with MLX's command buffers ending only where the script evaluates:
    - B2 = B, for the end-to-end GPU busy.
    - P = one command buffer per part, labelled in order. Parts: every quantized matmul, the MoE's routing sort /
      expert kernels / unsort, the HC reads and writes, PLE, the head, the commit replay, the acceptance read, and the
      ops between hooked calls in the attention / GDN / MoE modules (attention core incl. the indexer, GDN core,
      routing, combine).
    - The report matches the trace's buffers to the labels by order and count, and scales the slice to 48 layers (x4;
      the PLE layer, embedding, head and acceptance read x1).
- **Validity lines (a measurement, no kill line):**
  - P's tokens must equal A's, and every commit must land (smoke).
  - The parts must sum to B2's GPU busy within 5 %, after subtracting P's extra buffers at the marker-only floor.
    Otherwise the parts are reported as shares of B2's busy time.
- **Prior art:** command-buffer timing from Metal System Trace (standard); MTPLX's own roofline profiler (queued
  per-component benches, another model family); B5 here (GPU idle of the eager check). The anatomy of this model's exact
  check is ours.
- **Prediction** (16 rows, 1K context, 48 layers, today's MTPLX kernels: 4-bit experts at top-10, MLX's quantized
  matmuls):
  - GPU busy ~115-130 ms (B5: 31.9 ms per slice, x4 with the head counted 4 times).
  - Experts 45-55 ms; dense matmuls 20-25 ms; HC mixers 2-3 ms; head ~2.5 ms.
  - 'other' on the GPU 15-25 ms: GDN core 8-14, attention core incl. indexer 3-6, routing + combine 3-6, commit replay
    1-3.
  - Glue: A's GPU idle ~20 ms (B5: 21), B's ~3-4.
  - So NewIdeas' 'other 10-20 ms' is about right for the GPU side, and the eager engine adds ~20 ms of glue on top.
  - 8K context: attention core +2-5 ms. 32 rows: 'other' x1.3-1.6 (the GDN recurrence runs row by row).
  - Drafting 3.5-5 ms at block 16, 4-6 ms at block 32 (the selector's per-position loop is Python).
- **Interim result (Mon 14:18; data/ca_time.txt, ca_draft.txt, ca_smoke.log): end-to-end and drafting measured; the
  per-part split is NOT yet measured.**
  - Smoke passed: P's tokens equal A's, every capture-commit landed, 404 parts per 12-layer check.
  - Per 12-layer check (9 GDN + 3 QSA; exact; MTPLX's kernels, 4-bit experts at top-10), median of 10:

    | context | rows | A eager (the engine) | B async every 4 layers |
    |---|---|---|---|
    | 1K | 8 | 25.40 | 22.91 |
    | 1K | 16 | 38.87 | 36.27 |
    | 1K | 32 | 49.65 | 51.27 |
    | 8K | 8 | 25.70 | 23.65 |
    | 8K | 16 | 40.28 | 39.02 |
    | 8K | 32 | 52.80 | 56.74 |

    - x4, less ~3 x 3-5 ms for the once-per-check parts (embedding, head, PLE, acceptance read): the engine's exact
      resident 16-row check is ~141-147 ms eager and ~130-136 ms overlapped. That is twice price350's 'today's
      kernels' 68.5 ms, which used D32's exact dense kernels and L1-30c's expert kernels (top-8, ~2.7 b), not MTPLX's.
    - 8K context adds ~1.4-2.8 ms per slice at 16 rows (~0.5-0.9 ms per QSA layer). B is no faster than A at 32 rows.
  - **Drafting costs ~3.6x what price350 assumed.** DFlash2's 1-step draft (L1-20 weights, bf16, refine19's path):

    | block | drafter forward (5 layers) | head (4-bit, 248K vocab) + top-16 | selector chain | all, one sync |
    |---|---|---|---|---|
    | 16 | 7.46 ms | 4.46 ms | 0.83 ms | 12.52 ms |
    | 32 | 7.33 ms | 6.84 ms | 1.63 ms | 15.45 ms |

    - The forward reads ~1.1 GB of bf16 layers (~4.4 ms at 250 GB/s), and the head ~0.36 GB (~1.4 ms). An 8-bit
      drafter plus a fused head top-k would be ~4-5 ms. At today's 12.5 ms, price350's 16-row floor (34.3 ms with
      3.5 ms of drafting) becomes ~43 ms, so 350 would need ~15 tokens per check, about a perfect drafter.
  - **Why the per-part split failed twice:** the Metal trace (run 1 14:07, run 2 14:16) was stopped both times by the
    health monitor at ACT (disk < 100 GB). xctrace streams GPU counters to the per-user temp dir at ~120 MB/s (4.7-4.8
    GB of xrgpu_aps files + a 1.5 GB kernel trace in 40 s), whatever --output says, even with only the GPU
    instrument. My 3 s test missed it because xctrace deletes them on a normal exit. Deleted both times.
- **Result, run 3 (Mon 19:04; 1 min of GPU, no trace; ca3_batch.sh -> ca_anat.py diff; data/ca_diff.txt, ca_diff.json,
  ca_diff_report.txt): measured. 'Other' is ~45-65 ms at 16 rows, not 10-20, and hyper-connections (HC) alone cost
  31-34 ms.**
  - Method (knockouts, replacing the planned duplicates). One prefill per config, then rounds of checks cycling
    through variants:
    - A = the engine's eager path; D = my async-every-4-layers loop; one variant per part class; 'all' = every class
      knocked out; L0 = no layers.
    - A knocked-out call still builds its graph but returns its own output from an earlier check. Its GPU work is
      pruned, and its inputs are kept alive.
    - Class time = the paired wall D - D-class, 6 rounds after 1 warm-up. D is GPU-bound: with every class knocked
      out, the same Python graph build runs in under 6 ms of a 35 ms check.
    - Walls exclude each check's PLE n-gram staging, which is CPU time from the file-backed table before any GPU
      work. Including it doubled the paired noise; it is reported as its own row.
  - Per exact resident 16-row check, 48 layers (12-layer slice; per-layer parts x4; embedding / head / acceptance
    read once). MTPLX's kernels, 4-bit experts at top-10. Paired sd per class: 0.6 ms (1K), 1.2 ms (8K) per 12 layers.

    | part | 1K | 8K | note |
    |---|---|---|---|
    | routed experts (MLX sorted gather_qmm) | 58.7 | 56.8 | 1.22 ms / layer, as L1-24's microbench (1.21-1.26) |
    | HC reads + writes | 31.0 | 33.9 | eager at 16 rows, ~11 kernels per read; bf16 mixer matmuls ~12.7 of it (kern23) |
    | GDN projections | 16.8 | 19.1 | kern23's MLX 16-row shapes: ~15 |
    | QSA module (projections, indexer, core, KV writes) | 7.3 | 19.2 | its projections 2.5-5.3; core ~2 at 1K, ~17 at 8K |
    | router + shared expert | 4.3 | 1.2 | within noise (+-2.5) |
    | GDN recurrence kernel | 3.5 | 4.2 | |
    | expert sort / unsort | 3.2 | 0.7 | within noise |
    | PLE layer | ~0 | ~0 | |
    | once: embedding, head (1.4-2.1), acceptance read, sync | 2.8 | 2.7 | |
    | residual: norms, GDN conv / gating, softmax / top-k, combine, gates | ~0 (0-9) | ~0 (0-9) | 'all' leaves ~7-9 ms |
    | **verify forward, async (D)** | **127.1** | **134.7** | parts sum 127.0 / 137.9 (0.0 % / +2.4 %) |
    | engine's eager forward (A) | 132.9 | 140.6 | eager glue +5.8 / +5.9 |
    | commit replay, timed on its own | 9.7 | 9.1 | GDN recurrence over the kept rows + the PLE layer |
    | PLE n-gram staging (CPU) | 0.75 | 0.72 | 3.1-3.5 at 8 rows |
    | **check, engine path (A + commit + staging)** | **143.4** | **150.4** | **+ drafting 12.5 = 156-163 ms per ~10 tokens: ~64 tok/s, today's figure** |

  - 8 rows: engine check 88.9 ms (1K), HC 13.2 (MTPLX's fused 3-kernel HC read serves 2-8 rows). 32 rows: 215 ms,
    experts 103. At today's kernels, block 8 (~6.9 tokens) and block 16 (~10) give the same tok/s.
  - Validity: every commit landed in every variant. The parts sum to D within 0.0 % / 2.4 % (lines: 5 %). Knocking out
    everything leaves ~7-9 ms of small ops that the per-class knockouts give to their neighbours, so each class carries
    up to ~1 ms of nearby small-op latency. The 32-row configs are noisier (sd 1.4 ms; 1K residual 16 ms).
  - vs prediction:
    - Total: GPU ~115-130 predicted, 127-135 measured. Experts 45-55 predicted, 57-59 measured.
    - Wrong on HC: 2-3 ms predicted, 31-34 measured. I priced HC from the fused 1-8-row read; at 16 rows the engine
      runs the eager chain.
    - Wrong on the GDN core: 8-14 predicted, 3.5-4.2 measured (plus the commit replay, ~9).
    - Attention at 8K: +2-5 predicted, +12 measured.
    - Glue: ~20 predicted, 5.8 measured. Most of B5's 'idle' was GPU work on the HC chain.
  - **For price350, today's 16-row check splits as follows (1K / 8K):**
    - Matmuls in situ ~40 ms: GDN 16.8, QSA 5.3, router + shared 4.3, HC mixers ~12.7, head ~1.4. D32's exact kernels
      did the same set in 18.0.
    - Experts 58.7 (price350 used 26.2: L1-30c, top-8).
    - Other ~45-50 / ~60-65: HC elementwise ~18, commit 9.7, attention core ~2 / ~17, glue 5.8, GDN recurrence ~4,
      sort / unsort ~2, small ops 0-9, once ~1.4, staging 0.75.
    - So price350's 'other 20' is ~2.5-3x short. NewIdeas' rule: >= 15 ms means glue / HC / attention come first.
  - **Check levers, ranked (ms per 16-row check, 48 layers; exactness is each lever's own question):**
    1. HC: 31 -> ~6-8 with a fused 16-row read and write. The bf16 mixer bytes are 1.27 GB per check (floor ~4.5 ms).
       MTPLX already has a bit-exact 3-kernel read for 2-8 rows, but only in the compiled lane. A 16-row version is
       bitwise only if it repeats MLX's 16-row bf16 GEMM arithmetic. **-23 to -25.**
    2. Experts: 58.7 -> 33-37 with L1-25 / L1-27's NAX kernels. They are batch-invariant but not bitwise vs MLX, so
       they change the reference (Kamen's call). **-22 to -26.**
    3. Matmuls: ~40 -> ~18 (D32's qmm16c / mr2; MLX's 16-row call is padding-independent; the same reference
       question). **~-22.**
    4. Attention core at long context: ~17 at 8K. Context-dependent: **~-10 at 8K.**
    5. Commit replay: 9.7. Most of it is a second pass of the GDN recurrence + the PLE layer over the kept rows.
    6. Drafting: 12.5 (DR / FR-Spec screens, R-06 / R-07). **-7 to -9.**
    7. Eager glue: 5.8 (async every 4 layers). **-6.**
    - All together: ~156 -> ~55-60 ms, ~2.7x. 350 at 12-14 tokens needs ~34-40 ms, so the check stays ~1.5x short even
      with every measured lever.

## SD (NewIdeas Mon 12:30, QUEUE row SD; Leap, Mon 14:20 EET): structurally different drafters and verify shapes
- **The need** (NewIdeas' repricing, 13:46): with the check near its floor, 350 needs 12.0-13.7 tokens per 16-row check
  (+26-43 % over L1-20's 9.54; lattice oracle 14.1, perfect 15.6) or 15.5-17.6 per 32-row check. Block 16 wins at every
  kernel level, and a design's own drafting ms counts against it.
- **What our own data already says.** NT-1's replay of L1-20 (block 32) gives the hit rate at each depth, given hits
  before it: 0.94 0.92 0.90 0.93 0.95 0.93 0.93 0.94 0.95 0.91 0.91 0.89 0.87 0.87 0.88 (depths 1-15).
  - These rebuild L1-20's block-16 figure: 1 + the sum of the running products = 9.53 (measured 9.54).
  - With no depth decay at all (the shallow rate ~0.93 held at every depth) block 16 gives (1 - 0.93^16) / 0.07 =
    9.81, only +3 %.
  - So at block 16 the gap is per-token quality, not depth. 12.0 needs a flat 0.96 per token, 13.7 a flat 0.98. The
    near-copy (the 125B with 10 % of its experts, Chat1000) holds 0.983 and gives 13.4.
  - Families whose gain is depth robustness (AR drafters, semi-AR, exposure-bias fixes) can add at most ~+3-5 % at
    block 16. At block 32 depth matters more (a flat 0.93 gives 12.9, +26 %), but block 32 loses to block 16 anyway.
- **Families** (code / HumanEval, temperature 0; published gains are over a parallel-block baseline):

  | family | published | here | drafting cost per check | ceiling at block 16 | verdict |
  |---|---|---|---|---|---|
  | (i) AR drafters: EAGLE-3 training-time test, HASS, GRIFFIN | DFlash beats EAGLE-3 in tokens per check (2602.06036; AdaFlash 2607.19223 Table 4, Qwen3-4B HE: 4.59 vs 3.34). EAGLE-3 HE 6.52-6.74 with trees (~0.92-0.93 per token, inflated by the tree); EAGLE chained 0.76 per token on HE (Falcon's table); the 122B card's native MTP 7.68 at 15 steps (~0.89) | DFlash2 already holds ~0.93 at shallow depths | 15 sequential steps: native MTP 2.2 ms each (+33 ms); a 1-layer 4-bit drafter in a native loop >= ~0.3 ms (+4.5 ms) | 9.81 (+3 %) if it matched 0.93 flat | dead (our data + arithmetic) |
  | (i) AR tails / causal correction in the block: Domino, DSpark's Markov head, xPress, DARTree, H-Spec | over vanilla DFlash on HE: Markov head +20 %, xPress +27 % (2608.02438: 6.44 -> 7.76 -> 8.15), i.e. +5 % over a Markov head; Domino +7 % (2608.13524) | DFlash2 has the first-order (Markov) selector already; L1-21's picker +1 % | ~0 (xPress: ~6 Jacobi passes of a low-rank mixer) | ~+5 % | dead for 350 (+1-5 %) |
  | (i) semi-AR / PARD-style: Falcon, PARD, PARD-2, Gumiho | same class as DFlash: PARD-2 HE 9.19 at k 16 (Llama-3.1-8B, ~0.915 per token); Falcon HE 0.80 vs EAGLE 0.76 per token | re-drafting the tail after the drafter's confident prefix: +3 % at block 32, 0 at block 16 (L1-20) | +3.5 ms per extra drafter pass | ~+3 % | dead |
  | (ii) selectors stronger than first order over the lattice | LiLiCorr level with DFlash2's selector on HE (2608.20530); Domino / DSpark +1 % at best; xPress +5 % | L1-21: +1 % (12 M picker, plateaued at once) | ~0-1 ms | ~+5 % | dead |
  | (iii) exposure bias / on-policy | AdaFlash OPD: in-domain +26 % over off-the-shelf DFlash on code, cross-domain +4-5 %; acceptance-aware losses (2606.11552: position weights, a first-error focal loss and a chain loss) +21-76 % over a POSITION-UNIFORM baseline, no extra forward | a one-pass drafter never conditions on its own drafts, so on the accepted path there is no exposure bias; our fine-tunes on the 125B's own outputs took the domain gain (+7-11 %); DS: 2x on-policy data +1 %. Our trainer IS position-uniform: untested here | 0 | +3-8 % (moving accuracy to shallow depths: a +0.01 rise at depths 1-8 is worth +5 %) | the one free untested lever: SD1 below |
  | (iv) capacity: bigger drafters | DFlash Table 6, 3 / 5 / 8 layers: Math500 5.64 / 5.99 / 6.33 tokens, speedup 4.69 / 4.71 / 4.64x (flat); MT-Bench 3.37 (5) vs 3.50 (8). Scylla (2505.07858): log-linear in draft capacity to 10 layers | the shrink test NewIdeas proposed is answered by prior art: 3 layers = 0.94x of 5, at the 0.95 line | ~linear in layers (3.5 ms at 5) | +6 % per 1.6x layers; a flat 0.96 needs ~4-5 doublings (~60-110 ms of drafting) | dead (net ~0) |
  | (v) verify shapes: trees | DDTree (2604.12989, best-first over DFlash's marginals): HE 6.62 -> 9.44 (+43 %) at its best budget (256-512 nodes, datacenter GPUs), +33 % at 64 nodes; DARTree (Domino head + tree, 64 nodes): +51 % | L1-22: a 16-row tree (15 nodes) 10.43 vs the chain's 9.54 (+9 %); 32-row tree at block 32 12.14 | ~0; but every row costs expert bytes (union ~rows^0.6-0.72), and the 36 GDN layers need a tree-aware verify (unbuilt) | 10.43 at 16 rows: 304 tok/s at NewIdeas' floor vs the chain's 278 | best exact lever measured: +9 % |
  | extra: retrieval / copy drafts merged into the tree | Draft Less, Retrieve More (2605.20104); SuffixDecoding 7.82 tokens per step on SWE-bench agent traces | NewIdeas' F lever covers agentic copying | ~0 | small on fresh code; large on agentic edits | not ranked here |

- **Ranked for 350 at block 16:**
  1. Trees at 16 rows: +9 %, measured (L1-22). Needs a tree-aware GDN verify.
  2. An acceptance-aware loss: +3-8 % expected, free at inference, untested here (SD1).
  3. NT-1's lossy acceptance (NewIdeas; non-exact, Kamen's call): +10-21 % (MOD) / +1-2 % (LB).
  4. A causal refiner over the selector (xPress): <= +5 %; L1-21 says ~+1 %.
  5. Capacity: net ~0. AR drafters and semi-AR: net negative at block 16.
- **Stacked:** the exact levers (1 x 2 x 4) give ~+12-24 %, so 10.7-11.8 tokens per 16-row check. That is ~310-345
  tok/s at NewIdeas' floor (2-11 % short of 12.0) and ~275-300 at 90 % kernels (14-22 % short of 13.7). With NT-1 on
  top, 350 at the floor becomes reachable; at 90 % kernels it stays marginal.
- **Conclusion:** no published drafter family delivers the +26-43 % at block 16. Below 1B parameters the per-token
  quality sits at ~0.93; 13.7 needs ~0.98, which only target-class models reach (the near-copy). So the drafter half
  cannot close 350 alone. The check half (CA) has to carry most of the rest, with NT-1 (non-exact) as the swing.

## SD1 (Leap, pre-registered Mon 14:20 EET; SD's one test, waits for NewIdeas' GO): an acceptance-aware loss at block 16
- **Bet:** the trainer weights every depth equally. Accepted length rewards shallow depths far more: at block 16 the
  gradient of the expected length with respect to the log-probability at depth j is R_j = the sum of the running hit
  products from j on (~8.5 at depth 1, ~3.3 at depth 8, ~0.3 at depth 15). Weighting each depth's loss by R_j (stop
  gradient; the chain loss of 2606.11552) moves capacity to where it pays.
- **Test** (~50 min of GPU, no new 125B data; the trainer gains one flag):
  - Arm U: L1-20's recipe at block 16 (--bs 16, 2 epochs = 201 steps on data/l120, no partial masking), uniform weights.
  - Arm W: the same, each depth's loss weighted by its R_j from the drafter's own probability of the true token.
  - Both scored with the replay on the 24 held-out HumanEval samples at block 16 (1-step), per sample.
- **Lines:** W / U >= 1.10 (90 % CI above 1.03): PASS, stack it (free at inference). W / U < 1.05: KILL, the plateau is
  not a loss-weighting artifact. Between: keep it as a +5-10 % stackable lever.
- **Prediction:** W / U 1.03-1.08 (U ~9.6-9.9). 60 % between or kill, 15 % pass.

## L1-33 (NewIdeas' queue, MoonShots' idea; taken by Leap Sun 04:12 EET, PARKED 04:13 behind B1 and B5): screened argmax-only lm_head
- **Idea:** a low-bit screen of the 4-bit head (248320 x 2560) plus a provable per-row bound. Read exact 4-bit rows only
  for tokens the bound can't exclude, so the greedy argmax stays provably identical.
- **Prize:** at most ~1.1 ms per check (the head's 16-row qmm is ~2.2 ms), ~1.4 %.
- **Kill line (the queue's):** exact rows > 25 % of the vocab at the median position, or screen + exact reads > 50 % of
  the head's 16-row time (1.1 ms).
- **Prior:**
  - MoonShots' 2B proxy (real hidden states, bf16 head) found that 2- and 3-bit screens with a provable bound rule out
    nothing.
  - Arithmetic: the group-L1 bound is ~err x ||h||_1, i.e. (L1/L2 of h) x 0.2 (3-bit) or 0.6 (2-bit) logit standard
    deviations wide; L1/L2 is ~40 for a flat h. A 3-bit screen alone reads 78 % of the head's bytes (over the time
    line). A 2-bit screen (56 %) would need >= ~80 % of the rows excluded.
  - Prediction: dead unless the 125B's final hidden states are spiky (L1/L2 <~ 10).
- **Cheapest decisive test:**
  - (1) h33_extract.py: capture the 125B's real final hidden states (the head's input) on 4 public HumanEval windows,
    one streamed prefill at C 96 (GPU ~3-5 min).
  - (2) screen33.py, CPU: exact logits from the real 4-bit head; MSB-3 / MSB-2 screens with the group-L1 bound;
    candidates per position (median / p90); L1/L2 of h.

## Where an exact 16-row check stands (Leap, updated Sun 03:20 EET after L1-23..L1-32; measured where marked)
| part (16 rows, resident, top-8) | ms | basis |
|---|---|---|
| routed experts, 62 distinct per layer x 48 | 4-bit ~35, 3-bit ~33, 2-bit ~25 | measured per expert at 16 rows: 11.9 / 11.1 / 8.5 us (L1-30, L1-30c K-paired) |
| dense layers (full dense 3.44 GB; ~2/3 with the diet) | ~18 | measured: L1-23 + bf16 forms + native layout + L1-29 |
| GDN core (prework, recurrence, gated norm) + capture-commit replay | 6.2 | measured (L1-31; floor 0.97) |
| attention core + indexer, HC elementwise, routing, misc | ~6-9 | estimated |
| check edges, sync, Python (pass edges 8.7 ms at 4 rows) | ~9-10 | Fit12; mend (c): ~3,000 ops x 3.1-3.4 us eager |
| drafting (DFlash2) | ~3.5 | |
| **total at 3-bit experts** | **~77-82** | **-> ~115-125 tok/s at 9.5 tokens per check** |

- Today's real engine: 53.5 tok/s (RES3, MTP, C_m 448).
- Kernel status: matmuls at 64-88 % of their byte floors. Below 4 bits the per-expert cost stops following bytes
  (L1-30: 2-bit x1.38, 3-bit x1.08 vs 4-bit). Each remaining kernel item is <= ~5 % of the check (GDN fusion ~3-4 ms,
  exact grid ~1 %). The exception is the per-check dispatch / Python (~7-8 ms), which needs engine work: a compiled or
  native loop (MTPLX has a compiled fixed-4-row lane).
- 350 at 9.5 tokens per check needs ~27 ms per check. With every byte-bound part at 100 % of its floor, the bytes
  alone (experts 6.4 GB at 3-bit + dense 2.1-3.4 GB) are ~31-35 ms, and the non-matmul + edges (~21-25 ms) alone cap
  9.5 tokens at ~400 tok/s. So 350 needs fewer bytes AND more tokens per check AND a leaner loop together.
- Model-level levers measured tonight:
  - top-8: union -17 % (L1-17: MoE error 0.78x MSB-3's; Eval-suite verdict pending).
  - Parallel streams of one answer (L1-32): 0.59x DRAM reads per token, resident only. At 24 GB they raise SSD misses
    1.7-1.9x.
  - Frugal wait-top-2 at 24 GB: fails IFEval.
  - Pruning: rejected by Kamen.
- At 24 GB the SSD wall moves only with residency, bits per expert, top-k and routing. Misses per token don't depend on
  drafting (resid27), and anything that widens the per-check working set costs misses (L1-32b).

## Baseline (the earlier table, for reference): code tok/s at the bytes floor, 280 GB/s, everything resident
| route | resident | tokens / pass | GB / pass | tok/s at floor |
|---|---|---|---|---|
| DFlash2 block 8, 4-bit | 71.7 GB (cannot fit) | 6.87 | 9.4 | 186 |
| block 16 FT + truncation, 4-bit + dense diet | 71.7 GB | 9.38 @ 10.3 rows | 9.2 | 258 |
| block 16 FT + truncation, v3 tiers + diet | ~56 GB | 9.38 @ 10.3 rows | 7.6 | 305 |
| block 16 FT, v3 + diet + budget (32 experts / layer) | ~56 GB | 9.88 | 5.4 | 433 |
- The 35B section is retired (Kamen: the 125B only). The 35B work is a tool now. The GDN kernel program and the rollback
  finding port to the 125B: Fit12 has the port.

## R-06 + R-07 (NewIdeas' rapid screens; Leap, pre-registered Mon 19:15 EET): drafting at the speed of its bytes
- **Why:** DFlash2's 1-step draft costs 12.5 ms at block 16 (CA: forward 7.46 ms, head + top-16 4.46, selector 0.83).
  That alone is ~1/3 of the 34-40 ms check that 350 needs.
- **Screen** (scripts/dr_screen.py, one runner, <= 10 min; L1-20 weights):
  - Drafter layers + fc in bf16 / 8-bit / 4-bit (group 64; selector, embedding and target head unchanged).
  - x three ways to get the top-16 candidates:
    - argp: argpartition over all 248K logits (today).
    - 2s: exact two-stage (max per 128-token block -> top-16 blocks -> top-16 of their 2,048 logits). Same candidate
      set, so the same tokens.
    - v32k: FR-Spec 2502.14856. The head runs over 32K tokens only, ranked by counts on L1-20's training tokens, then
      by id.
  - Draft ms: one graph, one sync, median of 30 on a held-out sample.
  - Tokens per check: refine19's teacher-forced replay on the 24 held-out HumanEval samples, blocks 16 and 32.
- **Lines (NewIdeas' QUEUE):**
  - R-06 PASS: an argp / 2s arm <= 5 ms at block 16 with tokens within 1 % at both blocks. KILL: the best arm within
    1 % > 8 ms, or every arm <= 5 ms loses > 3 %.
  - R-07 PASS: v32k head <= 1.5 ms at block 16 and tokens >= 99 % of the full head. KILL < 98 %.
- **Prediction:**
  - Head pieces at block 16: lm_head ~2.2 ms, argpartition ~2 ms, 2s ~0.3 ms.
  - Arms: bf16 2s ~10.5 ms. 8-bit 2s 7.5-10.5 ms (MLX's 8-bit matmuls ran at 20-24 % of their floor in D32). 4-bit 2s
    ~5.5-6.5 ms.
  - Tokens: 8-bit within 0.5 %; 4-bit -1 to -4 %.
  - R-06: BETWEEN, 60 %. PASS 15 %: only 4-bit at <= 5 ms if its tokens hold.
  - R-07: head ~0.6-0.9 ms. The held-out code tokens are ~99.5 % inside the set, so tokens >= 99 %: PASS 70 %. This
    is in-domain (HumanEval-like counts on HumanEval), so it is optimistic for chat.
- **Result (Mon 19:16, 2 min of GPU; data/dr_screen.txt): R-06 KILL, R-07 KILL.**

  | b16 draft (ms) / tokens vs bf16 argp | argp | 2s | v32k |
  |---|---|---|---|
  | bf16 | 12.91 / 1.000 | 10.92 / 1.000 | 9.13 / 0.960 |
  | 8-bit | 12.02 / 1.009 | 9.98 / 1.009 | 8.10 / 0.965 |
  | 4-bit | 11.68 / 0.996 | 9.64 / 0.996 | 7.79 / 0.953 |

  - Two-stage top-16 is exact (tokens 1.0000 at both blocks) and saves 2.0 ms at b16, 4.2 ms at b32. Keep it.
  - Bytes are not the drafter's cost: 4-bit saves only 1.3 ms. Its forward (~6-7 ms) is launch-bound.
  - FR-Spec's 32K head runs in 0.85 ms but loses 4-6 % of tokens, although 99.2 % of the held-out tokens are inside
    the set. The draft's other candidates matter, not only its top-1.
  - vs prediction: head pieces right (lm_head 2.57, argp 2.23, 2s ~0). 8-bit / 4-bit far slower than predicted
    (launch-bound, not byte-bound). v32k tokens wrong: I predicted PASS at 70 %.

## R-09 HCF (NewIdeas GO Mon 19:12; Leap, pre-registered Mon 20:51 EET): the 16-row hyper-connection read, fused
- **Why:** CA run 3: HC reads / writes = 31-34 ms of the 143-150 ms 16-row check. At 16 rows the engine runs the eager
  chain, ~14 kernels per read. MTPLX's 3-kernel read serves 2-8 rows only, and it equals the *compiled* chain (wide
  gemv sums), not the 16-row GEMM.
- **Screen (scripts/r09_screen.py, ~1 min):** 96 reads (48 layers x 2 mixers, the model's own weights), each read's
  pending write from the previous read, 16 rows, interleaved arms, median of 20.
  - E: the engine's eager chain (GatedResidual itself, decode_verify phase).
  - X: exact by construction, 6 kernels. MTPLX's norm kernel (write + grouped RMS norm) + MLX's own down / inject / up
    matmuls (the same calls, so the same sums) + one gate kernel and one tail kernel. Both read tables of the stock
    elementwise functions at every bf16 input.
  - X5: X with down + inject as one matmul (5 kernels). F8 / F16: MTPLX's 3-kernel read as 2 x 8 rows / 1 x 16.
  - Bitwise: all 96 (mixed, written, inject) vs E.
- **Lines (QUEUE R-09):** PASS >= 15 ms saved per check AND bitwise = E. KILL < 8 ms. Fast but not bitwise = a
  reference change, logged and priced like NAX.
- **Prediction:** E isolated ~25 ms (20-32). X ~13 ms (9-17), so ~12 ms saved: BETWEEN 45 %, PASS 30 %, KILL 25 %. The
  three matmuls (~13 ms per check in kern23) stay as they are. X bitwise = E: 75 %. X5 bitwise: 50 % (the GEMM may
  tile N = 324 differently). F8 / F16: not bitwise (95 %), F16 saves ~18 ms.

## R-10 DCOMP (NewIdeas Mon 20:40; Leap, pre-registered Mon 20:51 EET): the drafter compiled
- **Why:** R-06: the draft costs 10.92 ms (bf16) / 9.64 ms (4-bit) at block 16 with the exact two-stage head. Weight
  bytes are not its cost. ~125 small kernels per layer are: the grouped dynamic convolution is ~23 per call, 4 calls
  per layer.
- **Screen (scripts/r10_screen.py, ~4 min):** mx.compile of each layer's fixed-shape parts (norm + conv prepare +
  q / proposal k, v; o_proj + conv finish + MLP part) and of the tail (norm, head, two-stage top-16, selector chain).
  The attention core stays eager because its shapes and RoPE offsets change every check. Arms E / C / P (C with the
  convolution as views of one padded copy) in bf16, E4 / P4 in 4-bit. Draft ms on one sample, tokens on the 24
  HumanEval replays, blocks 16 and 32.
- **Lines (QUEUE R-10):** PASS draft <= 6.5 ms at b16, tokens within 1 %. KILL > 8.5 ms.
- **Prediction:** C bf16 ~8.5-9 ms, P4 ~7.5 ms (6.8-8.5): BETWEEN 55 %, KILL 35 %, PASS 10 %. Compiled drafts bitwise
  = eager: 60 %. Tokens within 1 %: 95 %.
- **R-09 result (Mon 20:52, 6 s of GPU; data/r09_screen.txt): BETWEEN.** Exact X5 saves 9.67 ms per check, bitwise.

  | arm (96 reads, 16 rows) | ms | saved vs E | bitwise = E |
  |---|---|---|---|
  | E eager chain (~14 kernels / read) | 24.37 | - | - |
  | X 6 kernels (norm, 3 MLX matmuls, gate, tail) | 22.55 | 1.81 | yes, all 96 reads |
  | X5 5 kernels (down + inject as one matmul) | 14.70 | 9.67 | yes, all 96 reads |
  | F8 MTPLX read, 2 x 8 rows | 10.51 | 13.85 | no (from read 0) |
  | F16 MTPLX read, 16 rows | 11.79 | 12.58 | no |

  - Fusing the elementwise ops is worth only 1.8 ms. The read is bound by its skinny matmuls, as MTPLX found at
    prefill.
  - The inject matmul (N = 4, K = 10,240) alone costs ~8 ms per check. Folding it into the down matmul (N = 324) is
    free and bitwise: MLX's GEMM sums each output in the same K order whatever N is.
  - The next exact step is that same K order with more threadgroups for the down projection (N = 324 runs on only ~11
    output tiles over K = 10,240). That is the new idea R-09b.
  - vs prediction: E 24.4 (pred ~25, right). X saved 1.8, not ~12 (wrong: the elementwise ops are not the cost). X5
    bitwise (I gave it 50 %). F16 saved 12.6 (pred ~18).
  - In the check, HC costs 31-34 ms (CA knockouts) vs 24.4 here, so the in-check gain may be larger (~x1.3).
- **R-10 result (Tue 19:24-19:26, ~2 min of GPU; data/r10_screen.txt): KILL.** Compiling is exact but saves 0.7 ms.

  | arm | b16 ms | b32 ms | graph build (b16) | tokens per check vs E |
  |---|---|---|---|---|
  | E eager, bf16 | 11.24 | 12.21 | 0.46 ms | 1.0000 |
  | C compiled blocks + tail | 10.58 | 11.55 | 0.17 ms | 1.0000 (drafts bitwise = eager) |
  | P C + conv as views of one padded copy | 10.50 | 11.54 | 0.18 ms | 1.0000 (bitwise) |
  | E4 eager, 4-bit | 9.81 | 11.44 | 0.43 ms | 0.9940 |
  | P4 compiled, 4-bit | 9.37 | 10.96 | 0.17 ms | 0.9940 (bitwise = E4) |

  - Python builds the drafter's graph in 0.46 ms of 11.2. The drafter is GPU-bound: its ~500 small kernels cost GPU
    time, not CPU time. Fusing elementwise ops removes 0.7 ms of that.
  - vs prediction: C ~8.5-9 predicted, got 10.6 (wrong: I thought the CPU side was the cost). Bitwise (60 %): yes.
  - It ran during soft_end's ramp (duty ~55 %), which SIGSTOPs new children of held runners too (Fit12). The medians
    of ~30 drafts per arm hold: E4 9.81 vs R-06's 9.64 unramped (+1.8 %). Even 2 % off, P4 is ~9.2 > 8.5: KILL holds.
    R-09b step 1 (19:18:59-19:19:05) was not stopped: a stop would have added ~15 s to its 6 s.

## R-05 (SD1 resized; NewIdeas' rapid screen; Leap, pre-registered Mon 22:33 EET): acceptance-weighted depth loss
- **Screen (r05_batch.sh, ~9 min):** arms U (L1-20's recipe: bs 32, pmask 0.5, micro 1, lr 2e-5) and W (each depth's
  loss x R_j, the stop-gradient suffix sum of the running hit products of the drafter's own true-token probabilities,
  scaled to keep each micro-batch's total weight; scripts/r05_train.py --wloss, checked by hand on CPU). Each runs 21
  steps (~3 min, warm-up 3, cosine) from the L1-20 checkpoint. Scored by the 1-step replay on the 24 held-out
  HumanEval samples at block 32 (primary) and 16. Weights go to a RAM disk.
- **Lines (QUEUE R-05):** PASS W / U >= 1.05 at block 32 (then the 50-min run). KILL < 1.02.
- **Prediction:** U ~10.2 (= L1-20, which has plateaued: DS2). W / U 1.00-1.03: KILL 55 %, BETWEEN 35 %, PASS 10 %. 21
  steps is little time to move a plateaued drafter.
- **Result (Tue 19:51-19:58, ~7 min; data/r05_report.txt): KILL.** W / U = 0.970 at block 32 (90 % CI 0.954-0.986).
  At block 16 it is 0.976 (0.963-0.989).
  - U 9.68 / W 9.40 tokens per pass at block 32. U 9.37 / W 9.14 at block 16.
  - Both are below L1-20 itself (10.23 / 9.54 in R-10's replay), so 21 more steps of the same recipe already cost
    ~5 %.
  - W's printed loss is ~0.2 -> 0.03, against U's 3.3 -> 1.6. The R_j weights pile onto positions the drafter
    already gets right, so W trains on the easy depths and loses the hard ones.
  - vs prediction: KILL (55 %): yes, and W / U came in below 1, worse than the 1.00-1.03 I gave.

## R-09b step 1 (NewIdeas GO Mon 20:56; Leap, pre-registered Mon 22:33 EET): is there room for a same-order GEMM?
- **Doubt about my own premise:** MLX may already split K for M 16 x N 324 x K 10,240 (16 partitions x 11 tiles =
  176 threadgroups). Then "more threadgroups" is not the lever.
- **Screen (scripts/r09b_screen.py, ~30 s):**
  - Part 1, bits: the merged GEMM at 16 rows vs the same rows inside a 32-row GEMM, a batch of two, and 2 x 8 rows.
    This shows whether the 16-row path is the regular GEMM.
  - Part 2: X5 knockouts per stage (norm, merged GEMM, gate, up GEMM, tail).
- **Decision:** R-09b's ceiling = 9.67 + (merged GEMM in-chain cost - ~2.5 ms bytes floor). Below 11: KILL now.
  At 15 or more: build the GEMM. Otherwise BETWEEN at best (parked behind R-12 / R-11).
- **Prediction:** 16 rows = 32-row rows bitwise: 50 %. Merged GEMM in-chain ~5 ms, so the ceiling is ~12 (BETWEEN at
  best): 55 %. Build it: 20 %. KILL now: 25 %.
- **Result (Tue 19:19, 6 s of GPU; data/r09b_screen.txt): build it (17.0 by the rule; ~15 by a second estimate).**
  - Bits: the merged GEMM (and the up GEMM) at 16 rows = the same rows of a 32-row GEMM and of a batch of two,
    bitwise. Two 8-row calls differ. The 16-row path is MLX's regular GEMM.
  - Per 96 reads: E 24.62 ms, X5 14.81 (R-09: 24.37 / 14.70). Knockouts (X5 minus X5 without the stage): norm 3.01,
    merged GEMM 9.87, gate 9.98, up GEMM 4.01, tail 4.25.
  - The knockouts sum to 31.1 against a 14.8 chain, so they are not additive. Removing a stage splits the dependent
    chain into two halves that the GPU overlaps. The gate has almost no work yet "costs" 9.98.
  - Rule: 9.67 + (9.87 - 2.5) = 17.0 >= 15, so build it. Second estimate: the merged GEMM walks K = 10,240 per output
    tile like the N = 4 inject (~82 us per call, R-09) -> ~7.9 ms per 96 reads, floor ~2.5 -> <= 5.4 more -> ~15.1
    vs E. Both say >= 15, the second only just.
  - vs prediction: bitwise 16 = 32 rows (50 %): yes. Build it (20 %): yes. In-chain ~5 predicted, 9.87 measured, but
    inflated by the overlap.
  - Next (the build): find MLX's K order for this shape (plain or split-K, by bitwise candidates in a custom kernel),
    then the same order on narrower output tiles (BN 8: ~41 threadgroups). Lines unchanged: PASS >= 15 ms vs E AND
    bitwise, KILL < 11.

## R-13 OCC (NewIdeas Mon ~22:30): occupancy audit of the skinny GEMMs, answered from data (Leap, Mon 22:35, no GPU)
- **Source:** kern23 (L1-23: the 125B's dense shapes and per-check counts, 16 rows, cold) and CA run 3 (in the check).
- **Excess over the bandwidth floor per 16-row check, HC excluded:**

  | shape (count) | floor | shipped (chunked) | excess | MLX 16-row call | batch-invariant best |
  |---|---|---|---|---|---|
  | GDN in_proj_qkv (36) | 1.90 | 9.23 | 7.33 | 3.31 | 2.53 |
  | o_proj + out_proj (48) | 1.52 | 6.87 | 5.35 | 7.51 | 2.39 |
  | lm_head (1) | 1.28 | 5.69 | 4.41 | 2.21 | 1.53 |
  | GDN in_proj_z (36) | 1.14 | 5.18 | 4.04 | 5.65 | 1.79 |
  | QSA q_proj (12) | 0.76 | 3.45 | 2.69 | 1.44 | 1.29 |
  | shared gate/up (96) | 0.32 | 1.84 | 1.52 | 1.82 | 1.82 |
  | shared down (48) | 0.16 | 1.41 | 1.25 | 1.39 | 1.12 |
  | k/v, a/b, shared gate, PLE (various) | 0.32 | 2.73 | 2.41 | 2.20 | 2.16 |
  | **sum** | **7.4** | **36.4** | **29.0** | 25.5 | 14.6 |

  - The 8-bit router / indexer (~1.7 ms) is not in kern23. In the check (CA run 3, 1K) the same matmuls cost ~25 ms
    against ~7.4 of floors, so ~18 ms of excess remains after concurrent dispatch overlaps some of it.
- **Verdict: LEVER (>= 10 ms per check).** But most of it is already priced:
  - L1-23's batch-invariant set recovers ~21 ms. It is exact under its own numerics, but not bitwise vs the shipped
    chunked path, so it is a reference change.
  - The skinny shapes R-13 names (shared-expert gate, GDN a/b) are small: ~0.9 ms of excess together.
  - The big excess is on the wide 4-bit shapes, under the chunked path.
- **Bitwise-vs-shipped route:** keep each shipped kernel's per-output K order and give it more parallelism. That
  needs the dispatch question R-09b step 1 answers first. Until then it is unpriced.

## R-12 step 1 (NewIdeas Mon ~22:25; Leap, pre-registered Mon 22:38 EET): what the 9.7 ms commit is made of
- **Why first:** the commit replays each pure-GDN layer's recurrence over the kept rows (gated_delta_update + the
  conv tail, ~5-7 kernels per layer) and the PLE layer's whole forward. CA scaled the 12-layer commit (2.42 ms) x4,
  which counts the one PLE layer four times, so 9.7 may be too high.
- **Pricing NewIdeas' select variant from bytes:** a GDN state is 48 x 128 x 128 fp32 = 3.1 MB. Keeping all 16 rows'
  states writes ~1.8 GB per check (~7 ms), more than it saves. Every 4th row writes ~0.45 GB (~1.8 ms) and still
  needs a <= 3-row replay, which costs about what a 10-row one does if the replay is launch-bound.
- **Screen (ca_anat.py commit, ~1 min):** after real 16-row checks at 1K, keeps 1 / 4 / 10 / 15. Each commit part is
  evaluated alone: pure-GDN replays, the PLE layer, the rest. 48-layer estimate = GDN x 35/8 + PLE + rest x 4. Also
  the state bytes per layer.
- **Decision:** R-12's select variant is KILL by arithmetic if its added writes reach the replay's cost. A launch-bound
  replay (GDN time at keep 1 >= 0.7 x keep 15) points instead to one fused / batched replay kernel, priced as GDN time
  minus its bytes floor. If the true 48-layer commit is <= 5 ms there is no lever.
- **Prediction:** 48-layer commit ~7 ms (5.5-8.5): GDN ~5.5 (0.16 per layer), PLE ~1, rest ~0.5. keep 1 / keep 15 ~0.8
  (launch-bound). Select variant: KILL (75 %).
- **Result (Tue 19:45, 11 s of GPU; data/ca_commit.txt): KILL for COMMIT-SELECT. The commit is ~4.7 ms, not 9.7.**

  | keep | GDN replays (8 layers) | per layer | PLE layer | rest | 48 layers | CPU build (12 layers) |
  |---|---|---|---|---|---|---|
  | 1 | 0.57 | 0.072 | 0.99 | 0.09 | 3.87 | 0.47 |
  | 4 | 0.59 | 0.074 | 0.99 | 0.09 | 3.93 | 0.50 |
  | 10 | 0.74 | 0.093 | 1.09 | 0.09 | 4.72 | 0.57 |
  | 15 | 0.90 | 0.112 | 1.16 | 0.09 | 5.44 | 0.62 |

  - CA's 9.7 ms counted the PLE layer's full replay (~1.1 ms) four times. At L1-20's ~10 tokens per check the
    48-layer commit is ~4.7 ms of GPU, already under R-12's 5 ms PASS line before any change.
  - Select loses on bytes. Keeping all 16 rows' states writes 1.81 GB per check (7.2 ms), against a GDN replay of
    3.3 ms. Every 4th row writes 0.45 GB (1.8 ms) and still needs a <= 3-row replay (~0.07 ms x 35 = 2.5 ms,
    launch-bound): 4.3 ms > 3.3.
  - The GDN replay is launch-bound: keep 1 / keep 15 = 0.64, with ~0.07 ms fixed per layer.
  - Spin-off lever (exact, new): the 35 layers' replays are independent of each other. One batched kernel for all of
    them, with the same per-layer arithmetic, would remove most of the ~2.5 ms of fixed cost, so ~-2.5 ms per check.
  - The commit's CPU build is ~2.3 ms at 48 layers (0.57 x 4). That is part of R-11's idle.
  - vs prediction: 48 layers ~7 predicted, 4.7 measured (my per-layer GDN guess was 0.16, it is 0.09). Launch-bound:
    yes (0.64 vs 0.8). Select KILL: yes.

## R-21 GDNB step 1 (NewIdeas' QUEUE row R-21, from R-12; Leap, pre-registered Thu 2026-10-29 17:35 EET): the floor of a batched replay
- **Row's lines (NewIdeas):** PASS = commit -2 ms or more per check AND bitwise. KILL = < 1 ms.
- **Prior art:** batching independent small recurrences into one launch is standard (fla's varlen / batched chunk
  kernels; CUDA graphs and persistent kernels for launch-bound decode). Applied here to the 35 commit replays: untested.
- **Why a bound first:** a batched replay needs its inputs in one place. mx.concatenate copies each input with its
  own launch, and stacking the 35 fp32 states moves 110 MB, so the naive way may give the gain back. The real kernel
  (grouped multi-pointer, <= 31 Metal buffers per launch) is only worth writing if even the ideal batch clears the
  line.
- **Screen (scripts/r21_screen.py, ~1-2 min GPU, synthetic values on the 125B's GDN shapes):**
  - per-layer = MTPLX's own replay code for 35 layers (sigmoid, compute_g, the gated-delta kernel, conv concat +
    contiguous);
  - ideal = the same ops once on inputs already stacked [35, ...] (B = 35 in the same kernels): the floor of any
    batched replay;
  - stacked = ideal + the mx.stack of the 35 layers' inputs inside the timing.
  - Measured: the lazy graph is built, then mx.eval is timed. 40 interleaved reps at keep 1 / 4 / 10 / 15. Each
    layer's new state and conv state from ideal / stacked is compared bitwise with per-layer.
- **Lines for this step:** ideal saves < 1 ms at keep 10 -> KILL R-21 (no batched kernel can pass). >= 2 ms AND
  bitwise -> build the batched kernel. BETWEEN otherwise.
- **Prediction:**
  - per-layer at keep 10 ~3.3 ms (R-12: 0.093 x 35). Ideal ~1.0-1.3: the states' 217 MB read + write alone is ~0.8 ms.
    Saves ~2.0-2.3.
  - Stacked saves < 1 (the 110 MB state copy + per-input copies).
  - Bitwise ideal 95 % (the kernel's arithmetic per (layer, head) does not depend on B).
  - BUILD 50 %, BETWEEN 35 %, KILL 15 %.
- **Result (Thu 18:16, runner 20, rc 0, seconds of GPU; data/r21_screen.txt): KILL. The replay is already at its bytes
  floor, so batching has nothing to remove.**

  | keep | per-layer (35 launches each) | ideal batched | naive stacked | ideal saves (GPU + CPU build) |
  |---|---|---|---|---|
  | 1 | 1.194 ms | 1.091 | 2.343 | 0.10 + 0.23 |
  | 4 | 1.247 | 1.173 | 2.416 | 0.08 + 0.22 |
  | 10 | 1.586 | 1.543 | 2.756 | 0.04 + 0.22 |
  | 15 | 2.038 | 1.963 | 3.140 | 0.08 + 0.22 |

  - The 35 replays move 220 MB of fp32 state (read + write) per commit: ~1.1 ms at ~200 GB/s. Per-layer already runs
    there (keep 1: 185 GB/s).
  - The ideal batch (one launch per op over pre-stacked inputs) saves <= 0.1 ms on the GPU and ~0.22 ms of graph
    build. Stacking the inputs costs +1.1-1.2 ms. Every variant is bitwise equal to per-layer.
  - R-12's "launch-bound, ~0.07 ms fixed per layer" was an 8-layer-slice artifact. Its keep1 / keep15 = 0.64 is the
    state bytes, which don't depend on keep, not launch overhead. Its 0.072 ms per layer included the slice's
    fixed per-eval costs.
  - What would move the commit: fewer state bytes. One example is keeping the GDN states resident in a cheaper form,
    but fp32 states are part of exactness. Not a kernel lever.
  - vs prediction: per-layer 1.59 ms at keep 10, not 3.3 (R-12's slice scaling); KILL (15 %).

## R-11 ONE-SUBMIT (NewIdeas Mon ~22:25; Leap, pre-registered Tue 19:33 EET): the GPU's idle time per cycle
- **From data (CA diff run 3, 16 rows at 1K, 12 layers):** the engine's eager check builds its graph in 2.11 ms while
  the GPU waits (PLE staging 0.76 of it). The draft build is ~0.46 ms (R-10's E). The commit's build is not yet
  measured.
- **Constraint:** PLE staging reads the check's tokens on the CPU (n-gram rows from the 29.8 GB file-backed table).
  So the draft tokens must reach the CPU before the check graph is built. A GPU-side accept removes only the idle
  after the acceptance read.
- **Screen (scripts/r11_screen.py, ~1-2 min; CA's slice + the DFlash2 drafter):** keep fixed at 10, the text's own
  tokens in every check, the drafter fed from the check's logits. Four arms, 5 rounds x 8 cycles, interleaved:
  - S: the engine's order.
  - SB: the check built async every 4 layers + the commit replay submitted on its own (the cheap exact variants).
  - G1: GPU-side accept, one sync per cycle.
  - O: no sync, the bound.
  - S's idle is measured directly from the CPU spans with nothing queued: draft build, PLE + check build, accept +
    commit build. It is cross-checked by S - O.
- **48 layers:** after the acceptance read = draft build + 4 x commit build. After the draft read = PLE + 4 x (check
  build - PLE). The second overcounts by up to ~0.9 ms (non-layer build x 3).
- **Lines (QUEUE R-11):** lever if the 48-layer idle >= 8 ms per cycle; KILL < 3 ms.
- **Prediction:** 12-layer idle ~3.4 ms (draft build 0.5, check build 2.1, commit build ~0.8). 48 layers ~10 ms
  (7-11): LEVER 55 %, BETWEEN 40 %, KILL 5 %. Of that, ~3.7 is after the acceptance read and ~6.2 after the draft
  read. S - SB ~2 ms at 12 layers. O's PLE staging hides a sync (it reads the PLE history back): 40 %.

## R-09b build (Leap, pre-registered Tue 19:41 EET; step 1 said build it): MLX's own sums, loads prefetched
- **What MLX runs (matmul.cpp at v0.32.2):** gemv_wide declines at 16 rows (4 passes > 3). This shape takes the NAX
  split-K kernel: 64 x 64 tiles, 2 x 2 simdgroups, BK 256, K partitions of 4,096 (4,096 + 4,096 + 2,048), float
  partials, then gemm_splitk_accum (out = 0, += each partition in order, cast).
  - That is 18 threadgroups; half their simdgroups have no rows (M = 16 in a 64-row tile).
  - Each simdgroup walks its partition in 32-wide k steps, loading A and B straight from device memory before the
    fixed 16 x 32 x 16 NAX op. The loop is no-unroll with a volatile barrier per step, so every step waits on memory.
- **Why tiling cannot help:** the working simdgroups are capped at 11 column tiles x 3 partitions = 33 (36 now).
  The per-output order allows no K split, so the lever is latency hiding: the same op sequence, loads issued ahead.
- **Screen (scripts/r09c_screen.py, ~1 min):** custom kernels with MLX's nax.h inlined.
  - S1 = MLX's loop replicated. It must equal MLX bitwise, else BLOCKED.
  - S2 / S4 / S8 = loads 1 / 3 / 7 steps ahead.
  - F4 / F8 = one simdgroup walks all three partitions and adds them in MLX's order in registers: no accumulation
    kernel.
  - Bitwise: the 96 projections vs MLX's, and the X5 chain with each kernel vs X5 (all 96 reads). Timing: E, X5 and
    every arm interleaved, 96-read chains.
- **Lines (QUEUE R-09b):** PASS >= 15 ms saved per check vs E AND bitwise. KILL < 11. A compile failure (no Metal 4 /
  MetalPerformancePrimitives in custom kernels) or a non-bitwise replica is BLOCKED, not a verdict.
- **Prediction:** builds 60 %. S1 bitwise given a build 70 %. Prefetch brings the merged GEMM from ~80 us to ~40 us per
  call: ~-4 ms per 96 reads, ~13.7 saved vs E. BETWEEN 55 %, PASS 25 %, KILL 20 %.
- **Why it matters beyond HC:** if prefetching MLX's own NAX loop is bitwise and faster here, the same move applies to
  the 16-row dense quantized GEMMs (R-13: 29 ms over their floors per check), exactly.
- **Run 1 (Tue 20:16): BLOCKED by the harness, not the idea.**
  - Metal said my integer template arguments NT and N were undeclared in a kernel with a header (K, PS, NP, PF and T
    were fine).
  - The inlined nax.h itself compiled. MetalPerformancePrimitives builds in a custom kernel, and nothing in that
    header failed.
  - Fix: integer constants as `constant constexpr int` in the header, as MTPLX's kernels do; only T stays a template
    argument. R-13b had the same pattern and is fixed too. The re-run is queued in runner_r3.
- Note, family 1 (NAX qmm) for R-13c: these shapes run qmm_t_nax, MLX's non-split NAX kernel. The gemm_loop is the
  same, so this run's result prices R-13c's move directly.
- **Result, run 2 (Tue 20:40, 10 s; data/r09c_screen.txt): BETWEEN by 0.32 ms. Every arm is bitwise. S4 saves 14.68 ms
  per check vs E, 4.87 ms beyond X5.**

  | arm (96 reads) | ms | saved vs E | projections bitwise = MLX | chain bitwise = X5 |
  |---|---|---|---|---|
  | E (engine) | 24.51 | - | - | - |
  | X5 (MLX's NAX split-K) | 14.70 | 9.81 | - | (= E) |
  | S1 MLX's loop, 33 one-simdgroup tiles | 12.28 | 12.22 | 96 / 96 | yes |
  | S2 loads 1 step ahead | 10.22 | 14.28 | 96 / 96 | yes |
  | **S4 loads 3 steps ahead** | **9.82** | **14.68** | 96 / 96 | yes |
  | S8 loads 7 ahead (register pressure) | 15.09 | 9.41 | 96 / 96 | yes |
  | F4 / F8 (all partitions per simdgroup, 11 tiles) | 15.93 / 25.69 | 8.57 / -1.18 | 96 / 96 | yes |

  - MLX's NAX split-K arithmetic can be replicated in a custom kernel, bit for bit. Prefetching loads keeps the bits.
    So R-13c's premise holds.
  - The replica alone (S1, 33 one-simdgroup tiles instead of 18 x 4-simdgroup tiles with half of them idle) is
    already 2.4 ms faster than MLX. The prefetch adds 2.5 more.
  - Fusing the partitions (F) loses: 11 simdgroups are too few.
  - In the check, HC costs 31-34 ms (CA) vs 24.5 here, so the in-check gain may be ~x1.3.
  - vs prediction: builds (60 %) yes; S1 bitwise (70 %) yes; ~13.7 saved predicted, 14.68 measured; BETWEEN (55 %)
    yes.
  - Exact per-check savings for FOR_KAMEN decision 4: HC read X5 -9.81 -> S4 -14.68.

## R-15 DSK step 1 (NewIdeas Tue ~19:35; Leap, pre-registered Tue 19:44 EET): the draft's GPU time by kernel class
- **Why:** R-10 shows the drafter is GPU-bound (build 0.46 of 11.2 ms) and compiling its elementwise ops (the
  convolution included) saves only 0.7 ms. So the ~460 convolution kernels are probably cheap.
  - The bf16 drafter reads 1.21 GB per draft: ~115 GB/s at 10.5 ms. The 4-bit one reads 0.52 GB in 9.4 ms: ~55
    GB/s. That points at the skinny 16-row projections, as in R-13.
  - The drafter only proposes tokens, so any faster kernel is allowed (tokens within 1 %).
- **Screen (scripts/r15_screen.py, ~2 min):** CA's knockout method on one block-16 draft.
  - Classes: dense projections, convolution, attention core, norms, head, two-stage top-16, selector chain, all.
  - 10 real context rows per draft, so shapes stay fixed. 30 interleaved rounds. bf16 and 4-bit.
- **Lines (QUEUE R-15, for step 2):** PASS draft <= 6.5 ms at b16 with tokens within 1 %. KILL > 8.5 ms after the top
  two classes are swapped.
- **Prediction (4-bit, ~9.8 ms):** dense ~4.5 (top class, 70 %), head + top-16 ~2, attention ~0.8, convolution
  ~0.8, selector ~0.8, norms ~0.5. Removing the top two leaves ~3.5 ms, so step 2 has room if the dense projections
  can run near their bytes.
- **Result (Tue 20:28, ~1 min; data/r15_screen.txt): dense + head = 63 % of the draft. The drafter is GPU-bound.**

  | class (ms) | dense | head | attn | conv | select | norm | top-16 | rest (all out) | draft |
  |---|---|---|---|---|---|---|---|---|---|
  | 4-bit | 4.66 | 2.27 | 0.93 | 0.84 | 0.75 | 0.20 | 0.07 | 1.60 | 10.93 |
  | bf16 | 5.99 | 2.31 | 0.88 | 0.73 | 0.76 | 0.53 | 0.10 | 1.65 | 12.15 |

  - The draft's eval is 20-21 % main-thread CPU (2.2 of 10.5 ms). It is not encoding-bound, so R-10's reading holds:
    its kernels cost GPU time.
  - Why dense is slow, from MLX's dispatch at M = 16:
    - q (4096 x 2560), k / v, o (2560 x 4096) and the conv kernel projections run qmv_wide with 4 vector groups, so
      the weights are read 4 times.
    - The MLP runs the non-NAX split-K kernel: gate / up 7680 x 2560 at split 2, down 2560 x 7680 at split 6. These
      are R-13's two slow families, on the drafter.
  - Step 2 candidates (any numerics allowed):
    - NV16 qmv_wide for the qmv_wide shapes (R-13b's kernel).
    - The MLP off split-K: gate + up merged (N = 15,360 makes 480 tiles, split 1, NAX qmm). Down via NV16, or a NAX
      path.
    - The head at its floor (~1.3 ms).
    - At the floors, dense ~2 + head ~1.3 -> ~7.3 ms: still above 6.5. Passing also needs the small ops fused (attn +
      conv + select + norms + rest = 4.4 ms).
  - vs prediction: dense 4.66 vs ~4.5, head + top-16 2.34 vs ~2, attention 0.93 vs ~0.8, conv 0.84 vs ~0.8, selector
    0.75 vs ~0.8: right. Norms 0.20 vs ~0.5.

## R-13's exact half, by kernel family (Leap, Tue 20:05, no GPU; MLX 0.32.2 quantized.cpp at the tag + installed kernels)
- The shipped 16-row check runs MLX's natural 16-row calls. MTPLX's exact_verify does nothing unless
  MTPLX_EXACT_T0_GUARD=1. CA's ~25 ms in-check matches kern23's "MLX 16" column (25.5 ms), not the 12 + 4 chunked path.
  So "exact" = bitwise vs those calls.
- At M = 16 each 4-bit shape lands in one of three kernel families (split_k = max(1, 512 / tiles at bm = bn = 32)):

  | family | shapes (count) | MLX 16 | floor | excess | the exact move | row |
  |---|---|---|---|---|---|---|
  | NAX qmm, no split | qkv (36), q_proj (12), lm_head (1) | 6.96 | 3.94 | ~3 | R-09b's prefetched loads | R-13c |
  | non-NAX qmm_t_splitk | in_proj_z (36, split 2), o / out_proj (48, split 6) | 13.16 | 2.66 | ~10.5 | MLX's qmm_t_impl at BM 16 + fused partition sum | R-13d |
  | qmv_wide, 4 vector groups | shared gate / up / down, k / v, a / b, shared-expert gate | ~4.7 | ~0.5 | ~4.2 | one pass, vecs_per_tg 16 | R-13b |

  - qmm_t_splitk is MLX's standard qmm_t_impl at BM = BK = BN = 32, so half of each 32-row tile is wasted at M = 16.
    It writes bf16 partials, and a separate sum-reduction launch runs per call (84 per check).
  - In qmv_wide, k_lanes is always 8 for affine and vecs_per_tg never enters a vector's arithmetic (MTPLX's M4 route
    kernel relies on this).

## R-13b WIDE16 (Leap, pre-registered Tue 20:08 EET; NewIdeas accepted the lines 20:05): the small shapes in one pass
- **Screen (scripts/r13b_screen.py, ~2 min):** MLX's own dequantize + qmv_wide_impl, text from the installed
  quantized.h, at NV 4 / 8 / 16.
  - NV 4 must equal MLX: it checks the transcription.
  - kern23's small 4-bit shapes and counts, random weights, cold. Bitwise vs MLX's 16-row call on 8 instances per
    shape.
- **Lines:** PASS >= 2 ms saved per check AND bitwise on every shape. KILL < 1 ms or not bitwise.
- **Prediction:** builds and NV16 is bitwise: 80 %. MLX ~4.7 -> NV16 ~2.2 ms per check: ~-2.5. PASS 55 %, BETWEEN
  30 %, KILL 15 % (these shapes may be launch-bound rather than read-bound).
- **Result (Tue 20:32, ~1.5 min; data/r13b_screen.txt): KILL. The transcription is bitwise; one pass is 2x slower.**

  | shape (count) | MLX 16-row | NV4 | NV8 | NV16 |
  |---|---|---|---|---|
  | shared gate / up 640 x 2560 (96) | 1.85 | 1.94 | 1.77 | 4.45 |
  | shared down 2560 x 640 (48) | 1.51 | 1.51 | 1.34 | 3.61 |
  | k / v 512 x 2560 (24) | 0.61 | 0.57 | 0.54 | 1.10 |
  | in_proj_a / b 48 x 2560 (72) | 0.47 | 0.56 | 0.52 | 0.65 |
  | shared-expert gate 1 x 2560 (48) | 0.31 | 0.35 | 0.33 | 0.37 |
  | **per check (ms)** | **4.76** | **4.93** | **4.50** | **10.17** |

  - All NV are bitwise = MLX on every shape: vecs_per_tg never enters the arithmetic. MLX's quantized.h functions
    transcribed into a custom kernel (integer constants in the header) build and match MLX. R-13d can reuse that.
  - The premise was wrong. MLX's 4 vector groups read the same small weights at the same time and hit cache, so the 4
    reads are not 4x DRAM. One 16-vector pass has 4x fewer threadgroups and 4x the registers, and loses its latency
    hiding.
  - NV8 is the best exact option at -0.26 ms per check: below the KILL line.
  - vs prediction: bitwise (80 %) yes; ~-2.5 ms predicted, measured +5.4 (wrong: these shapes are latency-bound, as I
    feared in the prediction's note).
  - Consequences: R-15 step 2 should not use NV16 for the drafter's qmv_wide shapes. For R-13d, cutting the wasted
    rows (BM 16) may not help if the split-K kernel is latency-bound, so time it before building more on it.

## R-11 result + R-11b (Leap, Tue 20:10 EET)
- **R-11 result (Tue 20:07, 8 s of timed cycles; data/r11_screen.txt): LEVER by its line (11.49 ms), but most of it is
  not removable by one submission.**
  - Per-cycle wall at 12 layers + the drafter: S 52.92, SB 51.46, G1 52.48, O 51.44 ms. Paired: S - SB 1.36, S - G1
    0.24, S - O 1.63.
  - S's CPU spans: draft build 0.45, PLE + check build 4.68 (PLE staging 3.38), accept + commit build 0.61. That is
    5.75 ms of direct "idle", and 11.5 at 48 layers. The share after the acceptance read is 2.90; after the draft
    read it is 8.59 (PLE needs the tokens on the CPU), logged separately as NewIdeas asked.
  - But O (no sync at all) recovers only 1.63 of the 5.75. The no-sync bound is barely faster, so the CPU spans cannot
    hide behind GPU work: the evals themselves (draft 13.2, check 33.8 ms) must be mostly CPU. MLX's C++ encodes each
    kernel on the calling thread, and R-10's "Python build 0.46 ms" did not count that.
  - The cheap exact variants (SB: the check async every 4 layers + the commit submitted alone) give 1.36 ms per cycle
    at 12 layers, more than G1's GPU-side accept (0.24).
  - PLE staging is 3.4 ms per check here (0.76 in CA's run). It is CPU work on the critical path in every arm.
- **R-11b (scripts/r11b_screen.py, ~1.5 min; S and O):** main-thread CPU time inside each eval.
  - Line: a stage is encoding-bound if its eval CPU >= 80 % of its eval wall.
  - Prediction: check eval ~70 % CPU, draft eval ~80 %: one of the two encoding-bound, 60 %.
  - If so, the lever is fewer kernels per check (fusion), not overlap.
- **R-15 step 1 now also records the drafter's eval CPU time (same line).**
- **Added to R-11b (NewIdeas ~20:13):**
  - Dispatch counts: one untimed cycle's draft and check graphs, exported with mx.export_to_dot (view primitives not
    counted).
  - Pricing per stage: eval CPU per primitive; an indirect command buffer recorded once and replayed per cycle (~1.5 us
    per dispatch, my Sun 05:50 floor); encoding on 4 threads.
  - For MoonShots' R-24: per-row routing (top-10 expert ids for all 16 rows, 12 MoE layers) and the accepted length of
    20 real drafted checks. Drafts come from DFlash2 on dfl2d_code sample 0, acceptance from the sample's true tokens,
    and routing from the slice's verify forward. Saved to data/r24_routes.npz, with the union share from rejected rows.
- **R-11b result (Tue 20:30, ~1 min; data/r11b_screen.txt): not encoding-bound. Overlap is worth ~1.3 ms per
  12-layer cycle, not ~6.**
  - Draft eval 12.65 ms wall with 2.56 ms of main-thread CPU (20 %). Check eval 33.52 ms with 5.21 ms CPU (16 %).
    Both are GPU-bound, so R-11's "evals look CPU-bound" guess was wrong.
  - Dispatches: draft ~595 kernel-launching primitives (3.8-4.3 us of eval CPU each); 12-layer check ~1,384.
  - An indirect command buffer replayed per cycle would cost ~0.9 ms (draft) + ~2.1 ms (check, x4 at 48 layers) of
    CPU. Encoding on 4 threads would cost ~0.6 + ~1.3 ms.
  - But encoding already overlaps the GPU (16-20 % CPU during evals), so neither buys wall time: priced at ~0 for the
    cycle.
  - The open puzzle: S has 6.18 ms of CPU spans with nothing queued, yet O (no sync) wins only 1.29 ms (R-11: 1.63).
    So ~5 ms of GPU work happens during S's CPU spans, or a fully busy GPU runs slower (clock / power). This screen
    cannot tell which. The removable idle as measured is ~1.3-1.6 ms per 12-layer cycle.
  - R-24 dump (MoonShots): 20 real drafted checks (HumanEval_10, only 184 context tokens), 9.40 drafts accepted per
    check. Per MoE layer the expert union is 89.3 over all 16 rows vs 63.8 over the kept rows: 29 % of the union comes
    only from rejected rows. data/r24_routes.npz holds routes [20, 12, 16, 10] and accepted [20].

## R-32 STOPPRED (NewIdeas Tue ~20:35, ahead of R-21 and SB; Leap, pre-registered Tue 20:37 EET)
- **Why:** MoonShots' R-24 on my dump: rejected-only experts are 28.2 % of a check's per-layer union. A cut from
  layer 12 with a perfect stop row reaches 75 % of that (union -21 %, pairs -26 %). Kept rows stay exact (causal
  layers); a wrong early stop costs tokens only.
- **Screen (scripts/r32_screen.py, ~2 min + CPU):** CA's 12-layer slice on real drafted checks: DFlash2 on all 24
  held-out code samples, acceptance from the samples' true tokens, real commits.
  - The slice's head is the logit lens at layer 12: per row p(next draft), top-1 p, entropy.
  - Layer-0 baseline: the drafter's p of its chosen token and its selector margin.
  - Routing at layers 0-11 is the union proxy for layers 12-47.
  - Predictors: stop at the first draft whose score < tau. Scores: lens p alone; a logistic probe on lens + drafter
    features; the drafter p alone; a probe on drafter features. tau and the probes are fit on samples 0-11 (<= 2 %
    token loss) and scored on 12-23.
- **Lines (NewIdeas):** PASS: layer-12 whole-check union cut >= 10 % at <= 2 % token loss AND >= 5 points better than
  layer 0. KILL: < 5 % at <= 2 % loss, or no better than layer 0.
- **Prediction:** the oracle gives ~21 %. Layer-12 lens p at <= 2 % loss: ~8-12 %; layer-0 drafter p: ~4-7 %.
  BETWEEN 45 %, PASS 35 %, KILL 20 %. The lens at layer 12 (a quarter of the depth) may be weak; the probe should
  help.
- **Result (Tue 20:43, 40 s of GPU; data/r32_screen.txt): BETWEEN by the letter. Layer 12 adds nothing over the
  drafter's own confidence.**
  - 628 real drafted checks on 24 samples. 8.80 accepted drafts per check (9.80 tokens).
  - Oracle stop: whole-check union cut 23.7 %, pairs 29.1 %.

  | predictor (tau on samples 0-11, scored on 12-23) | token loss | whole-check union cut (x 36/48) | pairs cut |
  |---|---|---|---|
  | layer 12: lens p(draft) alone | 0.0 % | 0.0 % (no tau within 2 %) | 0.0 % |
  | layer 12: probe (lens + drafter features) | 2.0 % | 11.1 % | 14.2 % |
  | layer 0: drafter p alone | 1.6 % | 9.0 % | 12.1 % |
  | layer 0: probe (drafter features) | 2.1 % | 11.0 % | 14.1 % |

  - The logit lens at layer 12 (a quarter of the depth) carries no usable signal. Everything the layer-12 probe knows
    comes from the drafter's own p and margin.
  - Because that signal exists at layer 0, the cut belongs in the drafter: truncate the proposal at the predicted stop.
    The rows then leave all 48 layers, not 36: ~14.7 % union cut (~18.8 % of pairs) at ~2 % token loss, about 62 %
    of the oracle's 23.7 %.
  - Not tried: a probe on the raw layer-12 hidden states (2,560-d), which needs the dump extended. With lens scalars
    adding 0.1 points, I doubt it.
  - For MoonShots: data/r32_routes.npz (r24's layout + gates): routes and softmax gate weights [628, 12, 16, 10],
    accepted [628], sample [628].
  - vs prediction: lens ~8-12 % predicted, 0 measured (wrong); drafter ~4-7 % predicted, 9-11 % measured (wrong: the
    drafter's confidence is better than I thought).
  - R-35 TRUNC2 (NewIdeas ~20:47: a real-loop A/B of probe-based truncation): dead as specified, because DFL2E
    covered it and does better.
    - DFL2E's running-product rule (data/trunc_l5ft16_code.txt): conf >= 0.3 cuts 26.7 % of rows at -1.8 % tokens,
      vs this probe's 18.8 % at -2.0 %. The probe's first-position-below-tau rule loses the cumulative signal.
    - What stays open is DFL2E's engine test of that rule (variable rows per check above 9), in BigMoE's or Fit12's
      loop.

## R-11c SB (NewIdeas Tue ~20:31; Leap, pre-registered Tue 20:34 EET): the cheap exact variants, proved and timed
- **Screen (scripts/r11c_screen.py, ~2 min):** S vs SB (the check async every 4 layers + the commit replay alone) on
  two identically prefilled lanes per context (1K, 8K), same inputs, alternating 8-cycle blocks. Bitwise: drafts and
  tokens of every cycle, plus the final states.
- **Lines:** PASS: bitwise AND S - SB >= 1.0 ms per 12-layer cycle at both contexts. KILL: not bitwise, or < 0.3 ms
  at both.
- **Prediction:** bitwise 95 %. S - SB ~1.4 at 1K (R-11: 1.36), ~1.0 at 8K: PASS 55 %, BETWEEN 35 %.
- **Result (Tue 20:59, re-run after the swap-guard stop at 20:54; data/r11c_screen.txt): PASS. Bitwise, 4.0-4.2 ms
  per 12-layer cycle.**

  | context | S (ms per 12-layer cycle) | SB | S - SB (paired median, 5 x 8 cycles) | bitwise (43 cycles + final states) |
  |---|---|---|---|---|
  | 1K | 51.86 | 47.82 | **4.04** | drafts, tokens, states: yes |
  | 8K | 53.96 | 50.09 | **4.21** | yes |

  - 3x R-11's 1.36. R-11 ran four arms on one cache in rotating blocks; here each arm has its own identically prefilled
    lane, so this is the cleaner number.
  - The gain is larger than the check's Python build (~1.4 ms per 12 layers), so async every 4 layers also helps the
    GPU side (it starts earlier and works on smaller buffers), not only the CPU overlap. CA's earlier A vs async/4
    gave 2.6 (1K) / 1.3 (8K).
  - At 48 layers: ~16 ms per check if all of it scales with layers. The commit-alone part does not scale, so call it
    ~11-16. That needs the 48-layer engine to confirm.
  - Exact and trivial: async_eval every 4 layers in the verify forward + submit the commit's state arrays right after
    the acceptance read.
  - vs prediction: bitwise (95 %) yes; 1.4 / 1.0 predicted, 4.0 / 4.2 measured; PASS (55 %) yes.

## R-13c NAXP (R-13's family 1; NewIdeas' lines; Leap, pre-registered Tue 20:44 EET)
- **What MLX runs:** qkv, q_proj and lm_head at M = 16 take qmm_t_nax (no split; bm 32, bn 64, bk 64, wm 2, wn 2).
  - Per 64-wide k step, all 128 threads dequantize a 64 x 64 weight tile into threadgroup memory between two
    barriers. Each simdgroup then runs two 32-wide sub-steps of the NAX op.
  - The second simdgroup row (rows 16-31) has no rows at M = 16 but still runs.
- **Screen (scripts/r13c_screen.py, ~1.5 min):** MLX's nax.h + quantized_nax.h's loader inlined, integer constants in
  the header.
  - C1 = the replica. It must equal MLX, else BLOCKED.
  - C2 = BM 16, BN 64. C3 = BM 16, BN 32 (2x the threadgroups). C4 / C5 = C2 / C3 with double-buffered weight tiles
    (the next tile dequantizes while this one computes; one barrier per step).
  - kern23's shapes and counts (random 4-bit weights, cold). Bitwise vs MLX on 4 instances per shape.
- **Lines (NewIdeas):** PASS >= 1.5 ms saved per check AND bitwise on every shape. KILL < 0.5 ms or not bitwise.
- **Prediction:** C1 bitwise 80 %. Today's family total is ~7.0 ms (kern23: qkv 3.31 + q 1.44 + lm_head 2.21; floor
  3.94). Best arm ~5.2: -1.8. PASS 50 %, BETWEEN 35 %, KILL 15 %.
- **Result (Tue 20:56, ~1 min; data/r13c_screen.txt): KILL. Every arm is bitwise, none is faster.**

  | shape (count) | MLX | C1 replica | C2 BM16 | C3 BM16 BN32 | C4 C2 + double buffer | C5 C3 + double buffer |
  |---|---|---|---|---|---|---|
  | qkv (36) | 3.38 | 3.47 | 3.40 | 3.81 | 3.79 | 4.07 |
  | q_proj (12) | 1.52 | 1.53 | 1.48 | 1.49 | 1.63 | 1.64 |
  | lm_head (1) | 2.31 | 2.27 | 2.18 | 2.11 | 2.55 | 2.49 |
  | **per check (ms)** | **7.20** | **7.26** | **7.06** | **7.41** | **7.97** | **8.19** |

  - MLX's quantized NAX GEMM replicates bit for bit at any tiling and with double-buffered weight tiles.
  - Unlike R-09b's bf16 kernel, which loads straight from device memory every step and gains from prefetch, this one
    dequantizes each tile into threadgroup memory. Overlapping the next tile's loads buys nothing (it costs 0.8-1
    ms), so the cost is the dequantize / NAX work, not load latency.
  - The best exact arm (C2) saves 0.14 ms per check. vs prediction: -1.8 predicted (wrong); bitwise (80 %) yes.
  - This family's 3.3 ms over its floor is not reachable with MLX's arithmetic order. Only a different dequant /
    MMA scheme (a reference change, decision 4's non-identical side) gets it.

## R-13d SPK16 (R-13's family 2, the biggest exact piece; NewIdeas' lines; Leap, pre-registered Tue 20:47 EET)
- **What MLX runs:** in_proj_z (6144 x 2560, split 2) and o_proj / out_proj (2560 x 6144, split 6) at M = 16 take
  affine_qmm_t_splitk.
  - That is qmm_t_impl at BM = BK = BN = 32 with 2 x 2 simdgroups. Each split partition writes a bf16 partial; then a
    strided sum over axis 0 runs as a second launch.
  - At M = 16 half of each 32-row tile is padding.
  - The per-output order is steel's BlockMMA: 8-wide simdgroup-MMA k-steps through each 32-wide block, independent of
    the tile layout.
- **Screen (scripts/r13d_screen.py, ~1.5 min):** MLX's steel gemm.h + quantized.h inlined. Each kernel is followed by
  mx.sum(partials, axis=0), MLX's own reduction.
  - D1 = MLX's qmm_t_impl at BM 32 (the replica: must equal MLX, else BLOCKED).
  - D2 = MLX's qmm_t_impl at BM 16.
  - D3 = D2's order with double-buffered X / W tiles.
  - kern23's shapes and counts (random 4-bit, cold). Bitwise vs MLX's 16-row call on 4 instances per shape.
- **Lines (NewIdeas):** PASS >= 4 ms saved per check AND bitwise on both shapes. KILL < 2 ms or not bitwise.
- **Prediction:** D1 bitwise (incl. mx.sum = MLX's internal reduce) 65 %. MLX 13.2 ms per check (kern23), floor 2.7.
  D2 ~10.5, D3 ~8.5: ~-4.7. PASS 40 %, BETWEEN 30 %, KILL 30 % (R-13b showed latency-bound kernels can lose when
  the work per threadgroup changes).
- **Result (Tue 20:57, ~1 min; data/r13d_screen.txt): PASS. D2 saves 4.52 ms per check, bitwise.**

  | shape (split, count) | MLX | D1 replica BM32 | D2 BM16 | D3 BM16 + double buffer |
  |---|---|---|---|---|
  | in_proj_z 6144 x 2560 (2, 36) | 5.67 | 5.78 | **3.73** | 3.82 |
  | o_proj + out_proj 2560 x 6144 (6, 48) | 7.48 | 7.63 | **4.90** | 4.97 |
  | **per check (ms)** | **13.15** | **13.41** | **8.64** | **8.79** |

  - D2 is MLX's own qmm_t_impl at BM 16, followed by mx.sum(partials, axis=0), MLX's own reduction. It is bitwise =
    MLX's 16-row call on both shapes; D1 confirms that mx.sum reproduces MLX's internal strided reduce.
  - The win is the padding: at M = 16 MLX multiplies 32-row tiles. Double buffering adds nothing (as in R-13c).
  - vs prediction: D1 bitwise (65 %) yes; D2 ~10.5 predicted, 8.64 measured; PASS (40 %) yes.
  - Exact per-check savings so far (standalone kernels): HC read S4 -14.68, R-13d -4.52 (R-13b -0.26 and R-13c -0.14
    are below their lines).
  - Next, the confirmation: both swaps inside CA's real check (R-36), bitwise logits + states and the paired time.

## R-36 INCHECK (NewIdeas GO Tue ~20:59; Leap, pre-registered Tue 21:01 EET): S4 + D2 inside the real check
- **Screen (scripts/r36_screen.py with scripts/leapk.py, ~2.5 min):** CA's 12-layer slice running the engine's eager
  16-row check (acceptance read, lazy commit, keep 10). Four lanes, each identically prefilled, the same tokens,
  alternating 8-check blocks, 5 rounds, at 1K and 8K:
  - base: the engine as shipped.
  - S4: GatedResidual's 16-row decode_verify read -> R-09b's S4.
  - D2: mx.quantized_matmul calls MLX would serve with split-K at M = 16 -> R-13d's D2.
  - both: S4 + D2.
  - Bitwise: every check's logits and all cache states vs base.
- **Lines (NewIdeas):** PASS = every bit equal AND 'both' saves >= 80 % of the standalone sum (>= 15.4 ms per 48
  layers = >= 3.84 per 12-layer check) at 1K and 8K. KILL = any bit differs. Also reported: each swap's in-check
  factor (4 x its 12-layer saving / standalone), including S4's x1.3 question.
- **Prediction:** bitwise 85 % (S4's no-pending norm path and D2 inside MTPLX's own calls are new). S4 in-check factor
  ~1.1 (12-layer saving ~4.0 ms), D2 ~0.9 (~1.0 ms), both ~5.0 per 12 layers -> ~20 ms per 48: PASS 60 %.
- **Result (Tue 22:12, ~40 s of GPU; data/r36_screen.txt): PASS. Every bit equal; -18.2 to -18.4 ms per 48-layer
  check in-check.**

  | context | base (ms / 12-layer check) | S4 | D2 | both | saved S4 / D2 / both | both x4 (48 layers) |
  |---|---|---|---|---|---|---|
  | 1K | 36.09 | 32.92 | 34.42 | 31.51 | 3.56 / 1.62 / 4.60 | **18.42** |
  | 8K | 38.30 | 35.02 | 37.00 | 33.79 | 3.33 / 1.47 / 4.55 | **18.19** |

  - Bitwise vs the shipped engine: the logits of all 43 checks and every cache state, for S4, D2 and both, at both
    contexts. The swaps engaged on 4,128 HC reads and 3,612 split-K matmuls.
  - In-check factors (4 x the 12-layer saving / standalone):
    - S4 0.97 / 0.91: it lands as measured standalone, not x1.3.
    - D2 1.43 / 1.30: it does better in the check than standalone.
  - The check falls 12.7 % (1K) / 11.8 % (8K) with identical bits. This is not a reference change. SB (R-11c,
    -4.0 / -4.2 ms per 12-layer cycle) is separate and should stack; a combined run would confirm it.
  - vs prediction: bitwise (85 %) yes; S4 ~1.1 predicted, 0.9-1.0 measured; D2 ~0.9 predicted, 1.3-1.4 measured;
    ~20 per 48 predicted, 18.2-18.4 measured; PASS (60 %) yes.
  - For the engine owner: scripts/leapk.py (hc_read, splitk_d2, routes_splitk) and r36_screen.py's two patch points
    (GatedResidual.__call__ at 16 decode_verify rows with a block inject; mx.quantized_matmul where MLX takes split-K
    at M = 16).

## R-36b = R-39 STACK (NewIdeas GO ~22:15; Leap, pre-registered Tue 22:13 EET): do the three bit-identical levers stack?
- **Screen (scripts/r36b_screen.py, ~3 min):** R-36's four-lane design with arms base, SB (check async every 4 layers
  + commit replay alone), S4D2 (R-36's both) and all3 (SB + S4 + D2). Bitwise: logits of every check + all cache states
  vs base, at 1K and 8K.
- **Lines (NewIdeas, R-39):** PASS = every bit equal AND all3's paired saving >= 6.5 ms per 12-layer check at 1K and
  8K. KILL = any bit differs. My earlier 80 %-stacking line is still reported.
- **Prediction:** bitwise 90 %. SB here (checks only, no drafter) ~3 ms; S4D2 ~4.6; all3 ~7 ms per 12-layer check
  (~28 per 48): stacking ratio ~0.9, PASS 70 %.
- **Result (Tue 22:25, ~3 min; data/r36b_screen.txt): PASS. Every bit equal; all three stack (ratio 0.97-1.0):
  -26.5 / -30.9 ms per 48-layer check.**

  | context | base | SB | S4D2 | all3 | saved SB / S4D2 / all3 (sum of parts) | all3 x4 |
  |---|---|---|---|---|---|---|
  | 1K | 36.46 | 34.37 | 31.55 | **29.98** | 2.09 / 4.71 / **6.63** (6.80) | **26.5** |
  | 8K | 39.65 | 36.45 | 33.64 | **31.76** | 3.20 / 4.49 / **7.73** (7.69) | **30.9** |

  - Bitwise vs the shipped engine: logits of all 43 checks + every cache state, for SB, S4D2 and all3, at 1K and 8K.
  - The check is 18.2 % (1K) / 19.5 % (8K) faster with identical bits.
  - SB here is checks-only (2.1-3.2 ms). In R-11c's full draft + check cycle it gave 4.0-4.2, because the commit
    replay then overlaps the draft build. In the engine's real loop the total may be nearer -30 to -34 ms.
  - vs prediction: bitwise (90 %) yes; all3 ~7 per 12 layers predicted, 6.6 / 7.7 measured; PASS (70 %) yes.
  - Hand-off note for BigMoE: HANDOFF_EXACT16.md.

## R-15 step 2 (NewIdeas Tue ~22:27; Leap, pre-registered Tue 22:28 EET): the drafter's dense class on faster kernels
- **Screen (scripts/r15b_screen.py, ~4 min; R-10's harness, 4-bit eager drafter):**
  - E4: as is.
  - D2: its split-K MLP calls via R-13d's D2.
  - GU: gate | up merged into one 15,360-wide matmul (480 tiles, so MLX's NAX qmm, no split).
  - GUD2: GU + down via D2.
  - Draft ms on one sample (b16, b32) and tokens per check over the 24 held-out replays.
- **Lines (QUEUE R-15):** PASS <= 6.5 ms at b16 with tokens within 1 %. KILL > 8.5 ms. The head (2.27) is untouched,
  so this tests the dense half only.
- **Prediction:** E4 ~9.8; D2 ~9.1; GUD2 ~8.6 ms (dense 4.66 -> ~3.5). KILL 55 %, BETWEEN 45 %. Tokens: D2 exact,
  GU within 1 % (99 %).
- **Result (Tue 22:35, ~4 min; data/r15b_screen.txt): BETWEEN. GUD2 7.93 ms at b16 (-20 %), tokens x0.9955.**

  | arm | b16 ms | tokens x | b32 ms | tokens x |
  |---|---|---|---|---|
  | E4 | 9.93 | 1.0000 | 11.40 | 1.0000 |
  | D2 (bit-identical) | 8.47 | 1.0000 | 11.40 | 1.0000 |
  | GU (gate / up merged, NAX) | 8.29 | 0.9955 | 10.06 | 0.9984 |
  | **GUD2** | **7.93** | 0.9955 | 10.31 | 0.9984 |

  - D2 alone is exact for the drafter too: -1.46 ms per b16 draft. At b32 nothing changes, because the MLP calls have
    32 rows and leave split-K.
  - Merging gate | up (N 15,360, NAX qmm) moves the drafter's numerics by rounding only: tokens -0.45 % at b16.
  - What is left at b16 (~7.9 ms): the head (~2.3), attention / conv / select / norms (~2.7), the rest (~1.6), and
    ~1.3 ms of dense. Reaching 6.5 needs the head or the small ops fused, a different kind of change.
  - vs prediction: E4 9.8 predicted (9.93); GUD2 ~8.6 predicted, 7.93 measured; BETWEEN (45 %) yes.

## R-40 REF2 (Kamen approved decision 4, Tue 22:40; NewIdeas row R-40): plan (Leap, Tue 23:13, no GPU yet)
- **Goal:** one patch module that makes the 45 GB engine's exact reference batch-invariant and faster:
  - 4-bit dense via qmm16c (L1-23: -29 ms per 16-row check vs chunked; one fixed 16-row tile, decode pads);
  - routed experts via experts_nax (qmmg25, L1-25);
  - HC reads via S4; bf16 matmuls via MLX's 16-row calls (padding-independent per L1-23);
  - SB scheduling;
  - a load-time self-test.
  - PASS (NewIdeas): bitwise batch-invariant AND <= today - 40 ms per 48-layer check.
- **The real question is not the matmuls.** Today's 1-row decode runs other kernels than the 16-row verify for
  attention (sdpa_vector vs the verify kernels / QSA selection), the GDN recurrence (decode step vs the T-step
  kernel; the state is fp32, so the round trip itself is exact) and the router. Batch invariance needs either:
  - (a) decode run as a 16-row verify with one real row (rows 1-15 padding, masked from the GDN / KV / conv state;
    commit keep 1), so every token goes through the same kernels; or
  - (b) proof that each of those ops gives the same bits per row at M = 1 and inside M = 16.
- **First screen (INV0, Wed 08:02):** in CA's slice, a 16-row verify of 16 real tokens vs 16 AR steps of 1 token
  (the engine's decode path), today's kernels and with the REF2 matmul patches. Per-layer row outputs are compared
  bitwise to map where the rows first diverge (layer, op class). That decides (a) vs (b) before the module is built.
- **INV0 (pre-registered Tue 23:16; scripts/inv0_screen.py, ~3 min):** AR16 (16 one-token steps through the engine's
  verify path, commit keep 1) vs V16 (one 16-row verify), on identical 1K caches in CA's slice.
  - Records in forward order: PLE, attention / GDN block, MoE and layer outputs. Bitwise per row; first divergence per
    row.
  - Run twice: today's kernels, and REF2's padding patches: every quantized matmul / bf16 Linear / HC read with < 16
    rows padded to 16; 4-bit g64 N >= 1024 via qmm16c, the rest via MLX's 16-row calls; HC via S4; attention / GDN /
    experts as shipped.
  - Prediction: today, rows diverge at layer 0's first matmul. REF2: logits equal 30 % (attention / GDN already
    position-invariant through the verify path at M = 1); otherwise the first divergence is in attention or GDN (70
    %), which means route (a).
  - **Result (Wed 15:47, 10 s; data/inv0_screen.txt): only attention breaks batch invariance under REF2.**
    - today: all 16 rows diverge at record 0 (layer 0's GDN block: 1-row vs 16-row matmul kernels). Logits equal
      0 / 16.
    - REF2: records 0-9 equal for every row, i.e. layers 0-2 complete: GDN blocks incl. the recurrence, the PLE
      layer, MoE (MLX's gather experts), HC reads (S4) and the layer outputs. The first divergence for all 16 rows is
      record 10, layer 3's attention (the first QSA layer). Logits equal 0 / 16 (downstream of it).
    - So the matmul / HC / expert side of REF2 is batch-invariant as built. What is left is the attention path: at
      M = 1 vs inside M = 16 it runs other code / kernels (QSA block selection + SDPA).
    - Next (KERNELS, Kamen's 15:47 role): find which part of QSA differs (the indexer's block selection, which may
      depend on the total key count, vs the SDPA kernel: vector for 1 query vs full for 16), then either a
      batch-invariant attention kernel (per-row selection + one key blocking from position 0) or decode as a 16-row
      padded step. Only an attention fix is needed, not route (a) in general.
    - vs prediction: today at layer 0 (yes, the GDN block's matmuls); REF2 "first divergence in attention or GDN"
      (70 %): attention.
- **INV1 (pre-registered Wed 15:50; scripts/inv1_screen.py, ~2 min):** INV0's REF2 run plus one patch. An attention
  call with S < 16 rows is padded to 16 (copies of its last row) through the same 16-row verify path, the QSA cache
  is trimmed back by the padding (QSACache.trim: KV + pooled blocks), and the real rows are kept.
  - Pass for the route: every record and all 16 logits rows bit-identical AR vs verify.
  - Prediction: 55 % all equal. Risk: the indexer's pooled current block includes the padding rows (future positions)
    in AR but real rows in the verify; if the selection reads it, rows still differ.
  - **Result (Wed 16:07, 11 s; data/inv1_screen.txt): REF2 IS BATCH-INVARIANT.** Logits of all 16 rows and every
    one of the 37 records per forward (GDN, PLE, attention, MoE, layer outputs) are bit-identical between 16 one-token
    AR steps and the 16-row verify, on the 12-layer slice at 1K.
    - The REF2 patches that do it: every quantized / bf16 matmul and HC read with < 16 rows padded to 16 (qmm16c for
      4-bit g64 N >= 1024; MLX's 16-row call otherwise; S4 for the HC reads), plus attention with < 16 rows padded to
      16 through the verify path and the QSA cache trimmed back. GDN and MLX's gather experts are invariant as shipped.
    - So spec decoding against REF2's own AR output is exact by construction, as far as this slice shows. The indexer
      risk did not bite.
    - vs prediction: 55 % all equal: yes.
    - Next: the module (scripts/ref2.py: these patches + experts_nax + SB + a load-time self-test), then R-40's timing
      screen vs today (-40 ms per 48 layers = -10 per 12-layer check), then EvalSuite's hand-off.
- **R-40 screen (pre-registered Wed 16:10; scripts/ref2.py + scripts/r40_screen.py, ~4 min):** the module on CA's
  slice, loaded with MTPLX_FUSED_GATE_UP=1 (MTPLX's sanitize-time gate | up fusion, no copy) so the experts run
  experts_nax.
  - Part 1: batch invariance, AR16 vs V16 with REF2 on (records + logits, all 16 rows) at 1K.
  - Part 2: speed per 12-layer check, today's kernels (eager) vs REF2 + SB, paired lanes at 1K and 8K. Both lanes
    load the fused gate | up option, so "today" here is the shipped kernels with MTPLX's fused-MoE option.
  - Plus ref2.self_test().
  - **Lines (NewIdeas):** PASS = bitwise batch-invariant AND >= 10 ms saved per 12-layer check (40 per 48) at 1K and
    8K. KILL = not invariant.
  - Prediction: invariant 85 % (experts_nax is new in the chain; L1-25 showed it row-invariant). Saving per 12 layers:
    qmm16c dense ~3, experts ~5.5, S4 ~3.5, SB ~2-3 -> ~14 (~56 per 48): PASS 65 %.
  - **Result (Wed 16:28, ~25 s of GPU; data/r40_screen.txt): PASS. REF2 is batch-invariant and -51 / -60 ms per
    48-layer check.**
    - self-test: qmm16c, MLX 16-row, bf16 Linear all invariant. All 12 MoE blocks loaded fused (experts_nax engaged).
    - Part 1 (1K, with experts_nax in the chain): logits of all 16 rows and every record bit-identical between 16
      one-token AR steps and the 16-row verify.

    | context | today (ms / 12-layer check) | REF2 + SB | saved (paired) | x4 (48 layers) |
    |---|---|---|---|---|
    | 1K | 39.60 | 26.78 | 12.82 | **-51.3** |
    | 8K | 41.88 | 27.36 | 14.91 | **-59.6** |

    - The check is 32 % (1K) / 36 % (8K) faster.
    - vs prediction: invariant (85 %) yes; ~14 per 12 layers predicted, 12.8 / 14.9 measured; PASS (65 %) yes.
    - Next: EvalSuite runs the same-scores verdict (math / code / instructions) with ref2.generate (AR = each token a
      1-row verify + commit keep 1). Hand-off: REF2_HANDOFF.md.
- **EvalSuite's catch (Wed ~16:30): the full 125B only runs streamed (bigmoe.loader.load_streamed), whose expert
  pools hold separate gate / up / down. REF2's experts patch needed MTPLX's fused gate | up module.**
  - Fix in ref2.py (no BigMoE change): experts_split = qmmg on separate gate and up calls (= the fused halves, bit for
    bit), patched into mlx_lm's SwitchGLU.__call__, which BigMoE's streamed subclass calls with slot ids. Also patched:
    bigmoe.stream._switch_sorted. Self-test gains "experts split == fused" and "experts rows alone". BIGMOE_TIER is
    not supported.
  - **Streamed check (pre-registered Wed 16:35; scripts/ref2_stream_check.py, ~5-8 min):** the full 125B in
    EvalSuite's exact REF harness (evalsuite.env + evalsuite.model.load('ref', C = 192): load_streamed + its
    gate-rank combine patch; switched at 16:4x on EvalSuite's note, before any run). ref2 now exposes per-path call
    counters (ref2.counts()). Same 1K prefill on two caches; 16 one-token AR steps (ref2.ar_step) vs one 16-row verify.
    Every record of all 48 layers and every logits row, bitwise. Pass = self-test + all equal; then EvalSuite runs.
  - Prediction: PASS 75 %. Risks: the streamed MoE's routing / slot path or a layer-specific QSA difference beyond
    layer 11.
  - **Result (Wed 16:52, 30 s; data/ref2_stream_check.txt): PASS. REF2 is batch-invariant on the FULL streamed 125B.**
    - Harness: EvalSuite's REF (C = 192, SwitchGLUStreamed, 29.4 GB active). self-test all True (incl. experts split
      == fused, experts rows alone).
    - 16 one-token AR steps vs one 16-row verify: all 145 records per forward (48 layers) and all 16 logits rows
      bit-identical.
    - Counters per AR token: experts 48 (every layer on the REF2 kernel), attention 12 (padded QSA), hc_read 96,
      hc_eager16 1, qmm16c 181, mlx16 300.
    - EvalSuite has the green light for the same-scores verdict (ref2.generate).
    - vs prediction: PASS (75 %) yes.
- **R-40 step 2, EvalSuite's same-scores verdict (Wed ~21:10; ~/EvalSuite/docs/RESULT_ref2_2026-10-28.md, EvalSuite
  commit 84ebfba): PASS.** REF2 scores the same as today's stock reference (ref.r2, re-decoded the same day), 210 core
  rows each, on Leap's 86ca915 frozen at ~/EvalSuite/snap/leap-86ca915 (all 13 chunks had the same file hashes).

  | set | REF2 - today | p |
  |---|---|---|
  | math | +8.0 | 0.12 |
  | IFEval | -3.0 | 0.38 |
  | HumanEval+ | 0.0 | 1.00 |

  - All three are within the floor's rounding noise.
  - REF2 ran on every AR step of all 210 rows, at exactly 48 experts / 12 attention / 96 hc_read / 1 hc_eager16
    calls per step.
  - Flags (information only): 4 rows looped, 3 of them also loop in the stock arms. IFEval agreement is 95 %,
    against the floor's 97 %.
  - REF2's plain AR runs at 11.65 tok/s vs 15.50 for today's AR (x0.75), as the hand-off expected. Spec checks are
    the fast path. Peak 35.6 GB (+0.6 GB).
  - Decision 4's condition is met: REF2 may replace the 45 GB line's exact reference. NewIdeas has the verdict.
- **R-40 STEP 3 (NewIdeas Wed 21:2x, QUEUE row R-40 'STEP 3'; pre-registered Thu 2026-10-29 01:55 EET;
  scripts/r40s3_screen.py, ~6 min): REF2 + SB vs today's shipped exact check, all 48 layers, streamed, with live
  drafting.**
  - Harness: EvalSuite's REF harness (evalsuite.model.load('ref'): load_streamed + its install), at C = 176 instead of
    the line's 192. That keeps the process under the 36 GB rule with the drafter loaded; numerics don't depend on C.
  - Drafter: DFlash2 L1-20 (bf16, block 16, the official 1-step loop with R-06's exact two-stage head), run live on
    taps from the verify forward (layers 5 / 15 / 25 / 35 / 45). It always runs on stock kernels.
  - Loop: greedy, keep = accepted + 1, commit_verified_window.
  - Arm today: stock kernels, the engine's exact verify scopes, the commit left lazy.
  - Arm REF2 + SB: ref2 on for prefill and checks. SB = the mid-loop async every 4 layers (R-37's
    BIGMOE_VERIFY_MIDLOOP, patched per arm) + the commit replay submitted alone.
  - Prompts: 2 code (HumanEval/68, /91) + 2 math (math500 prealgebra/1640, algebra/694), interleaved, up to 256 new
    tokens. ABBA / BAAB per prompt. No new prompt after 420 s; the verdict then uses the prompts done.
  - 16-row checks with > 92 ids take BigMoE's _sync_call path: a per-layer sync on the ids, ensure, scratch gather.
  - Exactness:
    - every REF2 run's tokens equal EvalSuite's REF2 AR rows (results/ref2, Leap 86ca915 = today's kernel files);
    - the REF2 runs agree with each other;
    - on HumanEval/68 and prealgebra/1640, every kept check row's logits are bit-identical to in-process ref2.ar_step
      logits for the first 48 tokens.
  - Speed: the median cycle (draft + check + commit) per run. Saving = today - REF2, averaged per prompt, then the
    median over prompts. Also reported: prefill s, kept per check, tok/s per domain, REF2 counters per check.
  - **Lines (NewIdeas):** PASS = exact AND >= 40 ms saved per 48-layer check. KILL = < 20 ms saved. BETWEEN otherwise.
  - Prediction: exact 90 %. Saving ~30 ms (range 20-45):
    - SB's mid-loop async can hide little behind the per-layer id syncs of the _sync_call path;
    - the experts run experts_split on the scratch pool (two qmmg) instead of the fused experts_nax;
    - the slice's kernel-only share was ~8.8 ms per 12 layers (~35 per 48).
    - PASS 30 %, BETWEEN 55 %, KILL 15 %.
  - **Result, first attempt (Thu 01:57-02:06; logs/job.out; stopped by run_locked's 9-min limit after 1.3 prompts, no
    verdict file):**
    - Speed: a 16-row check takes ~3.9 s in this engine (today median cycle 3910 / 3906 ms, REF2 + SB 3830 / 3860 on
      HumanEval/68). Drafting is 25 ms. 16 rows x 10 ids exceed MAXK 40, so every check takes BigMoE's _sync_call:
      per layer an id sync, synchronous SSD loads (ensure), a scratch gather and clear_cache.
    - tok/s with live DFlash2: code 3.8 (kept 15.41 per check: /68's answer repeats the prompt's code); math 1.2 (kept
      5.24). For comparison, REF2's plain AR runs at 11.65 tok/s (EvalSuite).
    - REF2 saved 80 / 46 ms per check on /68, ~1-2 % of the check and inside its noise.
    - **Not exact:**
      - 0 / 48 logits rows bit-identical to in-process ref2.ar_step, on both prompts;
      - tokens equal EvalSuite's REF2 AR rows 256 / 256 on /68 but only 199 / 256 on prealgebra/1640.
      - This contradicts the 16:52 stream check (C 192, P 1024, no SB, no drafter).
    - The runner's rc-143 retry repeated the same run; Leap stopped it by PID at 02:09 (it held the shared lock for
      nothing).
- **R-40 STEP 3x (pre-registered Thu 02:12; scripts/r40s3x_screen.py, ~3 min): localize the exactness break, and price
  warm checks.**
  - Setup: as STEP 3, HumanEval/68 at C = 176. AR = 16 ref2.ar_step on the REF2 row's first 16 tokens.
  - Part A, variants (each with its own prefill, then one 16-row verify of the same tokens, rows vs AR, bitwise):
    - V0 = REF2 alone (= the stream check here); V0b = V0 again (determinism);
    - V1 = + SB mid-loop; V2 = + taps hook; V3 = drafter loaded + one draft before the verify; V4 = all of STEP 3.
    - V5 (added Thu 15:44, before any run): V0 after a stock 16-row check + commit on another cache. STEP 3 ran its
      today arm before the compared REF2 run; the 16:52 stream check kept REF2 on throughout.
  - Part B: warm 16-row checks (sliding by one token, commit keep 1, so each check's experts are mostly resident).
    Today vs REF2 + SB in alternating blocks of 4, 5 rounds.
    - Line (step 3's, on warm checks): PASS >= 40 ms saved, KILL < 20 ms.
    - Prediction: warm check ~0.8-1.5 s (syncs + gathers); saving ~35 ms: PASS 35 %, BETWEEN 45 %, KILL 20 %.
  - Prediction: V0 == AR 60 % (then V1, SB's mid-loop async, is the main suspect, 50 %); prefills deterministic 90 %.
    If V0-V4 all pass, V5 (order / first-call state) is the suspect.
  - **Result (Thu 16:12-16:17, rc 0; data/r40s3x_screen.txt):**
    - **Part A: REF2 is bit-exact in every live-loop variant; the break is a state leak from a stock check.**
      - V0, V0b, V1 (SB mid-loop), V2 (taps), V3 (drafter + a draft), V4 (all of STEP 3): 16 / 16 verify rows
        bit-identical to ar_step.
      - Prefills deterministic (last logits equal); verify deterministic (V0 == V0b).
      - V5: after one stock 16-row check + commit keep 5 on ANOTHER cache, a fresh prefill of the same prompt gives
        other bits (last logits differ). All 16 verify rows then differ from the AR, though the argmax still agrees.
      - So STEP 3's REF2 runs (after its today run) were judged against a cache from a changed prefill; REF2's
        kernels are not at fault.
    - **Part B: warm 16-row checks (0.0 expert misses per check):** today 4299.8 ms, REF2 + SB 4277.4 ms. Saved
      22.4 ms per 48-layer check: BETWEEN (step 3's line, applied to warm checks).
      - Spread today 4231-7390, REF2 4217-5722.
      - Even with every expert resident, a 16-row check costs ~4.3 s: the engine path, not the kernels, sets the
        price.
      - **Correction (Thu 17:2x, from ~/BigMoE/bigmoe/stream.py; replaces 'the _sync_call path').** The checks took
        PF1's _sync_call_staged -> _staged_full. EvalSuite's env forces BIGMOE_PF_PREFILL=1 + BIGMOE_PF_FULLSCR=1, and
        _stage_drop (the only exit) runs only on a gated call with <= MAXK (40) ids. An all-16-row loop never makes
        one, so every check inherits the prefill's staging.
        - Every layer then streams ALL non-resident experts of the next one: ~336 x 2.8 MB x 48 = ~45 GB per check,
          ~3.8 s at ~12 GB/s. That is the 4.3 s. The 0.0 misses are a counter artifact (staging bypasses ensure).
        - BigMoE's default (PF1 off) is the plain _sync_call: ensure loads only the needed experts. STEP 3z re-runs
          step 3 there.
    - vs prediction: V0 == AR (60 %) yes; SB the suspect (50 %) no, V5 was. Warm check ~0.8-1.5 s predicted, 4.3 s
      measured; saving 35 predicted, 22.4 measured (BETWEEN, 45 %).
- **R-40 STEP 3y (pre-registered Thu 16:25; scripts/r40s3y_screen.py, ~1-2 min): which call leaves the
  prefill-changing state, and which component carries it.**
  - Prefill A with per-layer records (GDN, attention, MoE, PLE, HC read, layer output: the last row + a checksum of all
    rows), REF2 off.
  - Then, in order, each trigger followed by a fresh prefill compared record by record:
    - t1 = REF2 16-row check + commit keep 5 (REF2's own loop);
    - t2 = stock 16-row check, no commit;
    - t3 = stock commit keep 5;
    - t4 = stock 1-row AR step.
  - A measurement, no lines.
  - Prediction: t1 leaves no state (70 %), since V0-V4 + the AR's commits didn't. t2 is the trigger (60 %), t3 (30 %).
    The earliest differing record is an HC read or the MoE (stock <= 16-row paths REF2 replaces) at layer 0 (50 %).
  - **Amended prediction (Thu 17:36, before any run; from reading ~/BigMoE/bigmoe/stream.py and ext/bigmoe_ext.cpp): a
    stale PF1 staging entry, i.e. wrong expert weights, not a numerics drift.**
    - The mechanism:
      - The commit's PLE-layer replay (zero-based layer 1; ple_layer_ids [2]) is a lone layer forward. At keep 5 its
        MoE call has 50 ids > MAXK 40, so under EvalSuite's PF1 it takes _sync_call_staged.
      - That call kicks layer 2's staging into buffer 0 (_STAGE["pending"][2]), which nothing consumes. Only
        _stage_drop clears pending, and only a gated call runs it.
      - The next forward's layer 0 restages buffer 0 with layer 0's experts, and layer 2 skips its own kick
        ("2 in pending").
      - ext.scratch_full copies staging slots by position, with no record of which layer filled the buffer. So
        layer 2's non-resident experts get layer 0's weights.
      - ext.adopt then writes them into layer 2's pool, so the damage persists in later forwards of the process.
    - Hence:
      - t1 leaks too (80 %): REF2's own commit keep 5 takes the same path. The leak is PF1 + a commit with keep >= 5,
        not stock kernels.
      - The earliest differing record is record 14 = layer 2's MoE (75 %).
      - t2-t4 also differ from A, because the pool stays damaged (70 %).
      - This fits STEP 3. /68 (kept 15.41, mostly full accepts, no replay) matched the REF2 row 256 / 256;
        prealgebra/1640 (kept 5.24) matched only 199 / 256.
    - Exposure: only PF1 with lone > MAXK calls. AR (keep 1) and full-accept commits go through the gated path or
      skip the replay. BigMoE's shipped engine has PF1 off. STEP 3z keeps PF1 off outside prefills.
  - **Result (Thu 17:46-17:48, runner 20, rc 0; data/r40s3y_screen.txt): a stale PF1 staging entry. REF2 is not
    involved; it is BigMoE's PF1 path under EvalSuite's env.**
    - Prefills are deterministic (A == B: all 242 records and the last logits).
    - After any 16-row check: nothing pending. After a commit keep 5, REF2's (t1) or stock (t3): pending = [2].
    - The next prefill then differs from A from record 14 = layer 2's MoE on (228 of 242 records, the last logits).
      Before it, every record is equal: layers 0-1 (incl. the PLE layer) and layer 2's HC reads + GDN.
    - t2 (stock check, no commit) and t4 (a 1-row gated step, whose _stage_drop clears pending): the next prefill
      == A, bit for bit.
    - **Stock kernels leave no state; neither do REF2's.** The trigger is the commit's lone PLE-layer replay with
      > MAXK ids under PF1. Its stale layer-2 entry is consumed by the next forward, whose layer 2 builds its scratch
      from buffer 0 after layer 0 has restaged it (ext.scratch_full has no layer tag): wrong expert weights.
    - Damage lasts one forward when a prefill consumes the entry: a long prefill uses every resident expert, so
      ext.adopt stops at once ("the tail is used this chunk"). If a 16-row check consumes it (STEP 3's live loop:
      every check after a partial accept of 5-15 rows), adopt can copy the wrong weights into layer 2's pool, where
      they stay while resident. That part is by the code, not measured here.
    - vs the amended prediction: t1 leaks (80 %) yes; record 14 = layer 2's MoE (75 %) yes; t2-t4 damaged (70 %) no
      (one forward only, as above). The original prediction (t1 clean 70 %; HC / MoE at layer 0, 50 %) was wrong.
    - Fix (BigMoE's engine; Leap edits nothing there): when _stage_kick stages layer M into buffer b, drop any other
      pending entry on buffer b (a stale one then re-stages at its own turn). Or tag each buffer with its layer and
      re-stage on a mismatch at pop.
    - Reported Thu 17:49 to BigMoE (engine owner) and EvalSuite (forces PF1). EvalSuite checked: no suite verdict is
      exposed (PF1 is on only for the ref family, all AR keep 1). MoonShots (PF1's origin) got a note.
- **R-40 STEP 3y-fix (BigMoE asked Thu 17:51; pre-registered Thu 17:52; scripts/r40s3y_screen.py 176 fix,
  runner_r21.sh after runner 20, ~1-2 min): the same bisection on BigMoE 7ac3192.** The fix does both:
  _stage_kick drops other pending entries on the buffer it restages, and a pop whose buffer holds another layer
  restages (_STAGE["blayer"]).
  - **PASS:** prefill determinism holds, AND after t1 and t3 (commit keep 5) the next prefill == A bit for bit (all 242
    records + the last logits), AND t2 / t4 stay == A. Any differing record = FAIL.
  - Expected: pending still shows [2] right after a commit keep 5. The fix drops that entry at the next forward's
    layer-0 kick, so the decisive line is the prefill. Safety-net restages 0. PASS 90 %.
  - **Result (Thu 20:52-20:53, third attempt, runner 25, rc 0, C 128; BigMoE b8bd33c, which contains 7ac3192;
    data/r40s3y_screen_fix.txt): PASS.**
    - Prefill determinism holds.
    - After a commit keep 5, REF2's (t1) or stock (t3): pending still [2], as expected. The next prefill == A now:
      all 242 records + the last logits (before the fix: 228 records differed from layer 2's MoE on).
    - t2 / t4 == A. Safety-net restages 0: the kick-time drop handled it.
    - The first two attempts (19:14, 20:06, C 176) were stopped by the swap guard during the load.
    - vs prediction: PASS (90 %) yes. Result to BigMoE.
- **R-40 STEP 3z = the STEP 3 re-run (orchestrator Thu 15:4x: 'with a longer chunk or fewer prompts'; pre-registered
  Thu 17:30; scripts/r40s3z_screen.py, ~7-9 min): REF2 + SB vs today on BigMoE's default check path, exactness in a
  REF2-only process state.**
  - Why a re-run, not a repeat: STEP 3 measured two harness artifacts.
    - (1) Its checks took EvalSuite's forced PF1 path, which BigMoE's default leaves off (see STEP 3x's correction).
      Here PF1 stays on for prefills (EvalSuite's REF2 rows were made so). It is off for checks and commits
      (BigMoE's default _sync_call: per layer an id sync, ensure of the needed experts only, a scratch gather). The
      switch-off does what the engine's first decode call does (_stage_drop).
    - (2) A stock check changes later prefills (STEP 3x V5). So exactness is judged in Part A, before any stock call.
  - Harness, drafter, loop and arms: as STEP 3 (C = 176, DFlash2 L1-20 on taps, greedy, keep = accepted + 1).
  - Part A (exact, REF2 only): HumanEval/68 and prealgebra/1640. 48 ref2.ar_step logits, then one REF2 + SB spec run
    of <= 256 tokens. Its tokens must equal EvalSuite's REF2 AR rows, and its kept check rows' logits the AR's, bitwise.
  - Part W (warm): 16-row checks sliding by one token along /68's REF2 row, commit keep 1. Today vs REF2 + SB in
    alternating blocks of 4, 5 rounds (STEP 3x's Part B, on the default path).
  - Part B (live A/B): /68 ABBA, prealgebra/1640 BAAB, then /91 and algebra/694 while the budget lasts. No new prompt
    after 330 s, no new run after 420 s, runs cut at 500 s are dropped. Saving = today - REF2 median cycle per prompt,
    median over prompts. Also prefill s, tok/s and kept per domain, and misses + SSD time per check.
  - **Lines (STEP 3's):** PASS = Part A exact AND >= 40 ms saved per 48-layer check (Part B). KILL = < 20 ms.
    BETWEEN otherwise.
  - Prediction:
    - Part A exact 85 %: the checks now run REF2's experts_split on a compact scratch instead of the full-E one.
    - Check + commit 0.5-1.2 s live (misses ~20-50 per check-layer at C = 176), 0.3-0.6 s warm.
    - Saving ~30 ms (range 15-50): every layer syncs on the ids, so the GPU sits on the critical path and kernel
      savings show undiluted; SB adds ~0. PASS 30 %, BETWEEN 50 %, KILL 20 %.
    - Part B's REF2 runs after stock runs may leave the REF2 rows (V5's leak): information, not the verdict.
  - **Result (Thu 18:38-18:47, runner 20, rc 1 = not exact; data/r40s3z_screen.txt; BigMoE 7ac3192 incl. the PF1 fix):
    FAIL (not exact). Speed: REF2 + SB saves 137.5 ms per 48-layer check (line 40); exactness breaks after
    partial-accept commits.**
    - Part A, REF2 only, before any stock call:
      - HumanEval/68: tokens == the REF2 row 256 / 256, logits rows bitwise vs AR 48 / 48. Its first 48 tokens came
        from full accepts (kept 15.41), so no commit replay ran.
      - prealgebra/1640: tokens == 55 / 256, logits rows 40 / 48. Kept 4.92, so a partial commit followed every check.
      - So a 16-row check right after a prefill, or after full accepts, is bit-identical to REF2's AR. The partial
        commit (trim + GDN replays + the PLE layer's lone forward over the kept rows) leaves state that is not.
      - REF2's AR never runs that path: its 1-row commit is a full accept (MTPLX returns early, no replay).
    - The stock arm on 1640 also leaves its own AR row (ref.r2) at exactly token 55. A common, non-REF2 commit path
      fits that; a near-tie token at 55 would also explain it.
    - Speed on BigMoE's default check path (PF1 only for prefills):
      - Live A/B: /68 saved 160.3 ms per check (today 1660 / 1855 ms per cycle, REF2 + SB 1551 / 1643). 1640 saved
        114.7 ms (today 1630, REF2 1515; 2 of 4 runs fit the budget). Median 137.5 ms, 3.4x the 40 ms line.
      - Warm (Part W): today 913.8 ms vs 746.9 ms, saved 166.9 ms. REF2's arm had fewer misses (74.7 vs 107.9 per
        check, ~10-30 ms of SSD).
      - Checks cost 1.2-1.9 s live: ~450-880 misses per check (8-18 per layer at C 176), io 130-230 ms, plus 48
        id syncs + scratch gathers. tok/s: code 8.6 -> 9.6, math 3.3 -> 3.3 (kept 5.12).
      - Prefill 4.2-9.1 s on PF1 (EvalSuite's env; the SSD was contended: runs varied 4.3 -> 9.1 s).
      - Step 3's 3.9-4.3 s checks were EvalSuite's PF1 staging.
    - vs prediction: Part A exact 85 % no. Check + commit 0.5-1.2 s predicted, 1.2-1.9 s live (misses ~2x
      the guess), 0.75-0.91 s warm. Saving ~30 predicted, 137.5 measured: PASS-size, but the verdict is FAIL on
      exactness.
    - Swap rose 5.97 -> 7.83 GB during the run (+1.86, under run_locked's 2 GB stop): keep the next screens light.
    - Next: STEP 3w bisects the partial commit, comparing per-layer cache state after a 16-row check + commit keep k
      with REF2's AR after k steps. Suspect: the PLE layer's lone replay at S = k rows runs MTPLX's small-S GDN
      path, whose fused conv + norm is not bitwise (L1-31: "the engine runs S > 6"). Alternatives: the QSA trim's
      pooled blocks, or an S-dependent HC / PLE path.
- **R-40 STEP 3w (pre-registered Thu 18:49; scripts/r40s3w_screen.py, runner_r22.sh after runner 21, ~2 min, no
  drafter): which part of a partial-accept commit leaves state REF2's AR does not.**
  - Setup: as 3z's harness, REF2 + SB on, PF1 for prefills only; prealgebra/1640 (P 62).
  - AR reference: REF2's AR, 17 steps from one prefill, every layer's cache state digested after k steps.
  - Spec path, for each k in {2, 5, 8, 12, 15, 16}: a fresh prefill, ONE 16-row check of the AR's own first 16
    tokens (every row matches), commit keep k, the same digests. Then one ar_step probe vs the AR's next logits.
  - State per layer, sha1 per slot:
    - GDN: conv + recurrent;
    - the PLE layer: all slots;
    - QSA: keys, values, raw indexer keys, pooled blocks.
  - k = 16 is a full accept (MTPLX returns early, no replay): the control.
  - A measurement; no lines. What it decides:
    - only the PLE layer differs -> make its replay take the verify's own path (REF2-side, exact);
    - QSA differs -> trim / pooled blocks (MTPLX's);
    - GDN layers differ -> the replay kernel vs the forward's.
  - Prediction:
    - k = 16 all equal, and check rows == AR 16 / 16 (95 %).
    - For k < 16, the pure GDN layers equal (85 %) and the QSA layers equal (75 %). The PLE layer (zero-based 1)
      differs (60 %), at least for k <= 6.
    - Probe logits differ wherever a state does (90 %).
  - **Amendment (Thu 18:51, before any run): arm B.** The same commits, wrapped in attention_phase("decode_verify") +
    model_forward_kind("target_verify"), without the capture scope.
    - Why: MTPLX branches on the phase in two places the PLE layer's replay crosses. The HC inject is
      _verify_inject in decode_verify and 2 * sigmoid(logits / hc) elsewhere (they differ in MLX 0.32.2). The MoE
      is _verify_call vs the stock combine. REF2's AR runs every step inside the verify scopes; STEP 3z's commits
      ran outside.
    - Also checked: MTPLX's fused 2-6-row conv + norm (MTPLX_FUSED_CONVNORM_VERIFY, not bitwise) is off in this env
      (only BIGMOE_GDN_EXACT turns it on), so the earlier S <= 6 suspect is unlikely.
    - Prediction: arm B makes every k equal to the AR, states and probe (55 %). Otherwise arm B narrows the
      difference to the PLE layer's non-phase parts.
  - **Result (Thu 19:54-19:55, runner 22, rc 0; data/r40s3w_screen.txt): no difference anywhere. One partial commit
    is exact.**
    - Both arms, every k in {2, 5, 8, 12, 15, 16}: prefill == the AR's, check rows == AR 16 / 16, and all 48
      layers' state after commit keep k == REF2's AR after k steps. That covers the 35 GDN, the PLE layer and 12 QSA,
      every slot. The probe step's logits == the AR's too.
    - The phase doesn't matter here: arm A == arm B.
    - So 3z's break needs something 3w held fixed:
      - (a) the rejected rows' CONTENT. 3w's rows k..15 were the AR's own next tokens; 3z's were wrong drafts.
        Stale buffer data beyond the trimmed frontier is outside the digested state (QSA raw keys / KV past
        kv.offset; pooled blocks are cut by length) but readable by the next forward. With 3w's tokens, stale ==
        what the AR writes there, so a leak would be invisible.
      - (b) multi-cycle effects, e.g. a lazy snapshot of a buffer that a later forward updates in place.
      - (c) the drafter between checks.
    - vs prediction: k = 16 equal (95 %) yes; PLE layer differs (60 %) no; arm B fixes it (55 %) moot.
    - Next: STEP 3v. Garbage in the rejected rows plus per-layer records of the NEXT check, clean (AR-built state)
      vs dirty (commit-built state). A multi-cycle synthetic loop without the drafter separates (a) / (b) from (c).
- **R-40 STEP 3v (pre-registered Thu 19:58; scripts/r40s3v_screen.py, runner_r24.sh after runner 23, ~2-3 min, no
  drafter): does a partial commit leak the REJECTED rows' content into the next forward?**
  - Setup: 3w's harness, REF2 + SB on, prealgebra/1640, REF2's AR for 48 steps.
  - Part 1, one cycle, k in (2, 5, 8, 12):
    - clean = prefill + k AR steps;
    - dirty = prefill + one 16-row check of toks[:k] + garbage in rows k..15 + commit keep k. Garbage = 'shift'
      (each row holds the AR token one position later) or 'const' (token 1000).
    - From each state the same 16-row probe check (AR tokens k..k+15) runs with per-layer records. Reported: dirty
      vs clean rows and the earliest differing record; clean rows vs the AR's logits.
  - Part 2, many cycles: ~7 checks of row 0 + m correct + (15 - m) wrong drafts (m from a fixed list), greedy accept,
    commit keep accepted + 1 (3z's loop without the drafter). Kept rows are compared bitwise with the AR logits.
  - A measurement; no lines. What it decides:
    - Part 1 dirty != clean -> a leak through stale rows. The earliest record names the component to fix in REF2's
      path or MTPLX's trim.
    - Part 1 equal but Part 2 off -> a multi-cycle effect.
    - Both equal -> the drafter or taps (c).
  - Prediction:
    - Part 1: dirty != clean for some k (70 %). The earliest differing record is the first QSA layer's attention
      (zero-based 3; raw indexer keys / pooled blocks past the frontier) (55 %).
    - Clean == AR 16 / 16 (95 %). Part 2 reproduces 3z's break (< all kept rows equal) (65 %).
  - **Result (Thu 20:36-20:38, runner 24, rc 0, 99 s, C 128; data/r40s3v_screen.txt): no leak, no multi-cycle effect.
    Everything exact.**
    - Part 1, all 8 cases (k 2 / 5 / 8 / 12 x shift / const): after a partial commit with garbage in the rejected
      rows, the 16-row probe check == the clean (AR-built) state's probe. All 16 rows and all 242 per-layer records
      are equal, and clean == AR 16 / 16.
    - Part 2, 7 checks in a row with synthetic drafts (kept 4, 1, 7, 2, 10, 3, 5; keep 1 = the gated path, keep 10 =
      _sync_call): kept rows == AR 32 / 32. Every accept landed exactly where designed.
    - So REF2's checks + partial commits + wrong rejected rows + multi-cycle are exact. 3z's 1640 break (40 / 48 rows)
      must come from what 3v leaves out: the DFlash2 drafter (its forward between checks, or its specific tokens),
      the taps hook, or ref2.remove() / apply() around drafting.
    - vs prediction: leak (70 %) no; QSA (55 %) no; Part 2 reproduces (65 %) no.
    - Next: STEP 3u runs 3z's REF2 loop on 1640 with per-check diagnostics in variants that switch those pieces
      one at a time.
- **R-40 STEP 3u (pre-registered Thu 20:40; scripts/r40s3u_screen.py, runner_r26.sh after runner 25, ~2 min, C 128):
  what in 3z's live loop breaks REF2's bit-identity?**
  - Setup: REF2's AR for 64 steps on prealgebra/1640, then four runs of 3z's REF2 + SB loop up to 48 tokens. Every
    kept row is compared bitwise with the AR; the first differing check (index, kept, rows, drafts) is reported.
    - U0 as 3z: the drafter drafts with REF2 switched off (remove / apply around it), taps on.
    - U1: the drafter drafts with REF2 left on.
    - U2: the drafter runs as in U0, but the check uses synthetic drafts from the AR tokens (3v's pattern).
    - U3: no drafter forward; synthetic drafts; taps on; REF2 switched off / on between checks.
  - A measurement; no lines. Reading:
    - U0 breaks, U2 breaks -> the drafter's execution touches the target (shared module / buffer / state).
    - U0 breaks, U2 exact -> the drafter's token choices (some ids / patterns) trigger it.
    - U0 breaks, U1 exact -> the REF2 switch matters. U3 isolates the switch + taps without the drafter.
  - Prediction: U0 reproduces 3z's break (85 %). U3 exact (70 %), U2 breaks (40 %), U1 exact (40 %).
  - First run (Thu 21:20-21:21, runner 26): died after the AR with a NameError (the bitwise-compare helper
    3z's harness never defined); no data. Fixed (92d5d30; every name in the script now resolves). Re-queued as
    runner 28, after R-50b.
  - **Result (Thu 22:00-22:01, runner 28, rc 0, 62 s, C 128; data/r40s3u_screen.txt): neither the drafter nor the
    REF2 switch nor the taps matter. The loop breaks without any of them.**

    | variant | checks, kept | kept rows == AR | first differing check |
    |---|---|---|---|
    | U0 as 3z | 10: 6 8 1 4 4 2 15 2 2 11 | 40 / 55 | #7 at relative 40 (kept 2, rows 0-1) |
    | U1 REF2 on while drafting | same as U0, bit for bit | 40 / 55 | #7, same |
    | U2 drafter runs, synthetic drafts | 11: 4 1 7 2 10 3 5 1 6 2 8 | 41 / 49 | #10 at relative 41 (kept 8, rows 0-7) |
    | U3 no drafter, synthetic, taps, switch | same as U2, bit for bit | 41 / 49 | #10, same |

    - Deterministic (U0 == U1, U2 == U3). Every check after the first differing one differs too.
    - 3v's synthetic loop stayed exact only because it stopped after 7 checks (relative 32).
    - The break is near relative 40-41 (absolute 102-103) in both loops, but it is not one position. Position 40
      is exact in U3 (row 1 of check #9) and differs in U0 (row 0 of check #7, right after a keep-15 commit). It
      follows the state some commit leaves.
    - Ruled out by reading MTPLX: the QSA indexer's sparse selection, which engages only past 2048 tokens
      (indexer_budget 2048 / ratio 4 = block_topk 512 blocks). The n-gram hot cache (exact byte copies).
    - vs prediction: U0 breaks (85 %) yes; U3 exact (70 %) no; U2 breaks (40 %) yes; U1 exact (40 %) no.
- **R-40 STEP 3t (pre-registered Thu 22:05; scripts/r40s3t_screen.py, ~1-2 min, C 128, gather path as 3u): where does
  REF2's loop leave its AR?**
  - U3's synthetic loop on 1640 (no drafter, no taps).
  - At every check start: all 48 layers' cache state digested vs REF2's AR at the same position. Reports the first
    check whose START state differs, and its layers / slots.
  - Per-layer records (sha1 of each row) for every check and every AR step. At the first check with a differing
    kept row: the earliest differing record of that row vs the AR step's.
  - A measurement; no lines. Reading:
    - start state differs -> the previous commit (layer / slot named) is the cause;
    - start state equal but a record differs -> the check's own computation at that layer / part.
  - Prediction: the start state of the breaking check differs (65 %), in a QSA layer's KV (45 %) or the PLE layer
    (30 %).
  - **Result (Thu 22:06, runner 29, rc 0, 32 s, C 128; data/r40s3t_screen.txt): the PLE layer's replay.**
    - Checks #0-#9 are exact, with start states equal to the AR's at every start. Check #9 started at relative 39
      (absolute 101) and kept 2.
    - Check #10's START state differs from the AR's in ONE layer: zero-based 1 (the PLE layer), slots 0, 1, 2. Those
      are its GDN conv state, its GDN recurrent state and the PLE short-conv state. Slot 3 (n-gram context ids) is
      equal, and all other 47 layers are equal.
    - Check #10's row 0 differs from the AR step first at record 7 = layer 1's GDN output. Everything before it in
      the forward is equal: layer 0, layer 1's PLE output and HC read.
    - So check #9's commit replay of the PLE layer (2 rows, from an equal start state, with row inputs equal to the
      AR's) produced a different layer-1 state than the AR's two 1-row steps.
    - Ruled out by reading MTPLX:
      - The n-gram row ids: the NumPy staging and the MLX graph path compute the same rows. The tokens at 39-40 are
        25 / 68490, no EOS.
      - The row bytes: both go through _SidecarGather.gather_np + mx.dequantize.
      - The lookahead lanes: they check the row ids before use.
    - Remaining suspect: mx.conv1d. REF2 pins matmuls / HC / attention / experts to 16 rows but not the PLE layer's
      depthwise dilated short conv (k 4, dilation 3) nor the GDN conv. Their output for a row may depend on how many
      rows the call holds (1 in the AR, keep in the replay), rarely, via the reduction order.
    - vs prediction: start state differs (65 %) yes; QSA (45 %) no; PLE layer (30 %) yes.
- **R-40 STEP 3s (pre-registered Thu 22:10; scripts/r40s3s_screen.py, seconds of GPU, no model): is MLX 0.32.2's
  conv1d row-invariant?**
  - Synthetic bf16 streams on the PLE conv shape (10240 channels, k 4, dilation 3, 9 state rows) and the GDN conv
    shape (10240, k 4, dilation 1, 3 state rows), 64 rows each.
  - AR = 64 one-row calls, each with its state window. Batched = the same rows in calls of S = 2..16 rows. Each output
    row is compared bitwise with the AR's.
  - Also the explicit form (sum of k shifted multiplies in fp32, fixed order), S rows vs 1.
  - A measurement; no lines. Reading:
    - conv1d differs at some S -> fix the replay inside REF2's path. Either the PLE replay runs as 1-row calls (the AR's
      own computation, so REF2's AR bits stay), or the conv goes to the explicit form everywhere (row-invariant by
      construction, but a reference change: EvalSuite re-checks REF2).
    - conv1d invariant -> look elsewhere in the PLE replay (record its tensors directly).
  - Prediction: conv1d differs for some S (70 %). The explicit form is invariant at every S (90 %).
  - **Result (Thu 22:13, a shared lock, 0.2 s; data/r40s3s_screen.txt): conv1d is row-invariant.** Both shapes, every
    S in 2..16: 0 of 64 rows differ from 1-row calls (0 of 655,360 elements). The explicit fixed-order form equals
    conv1d bit for bit at every S. vs prediction: conv1d differs (70 %) no.
- **R-40 STEP 3r (Thu 22:14, pre-registered in the script header; scripts/r40s3r_screen.py, 0.1 s, shared lock): the
  PLE layer's reductions.**
  - The gate's (key * query).sum(-1) over 2560, the full gate (sqrt|g| * sign), GroupedRMSNorm (mx.fast.rms_norm per
    2560-group), the HC mean over 4 streams and an fp32 sum. Each was run on S rows at once vs one at a time.
  - **Result: 0 differing rows for every op at every S (2..16).** Row-count arithmetic is not the cause.
- **Root cause (Thu 22:15, by reading MTPLX after 3r):** mtplx/attention_math.py's attention_gate, which the PLE layer
  uses for value * sigmoid(gate), picks its lowering by PHASE.
  - In decode_verify it runs a compiled fused expression; elsewhere, the stock output * mx.sigmoid(gate). Its docstring
    says the two differ in the last bit for some bf16 inputs (e.g. -6.84375).
  - REF2's AR steps and its 16-row checks run inside the verify scopes. The loop's commit_verified_window ran outside
    them, so the commit's lone PLE-layer replay took the stock sigmoid.
  - At gate values where the two lowerings differ, the replayed PLE output, and with it the PLE conv state (slot 2)
    and the GDN input (slots 0-1), left the AR's. That is 3t's signature: rare, data-dependent, deterministic, and
    invisible in 3w's 16 positions, which is why arm B (the phase) showed no difference there.
  - GatedResidual's inject and the MoE combine also branch on the phase. They shape the replayed layer's output,
    which the commit discards.
  - Fix (ref2.py, 22:15): REF2 wraps Qwen4ExpTextModel.commit_verified_window in attention_phase('decode_verify') +
    model_forward_kind('target_verify') when on (CFG commit_phase). REF2's AR bits don't change: its 1-row commit is a
    full accept and replays nothing.
- **R-40 STEP 3q (pre-registered Thu 22:14; scripts/r40s3q_screen.py, runner 30, ~4-5 min, C 144, ref2's defaults =
  slot call + commit phase): is REF2 bit-exact in the live drafting loop now?** (the wall Kamen named)
  - Part L, live DFlash2 loop up to 256 tokens on HumanEval/68, prealgebra/1640, HumanEval/91, algebra/694 while the
    budget lasts:
    - every token vs EvalSuite's REF2 AR row;
    - kept rows' logits vs 48 in-process REF2 AR steps, bitwise.
  - Part D: 3t's synthetic loop on 1640 (64 tokens). Every check start's 48-layer state digest vs the AR's.
  - Part C, control: Part L's 1640 run with commit_phase off. It must leave the REF2 row, as 3z / 3u did.
  - **PASS** = every compared token and logits row equal (L) AND every start state equal (D) AND the control breaks.
    **FAIL** = any difference in L or D. **VOID** = the control doesn't break.
  - Prediction: PASS 75 %. Other phase-dependent paths in the replay are possible but would also show as state
    differences in D.
  - **Result (Thu 22:16-22:19, runner 30, rc 0, 178 s, C 144; data/r40s3q_screen.txt): FAIL. The phase fix works, and
    3 of 4 prompts are now bit-exact in the live loop. algebra/694 has a second, separate break.**

    | prompt (kept per check) | tokens == EvalSuite's REF2 row | logits rows vs AR | tok/s (cycle) |
    |---|---|---|---|
    | HumanEval/68 (15.41) | 256 / 256 | 48 / 48 | 28.3 (505 ms) |
    | prealgebra/1640 (5.43) | 256 / 256 (was 55) | 48 / 48 (was 40) | 13.9 (368 ms) |
    | HumanEval/91 (8.09) | 256 / 256 | 48 / 48 | 18.5 (368 ms) |
    | algebra/694 (7.34) | 216 / 256 | 14 / 48 | 15.4 (460 ms) |
    | control: 1640, commit phase OFF | 55 / 256 | 40 / 48 | 13.6 |

    - Part D: the synthetic loop's start states == the AR's at all 11 check starts (3t broke at #10 without the fix).
    - The control reproduces the old break exactly, so the phase fix is what closed 1640.
    - On 694 only 14 of the first 48 kept rows match: an early break, within the first ~2-3 checks, and a different
      source. Tokens held to 216 (a numeric difference that rarely flips argmax).
    - Speed with ref2's defaults (slot call; C 144; live drafting): 28.3 / 18.5 tok/s on code, 13.9 / 15.4 on math.
    - vs prediction: PASS (75 %) no; the phase fix itself yes (1640 and Part D).
    - Next: STEP 3p, 3t's bisection on 694: AR digests at every position, the live drafter loop and the synthetic
      loop, the first check whose start state differs (layer / slot) and the first differing record.
- **R-40 STEP 3p (pre-registered Thu 22:22; scripts/r40s3p_screen.py, runner_r32.sh after R-50c, ~2 min, C 144, ref2's
  defaults): where does the loop leave REF2's AR on algebra/694?**
  - AR: 64 REF2 steps with per-layer row records and a 48-layer state digest at EVERY position.
  - Loop A (live DFlash2 drafter, as 3q) and Loop B (3t's synthetic drafts), up to 64 tokens each. At every check
    start: the state digest vs the AR's. Every kept row vs the AR logits. Reported: the first check whose start state
    differs (layer / slot) and, at the first differing kept row, the earliest differing record.
  - A measurement; no lines. Prediction: Loop A breaks within the first 3 checks (75 %); Loop B also breaks (50 %);
    the first differing start state is in one layer again (60 %), a QSA layer this time (P 98 + a check crosses
    absolute ~100-115) (35 %).
  - **Result (Thu 22:21-22:22, runner 32, rc 0, 39 s; data/r40s3p_screen.txt): the CHECK itself, at layer 35's
    attention, for rows past absolute 112.**
    - Loop A (live drafter): kept 6, 12, 14, 1, 7, 2, 4, 4; kept rows == AR 14 / 50.
      - Check #1 (start relative 6 = absolute 104, kept 12): rows 0-7 equal, rows 8-11 differ.
      - Row 8's earliest differing record is record 142 = layer 35's attention output; everything before it is equal,
        including the 8 earlier attention layers.
      - The next start state differs from layer 36 on (GDN 36-38, 40-42; QSA 39, 43), i.e. downstream of it.
    - Loop B (synthetic drafts): kept rows == AR 49 / 49, no start state differs. Its checks started at 110 (kept 2)
      and 112, so no kept row met the condition below.
    - The condition: a kept row at absolute p >= 112 in a check that started before 112. REF2's AR pads attention to
      16 rows (key length p + 16 >= 128), while the check's key length is start + 16 < 128. 694's first 48 rows
      cover 98-145, hence exactly 14 equal; 1640's first 48 cover 62-110, hence never.
    - vs prediction: Loop A within 3 checks (75 %) yes; Loop B breaks (50 %) no; one layer (60 %) yes, QSA (35 %) yes.
- **R-40 STEP 3o (Thu 22:24, pre-registered in the script; scripts/r40s3o_screen.py, 1.1 s, shared lock): is MLX's
  fused SDPA invariant to the key length?**
  - Synthetic: head_dim 256, 2 query heads on 1 KV head, 16 rows, bool causal mask (MTPLX's head-chunked verify
    shape). Rows near 90 / 104 / 112 / 120 / 240 / 500, N extended by 1..40 masked keys.
  - **Result: 0 N values change any row's bits.** The kernel alone is N-invariant here, so 3p's difference must
    enter through how MTPLX builds the call or its inputs.
- **R-40 STEP 3n (pre-registered Thu 22:28; scripts/r40s3n_screen.py, runner 33, ~1.5 min, C 144): inside layer
  35's attention.**
  - Every mx.fast.scaled_dot_product_attention call of layer 35 (12 head chunks) is captured for 3p's check #1 (row 8
    = absolute 112) and for the AR step at 112.
  - Compared bitwise: the row's q (post-rope), keys and values 0..112, and the row's output.
  - Reading:
    - q or k / v differ -> the inputs (rope / cache writes); the first differing key position names which write;
    - all inputs equal but the output differs -> the kernel in the real call shape (mask form / strides / chunking).
  - Prediction: a key / value at some position 104..111 (written by this check's rows 0-7) differs from the AR's (50
    %); q differs (15 %); inputs equal, output differs (35 %).
  - **Result (Thu 22:27 and 22:29 (3n-b: all 12 chunks + the gate + o_proj), runner 33, rc 0;
    data/r40s3n_screen.txt): the SDPA kernel itself.**
    - In all 12 head chunks, row 8's q, keys 0..112 and values 0..112 are bit-identical between the check (N 120,
      mask (1, 1, 16, 120)) and the AR's padded call (N 128).
    - In 11 chunks the row output is equal too. In chunk 5 (query heads 10-11) it differs.
    - attention_gate: the gate input is equal, the SDPA input differs, so the gated output and o_proj differ.
    - So MLX's fused vector SDPA gives a row different bits for equal inputs when the call's key length is 120 vs
      128, data-dependently (1 of 12 chunks here; 3o's random data never hit it).
    - vs prediction: inputs equal, output differs (35 %) yes.
    - The kernels (mlx/include/.../sdpa_vector.h): the 1-pass kernel stripes keys over 32 simdgroups (fixed, so
      N-invariant). The 2-pass kernel stripes them over `blocks` threadgroups, a function constant the host picks
      from the call. A host switch between the two, or a change of `blocks`, as N crosses ~128 would regroup the
      softmax partial sums. The host source is not on this Mac.
- **R-40 STEP 3m (pre-registered Thu 22:38; scripts/r40s3m_screen.py, runner 34, ~1 min): the real differing chunk
  replayed at other key lengths.**
  - From 3n's capture: the check's chunk-5 inputs re-run at N = 113..~190 plus 192 / 200 / 255 / 256 / 257 / 300 /
    512 / 1024 / 1100, extra keys zero-padded and masked. Reported: at which N row 8 equals the AR's output (N 128) or
    the check's (N 120).
  - Also: the check's call with contiguous k / v and with an additive float mask.
  - Reading: the N classes on real data give the rule a fix must respect (group a check's rows by the AR's class).
  - Prediction: the check's bits hold for N < 128 and the AR's for N >= 128 up to the next class boundary (65 %);
    contiguity / mask form doesn't matter (85 %).
  - **Result (Thu 22:33, runner 34, rc 0, 16 s; data/r40s3m_screen.txt): the boundary is N = 128 exactly.**
    - Chunk 5's row 8 from the check's real inputs equals the check's own bits at N 113..127 and the AR's at N 128..167
      (and on: the printout stopped at 40 matches).
    - Replays reproduce both original calls. Contiguous k / v and an additive float mask change nothing.
    - So MLX's vector SDPA switches mode at 128 keys for this call shape (2 query heads x 16 rows per call), and each
      side is N-invariant.
    - vs prediction: yes (65 %); contiguity / mask form doesn't matter (85 %) yes.
  - Fix design (keeps REF2's AR bits, so EvalSuite's reference stands):
    - REF2 wraps MTPLX's _verify_sdpa. In a 16-row call with N keys, row j sits at position N - 16 + j, so its AR
      step's call had N + j keys.
    - Rows whose AR class differs from the call's are recomputed in one extra SDPA call, with k / v zero-padded and
      masked to that class's lower bound.
    - The AR's own padded call never splits: its only real row is j = 0.
    - This needs the full class map (all boundaries, not just 128): STEP 3l.
- **R-40 STEP 3l (pre-registered Thu 22:36; scripts/r40s3l_screen.py, runner 35, ~1 min): the SDPA key-length class
  map on real data.**
  - Check #1's chunk-5 SDPA inputs from all 12 attention layers. Each is re-run at ~100 key lengths from 120 to
    33,000: dense around every power of two 128..32768, every 8 up to 300, plus 384 ... 24576, 33000.
  - Output hashes form the classes; reported: the boundaries.
  - Prediction: boundaries at 128 and at the 2-pass block-count switches (somewhere in 1024..16384) (60 %); no
    boundary between 128 and 1023 (80 %).
  - **Result (Thu 22:36, runner 35, rc 0, 17 s; data/r40s3l_screen.txt): one boundary only, at 128.** The 12 real calls
    (chunk 5 of every attention layer) at 57 key lengths from 120 to 33,000 fall into exactly 2 output classes:
    N <= 127 (2 probes) and N >= 128 (55 probes, up to 33,000). vs prediction: a second boundary (60 %) no; none in
    128..1023 (80 %) yes.
  - The fix is in ref2.py (22:3x, CFG sdpa_classes / sdpa_bounds = (128,)). REF2 wraps MTPLX's _verify_sdpa: rows of a
    16-row call whose AR step had >= 128 keys while the call has < 128 are recomputed with k / v zero-padded and
    masked to 128, then selected in. The AR's own call never splits (REF2's AR bits unchanged).
- **R-40 STEP 3k (pre-registered Thu 22:40; scripts/r40s3k_screen.py, runner 36, ~7 min, C 144; ref2's defaults =
  slot call + commit phase + SDPA classes): REF2 bit-exact in the live drafting loop? (the wall Kamen named)**
  - Part L, the live DFlash2 loop up to 256 tokens on HumanEval/68, prealgebra/1640, HumanEval/91, algebra/694, and
    the short prompts algebra/1282 (P 50) and gsm8k/708 (P 53), whose checks cross absolute 112 early:
    - every token vs EvalSuite's REF2 AR row;
    - kept rows' logits vs max(48, 152 - P) in-process REF2 AR steps, so the compared rows reach past absolute 112.
  - Part D: 3t's synthetic loop on 1640, every check start's state vs the AR's.
  - Part C, control: algebra/694 with the SDPA classes OFF. It must break as in 3q (14 / 48).
  - **PASS** = every compared token and logits row equal (L) AND D exact AND the control breaks. **FAIL** = any
    difference.
  - Prediction: PASS 70 %.
  - **Partial result (Thu 22:38-22:40, runner 36; logs/job.out): 4 of 4 finished prompts bit-exact with both fixes.
    Then the 22:40 quiet window's PAUSE froze the job (SIGSTOP).**

    | prompt (kept) | tokens == EvalSuite's REF2 row | logits rows vs AR | SDPA splits per check | tok/s (cycle) |
    |---|---|---|---|---|
    | HumanEval/68 (15.41) | 256 / 256 | 48 / 48 | 0.00 | 29.3 (520 ms) |
    | prealgebra/1640 (5.31) | 256 / 256 | 90 / 90 (past absolute 112) | 0.75 | 14.6 (342 ms) |
    | HumanEval/91 (8.09) | 256 / 256 | 48 / 48 | 0.00 | 18.9 (402 ms) |
    | algebra/694 (7.73) | 256 / 256 (3q: 216) | 54 / 54 (3q: 14 / 48) | 0.73 | 16.0 (465 ms) |

    - Not yet run: algebra/1282, gsm8k/708, Part D, Part C. The job resumes or loses its lock after 00:00 (Kamen: no
      new work before 00:00). Re-run 3k first thing.
    - Reported to the orchestrator as BIG (partial) at 22:42.
  - **Resumed run (Fri 00:04-00:06, in the soft-end ramp): no difference anywhere, but incomplete.**
    - The script's time budget counted the 84-min freeze. So algebra/1282 was cut after 2 checks (12 / 12 tokens,
      11 / 11 logits rows bitwise), and gsm8k/708 and Part C were skipped.
    - Part D: start states == the AR's at 11 check starts.
    - The printed "FAIL" comes from the cut rule (a cut prompt counts as not exact), not from a difference.
    - Totals so far: 1036 / 1036 tokens, 251 / 251 logits rows, D exact. The control has not run.
  - **Re-run (Fri 00:10, runner 37, same lines and prediction).** The budgets now count running time only: a ticker
    thread that a SIGSTOP freezes, like run_locked's limit. Each prompt line says "paused" if PAUSE appeared mid-run.
  - **RESULT (Fri 00:20-00:24, runner 37; data/r40s3k_screen.txt): PASS. REF2 is bit-exact in the live drafting
    loop.** 232 s running, no pause. Prediction PASS 70 %: right.

    | prompt (kept) | tokens == EvalSuite's REF2 row | logits rows vs AR | SDPA splits per check | tok/s (cycle), C 144 |
    |---|---|---|---|---|
    | HumanEval/68 (15.41) | 256 / 256 | 48 / 48 | 0.00 | 29.0 (539 ms) |
    | prealgebra/1640 (5.31) | 256 / 256 | 90 / 90 | 0.75 | 14.9 (348 ms) |
    | HumanEval/91 (8.09) | 256 / 256 | 48 / 48 | 0.00 | 19.2 (400 ms) |
    | algebra/694 (7.73) | 256 / 256 | 54 / 54 | 0.73 | 16.6 (451 ms) |
    | algebra/1282 (6.89) | 256 / 256 | 102 / 102 | 0.65 | 17.8 (362 ms) |
    | gsm8k/708 (6.16) | 197 / 197 (EOS) | 99 / 99 | 0.75 | 14.5 (423 ms) |

    - Totals: 1477 / 1477 tokens and 441 / 441 logits rows bitwise.
    - Part D: start states == the AR's at 11 check starts.
    - Control (algebra/694, SDPA classes OFF): 216 / 256 tokens and 14 / 54 logits rows. It breaks exactly as 3q did.
  - **CAVEAT (Fri 01:4x): EvalSuite's R-56 runs the same loop to full length (C 192, 36 suite prompts).**
    - So far 16 of 21 prompts are equal to the stored REF2 rows.
    - 5 diverge at 1-ULP near-ties (gap 0.125, or 0.0) at absolute positions 498 / 515 / 573 / 724 / 910, at check
      rows 11 / 0 / 2 / 11 / 1. Other prompts pass those positions exactly, so it is not a class in N.
    - ar_bad is 0 on the first 48-97 rows of every prompt.
    - So 3k's PASS holds as tested (256 tokens; logits on the first 48-128 rows). Full-length exactness is open: either
      the loop drifts after ~100 tokens, or the stored rows differ from today's AR. EvalSuite's bitwise diagnostic
      decides.
    - The orchestrator was told at 01:4x.
  - **EvalSuite 01:40: geometry/702 (P 520) has only 2/48 kept rows bitwise equal to the in-process AR.** All 8
    divergences sit at 498-1124 total keys. So the break shows within the first rows on a long prompt, not as slow
    drift.
- **R-40 STEP 4a (pre-registered Fri 02:10; scripts/r40s4a_screen.py, ~4 min, C 144): where the loop leaves the AR past
  ~512 keys.**
  - Method: REF2's AR, then from a fresh prefill the same AR steps up to a check whose drafts are the AR's own tokens.
    Every row of that check is then an AR row. Per row: logits bitwise, and the first DecoderLayer whose output differs.
  - Cases:
    - A: 702 right after its prefill (keys 536).
    - B: HumanEval/68 right after its prefill (keys 383): the control (3k: exact).
    - C1 / C2: HumanEval/68 with checks at absolute 500 and 520, so keys past 512 on a short prompt.
  - Prior reading: MTPLX's QSA indexer stays dense up to T // 4 <= block_topk (512), i.e. T <= 2051, on both of its
    lanes. 3l's probes showed one SDPA class boundary (128) up to 33K keys.
  - Prediction:
    - C fails like A (a keys-past-512 effect, not prompt length): 60 %. Only A fails (the > 512-token prefill): 30 %.
    - The first differing layer is a QSA attention layer (3): 60 %.
    - B is exact: 95 %.
  - **RESULT (Fri 02:06-02:07, runner 39; data/r40s4a_screen.txt): a keys-past-512 effect in the QSA attention call.**

    | case | check at (keys) | logits rows bitwise | first differing layers |
    |---|---|---|---|
    | A 702 (P 520) | 520 (536) | 2 / 16 (rows 0-1) | QSA 35 (row 2), 23 (row 4), 11 (row 6), 7 (row 12); the rest GDN, carried by the recurrence |
    | B HumanEval/68 (P 367) | 367 (383) | **16 / 16, every layer equal** | none |
    | C1 HumanEval/68 | 500 (516) | 2 / 16 | QSA 11 (row 2), 7 (row 6); the rest GDN |
    | C2 HumanEval/68 | 520 (536) | 2 / 16 | QSA 11 (row 2), 3 (rows 5, 6, 8, 13); the rest GDN |

    - Max |logit delta| per row: up to 0.59 (A), 3.1 (C1) and 4.6 (C2).
    - Predictions: C fails like A (60 %): right. First difference in a QSA attention layer (60 %): right. B exact: right.
    - Every first difference is a QSA attention layer. A GDN layer differs only for a row whose earlier row differed
      at the attention layer below it (the recurrence carries it).
    - Rows 0 and 1 always match. Their AR steps' calls have N and N+1 keys; rows >= 2 have N+2 and up.
    - **Reading:** for q_len 16, MLX 0.32.2 takes its full attention kernel (NAX on M5: bq 64, bk 32, head dim split in
      two), not the vector kernel. Past ~512 keys a row's bits depend on the call's key length on real data. 3o's
      synthetic data never showed it (start 500: 0 of 40 N differ).
- **R-40 STEP 4b (pre-registered Fri 02:30; scripts/r40s4b_screen.py, ~2 min, C 144): the class map past 512, real
  data.**
  - Capture: layer 35's _verify_sdpa inputs in 4a's case A (check at 520), and in AR step 2 (position 522).
  - Replays:
    - a: both calls reproduce their outputs.
    - b: check row 2 with the keys zero-padded to N' = 536..1100: which N' give the AR's bits.
    - c: the AR's call with the real row moved to q index 2.
    - d: the bucket fix, both calls padded to 1024 (and 1536).
  - Prediction:
    - b: classes are pairs of N (N and N+1 alike, per 4a's rows 0-1); the AR's N (538) gives the AR's bits: 70 %.
    - c: the row index does not matter: 80 %.
    - d: the bucket fix makes row 2 equal: 75 %.
  - If d passes, the fix pads every 16-row attention call past 512 keys to a 512-bucket. Rows whose AR bucket
    differs are recomputed in that bucket (ref2's sdpa_classes, with bounds at the bucket edges).
    - That changes REF2's AR bits past 512 keys, so the reference rows must be regenerated: EvalSuite's call.
    - The alternative keeps the AR: recompute every row j >= 2 at its own AR length, about 8 calls per chunk. Too
      slow to adopt; for a test only.
  - **RESULT (Fri 02:28, runner 40; data/r40s4b_screen.txt): the bucket fix holds on the captured call.**
    - Row 2's q, keys 0..522 and values are equal in the AR's call (538 keys) and the check's (536), yet the outputs
      differ. Both replays reproduce.
    - b: across N' = 536..1100 the row takes only TWO bit patterns. It flips between the AR's and the check's in
      even-length runs (2 to 26) with no period: 52 runs; 455 of 565 N' give the AR's bits.
    - c: moving the real row to q index 2 does not change it.
    - d: both calls padded to 1024, and to 1536: equal.
    - Predictions: two classes, not pairs (wrong in form, right that N alone decides); the index does not matter
      (right); the bucket fix works (right).
- **R-40 STEP 4c (pre-registered Fri 02:45; scripts/ref2_kb.py + r40s4c_screen.py, ~4 min, C 144): the key-bucket fix in
  the live loop.**
  - ref2_kb: every 16-row attention call with N >= 512 keys runs at the next multiple of 512 (zero-padded, masked).
    Rows whose AR step lands in another class (< 128, 128-511, or another 512-bucket) are recomputed in that class.
    Below 512 keys nothing changes.
  - It is a separate module, because ref2.py is pinned by EvalSuite's R-56. It changes REF2's AR bits past 512 keys,
    so the reference is the in-process REF2 AR with key buckets.
  - Harness = 3k's. Part L: 702 (AR rows 520..679) and HumanEval/68 (AR rows 367..586, across 512), the live loop to
    256 tokens, every kept row with an AR row compared bitwise. Part C: 702 with the buckets off must break.
  - PASS = every compared row equal AND the control breaks. Prediction: PASS 75 %.
  - **RESULT (Fri 02:43-02:47, runner 41; data/r40s4c_screen.txt): PASS. With key buckets the live loop is
    bit-exact past 512 keys.** 169 s running.

    | prompt | kept | logits rows bitwise vs the bucketed AR | tokens == stored REF2 row (old reference) | tok/s (cycle), C 144 |
    |---|---|---|---|---|
    | geometry/702 (P 520) | 5.12 | **160 / 160** (positions 520-679) | 118 / 256 | 14.9 (317 ms) |
    | HumanEval/68 (P 367) | 15.41 | **220 / 220** (367-586, across 512) | 256 / 256 | 27.0 (548 ms) |
    | control: 702, buckets off | 4.74 | 2 / 160 | 118 / 256 | 14.1 (301 ms) |

    - 3732 calls were bucket-padded, and 192 class-split recomputes ran (AR and loop together).
    - The bucketed AR leaves the stored row at 702's token 118, as the old loop did. That is the reference change
      past 512 keys (EvalSuite's diagnostic: the old AR ties there, 24.25 / 24.25).
    - EvalSuite's R-56 diagnostic (02:4x) agrees on the location. On all 8 failing prompts the loop leaves the old AR
      at the first row whose AR call has 513-514 keys. The stored rows reproduce with today's AR.
    - **Merged into ref2.py as REF2b (02:5x):** CFG key_buckets (default on), kb_from / kb_step 512. key_buckets off
      reproduces REF2's original AR.
    - Still open:
      - a bitwise microtest of the merge against the module that passed 4c;
      - the 1024 bucket edge (not reached here);
      - regenerating the reference rows on REF2b and the R-56 re-run with a bitwise check on every kept row
        (EvalSuite).
- **STEP 3y-fix, first attempt (Thu 19:14:47-19:15:04, runner 21): stopped by run_locked's swap guard, no result.**
  Swap went 9.8 -> 14.7 GB during the model load, right after the previous holder's job. Reported to the
  orchestrator as BIG (machine). Re-queued as runner 23, which waits 30 s after a lock grant and re-checks memory
  before loading.

- **STEP 3y-fix, second attempt (Thu 20:06:30-20:06:48, runner 23, after a 30 s settle): stopped by the swap guard
  again** (9.7 -> 13.0 GB in 18 s).
  - The machine held 13 GB in the compressor and ~24 GB free. A ~30 GB load at C 176 pushes compressed pages to
    swap, and the 58 %-free gate doesn't see it. STEP 3w's load at 19:54 grew swap by 0.
  - **Amendment (Thu 20:08, before their runs): C capped in the scripts.** 3y-fix and 3v run at C 128 (-6.5 GB;
    numerics don't depend on C). R-50 runs at C 144 (-4.3 GB; slot calls need <= 0.9 C = 129 experts, else the
    shipped path takes over and the script counts the fallbacks).
  - 3y-fix is re-queued as runner 25, after runner 24 (3v).
- **REF2 module notes (Wed 15:33, desk):**
  - experts_nax (qmmg25, L1-25: -22 ms per 16-row check, batch-invariant, 2.3x more accurate) needs gate | up
    concatenated per expert. A copy would cost ~1.3 GB per layer.
  - MTPLX already has _FusedGateUpSwitchGLU (qwen4_exp.py ~1499), which builds gu_weight at sanitize time and
    replaces the separate weights. So the module uses the engine's gu_weight when present, or fuses at load the
    same way; never a copy.
  - Projected REF2 saving per 48-layer check: dense via qmm16c ~-11.5, experts ~-22, S4 ~-14, SB ~-8 to -16 =>
    ~-56 to -64 ms vs the -40 line. That is conditional on batch invariance (INV0).
  - INV0 is queued (runner_r10). It waits for an open lid: closed since Tue 23:57.

## R-50 SLOTCALL (NewIdeas GO Thu 18:59, from Leap's 18:48 proposal; Leap, pre-registered Thu 2026-10-29 19:18 EET): 16-row checks without the scratch gather
- **Idea:** BigMoE's plain _sync_call (the 45 GB check path, PF1 off) runs per layer:
  - id sync -> ensure -> mx.take of the needed experts (up to 0.9 C = 158 x 9 pieces, ~2.8 MB each) into a scratch
    pool -> the MoE on compact ids -> clear_cache.
  - The gather keeps MLX's sorted gather on the reference's expert ORDER. REF2's expert kernel (qmmg25 / experts_split)
    computes every (row, expert) pair on its own (self-test 'experts rows alone'), so the order and the slot numbers
    don't matter.
  - So the experts kernel can run on the pool itself with slot ids, as the gated path already does: no gather, no
    scratch, no clear_cache.
- **Prior art:** this is what BigMoE's gated path and the 24 GB engine do for <= MAXK ids (slot ids into the pool,
  sync-free). Applied to the > MAXK plain path under a batch-invariant expert kernel: new here.
- **Screen (scripts/r50_screen.py, runner_r23.sh, one lock ~7 min):**
  - 3z's harness (C 176, PF1 for prefills only, live DFlash2 drafting), both arms REF2 + SB.
  - gather = _sync_call as shipped; slot = REF2-on calls with <= 0.9 C distinct experts on pool slots (the rest fall
    back). The patch lives in this process only; BigMoE's file is untouched.
  - HumanEval/68 ABBA, prealgebra/1640 BAAB, <= 160 tokens.
- **Lines (NewIdeas):** PASS = every token AND each run's first-check logits hash equal between the arms AND >= 40 ms
  saved per check (median cycle, gather - slot, median over prompts) -> into REF2's path. KILL = < 15 ms.
- **Prediction:** bit-equal 95 %. The gather moves ~100-158 experts x 2.8 MB per layer, ~13-21 GB of copies per
  check at C 176, ~65-100 ms at copy speed, partly hidden behind the id syncs. Saving ~60 ms (range 30-100):
  PASS 65 %, BETWEEN 25 %, KILL 10 %.
- **Amendment (Thu 20:08, before the run):** C 144 (see STEP 3y-fix's second attempt).
- **Result (Thu 20:16-20:18, runner 23, rc 0, 143 s; data/r50_screen.txt): PASS. Median saving 384.2 ms per 16-row
  check, about 6x the prediction; bit-equal.**

  | prompt (kept per check) | gather: cycle / tok/s | slot: cycle / tok/s | saved per check |
  |---|---|---|---|
  | HumanEval/68 (15.55) | 960.0, 1033.4 ms / 14.0-14.1 | 553.4, 537.9 ms / 25.7-25.8 | 451.1 ms (x1.83 tok/s) |
  | prealgebra/1640 (5.03) | 654.5, 659.9 ms / 7.4-7.5 | 344.1, 335.8 ms / 14.1-14.4 | 317.2 ms (x1.92 tok/s) |

  - Bit-equal: all 4 runs per prompt produce the same 160 tokens and the same first-check logits hash
    (25c55047fab416f5 / bb10459d7f720669).
  - Misses per check are equal across arms (~1266-1284 on /68, ~602 on 1640). So the saving is the gather path's
    own cost, not I/O.
  - Slot calls cover 48.1-48.4 MoE calls per check (every layer + the commit's lone PLE-layer replay). 2 fallbacks
    in two /68 runs (> 0.9 C = 129 distinct experts).
  - Why 6x the prediction: per layer the shipped path makes a fresh ~330 MB scratch (9 mx.take + concatenate) and
    frees it with clear_cache. That is ~16 GB of new buffers per check, paid at allocation / page-fault speed, not
    copy speed. Not profiled; the drop itself is measured.
  - REF2-only by construction. The slot ids keep exactness only where REF2's per-(row, expert) kernel runs: calls of
    <= 16 rows with REF2 on. Larger calls would hand stock MLX's sorted gather slot-ordered ids. The adopted form
    must gate on rows <= 16 (this screen's prefills ran PF1 and never reached it).
  - Context: in this harness REF2 + DFlash2 16-row checks go 14 -> 26 tok/s on code and 7.4 -> 14.3 on math, at C 144
    with 600-1280 SSD misses per check. That is not the shipped 45 GB line (~64 tok/s on code, MTP head, 2-7-row
    gated checks). The exact 16-row path is still the slower one; this removes ~40-48 % of its check.
  - vs prediction: bit-equal (95 %) yes. Saving 60 predicted, 384 measured: the scratch is an allocation cost, not
    a copy cost.
  - Next (NewIdeas' line: PASS -> into REF2's path):
    - ref2.apply() installs the slot call when BigMoE's stream is loaded, gated on REF2 on + rows <= 16 + PF1 off +
      <= 0.9 C distinct ids.
    - BigMoE (engine owner) gets the finding: its shipped plain-path gather costs ~8 ms per layer-call in
      allocations. For its own stock paths a reused scratch buffer (no clear_cache) may be a cheap exact win too;
      that is BigMoE's to screen.
> (a quote from the project owner, kept private)
  pre-registered Thu 21:02; scripts/r50b_screen.py, runner_r27.sh after runner 26, ~2 min):** warm 16-row checks,
  gather vs slot, at the 45 GB line's REF2 configuration (C 192), with the needed experts resident.
  - Setup: no drafter; REF2 + SB on; PF1 for prefills only.
  - Windows: 16-row checks sliding by one token along HumanEval/68's REF2 row, commit keep 1 (3z's Part W). Four
    passes over the same 14 windows, each from a fresh prefill, in the order gather, slot, slot, gather.
  - Logits per window are compared across all passes. Time = median over warm checks (window >= 2) per arm.
  - **Lines (R-50's):** PASS = every window's logits equal across the arms AND >= 40 ms saved per warm check. KILL =
    < 15 ms.
  - Prediction: the gather's cost scales with the distinct experts per layer-call (~100-130 for 16 rows), not with C,
    so the saving carries over. Warm gather ~700-900 ms vs slot ~350-500 ms, saving ~250-450 ms. Bit-equal 95 %;
    PASS 85 %.
  - The result goes to the orchestrator either way (Kamen asked directly), and to NewIdeas.
  - **Result (Thu 21:57-21:58, runner 27, rc 0, 65 s, C 192 = 29.4 GB active; data/r50b_screen.txt): PASS. A warm
    16-row check at the 45 GB configuration costs ~200 ms on pool slots vs 0.6-1.6 s with the gather. Bit-equal.**

    | pass (order G S S G) | warm check + commit, windows 2-13 | misses per check |
    |---|---|---|
    | 1 gather | 597-704 ms | 39-121 |
    | 2 slot | 164-240 ms | 9-49 |
    | 3 slot | 159-231 ms | 14-58 |
    | 4 gather | 1382-1604 ms | 17-66 |

    - Medians (pre-registered): gather 1043.1 ms, slot 200.3 ms, saved 842.9 ms per 48-layer check.
    - Against gather's better pass alone (~650 ms) the saving is still ~450 ms (x3.2).
    - Logits are equal across all 4 passes for every window.
    - The gather arm doubled between pass 1 and pass 4 at equal misses, while the slot arm stayed flat. That fits
      the allocation reading: fresh ~330 MB scratches get slower as memory fills. The slot path has no scratch.
    - Pass 1 had ~2x the misses of the later passes (its pool came cold from the prefill). The pool persists across
      passes; G S S G puts one gather pass at each end.
    - For the route to 350 at 45 GB:
      - An exact 16-row check now costs ~200 ms warm at C 192: ~32 misses per check (~90 MB from SSD) plus 48
        per-layer id syncs plus compute.
      - With DFlash2's kept per check (code 9.57, math 7.00) that is ~44 / ~33 tok/s for the exact 16-row path
        (with ~15 ms of drafting), vs ~14 with the gather. Today's 45 GB line (MTP head, 2-7-row checks) is ~64 on
        code.
      - 350 needs ~35 ms per 16-row check at ~12 kept. The next levers on this path: the per-layer id sync, i.e. a
        sync-free 16-row gate (BigMoE's MAXK build), plus REF2's kernel savings (137 ms measured in 3z) where they
        still apply.
    - vs prediction: PASS (85 %) yes, bit-equal (95 %) yes. Saving 250-450 predicted, 450-843 measured (slot 200 ms,
      better than the 350-500 guess).

## R-50c (Kamen via the orchestrator, Thu 22:1x: profile the new ~200 ms check end to end before a sync-free gate is built; NewIdeas GO 22:08; Leap, pre-registered Thu 2026-10-29 22:18 EET): where the warm 16-row check's time goes at the 45 GB configuration
- **Why:** R-50's waste went unseen because the check's time was never broken down: the copy was predicted at ~60 ms
  and the allocations cost 6x that. The next lever should come from a measured breakdown.
- **Screen (scripts/r50c_screen.py, runner_r31.sh after 3q, ~2 min):**
  - Setup: C 192; REF2 + SB with ref2's defaults (slot call, commit phase); no drafter; R-50b's warm windows (/68
    REF2 row, sliding, commit keep 1); 2 passes of 14, windows >= 2 counted.
  - Per MoE layer-call (an instrumented copy of ref2's slot path, the same work):
    - sync = the id sync's wait: GPU work up to the router + latency;
    - ensure = CPU + SSD for misses;
    - build = slot map + MoE graph;
    - moe = the expert kernels + their sync;
    - between = Python building the next layer's dense graph.
  - Per check: wall, head eval, the tail after the last MoE call, the commit, and a bare sync's latency.
  - Derived: GPU busy ~ sync + moe + head - sync latencies. The bubble (GPU idle) ~ ensure + build + between + tail +
    the latencies.
- **Lines (NewIdeas):** PASS = bubble >= 80 ms per check -> BigMoE builds R-55 GATE16 (sync-free 16-row gate, REF2's
  slot kernel unchanged). KILL = bubble < 40 ms -> compute is the lever (REF2 kernels). BETWEEN otherwise.
- **Prediction:** GPU busy ~70-90 ms (dense ~18, experts ~35, attention / GDN / HC / head ~20-35); bubble ~90-120 ms
  (48 x ~2 ms ensure + build + Python); commit ~5-10 ms. PASS 65 %, BETWEEN 25 %, KILL 10 %.
- **Result (Thu 22:20, runner 31, rc 0, 26 s, C 192 = 29.4 GB active; data/r50c_screen.txt): BETWEEN. Bubble ~60 ms
  of a 185 ms check.** Medians over 24 warm checks, ms:

  | stage | ms per check | note |
  |---|---|---|
  | expert kernels (48 MoE calls) + their sync | 77.7 | ~1.36 ms GPU per layer; ~100 experts x 2.8 MB = ~13.4 GB per check at ~173 GB/s, near the bytes floor |
  | GPU up to each router + sync | 64.6 | attention / GDN / HC / dense / shared expert / router: ~52 GPU + ~12 sync latency |
  | ensure (CPU + SSD) | 22.7 | 44 misses per check (~123 MB, ~10-12 ms of SSD) + 48 ensure calls |
  | Python building the dense graph between layers | 10.2 | |
  | head eval | 2.6 | |
  | slot map + MoE graph build | 1.9 | |
  | unexplained | 1.6 | |
  | **check** | **181.5** | + commit 3.6 = 185.1 wall |

  - GPU busy ~120 ms. The bubble (GPU idle) ~60 ms = ensure 22.7 + Python 12.1 + 97 sync latencies x 0.256 ms
    (~25).
  - A sync-free 16-row gate (R-55 GATE16) could take ~60 ms at most, 185 -> ~125 ms (-32 %), and only if ensure's
    misses are prefetched too. NewIdeas' line calls that BETWEEN (build at >= 80).
  - The floor under the exact 4-bit 16-row check: experts ~48 ms at 280 GB/s for 13.4 GB, plus dense ~20 ms, about
    70 ms. 350 needs ~35 ms per check at ~12 kept. So the route needs fewer bytes per check (or more kept per byte),
    not only fewer syncs.
  - vs prediction: GPU busy 70-90 predicted, 120 measured (the experts at 78, not 35: experts_split on ~100 experts at
    16 rows is bandwidth-bound); bubble 90-120 predicted, 60 measured; commit 5-10 predicted, 3.6 measured.
    BETWEEN (25 %).

## S45 (Leap, pre-registered Fri 2026-10-30 00:10 EET; Kamen 10-29 21:1x: 45 GB news first, in tok/s at 45 GB and ms per check): the exact path's live speed at 45 GB

- **What:** STEP 3k's live loop in speed mode (`r40s3k_screen.py 192 256 speed`, runner 37 after the 3k re-run).
  - EvalSuite's REF harness at C 192, REF2 + SB with ref2's defaults (slot call, commit phase, SDPA classes).
  - DFlash2 L1-20 drafting, up to 256 tokens on the 6 3k prompts (2 code, 4 math). No AR, Part D or control.
- **Reports:**
  - tok/s per prompt, and the means for code and for math;
  - per cycle (medians): drafter ms, and check + commit ms;
  - misses and SSD read ms per check;
  - every token vs EvalSuite's REF2 row (a free exactness check at C 192);
  - footprint now, lifetime peak and MLX peak.
- **A measurement, no lines.**
- **Prediction:**
  - median cycle 250-330 ms (R-50b: a warm check ~200 ms at C 192, plus the drafter);
  - code ~40 tok/s, math ~20 tok/s;
  - every token equal.
- **RESULT (Fri 00:24-00:26, runner 37; data/s45_speed.txt): code 25.1 tok/s, math 19.2 tok/s at C 192, exact.**

  | prompt | kept | tok/s | cycle (median, ms) | drafter (ms) | check + commit (ms) | misses per check | SSD reads per check (ms) |
  |---|---|---|---|---|---|---|---|
  | HumanEval/68 | 15.41 | 27.1 | 415.5 | 15.6 | 396.8 | 779 | 194.1 |
  | prealgebra/1640 | 5.31 | 18.2 | 280.6 | 12.5 | 266.1 | 414 | 110.7 |
  | HumanEval/91 | 8.09 | 23.1 | 302.7 | 13.4 | 292.8 | 563 | 144.4 |
  | algebra/694 | 7.73 | 20.3 | 359.6 | 12.4 | 350.1 | 644 | 162.2 |
  | algebra/1282 | 6.89 | 21.7 | 295.6 | 12.9 | 286.3 | 483 | 125.9 |
  | gsm8k/708 | 6.16 | 16.5 | 338.3 | 12.7 | 317.2 | 531 | 161.6 |

  - Median cycle 320.5 ms; kept 8.27 per check. Every token == EvalSuite's REF2 row at C 192 too.
  - Memory: footprint now 32.0 GB, lifetime peak 36.3 GB, MLX peak 35.9 GB. The "45 GB" configuration peaks at
    36 GB on these prompts.
  - Against the prediction:
    - cycle 320 ms, inside 250-330: right;
    - math 19 vs ~20: right;
    - code 25 vs ~40: wrong. HumanEval/68 keeps 15.4 per check but its checks cost 397 ms (779 misses).
  - **Where the cycle goes:**
    - The drafter takes ~13 ms; check + commit takes 266-397 ms.
    - R-50c's warm check (sliding by one token, 44 misses) took ~185 ms.
    - The live loop moves ~8 tokens per check and misses 413-779 experts per check, which costs 111-194 ms of SSD
      reads. That is about the whole gap to R-50c.
  - **So at 45 GB the misses are now the biggest exact-safe piece: ~1/3 to 1/2 of the check.**
    - Levers: a bigger C (~9 GB of headroom to 45 GB is ~+64 slots per layer at ~135 MB per slot), and overlapping
      the reads with compute (prefetch the next layer's experts from a lookahead router). Neither changes numerics.
    - BigMoE's own look-ahead (BIGMOE_PREFETCH) lives only on the gated path (<= 40 ids). 16-row checks reach
      ext.ensure synchronously, with no look-ahead, and EvalSuite refuses that env anyway. The overlap lever is
      BigMoE's engine work; proposed to NewIdeas and sent to BigMoE.

## S45b (Leap, pre-registered Fri 2026-10-30 00:45 EET; the orchestrator 00:3x: measure past 45 GB if it cuts misses): misses and speed vs C on the exact path

- **What:** S45's harness unchanged at C 224, 256 and 288 (runner 38). Each run is gated on memory_pressure's free
  share: 66 / 69 / 75 %. A C is skipped if its gates stay shut ~46 min. The swap guard (+2 GB) stays.
- **Expected peaks:** lifetime ~36.3 GB at C 192, plus ~135 MB per slot, so ~40.6 / 44.9 / 49.2 GB.
- **Prediction:** misses per check fall as C^-1.5, about -21 / -35 / -46 % vs C 192 (mean 569). SSD ms falls with them
  (C 192: mean ~150 ms per check). tok/s rises about +8 / +19 / +28 % over C 192 (code 25.1, math 19.2). Every
  token stays equal.
- **A measurement, no lines.** If C 256 (~45 GB) gives >= +15 % tok/s exactly, it goes to EvalSuite and BigMoE as a
  config candidate for the 45 GB line (their call), with its footprint.
- **NewIdeas GO (00:55, seen after the run):** C 192 / 224 / 256 only, and C 256 only with >= 50 GB free. My gates
  were looser (69 / 75 %, ~44 / 48 GB), and the runner went on to C 288 after the first trip.
- **RESULT (Fri 00:55-01:32, runner 38): C 224 measured; C 256 and C 288 stopped by the swap guard.**

  | C | lifetime peak | misses per check (mean) | SSD ms per check (mean) | median cycle | tok/s code | tok/s math | all-6 mean |
  |---|---|---|---|---|---|---|---|
  | 192 (S45) | 36.3 GB | 569 | 149.8 | 320.5 ms | 25.1 | 19.2 | 21.2 |
  | 224 | 40.5 GB | 432 (-24 %) | 115.7 (-23 %) | 270.9 ms (-15 %) | 24.7 | 21.9 | 22.8 (+8 %) |
  | 256 | stopped 17 s into the load: swap 11.3 -> 14.1 GB | | | | | | |
  | 288 | stopped 17 s into the load: swap 14.0 -> 18.3 GB | | | | | | |

  - Every token at C 224 == EvalSuite's REF2 row.
  - HumanEval/91 got slower at C 224 (21.5 vs 23.1 tok/s) although its median cycle fell 12 %. Swap grew 1.5 GB
    during that run, which may explain it.
  - Against the prediction: misses -24 % (predicted -21 %): right. All-6 tok/s +8 %: right. Code alone: not up.
  - Lesson (memory: swap-eats-the-disk): the free-share gate overstates room for one big load, and a sweep must stop at
    the first trip. Swap is 18.2 GB after this; the WATCH line is 20.
  - What remains: at C 224 the SSD reads are still ~116 ms of a ~271 ms cycle. The look-ahead prefetch on the
    16-row path is NewIdeas' R-57 for BigMoE.

## R-58 REF2-T (NewIdeas GO 10:20, from Leap's 10:0x answer; the tier plan 4433333333 is Kamen-adopted, STATE.md:31): REF2 on BigMoE's rank-tier engine

- **Desk port (Fri 16:1x, ref2.py; CFG["tiers"], on by default, inert without BIGMOE_TIER):**
  - BigMoE's own _sync_call runs every tier call with > 40 ids (5-16-row checks, big commit replays) as the exact
    4-bit prefill (tier.prefill_call). Its tiers apply to 1-4-row calls only, through the gated tier_call.
  - REF2-T's _tier_slots runs 5-16-row calls at the plan's tiers on pool slots, as tier_call does: base-pool (MSB-3)
    slots for every rank, LSB-sidecar slots for ranks 0-1, MoonShots' fused 3c kernel (tierglu.tier_glu).
  - 3c computes each (row, rank) pair in its own threadgroups, and its docstring states the pair arithmetic is
    subset-invariant. So rows should be batch-invariant.
- **INV-T (pre-registered 16:20; scripts/r58_invt_screen.py, runner 43 after runner 42, ~4 min, C 192):** STEP 4a's
  method with tiers on.
  - Cases: HumanEval/68 (keys 383) and 702 (keys 536, key buckets).
  - Counts: tier calls on the AR's gated path and on the check's REF2-T branch.
  - Control: the branch off, so the check's experts run exact 4-bit. It must differ.
  - **PASS** = 16/16 rows and every layer equal in both cases, with tier calls on both paths, AND the control differs.
    **KILL** = any row differs.
  - Prediction: PASS 75 %. The risk is a tier path state I haven't read, such as the gate's spin / first-pass logic.
- **INV-T RESULT (Fri 17:32-17:33, runner 43; data/r58_invt_screen.txt): PASS. REF2-T is batch-invariant on the tier
  engine.** 50 s.

  | case | logits rows bitwise | layers | tier calls: AR (gated tier_call) / check (REF2-T slot branch) |
  |---|---|---|---|
  | B HumanEval/68, keys 383 | 16 / 16 | every layer equal | 768 / 48 |
  | A 702, keys 536 (key buckets) | 16 / 16 | every layer equal | 768 / 48 |
  | control: tier branch off (the check's experts at exact 4 bits) | 0 / 16 | first difference at layer 0 | 768 / 0 |

  - Prediction PASS 75 %: right.
  - BigMoE runs the gate arm as a copy (~/BigMoE/scripts/r58g_invt_screen.py: BIGMOE_MAXK 160 + build_next, GATE_N 160).
    It checks that their sync-free 16-row tier gate gives the same bits.
- **S45-T (pre-registered Fri 17:40; scripts/r58_s45t_screen.py, a copy of r40s3k_screen.py with BIGMOE_TIER=4433333333
  / GLU 3c; runner 45 after runner 44):**
  - `224 256 speed 16,0.3,16 interleave`: a discarded warm-up, then per prompt tiers block 16, tiers t 0.3, tiers
    block 16, on S45's 6 prompts at C 224.
  - The reference is the REF2-T AR, so the stored 4-bit REF2 rows will not match. Exactness = tokens equal across the
    three runs of each prompt.
  - **PASS (NewIdeas' line)** = tiers block 16's all-6 wall-clock mean >= +10 % over 4-bit block 16 in S45c-2 (both
    after a warm-up), and tokens equal across the runs. **KILL** < +4 %.
  - Also reported: tiers + t 0.3, the stacked lever, vs 4-bit block 16.
  - Prediction: tiers block 16 +12 %; tiers t 0.3 +30 % over 4-bit block 16.
  - **Moved (Fri 18:3x, NewIdeas):** S45-T runs on BigMoE's gate, the path the line would adopt. BigMoE's R-58G PASS:
    their sync-free 16-row tier gate == _tier_slots bitwise (16/16 rows + layers at 383 / 536 keys).
    - Sizing per MoonShots' M-08 (~/MoonShots/data/split45.txt): C_m ~257-269 / C_l 96-128 at REF2's C 224 bytes. A
      small LSB pool (C_l 32) would swamp the reads (+25-145 %).
    - My slot-path S45-T is dropped; it was never queued. I suggested BigMoE add the tiers + t 0.3 arm.

## R-61 / R-62: the check's dense kernels after R-59 (Fit12's profile, Sat 02:2x; NewIdeas: fold any bit change into the next reference bundle with REF2-T, one same-scores test)

R-59 at C 224, warm 16-row REF2b check, ms per check (x its bytes floor at 280 GB/s):
- GDN 24.1 (5.7x, 36 layers);
- QSA attention 11.6 (9.1x, 12 layers);
- HC reads 7.95 / 7.22 (3.5x / 3.2x);
- non-expert total 53.1.

- **R-62 ATT16 (pre-registered Sat 02:4x):** one SDPA call per QSA layer for the 16-row check instead of MTPLX's 12
  head-chunked calls.
  - MTPLX chunks whenever q_len x GQA > 32, to stay on MLX's vector kernel. But MLX 0.32.2 sends q_len > 8 to its full
    (NAX) kernel, which takes GQA natively, so for 16 rows the chunking only multiplies dispatches.
  - Test (synthetic, the model's shapes; key lengths 383 / 536 / 1024 with REF2b's buckets): one call vs 12 chunks, ms
    and bitwise.
  - The full kernel computes each (head, query tile) on its own, so the bits should not change.
  - PASS = bit-identical AND <= 1/3 of the chunked time: exact-safe, adopt without a reference change.
  - BETWEEN = faster but bits differ: goes in REF2-T's bundle.
  - KILL = not faster.
  - Prediction: bit-identical 70 %, ~0.25 ms per layer (11.6 -> ~3 ms per check).
- **R-61 GDN16-125:** L1-15's GDN kernel program for the 35B (prework and gated norm fused, the recurrence with its
  gates inside, the state kept on-chip across the check's rows; 35B pass 44 -> 37 ms) ported to the 125B's GDN (Hk 16,
  Hv 48, Dk = Dv = 128).
  - Step 1: per-piece times at 16 rows on synthetic weights (projections, conv, recurrence, norm / gates), to see where
    the 24.1 ms goes.
  - Step 2: the kernel, bitwise vs today's GDN path.
  - PASS >= -10 ms per check; KILL < -4 ms. Bit-identical = exact-safe; else REF2-T's bundle.
- Both are small synthetic screens (< 2 GB), for a shared GPU by day (night order: big runs first).
- **K-60 RESULT (Sat 07:30, runner 51, shared GPU, rough; data/k60_experts_bw.txt): the experts kernel has ~29 % time
  headroom.**
  - A check's call: 16 rows x top-10 over 100 experts gives 76 distinct, 210 MB per layer call.
  - experts_split (REF2's path) takes 1.40 ms = 150 GB/s; a plain uint4 streaming read of the same bytes takes 0.99 ms
    = 210-212 GB/s.
  - At the read ceiling the experts would cost ~47 instead of ~67 ms per check: ~-20 ms, ~13 % of the stack's 154 ms
    cycle at C 224.
  - Exact-safe if qmmg's per-output accumulation order is kept: only memory access changes, e.g. more loads in flight.
    A candidate kernel row, after R-62 / R-61.
> (a quote from the project owner, kept private)
  hold_guard.sh). Reading, planning and writing only.
- **R-62 RESULT (Sat 19:58, runner 52, Kamen's 19:55-20:45 hour; shared GPU, rough; data/r62_att16.txt): BETWEEN.** One
  SDPA call per layer is 2.1-2.8x faster than MTPLX's 12 head chunks, but bit-identical only below 512 keys.

  | key length | bitwise equal | 12 chunks (ms per layer) | one call | x 12 layers per check |
  |---|---|---|---|---|
  | 383 | yes | 0.732 | 0.297 | 8.8 -> 3.6 ms |
  | 1024 | **no** | 0.707 | 0.254 | 8.5 -> 3.1 ms |
  | 2048 | **no** | 0.726 | 0.346 | 8.7 -> 4.2 ms |

  - The PASS line needed bit-identity AND <= 1/3 of the time. Got 0.36-0.48x, with bits that differ past 512: the same
    length-dependent NAX behaviour as 4b, now head-count-dependent too.
  - ~-5 ms per check (~3 % of the stack's 154 ms cycle).
  - It goes into the next reference bundle: REF2's AR and the check would both take the one call, and the key-bucket
    logic keeps them consistent. Prediction "bit-identical 70 %": wrong past 512.
- **R-61 step 1 RESULT (Sun 05:11, runner 54, shared GPU, rough; data/r61_gdn_profile.txt): no single piece dominates.**
  - One GatedDeltaNet built from the model's config: random 4-bit g64 projections, bf16, REF2 on, no cache. Each piece
    was timed alone with an eval after it.

    | piece (ms per layer) | 16 rows | 1 row |
    |---|---|---|
    | P: the 4 input projections (qmm16c, padded) | 0.740 | 0.590 |
    | C: conv chain (concat, conv1d, silu, split, l2norm x 2, scale) | 0.453 | 0.370 |
    | R: gated_delta_update (16 sequential steps) | 0.583 | 0.273 |
    | O: gated norm + out_proj | 0.336 | 0.363 |
    | W: the whole mixer call | 2.906 | 1.203 |

  - Timed alone, every launch waits on a sync, so the absolute numbers are ~4x R-59's in-model 0.67 ms per layer
    (24.1 / 36).
  - The shares are what count: P 35 %, R 28 %, C 21 %, O 16 %. The stock 16-row chain is ~35 small kernels per layer
    (the padded projections with their pad / slice ops, ~10 for the conv chain, the gate math around the recurrence,
    ~5 for the norm), ~1,260 per check. That fits R-59's 5.7x the floor.
  - **Step 2:** L1-15's 35B GDN16 program, ported. One prework kernel (conv + silu + l2norm + scale for any S <= 16),
    the recurrence with the gates inside, one gated-norm kernel; qmm16c projections stay.
    - Estimate ~-15 to -20 ms per check (~10-13 % of the stack's cycle).
    - Every op is row-independent, so the same kernels at S = 1 (the AR) keep REF2's batch invariance. The rounding
      differs from the stock chain, so it is a reference change, for the next bundle with R-62.
  - **Better: MoonShots already has EXACT fused GDN pieces for the 125B** (patches/planes_engine/gdn_exact.py, Oct 21;
    FINDINGS: bitwise == MTPLX's eager chain at S 1-8, model-level 4-row verify logits bitwise; -13 kernels per GDN layer,
    ~-3.5 ms per pass).
    - gdn_prework_exact: concat + conv1d + silu + l2norm + scale + the conv state in one dispatch, every rounding
      replicated. gated_norm_exact: SigmoidRMSNormGated in one dispatch.
  - **Tensor check at 16 rows (Sun 05:2x, shared lock; MoonShots' tests copied to scripts/ms_copy/, S up to 16): PASS.**
    The prework has 0 mismatches in q / k / v / state at S 1-16 and every scale; the gated norm has 0 of 98,304 at S 16.
    So for 16-row checks they are EXACT-SAFE (REF2b / REF2-T bits unchanged), no reference change.
  - **Step 2a (next):** route the 16-row path to them under REF2. Lift MTPLX's S <= 6 cap on the rows branch to 16, set
    MTPLX_FUSED_CONVNORM_VERIFY, install MoonShots' patch, and swap SigmoidRMSNormGated for gated_norm_exact.
    - Then one live-loop run: every token / logits row == without it, plus the check-time gain.
    - It is exact-safe, so it joins the exact-safe stack (gate + truncation) under one every-token check (FAST
      INTEGRATION rule 2).
    - The bigger GDN pieces (the padded projections' glue, the recurrence's gate math) come after.
  - **R-61a GDNX (pre-registered Sun 05:2x; ref2.py CFG gdn_exact, default off; runner 55):**
    - With the switch on, REF2 routes 1-16-row GDN calls to MoonShots' exact prework kernel (MTPLX's fused branches,
      the rows cap lifted from 6 to 16) and SigmoidRMSNormGated to gated_norm_exact.
    - Live A/B at C 224, CAP 192, quiet lock, NO-SWAP >= 44.5 GB: warm-up, then per prompt gate + t 0.3, gdnx gate +
      t 0.3, twice (A-B-A-B), on S45's 6 prompts.
    - Lines:
      - tokens equal across all runs (else FAIL);
      - adopt as REF2's default if the median cycle is not worse (exact-safe and fewer kernels);
      - its +5 % speed line is informative only. MoonShots' estimate is ~-3.5 ms of a ~154 ms cycle (~+2 %).
    - Prediction: exact; median cycle -2 to -5 ms.
    - **RESULT (Sun 05:24-05:32, runner 55, quiet lock; data/r61a_gdnx_c224.txt): exact, and a small gain. ADOPTED as
      REF2's default.** Tokens are equal across all 4 runs of every prompt.

      | run (C 224, gate + t 0.3) | median cycle | steady state | wall-clock |
      |---|---|---|---|
      | 0: without | 163.0 ms | 42.27 | 33.82 |
      | 1: gdnx | 157.6 ms | 43.97 | 36.42 |
      | 2: without | 156.1 ms | 43.84 | 29.91 (an outlier run) |
      | 3: gdnx | 155.5 ms | 44.16 | 37.23 |

      - The measure that counts: median cycle 159.6 -> 156.6 ms (-3.0 ms, -1.9 %); steady state 43.1 -> 44.1 (+2 %).
        This matches MoonShots' ~-3.5 ms and the prediction.
      - The printed wall-clock verdict ("+14-17 %, PASS") comes from run 2's low wall-clock (29.9 at an unchanged
        steady state), not from GDNX.
      - Per its pre-registered rule (exact and not slower), it is adopted.
        - ref2.py CFG gdn_exact = True by default.
        - The kernels come from a frozen, byte-identical copy (~/Leap/third_party/ms_gdn, README), so a later edit in
          MoonShots' tree cannot change REF2.
        - No bits change: REF2b / REF2-T rows stay valid.

## REF2c bundle + the experts kernel (NewIdeas 06:1x: (A) first; Leap)

- **K-60-T (Sun 06:09, runner 56, shared GPU, rough; data/k60t_tier_bw.txt): the 3c tier kernel has the same headroom
  as qmmg.**
  - It is the 45 GB default's experts path under REF2-T: MSB-3 base for every rank, LSB-1 sidecar for ranks 0-1.
  - A check's call (16 rows x 10 ranks, 76 base + 24 sidecar slots = 182 MB): tier_glu takes 1.209 ms = 150 GB/s,
    58.0 ms per check. A plain read of the same bytes takes 0.83-0.85 ms = 215-220 GB/s.
  - So ~31 % time headroom, ~-18 ms per check (~12 % of the cycle), on the default path.
  - Exact-safe if 3c's per-output accumulation order stays: only memory access changes (loads in flight, layout).
    That makes it the biggest check-time piece, and it could ship alone as exact.
- **Plan:**
  - (A1) 3c bandwidth, exact-safe: find the limiter in tierglu's kernels A (gate / up + SiLU) and B (down).
  - (A2) the REF2c bundle for the bit-changing pieces (R-62 attention, GDN in-proj fusion, gates in the recurrence,
    FMA dequant), one regen + same-scores test with REF2c-T.
- **K-61 TIERGRP (Sun 06:2x, shared GPU, rough; scripts/tierglu_grouped.py, k61_tiergrp_test.py, data/k61_tiergrp.txt):
  KILL.**
  - The lines were set in the session plan before the run (PASS = bitwise equal to tier_glu AND >= 25 % faster per layer
    call; KILL = not faster); they are logged here after it.
  - Design: 3c with the (row, rank) pairs grouped by (base slot, tier, sidecar slot); each weight chunk is loaded once
    and applied to every pair of the group in 3c's own per-pair order. Groups are capped at 2, 4 or 16 pairs.
  - Bitwise PASS: max |diff| 0 for M 1 / 4 / 16, three routings, both plans, ld 0 / 1 and caps 2 / 4 / 16.
  - Slower everywhere at M 16:

    | union | distinct experts | tier_glu | grouped |
    |---|---|---|---|
    | U 100 | 85 | 1.107 ms | 0.71-0.86x |
    | U 70 | 66 | 1.085 ms | 0.73-0.85x |
    | U 24 | 24 | 1.049 ms | 0.87-0.91x |

  - This agrees with MoonShots' 10-22 microbench (~/MoonShots/wide/MICROBENCH.md): 3c's cost is arithmetic per pair
    (~6.5 us each), not DRAM. tier_glu hardly moves from 85 to 24 distinct experts (1.107 -> 1.049 ms).
  - So K-60-T's "150 vs 218 GB/s" was not a bandwidth gap: 3c is ALU-bound.
  - The bit-identical route is closed: MoonShots' grouped3c ran -10-20 %, their mr3c 1.6-2x slower, mine 0.71-0.91x.
- **K-62 NAX3 (pre-registered Sun 06:2x EET): a dequantize-once tier kernel on the M5's matrix units.**
  - Probe (scripts/nax_probe.py, 06:2x): mx.fast.metal_kernel runs Metal 4's mpp::tensor_ops::matmul2d (16 x 32 x 16,
    bf16 in, fp32 accumulate) with MLX's BaseNAXFrag element map.
    - It is exact against numpy.
    - A row's output is bitwise the same across 6 different sets of other rows.
  - Design:
    - One threadgroup per (group of <= 16 pairs with the same base slot, tier and sidecar slot) x 32-column tile.
    - A = the group's x rows (bf16, padded to 16 rows).
    - B = the raw codes q as bf16, which are exact small integers (tier 2: q & 6; tier 4: 2q + lsb).
    - fp32 accumulators per 64-value group: C += s * (X Q_g) + b * xsum_g.
    - So each expert is dequantized once per check, not once per pair.
  - It is not bit-identical to 3c, so it is a REF2c item. It needs:
    - batch invariance: a pair's bits must not depend on its group's other pairs, the group size or its row in the
      tile;
    - the bundle's same-scores test.
  - Lines:
    - PASS = INV bitwise AND >= 1.3x faster than tier_glu at M 16, U 70 (a natural-like union) AND an error against an
      fp64 reference no worse than 1.5x 3c's (max and mean).
    - KILL = < 1.1x at U 70, or INV fails.
  - Expected: ~0.6-0.7 ms vs 1.085 ms (~-20 ms per check, ~13 % of the cycle). Per-pair cost also falls, so wider checks
    (trees, longer drafts) get cheaper per row.
- **K-62 result (Sun 06:3x, shared GPU, rough; scripts/tierglu_nax.py, k62_nax_test.py; data/k62_nax.txt v1,
  k62b_nax.txt v2): KILL at the pre-registered line (1.05x < 1.1x at U 70). A lead for wide checks.**
  - Accuracy: closer to fp64 than 3c in both versions. Max |err| / RMS 2.3e-2 vs 3c 3.3e-2; mean 3.2e-3 vs 4.2e-3.
    70.8 % of outputs differ from 3c's bits (bf16 rounding of h and y).
  - INV: PASS bitwise in both versions. The probe row gave the same bits alone, as row 0 or row 9 of 16 with other
    companions, and in a call where all 16 rows route alike (a group of 16).
  - Speed at M 16:

    | union | groups | tier_glu | v1 (per-value decode) | v2 (k-permuted, 2 values per 32-bit op) |
    |---|---|---|---|---|
    | U 100 | 101 | 1.087 ms | 0.75-0.83x | 0.92x |
    | U 70 | 87 | 1.075 ms | 0.84-0.94x | **1.05x** |
    | U 24 | 40 | 1.044 ms | 1.64-1.81x | **2.07x** |

  - Where v2's time goes at U 70: the MMA path alone (codes held constant) takes 0.52 ms; decode alone (no MMA) takes
    0.90 ms; both together take 1.03 ms. They barely overlap.
    - The 16-row MMA tile carries ~1.8 real rows per group, so the MMA path costs ~6 us per group even with free
      decode.
    - The DRAM floor for these bytes is ~0.62 ms.
  - Why it is a lead: the cost scales with groups, not pairs. At 2.5+ pairs per expert it wins 2x (U 24).
    - Wide checks give that sharing: S 32 has ~3.2 pairs per distinct expert (MoonShots' union table), and budget
      routing more.
    - So the kernel matters once checks get wider. Unexplored levers: B as uint8, or as uint4 from threadgroup memory
      (MPP supports bf16 x ui8 cooperative and bf16 x ui4 from memory); fewer registers so decode and MMA overlap.
  - Also new and reusable: custom kernels can drive the M5's matrix units with batch-invariant bits, and REF2c can use
    that elsewhere.
- **K-62c-e (Sun 06:3x-06:47, shared GPU, rough; NewIdeas told 06:40): tuning the lead. NAX4 is BETWEEN: 1.21-1.28x at
  U 70, 2.1-2.3x at U 24. A REF2c candidate.**
  - Probes:
    - Matrix-unit peak (scripts/nax_tput.py, nax_shapes.py): ~1.3 G ops/s of 16 x 32 x 16 (~10.5 T MAC/s) on register
      operands.
    - Integer decode of B overlaps fully with the MMAs; marshalling A with conversions costs ~25 %.
    - Shapes that build: 16 x 32 x 16, 16 x 32 x 32, 32 x 32 x 16. There is no 8-row shape, so the 16-row padding is
      fixed.
    - MPP takes B as packed uint4 straight from threadgroup memory (bf16 x uint4b_format -> fp32; K must be a multiple
      of 32). scripts/nax_u4_probe.py: exact. Layout: extents (k, n), strides (1, row nibbles), low nibble = even k.
      MPP's get_multidimensional_index confirms the A and C element maps.
  - c (prefetching the next group's codes): no gain, 0.98-1.02 ms. Load latency is not the limit.
  - **e = NAX4 (scripts/tierglu_nax4.py; data/k62e_nax4.txt):**
    - Lane r decodes weight row n0 + r per 64-value group: 6 aligned words, then 3c's spread8 to nibbles (~1.5 ops per
      value). The nibbles go to threadgroup memory and two 16 x 32 x 32 MMAs per group read them.
    - Accuracy is the same as K-62 (better than 3c against fp64). INV is PASS bitwise. Stage sizes 1 / 2 / 5 give equal
      bits.

    | union | groups | tier_glu | NAX4 | (v2) |
    |---|---|---|---|---|
    | U 100 | 101 | 1.097 ms | 1.024 (1.07x) | 0.89x |
    | U 70 | 82-87 | 1.07-1.09 ms | **0.84-0.90 (1.21-1.28x)** | 1.01x |
    | U 24 | 40-44 | 1.04-1.05 ms | **0.45-0.50 (2.1-2.3x)** | 1.97x |

  - Where NAX4's 0.897 ms goes at U 70 (82 groups):
    - kernel A (gate / up) 0.575 ms, at ~70 % of matrix-unit peak (floor 0.40);
    - kernel B (down) 0.273 ms, ~73 % (floor 0.20);
    - the two rowsum64 dispatches ~0.03 ms each.
    - Codes all read from one row (diag 3, no DRAM footprint): 0.70 ms. So scattered code reads cost ~0.15 ms.
    - Decode stages of 2 or 5 groups are slower (occupancy).
  - Floor: ~0.60-0.65 ms at U 70 (the 16-row MMAs and the DRAM bytes are equal), i.e. up to ~1.65x. Left: coalesced
    code staging, and fusing the rowsums (hsum in kernel A's epilogue, xsum via shuffles).
  - Per check at 45 GB: ~-10 to -12 ms (1.08 -> 0.85-0.9 ms x 48), ~7 % of the 155 ms cycle. It grows with check width
    (U 24-like sharing: ~-28 ms).
  - Lines: PASS needed >= 1.3x at U 70, KILL < 1.1x. Got 1.21-1.28x: BETWEEN. As a bit change it can only go in with
    the REF2c bundle (one regen + same-scores test). Integration needs:
    - both of REF2-T's paths on it (1-4-row gated tier_call and 5-16-row _tier_slots), so the bits do not depend on
      call size;
    - a GPU grouping for BigMoE's sync-free gate path (MoonShots' csr kernel, keyed by slot and tier);
    - a check on BigMoE's real pools and routing first.
- **K-62f / g (Sun 06:4x-06:5x, shared GPU, rough): NAX4 on REAL data. 16 rows ~1.4x (1.07-1.97x), 32 rows ~1.9x (1.30-2.77x);
  ~3.4x MORE accurate than 3c against fp64.** (scripts/k62f_real.py, k62g_acc_real.py; data/k62f_real_L{6,20,40}.txt)
  - Setup: one real 125B layer's v3 pools (records3: MSB-3 + LSB-1 of all 512 experts, slot = expert id; MoonShots'
    loader). Routing is real: Breakthrough's expA 125B decode traces, top-10 of S consecutive tokens, 12 passes per
    trace. Random x.

    | layer | S 16: code / chat / math / bulgarian | S 32: code / chat / math / bulgarian |
    |---|---|---|
    | L6 (75-89 distinct at S 16) | 1.19 / 1.08 / 1.07 / 1.15x | 1.50 / 1.34 / 1.30 / 1.39x |
    | L20 (49-69) | 1.27 / 1.40 / 1.36 / 1.69x | 1.72 / 1.89 / 1.82 / 2.07x |
    | L40 (40-68) | 1.31 / 1.94 / 1.46 / 1.97x | 1.81 / 2.77 / 1.91 / 2.70x |

  - tier_glu is 1.16-1.22 ms at S 16 and 2.28-2.38 ms at S 32. INV PASS in all 24 cells (a pass's row 0 alone == inside
    the pass, bitwise).
  - Accuracy, L20 S 16, against fp64 from the same dequantized weights, all four traces:
    - 3c: mean |err| 1.23-1.26e-2 x RMS, max 9.2e-2 to 1.05e-1, median relative 1.6e-2.
    - NAX4: mean 3.6e-3, max 2.5-3.4e-2, median relative 4.5e-3.
    - NAX4's error is smaller in 76 % of outputs and larger in 13 %.
    - Likely cause: 3c copies MLX's qmv load_vector, which sums the x values in bf16 (`x[i] + x[i + 1] + ...` on
      device bfloat) before the sum * bias term. NAX4 sums them in fp32 (rowsum64).
  - Per check at 45 GB (S 16, 48 layers): 3c ~57 ms -> NAX4 ~40 ms, ~-16 ms (~10 % of the 155 ms cycle, ~+11 % tok/s).
    That is an estimate from 3 layers, not measured end to end.
    - Wider checks get much cheaper: S 32 costs ~2.3 ms per layer on 3c vs ~1.25 on NAX4 (~-50 ms per check).
  - Not BIG yet: an end-to-end number (REF2c-T live loop) comes first. Next: integration under a ref2 switch.
    - The slot path (_tier_slots, host ids) is straightforward.
    - The 1-4-row gated tier_call and the sync-free gate need a GPU grouping (MoonShots' csr kernel keyed by slot and
      tier) so every call size uses the same kernel.
- **K-62h (Sun 06:5x, shared GPU, rough; scripts/k62h_gpugrp.py): the pair grouping on the GPU works, but small checks
  lose.**
  - tierglu_nax4.gpu_grp is one threadgroup that ranks pairs by (slot, tier, sidecar slot) and cuts groups of <= 16. It
    has no host sync; the grid is sized for P groups and the empty ones exit. The LP1 rule is in both the kernels and the
    key.
  - Bitwise == the host grouping in every cell (L20, 4 traces, S 1 / 3 / 16 / 32, 6 passes each). It costs
    ~0.02-0.035 ms per call.
  - Small calls are SLOWER than 3c: S 1 0.105 -> 0.18 ms (0.57-0.60x); S 3 0.25 -> 0.30 ms (0.79-0.86x). S 16 is
    1.36-1.61x and S 32 1.79-2.03x with the GPU grouping.
    - Why: a 1-pair group still pays the full 16-row MMA (~7.4 us), about 3c's per-pair cost, plus 4-5 dispatches.
  - For REF2c every call size must use NAX4, because the bits differ from 3c. With L1-14 truncation (t 0.3, ~9.8 rows per
    check on average, 1-16), the end-to-end gain is below the 16-row ~1.4x. Rough at ~10 rows: ~-11 ms per check. The
    live A/B decides.
- **K-62L (pre-registered Sun 06:58 EET; runner 57, quiet lock, night mode): NAX4 in the 45 GB default's live loop.**
  - ref2.py CFG tier_nax4 (off by default): bigmoe.tier._tier_glu -> NAX4 with the GPU grouping, so tier_call (1-4 rows
    and gated verify rows) and _tier_slots all take one kernel. It refuses BIGMOE_OOO.
  - Harness: scripts/k62_live.py, a copy of BigMoE's r57_screen.py (their R57_PRESET 45 settings: REF2-T 4433333333 / 3c,
    R-55 gate, L1-14 truncation t 0.3, C 224, DFlash2, CAP 256).
    - Arms gate+t (3c) vs gate+t+nax (NAX4, prefill included), ABBA per prompt, 4 of S45's prompts, warm-up discarded.
    - NAX4's tokens are compared only across its own runs (determinism).
  - Lines:
    - PASS = median cycle -8 % or better AND steady-state tok/s +8 % or better;
    - KILL = cycle better by less than 3 %;
    - else BETWEEN.
  - Prediction: cycle -5 to -10 %, steady +5 to +11 % (truncated checks ~10 rows, where NAX4 is ~1.2-1.3x on the
    experts, ~35-40 % of the cycle). PASS 35 %, BETWEEN 45 %, KILL 20 %.
  - The same-scores test (REF2c bundle) and an exactness check (NAX4's live loop == its own AR) come only after a PASS
    or BETWEEN.
  - **Runs:**
    - Runner 57 (quiet, C by memory) queued 06:58, then stopped at 07:05. The quiet grant waits while any other session
      uses >= 80 % CPU, and Breakthrough's 1-core job (99 %) held it. Its ticket also blocked this session's small GPU
      screens.
    - REF2-T's lifetime peak at C 224 is 38.0 GB (BigMoE's S45-T B run; 33.4 GB is MLX's active memory after load).
      So C 224 needs >= 42 GB NO-SWAP under a quiet lock, and 46 under a shared one.
    - Runner 58 (07:06): the same A/B at **C 160 on a shared lock**, rough (peak ~29.6 GB + 8 <= 38.3 available). A
      quiet confirmation at C 224 follows if it shows a gain.
    - NAX4's default decode stage was gs 2 (meant 1; gs 1 is ~5 % faster at U 70). Bits are equal across gs, so the
      K-62f/g/h numbers and runner 58 understate NAX4 slightly. The default is now gs 1.
- **K-62L RESULT (Sun 07:06-07:09, runner 58, SHARED GPU, C 160, rough; data/k62l_c160_screen.txt): KILL by its line.**
  - Median cycle 170.0 -> 179.7 ms (+5.7 %); steady-state tok/s 37.9 -> 38.6 (+1.8 %); check + commit 161.1 -> 169.4 ms.
    Line: PASS cycle -8 % AND tok/s +8 %; KILL cycle better by < 3 %.
  - The switch was active only in the NAX4 arm, and tokens were equal across the runs of each arm. Peak 29.7 GB.
  - Per prompt (steady tok/s 3c -> NAX4; median cycle):

    | prompt | steady tok/s | median cycle | misses per check |
    |---|---|---|---|
    | HE/68 (kept 15.4) | +0.4 % | 305 -> 302 ms | 1094-1294 vs 1082-1108 |
    | prealgebra/1640 (kept 5.2) | -1.4 % | 135 -> 138 ms | 281-350 vs 297-298 |
    | HE/91 (kept 7.0) | +0.9 % | 155 -> 154 ms | 426-449 vs 434 |
    | algebra/694 (kept 7.4) | +2.7 % | 185 -> 205 ms | 509-590 vs 619-635 |

  - Why no gain:
    - (1) At C 160 the loop is SSD-bound: demand SSD 57-200 ms per check, 280-1300 misses. With the sync-free gate, the
      GPU waits on loads, so expert compute is mostly off the critical path.
    - (2) NAX4 is a different reference. Its tokens differ slightly from 3c's (kept 7.43 vs 7.40 on 694), so the
      routing and the misses differ; algebra/694 had +23 % misses in the NAX4 runs.
    - (3) Truncated checks average ~7-10 rows, where NAX4 is ~1.1-1.2x and 1-3-row calls are slower.
  - So the experts' compute share of today's cycle is small: about 3-6 ms of ~150 at C 224 by estimate. A live A/B
    between two references cannot resolve it through SSD noise.
  - NAX4 stays a kernel lead for wide checks (S 32 ~1.9x) and compute-bound regimes. It is not a REF2c item now.
- **K-62i (Sun 07:0x, shared GPU, rough; scripts/tierglu_nax5.py, k62i_nax5.py): folding NAX4's two rowsum dispatches
  into its kernels (x sums from the A loads via lane shuffles; h sums in A's epilogue) changes nothing.** S 1 0.178-0.181
  vs 0.179-0.182 ms, S 16 within +-1 %. INV PASS. Small calls are latency-bound, not dispatch-bound: 10 groups x 20 tiles,
  each simdgroup walking 40 quant groups in sequence.
- **K-62j NAX6 (built 07:1x; scripts/tierglu_nax6.py, k62j_nax6.py; runs after runner 58):** NAX4 with a fixed split-K
  (kernel A: 4 chunks per tile, kernel B: 2). The chunk accumulators are summed in a fixed order in threadgroup memory,
  so the structure is the same for every call size and INV holds by construction. Aim: S 1-3 to 3c's level or better
  (S 1 0.105, S 3 0.25 ms), S 16 not slower.
  - **RESULT (Sun 07:1x, shared GPU, rough, L20 real data): a small gain, the small-call gap stays.** INV PASS.

    | rows | 3c | NAX4 (gs 1) | NAX6 |
    |---|---|---|---|
    | S 1 | 0.105 ms | 0.60x | 0.65x |
    | S 3 | 0.25 ms | 0.82-0.86x | 0.86-0.88x |
    | S 16 | 1.17 ms | 1.40-1.63x | 1.46-1.71x |
    | S 32 | 2.30 ms | 1.88-2.10x | 1.92-2.16x |

  - A 1-pair group pays the full 16-row MMA set (~7.4 us at peak), which sets the small-call floor. Splitting K hides only
    part of the latency.
  - **A1 closed for today's 45 GB operating point** (truncated checks of ~7-10 rows, SSD-bound cycles; K-61 and K-62L
    KILL). The matrix-unit tier kernel (NAX6 best) stays the lead for wide checks (S 32 ~2x) and compute-bound regimes.
    The ref2 switch (CFG tier_nax4) stays off.
- **R-62b INV (Sun 07:0x, shared GPU; scripts/r62b_inv.py, data/r62b_inv.txt): PASS.** The single-call 16-row SDPA (ref2
  CFG att_single) gives a row the same bits at any position among the 16 rows, with 5 different companion sets, and with
  the actual key count varied inside one padded length (L 384 / 512 / 1024 / 2048).
  - The control is not blind: chunked vs one call differs in 10 / 16 outputs at L 1024 / 2048 (0 at 512 in this draw).
  - So att_single is a valid REF2c piece under REF2b's key buckets. Its live check (live loop == its own AR past 512
    keys) comes with the bundle.
- **REF2c GDN in-proj (desk, 07:0x):** MTPLX_FUSED_GDN_INPROJ=1 concatenates qkv / z / b / a at load into one quantized
  matmul. Under REF2 only b and a (N 48, below CFG dense_min_n) would change kernel, onto qmm16c, so their bits change.
  - It is a load-time option, so it needs its own process (A-B-A processes).
  - RANKED_45GB estimates -2.5 ms per check.

## R-65 DISPATCH (Leap, pre-registered Sun 2026-11-01 07:2x EET): how many kernels one check of the 45 GB default launches

- Why: the check is GPU-bound (R-11b) with many tiny kernels. R-61 counted ~35 per GDN layer, and GDNX's -3.0 ms for
  ~470 removed kernels is ~6 us per launch. K-62L showed the experts kernel is not the limiter at this operating point.
  If the check launches thousands of tiny kernels, fusing them could be worth more than any single kernel.
  - mx.compile on elementwise chains is usually bit-identical (each op still rounds to its own dtype), so this could join
    the exact-safe stack.
- Census (scripts/r65_dispatch.py: BigMoE's r57 harness via k62_live.py, REF2-T on the gate + t 0.3, C 160, shared lock,
  one prompt, CAP 64):
  - During checks 3-5, every mx.eval / mx.async_eval first exports its pending graph (mx.export_to_dot).
  - Nodes are counted by primitive, views excluded (R-11b's method).
  - Host materializations outside mx.eval are missed, so the count is a lower bound.
- Lines:
  - >= 2,500 kernel-launching primitives per check -> build a fusion pass (exact-safe first), ~-6 ms per 1,000 removed;
  - < 1,500 -> park;
  - between: list the top 5 fusion targets with their counts and decide with NewIdeas.
- Prediction: ~4,000-6,000 per 16-row check (R-11b: 1,384 at 12 layers on the old engine).
- **RESULT (Sun 07:2x, shared, C 160; data/r65_dispatch.txt): ~3,944 kernel-launching primitives per 16-row check
  (lower bound; 2 checks, 28 eval segments). Over the 2,500 line: build a fusion pass, exact-safe first.**
  - Real work: CustomKernel 865 (~18 per layer: qmm16c, 3c tier A / B, GDNX, BigMoE's ext), Matmul (bf16) 388 (~8 per
    layer), QuantizedMatmul (MLX's own) 300 (~6 per layer).
  - Small elementwise and glue, ~1,700 per check (~35 per layer):
    - Multiply 421, AsType 274, Add 158, GatherAxis 144, Select 144, Sum 115, Concatenate 110;
    - Sigmoid 85, Negative 84, SliceUpdate 60, Divide 50, ArgPartition 48 + ArgSort 48, Arange 48, RMSNorm 40,
      Cos / Sin 24 + 24;
    - plus 3 compiled fusions (48 + 36 + 12).
  - Gate path: PostIds 96 + SpinWait 96 (2 per layer).
  - Views: Reshape 3,591, Broadcast 1,166, ExpandDims 618, Slice 565 (no launch unless a copy follows).
  - Priced at GDNX's ~6 us per removed tiny kernel: fusing ~1,000-1,500 of the ~1,700 is ~-6 to -9 ms per check (~4-6 %
    at C 224).
  - Bits: mx.compile is not always bit-identical to eager (R-40 3t: the PLE gate's compiled sigmoid differs in the last
    bit). Each fusion gets a bitwise test; the ones that differ go to REF2c.
- **R-65b ATTRIB (Sun 07:3x, shared, C 160; scripts/r65b_attrib.py, data/r65b_attrib.txt): where the kernels come from
  (per 16-row check, approximate).**

  | piece | kernels per check | per call | main primitives |
  |---|---|---|---|
  | attention (12 QSA layers) | ~1,500 | ~125 | Matmul 288, Multiply 264, AsType 168, Softmax 144, Select 144, Concatenate 108 |
  | MoE block (48) | ~1,344 | ~28 | QuantizedMatmul 192, CustomKernel 144, GatherAxis 144, Sum 96, PostIds / SpinWait 96 each, Softmax / ArgPartition / ArgSort / Divide 48 each |
  | GDN (36) | ~360 | ~10 | already lean after GDNX: 6 custom kernels, b / a projections, compute_g, sigmoid |
  | hyper_residual_write | 96 | 2 | |
  | the rest (HC reads / writes, norms) | ~600 | | |

  - **Attention is unfused.** A 16-row SDPA call (q_len 16, head_dim 256, bool mask) decomposes into scale, QK^T matmul,
    mask select, softmax and PV matmul.
    - MTPLX's 12 head chunks make 61 kernels per call; one call (R-62 att_single) makes 5. Checked by graph export at
      N 383 / 1024. This is why R-62 was 2.1-2.8x faster.
    - So att_single removes ~56 kernels per call. A custom fused attention kernel (one per call) could go further: with a
      fixed key-block order it is N-invariant by construction (masked keys add exact zeros), which could retire REF2b's
      zero-padded key buckets and class-split calls.
  - **MoE routing** is ~6-8 tiny kernels per layer (softmax precise, argpartition, take_along_axis, sum / divide,
    BigMoE's sort): ~350 per check. A fused router (bitwise replication -> exact-safe) is worth ~-2 ms.

## REF2c pieces measured (Leap, Sun 2026-11-01 07:2x-07:3x, shared GPU, rough; NewIdeas 06:1x: the REF2c bundle first)

- **R-62 att_single: READY.** R-62b INV PASS. R-62: -5 ms per check (12 attention layers, 61 -> 5 kernels per call).
- **R-67 GATES (scripts/gdn_gates.py, r67_gates_test.py; data/r67_gates.txt): KILL.** The GDN gate math (g, beta) moved
  inside mlx_lm's recurrence kernel in fp32.
  - INV PASS: 16 one-row calls == one 16-row call, bitwise.
  - The gates move up to 6 % of RMS from stock on random inputs. Stock rounds a + dt_bias to bf16 before the exp
    chain, so fp32 is the more accurate one.
  - Slower at 16 rows: 0.094 vs 0.081 ms per call, because every thread recomputes the transcendentals each step.
    Stock's two extra tiny kernels are worth < ~0.5 ms per check even if fixed. Not in the bundle.
- **R-68 INPROJ (scripts/r68_inproj_test.py; data/r68_inproj_pad32.txt): ~-0.5 ms per check (lower bound), only b's
  bits change.**
  - MTPLX's fused qkv / z / b / a projection is N 16480, not a multiple of 64. REF2 then sends it to MLX's own
    16-row matmul: 0.78x and every part's bits change.
  - Padded to 16512 rows (32 zero rows) it takes qmm16c: qkv / z / a bitwise equal to the separate calls, b differs,
    1.12x (0.128 -> 0.114 ms per GDN layer).
  - RANKED_45GB's -2.5 ms estimate is ~5x too high. The real dispatch gaps may add ~0.4 ms.
- **R-66 FATT16 (Sun 07:3x; scripts/fatt16.py, r66_fatt_test.py, nax_probe_nt.py; data/r66_fatt.txt): BETWEEN.** One fused
  kernel per 16-row attention call at head_dim 256 (MLX 0.32 falls back to 5 kernels).
  - Flash-attention-2 online softmax on the matrix units: S = QK^T per 32-key block, fp32 softmax, O += PV with V as the
    un-transposed right operand (layout probed: 2 key rows x 4 dims per lane, like the destination).
  - INV PASS on all four (bitwise): other rows' queries changed; the same query at another row; N grown by 37 / 300
    masked keys; an AR-style call (query alone, N + 7 keys). So it is batch- and N-invariant by construction and would
    retire REF2b's key buckets and class splits.
  - More accurate than MLX's call vs fp64: mean |err| 1.77e-3 vs 2.27e-3 x RMS, max 1.42e-2 vs 1.67e-2.
  - Speed, chained (8 calls per eval):

    | N keys | 12 chunks | one call (att_single) | FATT16 |
    |---|---|---|---|
    | 383 | 0.234 ms | 0.061 ms | 0.113 ms (0.54x) |
    | 1024 | 0.287 ms | 0.088 ms | 0.072 ms (1.23x) |
    | 2048 | 0.325 ms | 0.112 ms | 0.121 ms (0.93x) |

  - It is latency-bound (one simdgroup per head walks every key block). Unchained, its 1 dispatch vs 5 is worth
    ~20-30 us per call. Without key buckets it computes the real N instead of the next 512-multiple.
  - Next lever if wanted: split the key blocks over 2-4 simdgroups per head with a fixed-order merge (still invariant).
    Beyond att_single, attention is worth only ~1-2 ms per check, so parked as the REF2c attention alternative.
- **REF2c bundle as measured: ~-6 ms per check (~4 % of the C 224 cycle)**, nearly all from att_single. The fused
  attention kernel (head_dim 256 is unfused in MLX 0.32: R-65b) would add ~-2-3 ms; the router fusion (exact-safe if
  bitwise) ~-2 ms.

## R-69 OOO-EXACT (Leap, Sun 2026-11-01 07:3x; tensor level, shared GPU; scripts/r69_ooo_exact.py, data/r69_ooo_exact.txt): BigMoE's TIER_COMBINE and OOO are exact-safe under REF2-T

- Question: REF2-T today runs tier_glu without scores, then the verify call's MLX combine
  `(y * scores[..., None]).sum(-2)`. R-58G popped TIER_COMBINE, and the OOO path (resident pairs while the loader spins,
  then the rest) needs scores.
  - Do BigMoE's fused combine (3c dispatch B, scores in) and OOO (tier_glu_phase 1 / 2 + combine_phase) give the same
    bits?
- **Result: bitwise equal at S 1 / 2 / 4 / 7 / 10 / 16.** 3 trials each: synthetic pools, plan 4433333333, real top-10
  routing scores (precise softmax, argpartition, renormalized in bf16), a random ~30 % of pairs in phase 2.
  - combine_phase alone == MLX's combine at S 1-16 too (MoonShots' LY = 8 order holds at every width).
- So both can join the exact-safe stack on the 45 GB default:
  - TIER_COMBINE saves the combine's ~2 kernels per layer;
  - OOO overlaps each layer's SSD loads with its resident experts' compute (BigMoE: +4-6 % in v3 at 24 GB). At 45 GB
    the misses are 280-1,300 per check at C 160 (K-62L) and fewer at C 224.
- Open, BigMoE's side: the OOO path takes `post` / gate.spin. Does it run on the R-55 sync-free gate path? Then a live
  "every token == REF2-T" run with TIER_COMBINE + OOO on, and the speed A/B.
- **R-69L (pre-registered Sun 07:3x EET; runner 59, SHARED GPU, C 160, rough): TIER_COMBINE + OOO live.**
  - The gate path already passes the gate's post into tier_call (stream.py), and with TIER_COMBINE the MoE block hands
    scores down (tier.py), so BIGMOE_OOO=1 takes _ooo_call on the 45 GB default with no code change.
  - Harness: scripts/k69_live.py (k62_live.py's copy of BigMoE's r57; R69_KEEP_COMBINE stops the TIER_COMBINE pop).
  - One process per arm: A (default) - B (TIER_COMBINE=1 BIGMOE_OOO=1) - A, each 4 of S45's prompts, CAP 256, warm-up.
  - Lines:
    - PASS = every token of B == A on every prompt AND median cycle -3 % or better;
    - KILL = any token differs (not exact live) or the cycle worse by > 2 %;
    - else BETWEEN.
  - Prediction: tokens equal 90 %; cycle -3 to -8 % at C 160 (SSD-heavy).
  - **RESULT (Sun 07:31-07:34, runner 59, shared, C 160; data/r69l/): VOID. B never ran OOO.**
    - B's flags were on (TIER_COMBINE True, OOO True), but OOO calls were 0.
    - Tokens equal in A1 / B / A2 on all 4 prompts; cycle 176.3 / 177.7 / 176.8 ms. B was effectively A.
  - **R-69D diagnosis (Sun 07:3x, scripts/k69_diag.py, data/r69d/):**
    - Every tier_call got post (the gate path) but never scores, at all widths 1-16.
    - The active SparseMoeBlock.__call__ is EvalSuite's `block` (evalsuite/model.py:339; EM.load installs it), not
      BigMoE's `_call`. With STATE.sorted it routes itself (argsort by score) and combines with MLX ops.
    - So bm["tier"]["scores"] is never set, and TIER_COMBINE / OOO cannot engage in any EvalSuite-based run: the r57
      harness, R57_PRESET 45, the REF rows.
    - BigMoE's own `_call` takes only <= 8-row calls anyway.
  - The fix belongs to EvalSuite / BigMoE: in `block`, when the switch has a tier bm and TIER_COMBINE is on, set
    bm["tier"]["scores"] (and "shared" for OOO's spin edge) before the switch call, and skip the combine when the
    switch returns combined rows (as BigMoE's _call does).
    - Exactness carries over: R-69 showed both fused combines == MLX's combine for any rank order at S 1-16.
    - Sent to BigMoE and EvalSuite 07:4x.
  - **R-69b (pre-registered Sun 07:4x; runner 60, shared, C 160):** the fix tried in my harness process only
    (scripts/k69b_live.py, R69_PATCH_BLOCK=1; no edits to EvalSuite's or BigMoE's files).
    - A replacement block runs EvalSuite's sorted routing with the same ops, hands the scores to the tier path, and
      uses its combined rows.
    - Processes A1 (default) - B (patch + TIER_COMBINE + OOO) - A2, 4 prompts, CAP 256.
    - Lines:
      - PASS = OOO calls > 0 in B AND every token of B == A AND == BigMoE's REF2-T reference rows
        (data/r57/s45t_B_tokens.json, arm gate+t) AND median cycle -3 % or better;
      - KILL = any token differs;
      - BETWEEN = exact but < 3 %.
    - **RESULT (Sun 07:38-07:41, runner 60, SHARED GPU, C 160, rough; data/r69b/): PASS, exact, +13 % steady-state.**
      - B ran OOO: 7,479 _ooo_call calls; every tier call got scores (post + scores at every width 1-16).
      - Tokens: B == A1 == A2 on all 4 prompts, and all == BigMoE's REF2-T reference rows (s45t_B_tokens.json gate+t).

        | arm | median cycle | check + commit | steady tok/s | whole-run tok/s |
        |---|---|---|---|---|
        | A1 / A2 (default) | 177.8 / 176.7 ms | ~169-170 ms | 36.4 / 36.6 | 34.7 / 34.8 |
        | B (patch + TIER_COMBINE + OOO) | 156.9 ms (-11 %) | 146.8 ms | 41.2 (+13 %) | 38.9 (+12 %) |

      - Per prompt, steady state: HE/68 50.3 -> 57.5 (+15 %); 1640 33.2 -> 38.2 (+15 %); HE/91 36.5 -> 40.7 (+12 %);
        694 36.8 -> 41.7 (+13 %). Misses and demand SSD are unchanged (520, ~99 ms per check): OOO hides them, it does not
        remove them.
      - Measured at C 160 (more SSD-bound than C 224). BigMoE's quiet C 224 arm tonight gives the 45 GB number; it needs
        the block patch to engage.
      - Adoption is NewIdeas' call. EvalSuite will then port the patch into model.py and run a ~40-min R-56c-style audit
        vs results/ref2t.
    - **BigMoE follow-up (07:4x):**
      - Their r57 harness has the same fix (R57_COMBINE=1 -> _fused_block, aebc4e1). It also passes 'shared' for OOO's
        spin edge (no value change) and acts only on 1-16-row calls.
      - Tonight's quiet C45 turn runs C45_OOO=1 at C 224, checked token by token against s45t_B gate+t.
      - Native fix (01f652c): tier._call (BigMoE's own runtime, not EvalSuite's) now takes up to 16 rows on rank plans.
        Before, 9-16-row calls got tiers by argpartition position, not rank. Uniform plans keep 8, so the exact 24 GB
        stacks are untouched.
      - **Caveat:** EvalSuite's block computes the shared expert with a standalone sigmoid; native verify (tier._call
        and MTPLX's _verify_call) uses attention_gate's lowering, and MTPLX notes the two can differ at rare bf16
        inputs. So a native REF2-T run needs a native reference, not EvalSuite's token rows.

## REF2c FREEZE + R-70 FUSED ROUTER (Leap, Sun 2026-11-01 11:4x-11:5x EET; NewIdeas 07:5x routing; shared GPU, day rules)

- **REF2c frozen at 91cd316: ref2.CFG att_single + gdn_inproj** (on top of REF2-T). EvalSuite regenerates it tonight
  as REF2c-T (full regen, then the live audit) and runs one same-scores test.
  - FATT16 is left out. NewIdeas' rule was "only if it is free", and it is not: 0.54x vs att_single at 383 keys.
  - gdn_inproj (new ref2 switch, R-68's padded fused projection):
    - each GDN layer's qkv | z | b | a packs + 32 zero rows (16480 -> 16512, so it takes qmm16c);
    - built once per layer at its first call with the switch on; the four modules are then dropped;
    - with the switch off, a built layer runs the four calls on row slices (= the stock bits).
- **R-71 in-proj INV (scripts/r71_bundle_inv.py, data/r71_bundle_inv.txt): PASS.** Real 4-bit packs of GDN layers
  0 / 16 / 32 / 46, 4 trials each:
  - every row of a 16-row call == the same row run alone, and == the same row among 15 other companions;
  - switch off == the separate modules.
  - Bits vs REF2-T: qkv and z unchanged. b and a change in 0.1-0.8 % of their outputs (R-68's synthetic draw had a
    unchanged by chance).
- **R-71b GDN-layer INV (scripts/r71b_gdn_layer_inv.py, data/r71b_gdn_layer_inv.txt): PASS.** MTPLX's GatedDeltaNet
  with layer 0's real weights, under REF2 + GDNX (gdnx 34 calls) + gdn_inproj (built, 19 calls):
  - after a 37-token prefill, one 16-row call == 16 one-row calls, bitwise for y and for the conv and recurrent state;
  - on vs off: 45 % of y's outputs move in the last bits, so a full regen is needed.
  - With R-62b (att_single, SDPA level), the bundle's INV is covered piece by piece. The live check (loop == its own
    AR) is EvalSuite's audit after the regen.
- **R-70 FUSED ROUTER (scripts/router_fused.py, r70_router_test.py; data/r70_router.txt): exact-safe, bitwise PASS,
  ~-1.0 ms per check (shared GPU, rough).**
  - What it replaces: one kernel for the MoE block's routing chain, the same chain in EvalSuite's block and BigMoE's
    tier.install_block: softmax precise, argpartition, take, sum, divide, negate, argsort, take x 2 (~9 kernels per
    layer).
  - How each step is replicated from MLX 0.32's kernel sources:
    - softmax: softmax_single_row line by line (128 threads, 4 reads each, simd_max / simd_sum, fast::exp, 1 / sum);
    - argpartition: MLX runs it as its stable merge sort, so it is the top-k keys (bf16 bits << 16 | index) in
      ascending order;
    - sum: row_reduce_small's thread loop, which accumulates in the output type (bf16), left to right;
    - divide: bf16 / bf16;
    - argsort: stable, so rank = #larger + #equal-earlier.
  - Bitwise test: inds and score bits over 11 input families (normal sigma 0.5-8, logits on 1/4 and 1/16 grids, all
    equal, 3 levels, peaked) and the live shapes [1, R, 512] for R 1-16.
    - **0 of 540,672 rows differ.**
    - The control is not blind: an fp32 sum differs in 132,830 rows, so bf16 accumulation is required.
  - Speed, 48 dependent layers per eval: 21-22 us saved per layer at R 16 / 10 / 1, so **~-1.0 ms per check** (not
    the ~2 ms estimated at ~6 us per kernel; here the tiny kernels cost ~3.4 us each).
  - It ships alone (no regen) after one live token check vs the REF2-T rows. Then the drop-in goes to the block
    owners (EvalSuite model.py, BigMoE tier.py).

## R-72 ATT-ANAT + exact attention fusions (Leap, Sun 2026-11-01 11:5x EET; shared GPU, day-sized; scripts/r72_att_anat.py, data/r72_att_anat.txt)

- **Harness:** one real QSA attention layer (MTPLX's Attention + indexer, layer 3's 4- / 8-bit weights) under REF2,
  prefilled to 383-4,100 keys, then 16-row check calls in the verify phase. Each arm trims back, so all arms see the
  same cache. Census = kernel-launching primitives of the call's graph (R-65's method).
- **Two exact-safe fusions, both published inside MTPLX and wired for 1-8 rows only; here as ref2 switches at
  REF2's 16-row calls:**
  - CFG att_rope: the attention RoPE as one dispatch (mtplx.kernels.qwen4_m4_rope.rope_qk) instead of the stock
    table + two rotations.
  - CFG idx_prep: the indexer's RMSNorm + partial RoPE (queries and pooled block keys) through the shipped
    qsa_indexer_prepare kernel. It is enabled without MTPLX_FUSED_QSA_INDEXER, which would also switch the block
    selector.
- **Bits: PASS at T0 383 / 700 / 1,100 / 2,600 / 4,100.** The indexer selects blocks past ~2,048 keys. Checked
  bitwise: y and the KV / raw-key / pooled-block cache rows.
  - att_rope == REF2-T, idx_prep == REF2-T, both == REF2-T;
  - both on top of att_single == att_single.
- **Kernels per 16-row attention call:**

  | keys | REF2-T | + att_rope + idx_prep | att_single (REF2c) | att_single + both |
  |---|---|---|---|---|
  | 383-1,100 (dense) | 125-131 | 81-87 | 69-75 | 25-31 |
  | 2,600-4,100 (indexer selects) | 171 | 111 | 115 | 55 |

  - So the exact pair removes 44-60 kernels per call, ~530-720 per check over the 12 QSA layers.
  - What remains with att_single + both (31 at 700 keys):
    - the projections: qmm16c q / o, MLX's k / v / indexer qk;
    - the 5 SDPA kernels and the 3 key-bucket pads;
    - 5 cache slice-updates, the mask build (Arange, Full, LessEqual), 2 RMSNorms and the output gate.
- **Speed per call (T0 700; 8 calls chained, graph built per call; shared GPU, rough):** REF2-T 0.690 -> 0.598 ms
  (-1.1 ms per check). att_single 0.450 -> 0.380 ms (-0.8 ms per check).
- Status: exact-safe, so no regen is needed. It joins the router (R-70) in one every-token check vs the REF2-T rows
  before shipping (FAST INTEGRATION: exact-safe changes are stacked under one check).

## R-73 GLUE-ANAT + exact glue kernels (Leap, Sun 2026-11-01 12:0x EET; shared GPU, day-sized; scripts/r73_glue_anat.py, glue_fused.py, data/r73_glue_anat.txt)

- **Harness:** the per-layer glue around attention / GDN and the routed experts, with layer 3's real weights under
  REF2-T: two HC reads (REF2's leapk.hc_read), the MoE block minus the experts (8-bit gate, routing, shared expert +
  its gate, tail), the residual write. The attention output and the experts' combined rows are inputs.
- **Probe:** to replay MLX's eager bf16 sigmoid in a custom kernel, exp must be metal::precise::exp. That gives 0 of
  65,536 bf16 inputs off; the JIT default (maybe-fast) and fast exp each miss 1. MLX's own kernels are built with
  precise math, so later replays must use precise:: math.
- **Kernels per layer (16-row check):** base 30 = HC reads 10 + MoE glue 18 + write 2.
  - R-70's router (9 -> 1): -8.
  - moe_tail, y + sigmoid(g) * s in one kernel (3 -> 1): -2.
  - CFG hc_write, the residual write as one compiled kernel in MTPLX's op-diet view spelling (2 -> 1): -1.
    - The stock spelling's reshape between the ops blocks fusion; a plain mx.compile of it stayed at 2.
  - All three: **19 per layer, -11 per layer, ~-530 per check.**
- **Bits: PASS** for each piece alone and all three together, == base over 6 trials (input scales 0.5-8).
- **Speed (8 dependent layers per eval; shared GPU, rough):** base 0.441 -> all 0.401 ms per layer, ~-1.9 ms per
  48-layer check.
- **The exact-safe stack so far:**
  - R-72 attention: -44 to -60 kernels per QSA call;
  - R-70 router + R-73 tail + write: -11 per layer;
  - together ~-1,060 to -1,250 kernels per check, of ~3,944. REF2c's att_single (-672) comes on top.
  - Rough sum of the measured pieces: ~-3 ms per check at the microbench's ~3 us per tiny kernel. At GDNX's in-situ
    ~6 us per kernel it would be ~-6 to -7 ms.
- **R-74 SHARED-CAT (12:1x; scripts/r74_shared_cat.py, glue_fused.shared_cat; data/r74_shared_cat.txt): exact-safe,
  -2 kernels per layer.**
  - What it does: the MoE input's three small 4-bit projections (shared expert gate_proj N 640, up_proj N 640,
    shared_expert_gate N 1) as one MLX call on the row-concatenated packs. N is 1281, not a multiple of 64, so REF2
    keeps MLX's own 16-row call; padding to 64 would send it to qmm16c, with other bits.
  - Bits:
    - every part == the separate calls on the real packs of layers 3 / 20 / 44, input scales 0.5-4, 16- and 1-row
      (padded) calls, 0 / 7 / 31 / 63 zero rows: 12 / 12 configs;
    - in the glue harness, cat and all four == base over 6 trials.
  - **Glue per layer with all four pieces: 30 -> 17 (-13 per layer, ~-620 per check).**
  - Speed (less contended this run, rough): base 0.321 -> 0.301 ms per layer, ~-1 ms per check. Here a tiny kernel
    costs only ~1.5-3 us; the in-situ value comes from the C 224 run.
- **K-73L -> BigMoE's C45 turn (BigMoE 12:1x): K-73 rides tonight as arm K (tag c45k_224) at C 224.**
  - Their r57_screen at 3375145 carries the K73_EXACT block; order A no-OOO -> B default -> K -> A2 -> P.
  - Every token is checked vs s45t_B, under the quiet lock. B and K are single runs, so small gains read against the
    A / A2 spread.
  - Runner 61 was cancelled 12:09. R-74 was offered as a one-line add.
- **K-73L as first planned (runner_r61.sh, scripts/k73_live.py = a copy of BigMoE's r57_screen.py at b9049b0):**
  - processes A (R57_PRESET=45) - B (+ K73_EXACT=1: the whole stack) - A at C 160, 4 prompts, CAP 256;
  - every token vs the REF2-T rows (s45t_B gate+t), shared GPU, rough timing;
  - night only; waits for NO-SWAP >= 37.7 GB and for no other process over 8 GB.
  - On PASS: the ref2 switches default on, and the block pieces (router_fused, moe_tail) go to the block owners
    (EvalSuite model.py, BigMoE tier.py / r57_screen).

## R-75 / R-76 the drafter's share (Leap, Sun 2026-11-01 12:1x-12:2x EET; shared GPU, rough; scripts/r75_draft_anat.py, r76_draft_fuse.py, draft_fused.py)

- **Why:** in REF2-T's C 224 run (s45t_B) the cycle is 131.9 ms and check + commit 121.1 ms. So ~10.8 ms per cycle is
  the drafter plus bookkeeping, serial with the check.
- **R-75 anatomy:** DFlash2 (5 sliding-window layers, hidden 2560, ffn 7,680, selector rank 256) bound to the real
  4-bit embed / lm_head, driven by r57_screen's propose verbatim.
  - **649 kernels per propose:** hidden_states 469, head2s 12, chain 168.
  - Shared GPU (Breakthrough at ~50 %): hidden_states 10-12 ms, head2s 2.8, chain 1.4.
  - The 469 come mostly from the grouped dynamic conv (4 calls per layer, ~11 tiny ops each) and a per-layer mask
    rebuild.
- **R-76 draft_fused.install() (exact-safe; the drafts are bitwise identical):**
  - one kernel per conv call, replaying the eager bf16 ops in order (+0 above the block's first row);
  - the attention mask memoized per (ctx_len, rows, window).
  - 30 cycles on two identically built caches: hidden, candidates, path and confidences all bitwise equal.
  - **Kernels per propose 649 -> 364.** ms per propose 16.2 -> 14.5 (shared, rough); hidden_states 11.6 -> 10.7.
- **Reading, with October's R-06 / R-10 / R-15 (above):** the drafter's time is mostly GPU time in its dense
  projections and head.
  - bf16 at M 16 runs ~60 % of its bytes floor; an 8-bit / 4-bit drafter saved only ~1 ms in R-06. So no big lever
    here.
  - Left on the table (exact, from R-15 step 2):
    - D2 for its split-K MLP calls: -1.5 ms on the 4-bit drafter; check whether the bf16 path splits;
    - a fused selector chain: ~-1.3 ms, with bit-exact replication of MLX's rank-256 bf16 sum;
    - the head on qmm16c: ~-0.7 ms, but it moves the drafter's numerics.
- R-76 was offered to BigMoE as one more line in tonight's K arm (12:2x); BigMoE added it, K arm frozen.
- **R-77 CHAIN (scripts/chain_fused.py, r77_chain_test.py; data/r77_chain.txt): exact, the selector walk in 8 kernels
  instead of 167.**
  - Every (predecessor, successor) edge for all 15 steps is computed at once with the same MLX ops in the same order.
    MLX's own rank-256 sum stays row-independent.
  - The walk runs in one threadgroup: bf16 add, argmax with first-index ties, and MLX's softmax_single_row for 16
    values replayed as in R-70.
  - Bits: 60 real cycles (33 top-2 ties hit the tie rule) + 200 tie-heavy synthetic walks: PASS.
  - 1.45 -> 0.30 ms (shared, rough).
- **R-78 FORMS (scripts/r78_draft_forms.py; data/r78_draft_forms.txt): L1-23's trick on the drafter's bf16
  projections.**
  - At 16 rows, x @ Wt (pre-transposed, contiguous) and (W @ x.T).T are bitwise == nn.Linear's x @ W.T, and some are
    faster.
  - At 8 / 15 rows they take other kernels and differ, so the context rows keep nn.Linear.
  - The MLP matmuls already run near their bytes floor. ~-0.6 ms per propose (shared, rough).
  - draft_fused.install(forms=True, draft=...) puts x @ Wt on the MLP + q, replacing W (no extra memory), and
    (W @ x.T).T on k / v block rows, o and the conv projections.
- **R-79 DRAFT-STACK (scripts/r79_draft_stack.py; data/r79_draft_stack.txt): R-76 + R-77 + R-78 together.**
  - Path + confidences bitwise over 60 cycles.
  - **ms per propose 14.45 -> 12.41 (-14 %; 4 interleaved rounds, shared GPU with Breakthrough at ~50 %, rough).**
    In the quiet ~10 ms drafter that is likely ~-1.5 to -2 ms per cycle.
  - Tonight's K arm carries R-76 only. R-77 + R-78 join the next stacked check.


## R-80 / R-81 long context: the first call and the attention curve (Leap, Sun 2026-11-01 12:3x EET; shared GPU, rough; scripts/r80_first_call.py, r81_att_ctx.py)

- **Why:** Breakthrough's FIXED-PART result (12:16). The first check after a 9-12K-token prefill costs 1.6-3.4 s, vs
  82-191 ms steady; the second ~290-380 ms. That start-up excess is ~48 % of agent's per-check cost (8-18 % on
  code / math / chat).
  - Their candidates: the prefill's lazy tail, cold pools, one-time builds / compiles at the new context.
  - Separately, context costs +2.27 ms per K per check.
- **R-80 (one real QSA attention layer, fresh process, short warm-up at 700 keys, then 16-row checks at 10,000 /
  10,600 / 14,000 keys):**
  - First call at a new length = 1.4-1.7x the steady call: +1.4 / +1.6 / +8.5 ms per layer. The very first call in
    the process: 6.9 vs 1.3 ms.
  - **So one-time kernel builds on the attention side are tens of ms per check at most, not seconds.** The excess
    must be the prefill tail or cold pools (or the GDN / MoE side, untested here).
- **R-81 attention vs context (median of 5 after warm-ups at each length; selection off = the plain causal mask, same
  SDPA work):**

  | keys | REF2-T ms per call | att_single | selection's own cost | REF2-T x 12 layers per check | att_single x 12 |
  |---|---|---|---|---|---|
  | 2,000 | 1.05 | 0.83 | ~0 | 12.6 | 9.9 |
  | 4,000 | 1.34 | 1.08 | ~0 | 16.1 | 12.9 |
  | 8,000 | 1.89 | 2.10 | +0.66 | 22.7 | 25.2 |
  | 10,000 | 2.03 | 1.71 | +0.05 | 24.3 | 20.6 |
  | 14,000 | 2.69 | 2.43 | +0.12 | 32.3 | 29.2 |
  | 20,000 | 3.10 | 2.91 | +0.25 | 37.2 | 34.9 |

  - The 16-row verify computes dense QK^T over every key with the indexer's mask; the selection itself is cheap.
    So attention grows ~1.2-1.4 ms per K context per check.
  - A block-sparse 16-row kernel (only the selected blocks + the causal tail) would cap it near the 2K level:
    ~-12 ms per check at 10K and ~-24 at 20K (~-13 % of agent's steady 88.7 ms check at ~10K).
  - Natural form: FATT16 (R-66) with fully masked 32-key blocks skipped. That is exact under FATT16's own invariant
    numerics (a skipped block leaves (m, l, O) unchanged), so a REF2c / REF2d-type item; it also retires the key
    buckets. Not built.
  - R-80's 11 ms per call at 14K was an outlier (GPU contention or the first length).
- **Block-sparse FATT16 priced on paper (12:3x): small under exactness. Not built.**
  - The verify mask is per row at the indexer's 4-token blocks (top-512 per row of ~2,500 blocks at 10K: ~20 %
    density).
  - REF2 needs a row's bits to be independent of the other rows (AR == check). So a fused 16-row kernel must keep
    absolute-position key groups (masked keys = exact zeros, as FATT16) and can skip only a group that no row selects.
  - With overlapping rows (union ~30 % at 10K, ~15 % at 20K), a 16-key group is empty for all rows ~25 % of the time
    at 10K and ~50 % at 20K. That is a 1.3-2x SDPA gain at long context only, a few ms per check.
  - A compact union-list kernel would get the full ~4-5x, but its chunking depends on the other rows: not
    batch-invariant, so a non-exact fast-mode option only (Kamen's call).

## R-82 HC-GATE (Leap, Sun 2026-11-01 12:4x EET; shared GPU, real weights; scripts/r82_hc_gate.py, data/r82_hc_gate.txt): exact-safe, -1 kernel per HC read

- After REF2c and the exact stack, the two HC reads per layer are the biggest remaining kernel group (~5-6 each).
  REF2's S4 read runs: norm -> S4 split-K partials -> _ACCUM (the partition sum) -> gate1 (table lookups) -> MLX's
  bf16 up matmul -> tail.
- ref2 CFG hc_gate (leapk.hc_read fuse_gate) runs _ACCUM + gate1 as one kernel. It is the same fp32 partition sum in
  the same order, rounded to bf16, then gate1's lookups on those bits; the intermediate `both` is never written.
- **Bits PASS:** real HC weights of layers 3 / 20 / 44, attention and MLP reads, 4 inputs each, with and without the
  pending write: mixed / written / inject bitwise.
- **Kernels per read 6 -> 5, ~-96 per check.** The tail could join the up matmul only with a bitwise replica of MLX's
  bf16 GEMM at K 320: not done.
- Joins R-77 + R-78 in the next stacked check. ref2.self_test PASS with the switch off (BigMoE's harness imports
  ref2 / leapk live).

## R-83 GDN-INPROJ-X (Leap, Sun 2026-11-01 12:4x EET; shared GPU, real weights; scripts/r83_gdn_x.py, data/r83_gdn_x.txt): R-68's fusion without the reference change, exact-safe

- **NewIdeas 12:4x:** by day, cut chat's constant part (the drafter stack, then exact-safe fusions on the dense part and
  the syncs, stacked for one every-token check).
- ref2 CFG gdn_inproj_x:
  - qkv | z as ONE qmm16c call (N 16,384). qmm16c's per-column bits don't depend on N (R-68 / R-71).
  - b | a as ONE MLX 16-row call (N 96, below dense_min_n, so it stays on MLX's kernel).
  - Every part keeps REF2-T's bits, unlike REF2c's gdn_inproj, which moves b / a to qmm16c.
  - Built once per layer; the four modules are dropped, so no extra memory. With the switch off, a built layer runs
    the four calls on row slices.
- **Bits PASS:**
  - tensor level: real packs of GDN layers 0 / 16 / 32 / 46, 4 scales, 16- and 1-row (padded) inputs, all four parts;
  - layer level, layer 0 under REF2 + GDNX: 16-row call, 16 one-row calls (y + conv / recurrent state), and switch off
    after the build, all == stock.
- **Kernels per 16-row GDN call 10 -> 8, ~-72 per check.** ref2.self_test PASS with it off.
- Joins the next stacked check with R-77, R-78 and R-82.

## R-84 DRAFT-HEAD (Leap, Sun 2026-11-01 12:4x EET; shared GPU, rough; scripts/r84_draft_head.py, data/r84_draft_head.txt): the drafter's head on qmm16c

- The drafter's head is the target's 4-bit lm_head (248,320 x 2560) at 15-16 rows. MLX's 16-row call is used
  because r57 removes REF2 around propose. qmm16c (REF2's 16-row NAX kernel) is faster. The draft logits round
  differently, so drafts are not bitwise; the target still verifies every token, so outputs stay exact.
- head2s 2.72-2.81 -> 2.16-2.33 ms (~-0.5 ms per propose).
- **Kept per check, teacher-forced along the 125B's own tokens (D-02b's replay; L1-20 drafter with R-76 / R-77 / R-78
  in both arms):**
  - code 9.569 (666 checks) -> 9.555 (667 checks);
  - math 6.997 (2,046 checks) -> 7.007 (2,043 checks).
  - Unchanged within noise (+-0.15 %). 92 % / 97 % of the aligned checks have identical drafts.
- Side check: with R-76 / R-77 / R-78 installed, the replay reproduces D-02b's original numbers (9.57 / 7.00) exactly,
  consistent with those pieces being bitwise.
- Verdict: a small optional win (~0.5 ms per cycle, ~0.4-0.7 %). It is kept OUT of tonight's K83 arm so that arm's
  drafts stay bitwise and its A / K comparison stays clean. A later batch can take it.

## Chat's constant part: what is left after the two stacks (Leap desk notes, Sun 2026-11-01 12:5x EET; no GPU)

- **Commit replay (MTPLX capture-commit):** the verify stashes the rows the GDN recurrence consumed. A rejected window
  (most truncated checks) then replays only the recurrence over the kept rows from the pre-verify snapshot, plus a full
  forward of the one PLE layer.
  - That is ~5 tiny kernels per GDN layer (compute_g, sigmoid, the recurrence, conv concat, contiguous) x 35, ~175 per
    commit, serial on the GPU before the next draft.
  - Batching across layers is exact (same per-head arithmetic), but Metal's 31-buffer limit forces groups of ~4
    layers. Concatenating the 36 fp32 states into one buffer would copy ~110 MB per commit, eating the ~0.5-1 ms
    gain.
  - Worth it only with a persistent per-check state layout. Parked.
- **Syncs per cycle:** the GPU idles while the CPU builds the first verify segment after the draft's read-back, and
  during the acceptance bookkeeping + commit build before the next draft (~2-3 ms per cycle estimated, ~2-4 % of a
  chat cycle).
  - Truncation makes the row count data-dependent, so the draft-verify sync can't simply go.
  - K-80's per-check split (graph-build ms) will measure the build share tonight.
- Ranked for the next batch (exact): the drafter head on qmm16c (R-84, drafts not bitwise, -0.5 ms); attention k | v
  on one MLX call (-12 kernels per check); swiglu folded into qmm16c's down-projection prologue (-48). All small next
  to what tonight's K-73 / K-83 arms measure.

## Trees for the kept-per-check wall: prior art and the exactness cost (Leap desk notes, Sun 2026-11-01 05:4x; ranked row #10)

- **Bole** (arXiv 2608.01651): tree speculation for hybrid-attention models. The linear-attention recurrence is
  rewritten as a tree-structured closed form, in one kernel that verifies all nodes in parallel; speculative state
  updates are kept as token-level factors, and only the chosen state is rebuilt.
  - Tree verification 3.4-7.7x faster, transient state memory 82-99x smaller, decode up to 2.03x the best tree baseline
    (SGLang). It claims "losslessly encodes" the state; the abstract claims no bit-exactness.
- **TreeWY** (arXiv 2608.20961): a tree-structured WY transform of the gated delta rule, one triangular solve for all
  nodes, rebuilding only the accepted state. On Qwen3.5 35B / 397B, our model's family: memory saved; "a wider,
  higher-acceptance draft becomes possible, though not yet a throughput win"; "essentially identical acceptance".
- **For REF2's exact checks:**
  - Both reorder the recurrence's arithmetic, so neither is bit-exact against REF2's sequential AR.
  - An exact tree check needs:
    - (1) the GDN state of each node computed sequentially from its parent's state: one state buffer per node, ~3 MB
      per layer, so 16 nodes ~50 MB per layer, reused layer by layer;
    - (2) attention with each tree row's visible keys in the AR's own layout. Past 512 keys the NAX kernel's bits
      depend on where keys sit (4b), and a tree interleaves siblings. So each row needs a gather of its visible keys
      (prefix + ancestors) into contiguous order, ~16 extra SDPA calls per QSA layer, much like keep-the-AR.
  - The likely gain: a sibling only where the drafter is unsure (L1-14's confidence), i.e. EAGLE-2-style dynamic
    trees. DFlash2's top-K recall says the 2nd candidate often holds the token (oracle top-16 12.8 vs 9.5 picked).
  - The earlier L1-22 trees gave +5 %. Routed to NewIdeas 05:3x: the REF2c bundle (A) or exact trees (B) first.

## S45c TRUNC125 (NewIdeas 01:46; Leap, pre-registered Fri 2026-10-30 04:00 EET): L1-14's running-confidence truncation in the exact loop at 45 GB

- **What:** S45's harness at C 224 (`r40s3k_screen.py 224 256 speed 16,0.3,0.2`, runner 42), REF2b (ref2.py with key
  buckets). Three arms in one process: block 16, t 0.3, t 0.2.
  - A truncated check verifies 1 + the depths whose running product of the selector's confidences is >= t (L1-14).
  - REF2 pads every <= 16-row call, so a truncated row has the bits of a full check's row: exact by construction. Only
    the rows' experts shrink: fewer misses and fewer expert bytes.
- **Also:** REF2b's key-bucket cost per 16-row attention call (synthetic, the model's shapes, N 536 / 800 / 1000),
  buckets on vs off.
- **Lines (NewIdeas):** PASS = all-6 mean tok/s >= +5 % over block 16, with every token == the block-16 arm's.
  KILL = < +2 %.
- **Prediction:**
  - t 0.3: ~11 rows per check, kept ~8.0 of 8.27, misses -20 %, tok/s +8 %.
  - t 0.2: ~12 rows, +5 %.
  - Every token equal.
  - Bucket cost <= 3 ms per check.
- **First, the REF2b merge check (runner 42 KBTEST, 17:01): PASS.** The merged ref2.py == the 4c module bitwise at
  N = 100..1100. With buckets off it == MTPLX's plain call, wherever no class boundary is crossed.
- **RESULT (Fri 17:10-17:18, runner 42; logs/job.out; the swap guard stopped the job during t 0.2's 6th prompt,
  11.4 -> 14.5 GB): exact, and a large steady-state gain, but the pre-registered wall-clock verdict is not clean.**
  - REF2b bucket cost per 16-row attention call: N 536 0.94 -> 1.12 ms, N 800 0.66 -> 0.68, N 1000 0.78 -> 0.70. That
    is <= +2.2 ms per check: right.
  - Every token in both truncation arms == the block-16 arm's and EvalSuite's REF2 rows: exact.

    | arm (C 224) | rows per check | kept | median cycle | misses per check | SSD ms per check | wall-clock all-6 tok/s | steady-state tok/s (kept / median cycle, mean of 6) |
    |---|---|---|---|---|---|---|---|
    | block 16 | 16.00 | 8.27 | 293.2 ms | 432 | 116.5 | 22.80 | 26.9 |
    | t 0.3 | 9.84 | 7.91 | 205.1 ms (-30 %) | 330 (-24 %) | 91.3 | 23.40 (+2.6 %, but see below) | **32.6 (+21 %)** |
    | t 0.2 (5 prompts) | ~9.3 | ~7.3 | ~231 ms | ~318 | ~100 | - | 31.5 vs 27.8 on the same 5 (+13 %) |

  - **Why the wall-clock line misleads:**
    - t 0.3's algebra/1282 hit a PAUSE (flagged): 1.2 tok/s over the wall.
    - Order warm-up: block 16 ran first. HumanEval/68 truncates nothing in every arm (16.00 rows, kept 15.41), yet
      goes 27.6 -> 35.4 -> 38.5 tok/s, because the process's first prompt pays a 10.5 s prefill and cold checks.
    - Per prompt at t 0.3, the cycle falls 25-32 % and kept falls 3-9 % (1640, HumanEval/91, 694, 1282, 708).
  - Predictions: rows ~11 (got 9.8); misses -20 % (-24 %); tok/s +8 % (steady-state +21 %, wall-clock unclean); every
    token equal (right); bucket cost (right).
  - **Verdict: not called on the pre-registered line (the run was cut, paused and order-biased). The steady state
    says PASS-size.**
  - **S45c-2 (pre-registered 17:3x):** a clean re-run, t 0.3 vs block 16 only, at C 224 on the same 6 prompts.
    - Order: a warm-up prompt first (discarded), then per prompt A-B-A (block 16, t 0.3, block 16).
    - Reports wall-clock tok/s per run, PAUSE-flagged runs excluded.
    - Same lines: PASS = all-6 mean tok/s of t 0.3 >= +5 % over the mean of the two block-16 runs, every token
      equal. KILL < +2 %.
    - Prediction: +15 to +22 %.
    - **Attempt 1 (Fri 18:36, runner 44): stopped by the swap guard 15 s into the load** (10.4 -> 13.5 GB). Another
      session's GPU job and Kamen's apps were resident; the 66 % gate left ~2 GB of margin at C 224's ~40.5 GB peak.
      Re-queued as runner 45 with a 72 % gate (~46 GB free).
  - **S45-STACK (pre-registered Fri 20:5x; replaces S45c-2 under FAST INTEGRATION; NewIdeas' RANKED_45GB.md exact-safe
    stack, items 1 + 2 at C 224):**
    - One process at C 224, the gate switchable per run (BIGMOE_MAXK 160 + ext build_next; stream.GATE_N 160 / 40).
    - After a discarded warm-up, per prompt: slot block 16, gate block 16, slot t 0.3, gate t 0.3, slot block 16.
      CAP 192 tokens, so 30 runs fit one lock. Runner 45, gate 76 % free (the ~40.5 GB peak + 8 GB).
    - Reports: wall-clock tok/s per run (PAUSE-flagged runs excluded) and the steady state. Each arm vs the two slot
      block-16 runs: the gate alone, truncation alone, and the stack.
    - Exact: every token equal across all 5 runs of a prompt.
    - Lines:
      - the stack (gate + t 0.3) >= +40 % over slot block 16 = PASS;
      - truncation alone keeps S45c-2's lines (PASS +5 %, KILL < +2 %);
      - any token difference = FAIL.
    - Prediction: gate +30 % (R-55: +35-48 % live), t 0.3 +15-20 %, the stack +45-60 %.
    - **Attempt 1 (Fri 21:01, runner 45): stopped by the swap guard 20 s into the load** (14.7 -> 19.3 GB), although
      memory_pressure's free percentage passed 76 % at 21:00. vm_stat minutes later: wired 36.7 GB (the concurrent
      shared MLX jobs' Metal buffers), available (free + inactive + speculative + purgeable) ~14.5 GB.
      - The free percentage is not a usable gate for ~40 GB loads while other MLX jobs run.
      - Runner 45 now gates on vm_stat's available >= 49 GB (the peak + 8 GB), so it waits for a genuinely quiet Mac
        (likely the night). Fit12 was warned before its own C 224 load.
    - **Attempt 2 (Fri 23:30, runner 46, queue-first per the orchestrator's 23:2x rule):** granted, then released.
      After the settle, 37 GB was available: wired only 7 GB, but Kamen's apps and the system had 21.8 GB active. So
      while his apps are open, ~37 GB is the ceiling and no C 224 / C 192 run fits.
    - **Orchestrator 23:3x: run the stack as a proof of concept at a smaller C, labelled "not the 45 GB point"; stop
      probing C 224 until it says memory is back** (it asked Kamen to quit Chrome / Spotify and leave the lid open).
      - Quiet-lock runs may then gate on peak + 4 GB.
      - **C 160, not C 128:** BigMoE's R-55 gate needs base slots >= a call's distinct experts (<= 160 for 16 rows),
        so C 128 could evict a slot the same call still needs. C 160 peaks at ~32.5 GB, gate 37 GB (+4). Runner 47.
      - Same 5 arms and lines. The relative gains and the every-token check transfer; tok/s at 45 GB does not.
    - **RESULT (Fri 23:39-23:47, runner 47, quiet lock; data/s45c2_c160.txt): PASS on every line. The exact-safe stack
      is +50 % at C 160 (not the 45 GB point).** Every token is equal across all 5 runs of every prompt.

      | arm (C 160, CAP 192, after a warm-up) | wall-clock tok/s (5 prompts) | vs slot block 16 | steady state | median cycle | rows | kept | misses |
      |---|---|---|---|---|---|---|---|
      | slot block 16 (runs 0 + 4; one paused run excluded) | 19.65 | - | 20.0 / 20.8 | 405 / 386 ms | 16.0 | 8.37 | 757 |
      | gate block 16 (R-55) | 26.50 | **+34.8 %** | 26.8 | 294 ms | 16.0 | 8.37 | 758 |
      | slot t 0.3 (L1-14) | 22.99 | **+17.0 %** | 25.7 | 277 ms | 10.0 | 7.96 | 582 |
      | gate + t 0.3 (the stack) | 29.52 | **+50.2 %** | 34.1 | 209 ms | 10.0 | 7.96 | 582 |

      - Predictions: gate +30 % (got +34.8), t 0.3 +15-20 % (+17.0), stack +45-60 % (+50.2): all right.
      - Bucket cost: <= +2.3 ms per check.
      - It stacks nearly multiplicatively: 1.348 x 1.170 = 1.577 expected, 1.502 measured.
      - Sent to the orchestrator as BIG at 23:48, labelled C 160. The C 224 run waits for the orchestrator's "memory is
        back".
    - **C 224 attempt (Fri 23:50-23:51, runner 48, after the orchestrator's "memory is back"): stopped by the swap guard
      30 s into the load** (9.0 -> 12.6 GB). At the grant, vm_stat available was 45.0 GB against the 44.5 needed.
      - 'Inactive' includes anonymous app memory, which only swap frees. At 23:52: free 24.7, file-backed 9.0,
        anonymous 24.9, wired 7.1 GB.
      - The gate that frees without swap is free + speculative + purgeable + file-backed: ~42 GB at the grant, ~35 at
        23:52.
      - **No more C 224 attempts tonight** (four trips today). The C 160 stack stands; the orchestrator was told.
    - **Then (orchestrator 23:5x and 01:4x NIGHT MODE):** Chrome closed until 08:00 and Kamen's other apps quit, so
      NO-SWAP is ~48 GB with nothing running. The NO-SWAP gate is the rule for every big load.
      - Runner 49 (C 224 stack, NO-SWAP >= 44.5 after the grant) was released twice short (36.5 at 00:38, 41.2 at 01:21,
        before the apps were quit); queued again 01:36.
  - **C 224 RESULT (Sat 02:29-02:37, runner 49, quiet lock; NO-SWAP 49.9 GB at the grant; data/s45c2_c224.txt): PASS on
    every line at the 45 GB point. Exact:** every token is equal across all 5 runs of every prompt.

    | arm (C 224, CAP 192, after a warm-up) | wall-clock tok/s | vs slot block 16 | steady state | median cycle | rows | kept | misses |
    |---|---|---|---|---|---|---|---|
    | slot block 16 (runs 0 + 4) | 24.97 | - | 27.6 | 292 / 288 ms | 16.0 | 8.37 | 446 |
    | gate block 16 (R-55) | 32.91 | **+31.8 %** | 36.1 | 226 ms | 16.0 | 8.37 | 441 |
    | slot t 0.3 (L1-14; 5 prompts, 694 paused) | 28.90 | **+12.5 %** | 33.5 | 210 ms | 10.0 | 7.96 | 346 |
    | gate + t 0.3 (the stack) | **36.36** | **+45.6 %** | **43.5** | 154 ms | 10.0 | 7.96 | 345 |

    - Predictions: gate +30 % (right), t 0.3 +15-20 % (got +12.5: a bit low, as C 224 has fewer misses to save),
      stack +45-60 % (right).
    - Bucket cost: <= +2.7 ms per check.
    - Sent to the orchestrator as BIG at 02:37 (before its 04:15-07:45 quiet window).
  - **S45-LONG (pre-registered Sat 01:4x; the orchestrator's "a longer, steadier 45 GB number"; runner 50 after runner
    49):**
    - The same exact loop at C 224, CAP 256.
    - Per prompt: slot block 16, gate + t 0.3, slot block 16, after a warm-up.
    - 12 prompts not in S45's six, all with EvalSuite REF2 rows: HumanEval 101 / 27 / 117 / 4 / 86 / 143,
      MATH-500 intermediate_algebra 1422 / algebra 2232 / prealgebra 1655, GSM8K 705 / 715 / 724.
    - One 25-min quiet ticket (run_locked 20 min).
    - Lines: the stack >= +40 % wall-clock over slot block 16 = PASS (as S45-STACK); any token difference = FAIL.
    - Prediction: +40 to +55 % at C 224 (fewer misses than C 160, so a little below its +50 %); exact.
    - **RESULT (Sat 02:55-03:02, runner 50, quiet lock; NO-SWAP 48.4 GB at the grant; data/s45long_c224.txt): PASS,
      +45.3 %, exact** (tokens equal across all 3 runs of every prompt).

      | arm (C 224, CAP 256, 12 prompts) | wall-clock tok/s | steady state | median cycle | rows | kept | misses |
      |---|---|---|---|---|---|---|
      | slot block 16 (runs 0 + 2) | 27.58 | 30.4 | 295 / 300 ms | 16.0 | 9.26 | ~450 |
      | gate + t 0.3 | **40.09 (+45.3 %)** | 44.3 | 177 ms | 11.3 | 8.94 | 374 |

      - It matches S45-STACK's +45.6 % on the other 6 prompts: the exact stack is a steady ~+45 % at the 45 GB point
        over 18 prompts.
      - Sent to the orchestrator 03:02 (before its quiet window) and to NewIdeas.

## D-02 DOMAINS (NewIdeas' DRAFTER TRACK D98, Kamen Wed 15:33; split by Leap 15:44, agreed): DFlash2's gap per domain
- **Why:** Kamen's target is ~98 % per guess (~14 of 16 kept per check) on chat, math, code and agent. Chat1000
  REPORT 12: twins of the 125B reach 14-15 on code / math. On IFEval / chat / agent the 125B's own float-order floor
  is 13.2 / 13.7 / 11.0, so ~0.95 is the realistic top there.
- **D-02b (scripts/d02b_screen.py, ~2 min):** DFlash2 L1-20 (bf16, block 16, the official 1-step loop, exact
  two-stage head) teacher-forced along the 125B's own tokens: code (dfl2d_code, 24) and math (dfl2d_math, 60).
  - Reported per domain: kept per check, per-token rate, rate per depth 1-15, P(first 15 right).
  - The >= 0.99-sure share of first misses waits for D-02a (the stored features keep the 125B's top-1 token, not its
    probability).
- **D-02a then:** dfl2c_extract features + the 125B's probabilities for IFEval (30), multi (24), agent (12 turns), and
  probabilities for code + math. Then D-02b on those.
- **Expectation (a measurement, no lines):** code kept ~10.2 (R-10's 10.23 at b32 is the nearest; ~9.5 at b16),
  per-token ~0.93. Math kept ~8.5, per-token ~0.91.
- **D-02b result (Wed 15:48, 35 s; data/d02b_screen.txt):**

  | domain | checks | kept per check | per-token rate | P(first 15 right) | rate at depth 1 / 4 / 8 / 12 / 15 | near-copy 10 % per token (REPORT 12) |
  |---|---|---|---|---|---|---|
  | code (HumanEval+) | 666 | **9.57** | **0.924** | 0.276 | 0.941 / 0.946 / 0.939 / 0.906 / 0.858 | 0.983 |
  | math (GSM8K CoT) | 2,046 | **7.00** | **0.871** | 0.086 | 0.908 / 0.874 / 0.868 / 0.836 / 0.791 | 0.970 |

  - The gap to the near-copy is ~6 points per token on code and ~10 on math. Kamen's ~0.98 is twin territory there.
  - The rate falls with depth in both domains (code 0.94 -> 0.86, math 0.91 -> 0.79), so the deep positions are
    where a better drafter pays most.
  - Leap is the KERNELS session since Kamen's 15:47 role change. D-02a (IFEval / multi / agent features + the 125B's
    probabilities) and all further drafter work go to MTP Head (TrainLab).

## Runner 62 read-out (Mon 2026-11-02 07:2x-07:49 EET; quiet lock, C 160, R57_PRESET=45 with OOO + TIER_COMBINE; data/runner_r.log, data/k80/k80_diag_screen.txt, data/r73l/k83_{A,K}_screen.txt)

### K-83 stacked every-token check: EXACT, draft -1.2 ms, cycle -1.5 %, steady +0.7 %
| arm (own process, A first) | tok/s (steady) | median cycle | check + commit | draft | misses | demand SSD | every token == REF2-T |
|---|---|---|---|---|---|---|---|
| A = the default | 39.5 (41.6) | 153.7 ms | 145.9 ms | 11.4 ms | 469 | 94.2 ms | 6 / 6 prompts |
| K = K-73 + K-83 stacks | 39.5 (41.9) | 151.4 ms | 145.1 ms | 10.2 ms | 469 | 95.4 ms | 6 / 6 prompts |
- Draft per prompt 12.0 / 11.4 / 11.5 / 11.3 / 11.4 / 11.3 -> 10.3 / 10.2 / 10.0 / 10.3 / 10.2 / 10.2 ms: R-76 / R-77 /
  R-78 engage live. Engagement at graph build: att_rope 2,724, idx_prep 5,616, hc_write 11,035, gdn_inproj_x 8,311
  fused calls, hc_gate on (hc_read 22,070), draft_fused + forms, chain_fused.
- Why small: at C 160 the check is SSD-bound (demand SSD 94-95 of 145 ms), so the dense-side savings (-0.8 ms on
  check + commit) hide behind the loads. The drafter's -1.2 ms lands in full (the SSD idles while it drafts).
- NewIdeas' 01:28 question (C45 arm K: draft 11.5 -> 11.9 ms): arm K had R-76 (draft_fused.install(), BigMoE 2c7b871)
  but not R-77 (chain_fused) or R-78 (forms=True). So K-83's -1.2 ms comes from R-77 + R-78; R-76 alone moved nothing
  live.
- Verdict: PASS as exact-safe (every token equal, both stacks live). Small at C 160; the dense part shows more where
  the loop is GPU-bound (C 256+).

### K-80 first-check split: three named parts; the non-SSD one-time cost is the largest
| run (one process, after a gsm8k warm-up) | prompt | draft: 1st / median | check + commit: 1st / median | misses: 1st / median | demand SSD: 1st / median | first-cycle excess: SSD / draft / other |
|---|---|---|---|---|---|---|
| 1, cold pools | agent/bash18 (9,687 tok) | 39.1 / 13.0 | 223.2 / 117.6 | 856 / 260 | 153.6 / 57.0 | 131.7 ms: 73 % / 20 % / 7 % |
| 2, cold pools | agent/text23 (8,582 tok) | 36.9 / 13.1 | 314.0 / 207.1 | 1,202 / 792 | 215.8 / 140.1 | 130.7 ms: 58 % / 18 % / 24 % |
| 3, warm repeat | agent/bash18 (9,687 tok) | 41.8 / 13.0 | 190.6 / 91.9 | 127 / 134 | 40.4 / 36.6 | 127.5 ms: 3 % / 23 % / 74 % |
- 1. **Miss burst** when the pools hold another prompt's experts (runs 1-2): +76-97 ms of demand SSD, 71-91 % of the
  check's excess.
- 2. **The drafter's first call**: +24-29 ms in every run (37-42 vs 13 ms), outside the check. It is the window prefill
  (up to SWIN - 1 context positions through DFlash2): real compute that could run inside the target's prefill.
- 3. **A one-time non-SSD cost inside the first check** (run 3: normal misses, SSD +3.8 ms): +84 ms in the graph-build
  phase (159.6 vs 75.9 ms, the median of checks 2-7) + 15 ms in the tail. The build waits on each layer's router ids
  (the sync-free gate), so this is GPU progress plus CPU stalls inside the check. Not kernel builds: same prompt and
  shapes as run 1 in the same process (R-80: first-length builds are +1.4-8.5 ms per attention layer at most).
- **The first-cycle excess is ~128-132 ms in all three runs, cold or warm.** The miss burst and the one-time cost
  overlap, so warming the pools alone would likely leave ~95 ms. Part 3 is the first target, then 2, then 1.
- Candidates for part 3 (K-80 cannot separate them):
  - (a) the MLX buffer cache starts empty (run() ends with mx.clear_cache(); the prefill refills it with 1,024-row
    buffers a 16-row check cannot reuse), so the first check allocates every intermediate buffer fresh;
  - (b) the gate's first call after the slot-mode prefill (pf1(False) drops the staging buffers);
  - (c) first touch of decode-only buffers (page faults).
- Separating test (10 min, agent prompt after a long prefill): run the first check twice with a rollback between (if
  the second copy is at the median, part 3 is one-time setup); then no mx.clear_cache() between runs plus one 16-row
  dummy forward right after the prefill (allocator warm).
- 1-row checks take no gate: build 16.5-17.5 ms (pure CPU graph build, ~0.35 ms per layer).
- Line (ROUTE_350 A6 / DAY9 #7: KILL if the split names nothing >= 30 %): **PASS**. Part 3 is 74 % of the warm first
  cycle's excess, the miss burst 58-73 % of the cold ones', the drafter's window 18-23 % of all three. A6 goes to
  BigMoE by cause.

> (a quote from the project owner, kept private)
- **C0 / NEW-2 timeline (condition 5; scripts/c0_timeline.py):** BigMoE's r57_screen.py (e3210bf, the 45 GB default:
  R57_PRESET=45 = REF2-T + gate + t 0.3 + OOO + TIER_COMBINE + K-73) with probes. At checks 1, 2, 5, 10, 15, 30 the cache
  is rolled back (mtplx rollback_after_verify) and the check re-run:
  - warm repeats (all experts resident) as live;
  - one synced pass: pieces hc_a / mixer (gdn, att) / hc_m / moe_pre (router + shared) / experts (tier GLU + combine) /
    moe_post / hc_w / ple, each split into CPU build and eval (one sync taken off);
  - a row sweep 1-16 on the untruncated draft path, and synced passes at 1 and 16 rows;
  - a last forward equal to the real one bitwise, then the commit as usual.
  - By day at C 112 (fits ~29 GB NO-SWAP), code HumanEval/68 + chat multi/bg0; tonight at C 256 if a ticket allows.
  - Instrument, no line. Prediction: warm 16-row check 85-105 ms; experts 45-60 ms of it; dense (mixers + HC + router /
    shared) 30-40; per added row warm 2.5-3.5 ms (M-16's beta 2.64).
- **C2 (condition 4; scripts/k62p_floor.py):** NAX6 at S 16 on real layers 6 / 20 / 40 + real routing vs the byte floor
  (each distinct expert's MSB-3 2.15 MB + LSB-1 0.77 MB for experts at 4-bit ranks, at 300 and 215 GB/s).
  - Line: C2 (a floor kernel) is worth building only if NAX6 / floor >= 1.3x; else KILL.
  - Prediction: NAX4 ran ~180 GB/s in K-62f, so NAX6 / floor ~1.2-1.6x: BETWEEN-ish (KILL 40 %, PASS 60 %).
- **B3 (condition 9; same script):** expert slope from 16 to 32 rows (3c and NAX6, x 48) + dense matmuls 0.26 ms per row
  (D32) + GDN steps / attention rows ~0.3-0.4 = the NAX32 beta. Line: KILL if > 2.5 ms per row.
  - Prediction: NAX6 slope ~1.2-1.5 -> beta ~1.8-2.2: PASS (75 %).
- **A2 + A4 (condition 5; scripts/r86_a2a4.py):** bmm16 (L1-29) and SPK16 (R-13d D2) against the kernels the default
  runs now (S4 for the HC read's mix-down, qmm16c for the split-K shapes), plus bits.
  - Lines: A2 KILL if tokens differ or < -1.5 ms; A4 KILL if < -2 ms.
  - Prediction: both KILL. S4 and qmm16c already took these gains (R-09b / R-36, REF2), and bmm16's / D2's bits differ
    from S4's / qmm16c's (a reference change, not an exact-safe swap).
- **D7 (condition 6; scripts/r85_d7.py):** the L1-20 drafter in 4 arms on D-02b's teacher-forced replay (code 24, math
  60): A = today's K-83 drafter (bf16, R-76 / R-77 / R-78, MLX head); B = A + R-84 qmm16c head; C = 4-bit layers (R-06b's
  quantize) + R-76 + R-77 + qmm16c head; D = C with REF2 applied in the propose (16-row 4-bit projections on qmm16c).
  - Line: PASS if an arm proposes in <= 8 ms with kept within -1 % of A; KILL if every arm stays > 8 ms.
  - Prediction: A ~10, B ~9.5, C ~8.4, D ~7.5-8; kept -0.3 to -0.6 % in C / D (R-06b: -0.4 %). PASS 50 %.
- **C1 (condition 4, tonight, BigMoE's harness):** ref2 CFG tier_nax_ooo (101573d): NAX6 in OOO's two phases (phase-aware GPU
  grouping), bitwise == one NAX6 call in 96 / 96 real-data cases (data/k62o_nax_ooo_L20.txt). Kernel level (shared,
  rough): OOO's two NAX6 phases ~1.26x vs 3c's at S 16 (one call 1.75x); at S 8 0.85x (each phase pays whole 16-row MMA
  tiles for half-empty groups).

### DAY9 results 1 (Mon 11:02, quiet lock; data/k62p_floor.txt, data/r86_a2a4.txt)
- **A2 bmm16 HC mix-down: KILL (both clauses).** Per check (x 97 HC reads, 16 x 10240 -> 320): MLX 8.85 ms, bmm16 2.07,
  S4 (what REF2 runs, + the 4 gate rows) 2.29. bmm16 vs the default: -0.22 ms (line -1.5), and its bits differ from S4's
  (S4 == MLX bitwise). L1-29's -2.5 ms was against MLX's form; S4 (R-09b / R-36, in REF2) took it.
- **A4 SPK16 shapes: KILL.** Per check at 16 rows: in_proj_z (x 36) MLX 8.26 / D2 4.08 / qmm16c 1.48 ms; o_proj + out_proj
  (x 48) MLX 10.68 / D2 5.30 / qmm16c 1.97. SPK16 is +5.94 ms SLOWER than REF2's qmm16c (and bitwise MLX, not qmm16c).
  The guide's -4.5 ms was against MLX's split-K; qmm16c already saves ~15.5 ms there.
- **C2 the expert kernel vs its byte floor (S 16, real routing, layers 6 / 20 / 40 x 4 traces x 4 windows):**
  | rows | bytes per check (x 48) | 3c | NAX6 | floor at 300 / 215 GB/s | NAX6 / floor |
  |---|---|---|---|---|---|
  | 16 | 7.38 GB | 50.5 ms (146 GB/s) | 43.2 ms (171 GB/s) | 24.6 / 34.3 ms | 1.75 / 1.26x |
  | 24 | 9.82 GB | 74.7 ms (131 GB/s) | 56.1 ms (175 GB/s) | 32.7 / 45.7 ms | 1.71 / 1.23x |
  | 32 | 11.33 GB | 98.9 ms (115 GB/s) | 65.8 ms (172 GB/s) | 37.8 / 52.7 ms | 1.74 / 1.25x |
  - Quiet, NAX6 is only 1.17x faster than 3c at S 16 (L6 0.96x, L20 1.22x, L40 1.42x): layers with more distinct
    experts gain least. NAX6 holds ~171-175 GB/s at every width; 3c falls from 146 to 115 GB/s.
  - The line (C2 worth building if NAX6 / floor >= 1.3x) straddles the 215-300 GB/s range: **BETWEEN until the
    machine's achievable read bandwidth is measured** (bw_probe.py, next quiet slot). At ~250 GB/s it is ~1.46x.
- **B3 NAX32 beta: PASS.** Expert slope from 16 to 32 rows: 3c 3.03, NAX6 1.41 ms per added row per check. + dense
  matmuls 0.26 (D32) + GDN steps / attention rows ~0.3-0.4 (C0 measures) = **~2.0-2.1 ms per row vs the 2.5 line**.
  Bytes per added row fall from ~0.42 GB (1 -> 16) to 0.25 GB (16 -> 32): rows share experts, and NAX6 turns that
  into time while 3c, paying per pair, does not.
- **C0 at C 112: crashed in its own probe (not a finding).** At C 112 the tier base pool holds 103 experts per layer,
  so a 16-row probe call (up to 160 ids) skipped the sync-free gate (stream.py: > 92 ids only if the pool holds them)
  and REF2-T's slot path asserted 119 > 103. The live checks were fine. Re-queued at C 160 (C_m ~168).
- **D7 (R-85) fused 4-bit drafter: BETWEEN (data/r85_d7.txt; quiet; D-02b's teacher-forced replay).**
  | arm | propose code / math (median) | kept code / math | identical drafts vs A (code) |
  |---|---|---|---|
  | A today's drafter (bf16, R-76 / 77 / 78, MLX head) | 9.92 / 9.50 ms | 9.569 / 6.997 | - |
  | B A + qmm16c head (R-84) | 9.35 / 9.32 | 9.555 / 7.007 | 615 / 666 |
  | C 4-bit layers on MLX's kernels + R-76 / 77 + qmm16c head | 10.04 / 10.06 | 9.526 / 6.807 | 106 / 654 |
  | D C with REF2 in the propose (16-row 4-bit on qmm16c) | **7.17 / 7.41** | 9.498 / 6.820 (-0.75 % / -2.53 %) | 112 / 655 |
  - Time line (<= 8 ms): D passes. Kept line (within -1 % of A): code passes, math fails (-2.5 %). MLX's own 4-bit
    16-row calls are no faster than bf16 (C); the gain is qmm16c's (D).
  - Net per cycle, roughly: code +1 % (draft -2.75 ms, kept -0.75 %), math ~-1 %. Not adoptable as is; the kept loss is
    the 4-bit quantization (R-06b's -0.4 % was the older drafter). Next: mixed precision (the most sensitive layers bf16)
    or a short 4-bit fine-tune of l120 (MTP Head), then D again.

### DAY9 results 2 (Mon 11:04-11:09, quiet lock; data/c0/c0_c160_*.{txt,json}, data/bw_probe.txt)
- **Bandwidth probe: the M5 Pro streams 258-287 GB/s** (uint4 read kernel, 1-4 GB, medians; best 294; mx.sum 270-285).
- **C2 -> PASS.** The 16-row check's experts (7.38 GB) have a floor of 25.7-28.6 ms at 258-287 GB/s. NAX6's 43.2 ms is
  1.5-1.7x above it (1.4x at 85 % of streaming); 3c's 50.5 ms is 1.8-2.0x. Line >= 1.3x: a floor-level expert kernel
  has room. NAX6 holds ~171-175 GB/s (60-66 % of streaming) at every width.
- **C0 / NEW-2 timeline at C 160** (the 45 GB default's loop with K-73; code HumanEval/68 + chat multi/bg0; 15 probe
  checks; every probe's logits == the real check's bitwise; HumanEval/68 every token == REF2-T). Warm = all of the
  check's experts resident (0 misses in every warm pass).
  | piece (ms per check, synced, CPU build inside) | 1 row | 16 rows | per added row |
  |---|---|---|---|
  | experts (tier GLU + OOO + combine, 48) | 7.1-9.4 | 52.4-56.3 | **2.9-3.2** |
  | GDN mixers (36) | 12.3-13.4 | 12.0-13.1 | ~0 |
  | HC reads (hc_a + hc_m, 96) | 12.9-14.7 | 10.9-11.7 | ~-0.1 |
  | QSA attention (12; ~200-key contexts) | 7.4-8.9 | 7.1-9.0 | ~0 |
  | router + shared expert (moe_pre) | 2.9-4.3 | 2.4-3.1 | ~0 |
  | MoE tail (moe_post) | 1.0-2.0 | 1.8-2.4 | ~0 |
  | head + embed + final mixer | 2.2-2.5 | 2.6-2.7 | 0 |
  | PLE, residual write, glue | ~1.5 | ~1.5 | 0 |
  - One eval sync = 222 us (taken off each record). "glue" printed -10.3 is that subtraction on 48 already-evaluated
    layer outputs (no real sync): the true glue is ~+0.4 ms.
  - **Unsynced warm check: 36-38 ms + 2.9-3.2 ms per row over 2-16 rows** (84-89 ms at 16 rows). The synced 16-row sum
    is ~95 ms because the ~8.5 ms of CPU build overlaps the GPU when unsynced. **Dense constant ~32 ms of GPU**
    (cost model: 30.2): GDN 12.4, HC reads 11.4, attention 7.3, router / shared / tail 5.1, head 2.7, rest ~1.5.
  - **The row term is all experts.** Every dense piece is flat from 1 to 16 rows (the padded 16-row kernels). So the
    regression's 7.82 ms per row (C 224, no OOO) is ~3.0 ms of expert GPU + ~4.8 ms of SSD.
  - **Cold vs warm (the SSD's share at C 160):** code 16-row live checks 214-472 ms (972-2,449 misses, SSD 162-367
    ms) vs 84-89 warm; chat 4-6-row checks 66-108 ms vs 47-54 warm.
  - **1-row checks cost more than 2-row ones** (48-49 vs 43.5-44 ms warm): SB's mid-loop async_eval covers 2-16 rows
    only, so a 1-row check's 9.3 ms of graph build doesn't overlap the GPU. Exact fix: SB at 1 row. Small: 11 of 132
    chat checks (8 %) were 1-row, ~0.5 %.
  - **K-80 cross-check:** the immediate warm repeat of a run's FIRST check is at the normal warm level (code 84.9 /
    84.4 vs 88.7 at check 5; chat 60.2 then 52.6 vs 53.5). So the first-check excess is one-time setup + misses,
    paid once per run, not a property of that check's work. The drafter's first call: 24-26 vs 12-13 ms here.
  - **B3 refinement:** GDN / attention add ~0 per row up to 16 rows, so the NAX32 beta is ~1.7-1.9 (1.41 experts +
    0.26 dense + ~0-0.2), PASS with more margin.
- **Condition 5 (constant <= 13 ms) on these numbers:** today's dense constant is ~32 ms of GPU. A1 (REF2c-T failed
  IFEval), A2 and A4 are gone; A3 (K-73) is in. What is left is GDN 12.4 (A5, one-kernel GDN body) and the HC reads
  11.4 (~4x their byte floor: 0.64 GB per check, ~2.3 ms). Even both at ~4 ms plus attention / tail fusions lands
  ~15-18 ms. <= 13 needs more than the known exact levers.
- **D7b (R-85b) which parts can stay bf16 (Mon 11:17-11:19, quiet; data/r85b_d7b.txt).** All arms with REF2 in the
  propose (16-row projections on qmm16c / REF2's bf16 forms), R-76 conv kernel, R-77 chain, qmm16c head.
  | arm | 4-bit parts | propose code / math | kept code / math vs A |
  |---|---|---|---|
  | A | none (bf16) | 10.39 / 10.16 ms | 9.555 / 7.042 (ref) |
  | D | layers + fc | **6.08 / 5.92** | -0.60 % / **-3.14 %** |
  | E | layers (fc bf16) | **6.30 / 6.25** | **+0.00 %** / -3.39 % |
  | F | MLPs only | 8.13 / 7.98 | -0.30 % / -1.93 % |
  | G | MLPs + q / o | 7.60 / 7.46 | +0.47 % / -2.17 % |
  - Arm D took 7.17 / 7.41 ms in D7's run and 6.08 / 5.92 here (both quiet): read the 4-bit propose as ~6-7.4 ms.
  - **Code: PASS.** E proposes in 6.3 ms (from 10.4) with code kept unchanged; D in 6.1 ms at -0.6 %. Every 4-bit
    arm stays inside -1 % on code.
  - **Math: every 4-bit arm loses 1.9-3.4 % kept**, MLP-only included, so keeping parts in bf16 does not fix it: the
    MLPs' 4-bit rounding (RTN, g64) costs math drafts. A 4-bit-aware fine-tune of l120 (MTP Head) is the fix to try.
  - Net, roughly (draft -4 ms per cycle): code +1.5-2.5 % with E; math ~0 (draft gain eaten by kept). With a
    per-type drafter switch (NewIdeas' type rule exists for chat), code could take E alone.

## ANE-1: DFlash2 L1-20 on the Neural Engine (Fable mission 17, idea 2.3; Mon 2026-11-02, pre-registered 17:0x)
- Order: the orchestrator (Kamen approved coremltools 9.0 for this probe only; no torch). DRAFTER_WALL.md 2.3: a bigger
  drafter on the Neural Engine drafts the next block from the likely cut points while the GPU checks.
- Build: scripts/ane_probe.py, coremltools' MIL builder from the safetensors, int8 symmetric per channel (W8A16), body
  (fc + 5 DFlash2 layers, 2,031 cached + 16 new context rows, 16-row block) and head (target lm_head, 4-bit ->
  dequantized -> int8, 20 chunks + per-128 block max); embeddings, two-stage top-16 and the selector chain on the CPU.
- Step 1 line: one 16-row block > 30 ms on CPU_AND_NE = KILL. Step 2 line: the GPU's warm check time > +10 % while the
  Neural Engine drafts = KILL. Step 2 is judged with the ANE drafting back to back (worst case). The GPU check is a
  proxy: the expert part of a 16-row check (48 chained NAX6 calls, real layers 6/20/40 and real routing, 7.4 GB). The
  dense part is less bandwidth-bound, so the proxy's slowdown is an upper bound for the whole check. Shared lock: MTP
  Head's x4 training is on the GPU, so the A/B blocks are interleaved and the idle-block spread is reported.
- **Results (Mon 17:00-17:16, shared lock beside MTP Head's x4 training; data/ane_probe.txt):**
  | step | line | result | verdict |
  |---|---|---|---|
  | 1: one 16-row block on the Neural Engine | <= 30 ms | body 14.8 / 16.1 + block-max head 8.1 / 8.5 + CPU top-16 and chain 0.3 / 0.4 = **23.8 / 25.6 ms** (two runs) | **PASS** |
  | 2: GPU check proxy beside it | <= +10 % | back to back **x1.070** (blocks 1.066-1.083), 36 drafts/s x1.060, 12/s x1.013 | **PASS** |
  | kept per check (r84 replay, MLX bf16 vs ANE int8) | none set | code 9.569 -> 9.540 (-0.30 %), math 7.011 -> 7.010 (-0.01 %) | int8 + fp16 cost nothing |
  - Placement (MLComputePlan): all 620 body ops and the head's matmuls / maxes run on the Neural Engine. Only the
    head's two output concats run on the CPU.
  - Python copies the 248,320-column logits out as fp32, which adds 3-6 ms (26.8 / 31.6 ms with it). Native code reads
    the fp16 output in place and needs only the 16 best 128-blocks per row.
  - Bandwidth: the head streams 75-79 GB/s, the body only 30-32 GB/s. The likely causes, not yet split: 42 MB of cache
    inputs copied per call, and 620 small ops. A stateful cache (CoreML StateType) and fewer ops should bring a block
    to ~17-19 ms.
  - The ANE's own draft time barely moves while the GPU runs: 23.5 -> 24.1 ms.
  - **fp16 range (new; matters for any ANE or fp16 port of this drafter):**
    - The residual stream reaches ~35,000 after layer 0 and ~196,000 in layer 4 (MLX fp32), past fp16's 65,504. bf16
      has fp32's range, so MLX never shows it.
    - The first build drafted badly: hidden cos 0.889 vs MLX fp32, NaN on the CPU in fp16, with int8 and fp16 weights
      alike.
    - Exact fix: every norm on the stream is an RMSNorm (scale-invariant), so the body carries h / 32 (embeddings and
      o_proj / down_proj x 1/32) and prescales each norm's input.
    - After the fix: cos 0.9989 (min 0.9971), CPU fp16 0.9988, top-1 = MLX's on 95 % of rows, kept unchanged (above).
  - Caveats:
    - Step 2's check is a proxy: the expert part only, 61.5 ms here under MTP Head's load. The dense part is less
      bandwidth-bound, so x1.07 is an upper bound for the whole check. It ran on a shared GPU; a quiet re-run or the
      real engine at night would firm it up.
    - A 1B-class body (2x) at today's 30 GB/s is ~32 ms + the head's 8.5, so ~40 ms a candidate: two per ~85 ms day
      check, or ~25-30 ms each with the cache fix.

## ANE-2: the batched 3-candidate Neural Engine call (Fable mission 18 item 1; line <= 14 ms; Mon 2026-11-02 17:40-18:20)
Shared lock, beside MTP Head's x4 training and my own MLX replays: the timings are rough. Random weights of the real
shapes, since latency does not depend on the values.
- **Head (data/ane_head_bench.txt):**
  - The full 248K-row head, int8, takes 8.06 ms at 15 rows and 8.42 ms at 45 (three candidates add only +0.4 ms).
  - fp16 takes 10.6 / 13.2 ms, so the Neural Engine streams >= 120 GB/s. int8's 79 GB/s is a weight-rate limit
    (~79 G weights/s), not bandwidth.
  - The target's exact 4-bit g64 head (uint4 + fp16 scale / offset) falls back to the CPU: 131 / 173 ms. The
    Neural Engine runs int8, not int4 blockwise.
  - So the head needs fewer weights, not fewer bytes per weight.
- **Body ablations (ANE-1 graph; data/ane_body_bench.txt):**
  - The 2,031-row cache passed as inputs costs most of it: 20.5 ms vs 7.8 ms with a 15-row cache.
  - The DFlash2 convs cost ~4.9 ms there.
  - 48 rows add only +1.9 ms.
- **States (data/ane_state_bench.txt):**
  - The cache as CoreML states runs on the Neural Engine with a one-hot scatter write: 6.7 vs 10.8 ms as inputs.
  - A dynamic slice_update falls back to the CPU.
  - A V^T state written by `v @ onehot` fails at run time (bisect T1); the `onehot^T @ rows` write works.
- **v2 (scripts/ane_v2.py; data/ane_v2.txt):**
  - Design:
    - One stateful call drafts 3 x 16 rows.
    - Channels-first layout, every linear a 1x1 conv.
    - q_proj rows and o_proj columns permuted to (kv group, head dim, head), so GQA needs only small transposes.
    - Query-major attention over a 2,064-slot ring of K / V states.
    - DFlash2 convs by broadcasting.
    - The h / 32 fp16 fix; int8.
  - All 647 ops run on the Neural Engine.
  - **15.7-15.9 ms per batched call** with a top-32K head.
  - Back-to-back variants:
    - convs: ~0.2 ms;
    - 32K head: ~3.0 ms, of which ~0.8 is Python's fp32 copy of the logits;
    - in-model state write: ~2.4 ms;
    - attention over the 2,064-slot ring: ~6.5 ms (a 528-slot ring gives 9.3 ms in total).
  - **Native-path estimate ~13.1 ms**, not yet measured. The host writes the accepted rows into the state buffer
    (MLState; Python's write_state takes whole arrays), and the logits are read in place.
- **FR-Spec head (Zhao et al. 2025: frequency-ranked draft vocabulary; exact-safe; data/frspec_walk.txt):**
  - Ranking: 109K generated tokens from TrainLab captures, with the test samples excluded.
  - Kept per check vs the full head, MLX replay:
    | head | code | math | agent |
    |---|---|---|---|
    | top 32K static | -6.72 % | -7.39 % | (not run) |
    | top 32K + the session's own tokens (CPU) | **-0.89 %** | **-1.14 %** | **-4.07 %** |
    | top 16K + session | -1.48 % | -2.36 % | -4.93 % |
  - 86-88 % of code's misses are tokens already in the context (copied identifiers). The session extension costs
    3-11 extra tokens per sample.
- **Shorter attention window (MLX replay, full head):** agent 1,024: -3.92 %; 512: -8.61 %. Code 0.00 / -0.15 % (the
  code samples are <= 841 tokens).
- **Reading:**
  - <= 14 ms looks reachable on the native path with the full 2,048 window and a 32K FR-Spec head + session, at
    ~1 % kept on code / math and ~4 % on agent.
  - The full head does not fit (~19-20 ms).
  - Next: a Swift harness with host-side state writes and in-place logits, on the real weights (latency + cos +
    kept with the batched ring).
- **Native path, measured (18:2x; native/ane_native.swift, scripts/ane_v2.py parts=all+native, W 2,048):**
  - Setup:
    - A Swift CLI with preallocated fp16 inputs and output backings, so the logits are not copied out.
    - The model only reads the K / V ring states and attends over [ring | block i's 16 context rows | its own block
      rows].
    - After each call the host commits the accepted rows into the states (MLState.withMultiArray, 11 rows x 10
      states).
  - Random weights of the real shapes: predict 12.56 ms (p10 11.86, p90 14.03) + commit 0.11 = 12.67 ms.
  - **Real weights (top-32K FR-Spec head): predict 12.33 ms (p10 11.41, p90 12.92) + commit 0.09 = 12.42 ms per
    batched 3-candidate call: PASS (line 14).** Shared GPU (another session's walk at 60-98 %). The CPU's
    two-stage top-16, the session extras and the chain add ~0.3-0.5 ms natively.
  - **Correctness (real weights, vs the stock MLX drafter in fp32):** hidden cos mean 0.9988, min 0.9969 over 192
    rows (4 samples x 3 candidate slots x 16 rows), as ANE-1's 0.9989 / 0.9971.
  - **End-to-end kept on the Neural Engine (18:3x; ane_v2.py nwalk):**
    - Setup: v2 native on the real weights. The host keeps the ring, writes it into the states, passes the accepted
      rows since the last draft as context rows (prompts in 16-row chunks) and commits them after each call. Head:
      top-32K + the session's tokens on the CPU; two-stage top-16 + chain on the CPU.
    - Code 9.512 (670 checks), math 7.098 (968), agent 3.040 (631).
    - vs MLX with the same head: +0.3 % / -0.5 % / +5.2 %. vs the MLX full head: -0.6 % / -1.7 % / +1.0 %.
    - The agent edge is plausibly the ring's exact absolute-position window vs MLX's rotating cache on > 2,047-token
      contexts. Untested, and the set is 12 windows.
    - So the batched native call drafts as well as MLX with the same head. The head choice is the only kept cost.
  - Not run yet: a quiet re-time; 64K + session kept.

## Dense-pass check + GDNB: one-dispatch GDN body (cloud Fable's dense plan, ~/FableGod/CLOUD_ANSWER_DENSE.md; Mon 19:xx)
- **The cloud's diagnosis, checked against this table:** right in direction, wrong in its numbers. Sent to the
  orchestrator 19:0x.
  - Count: ~3,944 kernel launches per 16-row check (R-65), minus K-73's ~1,100-1,250, so ~2,700-2,850 now (not
    re-counted). The cloud said ~1,500.
  - Cost per launch: 1.3-1.6 us per dependent dispatch, ~6 us per removed tiny kernel in situ. The cloud said 10-15.
  - Host syncs: no blocking per-layer sync is left. SB queues async_eval every 4 layers, and R-55's gate does one
    PostIds / SpinWait handshake per layer, which OOO overlaps.
- **GDNB (scripts/gdnbody.py, gdnbody_test.py; ref2 CFG gdn_body, off by default):**
  - What it is: the 1-16-row GDN body as ONE dispatch, one threadgroup per value head. Inside:
    - the exact prework for every row;
    - the recurrence row by row, with the fp32 state in registers (mlx_lm's loop body verbatim);
    - the exact gated norm for every row;
    - g and beta in-kernel.
  - **Bitwise == REF2-T on layer 0's real weights:**
    - pieces q / k / v / y / conv state / recurrent state;
    - the whole layer at S 16 / 8 / 1;
    - one 16-row call == 16 one-row calls;
    - the capture-commit stash, through ref2's switch.
  - So this lever needs no REF3-T.
  - **g / beta in-kernel, proved by exhaustion (gdnb_gates_exhaust.py):**
    - compute_g replica: 0 / 113,246,208 differ (every bf16 a x 48 heads x all 36 GDN layers' A_log / dt_bias);
    - Sigmoid with metal::precise::exp: 0 / 65,536.
  - **Side finding:** a custom kernel's metal::exp is the FAST exp. Sigmoid with it differs from MLX's on 1 / 65,536
    bf16 inputs (-6.84375) and on 628 / 65,536 fp32 values. MoonShots' "exact" GDN kernels (TgSigmoid in the silu and
    the gate) inherit this, so they are not bit-identical to MLX's eager chain on rare inputs. REF2-T is unaffected:
    decode and verify run the same kernels, so SD == AR holds.
  - Kernels per 16-row layer call: 10 -> 8 (gates outside); -> 6 with the gates in.
  - **Speed:**
    - First quiet-ish read: whole layer in a 36-layer chain, reference 0.335 vs fused 0.328 ms at S 16 (-0.26 ms per
      check). Small: the fused kernel's 48 x 256 threads walk 16 state rows per simdgroup per step.
    - 16 / 32-simdgroup variants are ready (gdnbody_time.py). The quiet timing waits for MTP Head's run (ticket
      queued 19:0x).
- **GDNB quiet timing (Mon 19:13, quiet lock; data/gdnbody_time.txt):** every variant bitwise == REF2-T (y and state).
  | 16 rows, ms per layer (36 chained) | reference | fused NSG 8 | NSG 16 | NSG 32 | NSG 8, gates outside |
  |---|---|---|---|---|---|
  | GDN body | 0.0821 | 0.0861 | **0.0788** | 0.0784 | 0.0888 |
  | whole GDN layer | 0.3376 | 0.3261 | **0.3210** | 0.3351 | 0.3401 |
  - Best (NSG 16): -0.003 ms on the body, about -0.017 ms on the whole layer (the ranges overlap). So -0.1 to -0.6 ms
    per check. At 1 row, about zero.
  - **Verdict on the cloud's first kernel:**
    - Bits: PASS. Bit-identical 1-row vs 16-row and vs REF2-T, so no REF3-T is needed.
    - Speed: <= 0.2 ms per layer, PASS, but today's chain already meets it (GDNX, R-61, fused the body to 0.082 ms per
      layer).
    - Value: ~0.1-0.6 ms per check, not -3. The one-dispatch design's 48 threadgroups (one per head, so the gated norm
      stays in the threadgroup) give back in parallelism what the 4 removed launches save.
  - **Where a GDN layer's 0.338 ms goes:** the body is ~0.08 ms. The other ~0.256 ms is the projections: in_proj qkv /
    z / b / a + out_proj, ~29 MB of 4-bit weights per layer, ~2.2x the byte floor at 16 rows. GDN's 12.4 ms per check
    is ~9 ms of GEMMs, the same 16-row 4-bit GEMM efficiency problem as the HC reads (~4x floor) and the other dense
    projections.
  - GDNB stays as an exact-safe switch (ref2 CFG gdn_body, off), for a bundle if one is built.

## D1-B HC kernel B (Leap, pre-registered Mon 2026-11-02 20:5x EET; Fable mission 26 ~/FableGod/HC_READ.md design D1; orchestrator 20:4x, NewIdeas queue item 1; scripts/d1b_screen.py, data/d1b_screen.txt)

- **What:** kernel B of D1, the two-kernel bitwise HC read. One kernel replaces three: R-82's partition sum + gate
  (_ACCUM_GATE) in the prologue, a bitwise replica of MLX's K-320 bf16 up GEMM, and leapk's tail in the epilogue.
  Kernels per 16-row read 5 -> 3 (norm, S4 partials, B).
- **What MLX runs for up (M 16, N 10,240, K 320; matmul.cpp v0.32.2):** steel_gemm_fused_nax. gemv_wide declines at 16
  rows; no NAX split-K at K 320. Tiles 64 x 128 ('s' devices), 2 x 4 simdgroups of 32 x 32, BK 256 + a K tail of 64.
  Every output is the same ten 32-wide k steps, each two 16 x 32 x 16 NAX ops; the 16 rows fill the first 16-row
  fragment, whose ops don't see the zero-filled rows 16-31 (the same fact S4 relied on).
- **Design change vs Fable: no column interleave of W_up.** Each threadgroup's 4 simdgroups take the same 32 channels of
  the 4 streams: the 32-aligned column runs s x 2560 + ch0, the same runs MLX gives one simdgroup. So every output keeps
  its place in its 16 x 32 op (Fable's risk (b) does not arise) and no weight copy is needed. The 4 streams of a channel
  meet in threadgroup memory for the tail.
- **Arms:** B{CPT}p{GPF}: CPT = 32-channel groups per threadgroup (1 / 2 / 4: 80 / 40 / 20 threadgroups of 128 / 256 /
  512 threads), GPF = W_up loads issued GPF - 1 steps ahead (1 = MLX's loop, 4 = S4's); v = float4 prologue loads.
- **Checks:**
  1. The replica's up (a debug variant) == MLX's `mix @ W_up^T`, plus mixed / inject, for all 96 mixers.
  2. 96-read chains (real HC weights of all 48 layers, the pending write on, as R-09c): every read's mixed / written /
     inject == the S4 chain (leapk.hc_read with R-82 off and on).
  3. Kernels per read (census).
  4. ms per 96-read chain, arms interleaved, median of 40 after 3. Shared GPU, so rough.
- **Lines:** bits 96 / 96, else BLOCKED (one fix, not a verdict). PASS >= 1.9 ms saved per 96 reads vs S4 + R-82
  (Fable's bound for the B half: 9.9 -> 8.0). KILL < 0.8 ms. BETWEEN otherwise.
- **Prediction:** replica bitwise 85 %. Saving ~1.3 ms per 96 reads: the prologue re-reads the 61 KB of partials in
  every threadgroup (~3 us) and the tail's latency moves inside B, against two boundaries + the gate and tail kernels
  (~10-12 us per read). BETWEEN 55 %, PASS 30 %, KILL 15 %.
- **Result (Mon 21:02-21:13; data/d1b_screen.txt): exact everywhere; KILL on the pre-registered arms by 0.07 ms, ~1.0 ms
  with a post-hoc variant. Far below Fable's -2.6.**
  - **Bits:** the replica's up == MLX's `mix @ W_up^T` on all 96 mixers (scalar and float4 prologue), and mixed / inject
    too. Every arm's 96-read chain == the S4 chain (R-82 off and on), 96 / 96. Kernels per read: S4 6, S4g 5, B 3.
  - **Quiet timing (quiet lock 21:11, Kamen's apps open), ms per 96-read chain, median of 40:**

    | arm | ms | vs S4 + R-82 |
    |---|---|---|
    | S4 (today, R-82 off) | 10.59 | +0.08 |
    | S4g (S4 + R-82) | 10.51 | - |
    | pre-registered: B1p4 / B2p4 / B4p4 / B2p1 / B2p2 | 10.85 / 10.17 / 9.91 / 10.05 / 9.99 | +0.34 .. -0.60 |
    | pre-registered: **B2p4v** (best) | **9.78** | **-0.73** |
    | post-hoc: B4p4v / B4p1v / B2p1v | 9.56 / 9.66 / 9.64 | -0.95 / -0.85 / -0.87 |
    | post-hoc: **B4p2v** (best) | **9.48** | **-1.03** (-1.11 vs S4) |

  - **Verdict:** KILL on the pre-registered arms (best -0.73, line 0.8). The post-hoc best (B4p2v: 20 threadgroups of
    512 threads, float4 prologue, W_up loads 1 step ahead) is -1.03, in the BETWEEN band but not the verdict.
  - **Shared-GPU runs (rough):** -0.74 (21:02) and -1.71 (21:04, while the GPU was busy, S4 14.56 ms). The saving grows
    with GPU contention: fewer dependent launches. In the live check it may land between the quiet and the busy number.
  - **Why it is small:** per read, S4g 109.5 us vs B4p2v 98.8 us, so -10.7 us. The weights' floor is ~47 us (13.2 MB).
    The work fused away (the gate kernel, the tail kernel, two boundaries) was worth ~11-15 us. Kernel B pays some of it
    back: every threadgroup re-reads the 61 KB of fp32 partials and does 5,120 table lookups before its first MMA. The
    float4 prologue alone was worth 0.55 ms per 96 reads, so the prologue's latency is several us. The four streams
    also meet through threadgroup memory. The up GEMM runs at its bytes either way.
  - **What it says about D1:** Fable's -2.6 (bound -1.9) for the B half assumed ~25 us of removable chain tax per read
    in these kernels; the measured part is ~11. Kernel A has the same structure: the rms over full streams must come
    before the first scaled tile, ~3-7 us serial per one-simdgroup threadgroup. So D1 ~ -1.0 (B) + ~0.3-0.8 (A) =
    **~-1.3 to -1.8 ms per check, not -3.4.**
  - **Where the rest goes:** ~99 us per read vs the ~47 us floor. Mostly the two GEMMs' in-chain latency: S4's
    4,096-wide partitions each walk 128 k steps on one simdgroup, and the 2,048-wide one idles for half the walk. The
    rest is the norm kernel and the launches. A per-kernel GPU timeline (Metal System Trace) of one read would settle the
    split. An S4 whose W_d tiles are streamed into threadgroup memory by loader simdgroups (same MMA sequence, so
    bitwise) is the lever that view points at, up to ~-0.6 ms per check if S4 reaches its bytes.
  - **Keep as exact-safe:** ~1 ms per check (~1.2 % of the 84 ms check). Proposed as ref2 CFG hc_b (off by default) for
    the next exact bundle with GDNB.

## M27-P matrix-unit peak (Leap, pre-registered Mon 2026-11-02 21:1x EET; Fable mission 27 ~/FableGod/EXPERT_KERNEL.md section 0.3; orchestrator 21:1x; scripts/nax_peak.py, data/nax_peak.txt)

- **Why:** EXPERT_KERNEL.md says NAX6's ceiling is the matrix unit's rate on padded 16-row tiles, priced at Sunday's
  probe: 1.3 G op/s of 16 x 32 x 16, ~10.5 T MAC/s. That probe ran ONE dependent accumulator chain per simdgroup and
  reported only that shape; NAX6 issues 16 x 32 x 32 ops.
- **Probe:** every op shape that builds (16x32x16, 16x32x32, 32x32x16, 16x32x64, 32x32x32, 16x16x16, 16x64x16) x 1 / 2 /
  4 independent accumulator chains per simdgroup (each its own B and destination, A shared) x threadgroups of 1 / 4
  simdgroups. Operands in registers: no memory, no decode. 1,280 simdgroups x 256 ops per chain, median of 11.
- **Line (Fable's):** best >= 1.6 x 10.5 = 16.8 T MAC/s -> the NAX7 world (NAX7 alone can reach >= 85 % of streaming).
  Otherwise the NAX7 + hybrid world.
- **Prediction:** the per-MAC rate of 16 x 32 x 32 within +-10 % of 16 x 32 x 16; 2-4 chains +20-60 %. Best ~13-17 T
  MAC/s. P(>= 1.6x) = 35 %.
- **Result (Mon 21:22, quiet lock, Kamen's apps open; data/nax_peak.txt): the matrix unit tops out at ~14.0 T MAC/s =
  1.34x Sunday's 10.5, under the 1.6x line, so the NAX7 + hybrid world.**
  - Calibration: Sunday's kernel verbatim today reads 11.7 T MAC/s (Sunday 10.5), so ~10 % is run-to-run / thermal.
    Against this run's own calibration the best is 1.20x.
  - Best per op shape (1 chain, the best threadgroup size):

    | op | G op/s | T MAC/s |
    |---|---|---|
    | 16 x 32 x 16 | 1.57 | 12.8 |
    | **16 x 32 x 32 (NAX6's)** | **0.86** | **14.0** |
    | 32 x 32 x 16 | 0.76 | 12.5 |
    | 32 x 32 x 32 | 0.42 | 13.9 |

    16x32x64, 16x16x16 and 16x64x16 do not build.
  - **More independent accumulator chains do not raise the rate.** 2 chains are flat (12.1-13.9), and 3-4 chains
    collapse to 2-11 T MAC/s (register spills: the per-launch time grows 4-20x). One chain per simdgroup at 64
    simdgroups per core already saturates the unit. Threadgroup size (1-8 simdgroups) changes nothing.
  - **What it changes in EXPERT_KERNEL.md's accounting:** NAX6's op set per group, priced at 16x32x16 at 10.5 (7.32
    us), is 4,800 ops of 16x32x32 at 0.86 G op/s = 5.6 us here. That is 81 % of the group's byte time (6.95 us at 284
    GB/s), not 105 %. 85 % of streaming leaves 6.9 us per group for the body, i.e. 5.6 us of ops + ~1.3 us for
    everything else (exposed DRAM, marshalling, barriers; today ~2.8). NAX7 alone lands between Fable's 73 % (1x) and
    99 % (2x): ~80-83 % by interpolation. The hybrid (1-2-pair groups off the matrix unit) is needed for >= 85 %.
  - vs prediction: 16x32x32 within +-10 % of 16x32x16 per MAC (predicted) — it is +9 %. Chains +20-60 % (predicted) —
    wrong, 0 % or worse. Best 13-17 (predicted) — 14.0. P(>= 1.6x) 35 % — no.

## D3C 3c zero-DRAM diagnostic (Leap, pre-registered Mon 2026-11-02 21:2x EET; orchestrator 21:2x "a faster bitwise replica of 3c"; scripts/d3c_diag.py, data/d3c_diag.txt)

- **Question:** is REF2-T's expert kernel (3c: MoonShots' planes_engine tierglu, ~5.7-7.1 us per pair) bound by memory
  latency (then a bitwise replica with prefetched loads can win, as S4 did over MLX's split-K) or by ALU issue (then only
  fewer ops per weight help; Sunday's S11, decode shared across a group's rows, was slower)?
- **Test:** 3c itself on random pools (640 x 2560 experts, plan 4433333333, 16 rows x 10 ranks = 160 pairs, fused
  combine), routings: real (U 70 / 24 distinct), one (every pair reads slot 0: the same pairs, tiers and arithmetic, ~3
  MB of codes that stay in cache), all (160 distinct). Kernel A alone and A + B, interleaved, median of 25.
- **Lines (A + B at U 70):** 'one' >= 30 % faster -> latency / memory-bound, build the replica. < 10 % -> ALU-bound,
  drop the road. Between: partial, the gap is the replica's ceiling.
- **Prediction:** kernel A (25,600 short threadgroups, 4 rows per simdgroup) is latency-sensitive: 'one' 20-35 % faster.
  P(>= 30 %) = 40 %.
- **Result (Mon 21:25, shared GPU beside Fit12 at ~80 % GPU, rough; data/d3c_diag.txt): 3c is ALU / issue-bound at a
  16-row layer's union. DROP the bitwise-replica road.**

  | routing (160 pairs) | kernel A, ms | A + B, ms |
  |---|---|---|
  | real U 70 (68 distinct) | 0.844 | 1.456 |
  | real U 24 | 0.778 | 1.326 |
  | one (1 distinct: no DRAM) | 0.779 | 1.326 |
  | all (160 distinct) | 1.266 | 2.004 |

  - With every pair reading one cache-resident expert, 3c is only 8.9 % faster (A alone 7.7 %): under the 10 % drop
    line. At U 24 it is exactly as fast as with no DRAM at all.
  - So a replica with perfectly hidden loads gains at most ~9 % of 3c's time (~4-5 ms per check of ~50), and a
    prefetching replica would get only part of that. S4's win came from a latency-bound kernel with idle simdgroups;
    3c's 25,600 short threadgroups already keep the ALUs busy. Only fewer ops per weight would help, and Sunday's S11
    (decode shared across a group's rows) lost on occupancy.
  - DRAM does matter at the extreme: 160 distinct experts cost +38 % over the cache-resident case. That is the wide-
    check regime, where NAX6 / NAX7 share one decode across a group's rows.
  - Rough (shared GPU); the absolute times are inflated by Fit12's load. The direction is robust: even a quiet 2x larger
    gap would leave the replica's ceiling at ~15-20 %, short of the 1.3x the road needed.
  - vs prediction: 20-35 % predicted, 8.9 % measured. I was wrong: kernel A is not latency-bound.

## N7-0 / N7-1 NAX6 breakdown and kernel-A knockouts (Leap, Mon 2026-11-02 21:34-21:40; NAX7 groundwork, Fable mission 27 design A; scripts/n7_breakdown.py, n7_knock.py; data/n7_breakdown.txt, n7_knock.txt)

- **N7-0, the engine's C1 call by dispatch** (layer 20 real pools, real routing S 16, 16 passes, 72 groups / 60
  distinct experts per call; shared GPU beside Fit12, so inflated ~1.4x): full call 1.259 ms; kernel A 0.811, kernel B
  0.385; gpu_grp 0.042, rowsum64(x) 0.029, rowsum64(h) 0.034, combine 0.042 (the four small dispatches 0.147 ms, 12 %).
  The pieces reproduce the full call bitwise.
- **N7-1, kernel A alone with one piece knocked out** (same data, GPU otherwise free):

  | arm | ms per call | vs full |
  |---|---|---|
  | full (NAX6 kernel A) | 0.575 | - |
  | no DRAM footprint (DIAG 3) | 0.477 | -17 % |
  | **no matrix ops** | **0.368** | **-36 %** |
  | no A-operand loads | 0.571 | -1 % |
  | no decode (raw codes) | 0.534 | -7 % |

  - Kernel A's op set at M27-P's rate is 0.268 ms (72 groups x 3,200 ops of 16x32x32 at 0.86 G op/s). Its bytes are
    ~0.30 ms (60 experts x 1.43 MB at 284 GB/s).
  - **Without the ops the kernel is memory-bound** (0.368 ms ~ its bytes at ~235 GB/s). **With them it adds 0.207 ms
    of its 0.268**: the matrix unit is ~80 % exposed, not overlapped with the stream.
  - A loads cost nothing (L1 hits); the decode ~7 %.
  - So the lever is overlap, not fewer instructions: each simdgroup waits for its codes, decodes, runs two ops and
    consumes cg at once, and the 8 simdgroups of a 20 KB threadgroup (likely one per core) are too few to cover it.
    Perfect overlap would put kernel A near max(bytes, ops) ~0.37-0.40 ms (-30-35 %).
- **NAX7 step 1 (next):** kernel A software-pipelined, bitwise = NAX6: the codes of group g+1 (and g+2) loaded into
  registers before group g's ops, the decode of g+1 between op.run(g) and g's epilogue (two cg destinations
  alternating), the same per-output op sequence and chunk sums.

## N7-2 NAX7 step 1: kernel A software-pipelined (Leap, pre-registered Mon 2026-11-02 21:4x EET; scripts/tierglu_nax7.py, n7_screen.py; data/n7_screen.txt)

- **What:** NAX6's kernel A with the same per-output arithmetic and a new schedule. The codes / LSB / scale / bias of
  quant group g + 2 are loaded into registers right after group g + 1 is decoded, and that decode runs between
  op.run(g) and g's epilogue. So the loads get a full iteration to arrive and the decode overlaps the matrix unit.
  Every output keeps NAX6's sequence (cg = 0, two 16x32x32 ops, acc += s cg + b xsum; chunk sums in order; swiglu).
  Kernel B and the small dispatches stay NAX6's.
- **Test:** real layer-20 pools, real routing (4 traces). Bits on 96 cases (4 traces x S 8 / 16 / 32 x 8 passes):
  kernel A's h and the whole call's y vs NAX6. Time: kernel A alone and the whole call (+ combine), at S 16 / 8 / 32.
- **Lines:** bits 96 / 96, else BLOCKED. PASS kernel A >= 20 % faster at S 16. KILL < 5 %.
- **Prediction:** bitwise 90 %. Kernel A 0.575 -> ~0.45 ms (-20-25 %): about half of the 0.21 ms of exposed ops
  hidden. The call ~-10-15 %. PASS 50 %, BETWEEN 35 %, KILL 15 %.
- **Result (Mon 21:42, shared lock, GPU ~50 % free; data/n7_screen.txt): bits PASS 96 / 96 (h and y), but KILL: the
  pipelined kernel A is SLOWER.** S 16: 0.569 -> 0.632 ms (+11 %); S 8 +5 %; S 32 +6 %; the whole call +2-3 %.
  - The same as Sunday's K-62c (prefetching the next group's codes: nothing). So within one simdgroup the matrix op
    blocks: putting the decode after op.run overlaps nothing, and the prefetched registers cost occupancy.
  - So the overlap N7-1 asks for must come from OTHER simdgroups on the core. Next: is kernel A occupancy-bound
    (threadgroup memory 20 KB per 8-simdgroup threadgroup)? Then a producer / consumer split (loader simdgroups decode
    into a ring, op simdgroups only run ops) is the NAX7 structure to build.
- **N7-3 (Mon 21:45, shared lock; data/n7_occ.txt): kernel A is not threadgroup-memory-bound.** Padding its
  threadgroups from 20 to 24 / 28 / 32 KB changes nothing (0.562 / 0.558 / 0.567 / 0.566 ms; bits equal).
- **N7-4 (Mon 21:46, shared lock, rough; scripts/nax_tg_probe.py, data/nax_tg_probe.txt): the matrix unit runs NAX6's
  operand form at ~the register rate.** 16x32x32 with B in registers 0.77-0.81 G op/s; B as uint4 from threadgroup
  memory (NAX6's form) 0.76-0.77; B as bf16 from threadgroup memory 0.70-0.79; uint4 + NAX6's per-pair epilogue
  0.60-0.71. So kernel A's op set is ~0.33 ms with its epilogue, against 0.368 ms of memory-bound work: the two are
  simply not overlapped (0.575 ~ 0.37 + 0.21).
- **What N7-1..4 say together:** not DRAM-, decode-, occupancy- (threadgroup memory) or op-path-bound; within one
  simdgroup the op blocks (N7-2). The overlap has to come from simdgroup roles: **NAX7-WS**, loader simdgroups that
  fetch and decode two stages ahead into a ring in threadgroup memory while op simdgroups run only NAX6's op sequence.

## N7-5 NAX7-WS kernel A: loader / op simdgroups (Leap, pre-registered Mon 2026-11-02 21:4x EET; scripts/tierglu_nax7.py kernel_a7w, n7_screen.py N7_ARM=ws; data/n7_screen.txt)

- **What:** 16 simdgroups per threadgroup (NAX6: 8). Simdgroups 0-7 are NAX6's (projection x K chunk) and run only its
  op sequence on decoded tiles. 8-15 are their loaders: loader 8 + c fetches quant group g + 1's codes / LSB / scale /
  bias into registers and decodes it into a two-slot ring while op simdgroup c runs group g; one threadgroup barrier
  per stage, in uniform control flow. The chunk sums and the swiglu are NAX6's. Bitwise == NAX6 by construction.
- **Test, lines:** as N7-2 (bits 96 / 96 h and y, else BLOCKED; PASS kernel A >= 20 % faster at S 16; KILL < 5 %).
- **Prediction:** bitwise 90 %. Kernel A 0.57 -> ~0.40-0.45 ms (-20-30 %): the loaders' DRAM waits now overlap the
  op simdgroups' ops. The risk is the per-stage barrier: a stage lasts as long as its slowest loader, so the loaders'
  one-stage fetch lead may still expose DRAM latency. PASS 50 %, BETWEEN 30 %, KILL 20 %.
- **Result (Mon 21:49, shared lock, GPU ~50 % free): bits PASS 96 / 96, but KILL: 35 % SLOWER.** Kernel A S 16
  0.562 -> 0.758 ms; S 8 +43 %; S 32 +31 %; the whole call +18-23 %.
  - The stage barrier makes every stage as long as its slowest loader's DRAM wait plus the barrier, for all 16
    simdgroups. NAX6's simdgroups run free (simdgroup barriers only), so the scheduler already interleaves their
    phases. Lockstep staging throws that away.
  - Two schedule changes (N7-2, N7-5), both bitwise, both slower. NAX6's free-running structure is a local optimum for
    this op form. Next, measure what limits it: how many kernel-A threadgroups a core keeps resident (time vs grid
    size with identical per-threadgroup work).

## N7-6 / N7-7 residency and the matrix unit's arithmetic (Leap, Mon 2026-11-02 21:51-21:57; scripts/n7_resid.py, nax_order_probe.py, nax_order_fit.py; data/n7_resid.txt, nax_order_probe.txt, nax_order_fit.txt)

- **N7-6 (shared lock, rough):** NAX6 kernel A on G one-pair groups (20 G threadgroups of identical work), dependent
  chain. Below G ~ 8 the time is a ~40-70 us latency floor. From G ~ 10 (10 threadgroups per core) it rises linearly at
  **~6.6-7.6 us per group with ~no fixed cost**. The whole call adds ~4.4 us per group (kernel B) and only ~10-20 us
  per call for the small dispatches in a dependent chain (less than N7-0's standalone 0.147 ms suggested).
- **N7-7 (shared lock), the matrix unit's per-output arithmetic** (16x32x32 in NAX6's form; A bf16 with exponents
  2^-12..2^12, B nibbles, C fp32; 1,024 tests x 512 outputs):
  - The element mapping is right: integer operands (every order exact) match 100 %.
  - Adversarial operands: **no fp32 order matches.** Sequential with C first 38 %, C last 33 %, pairwise tree 37 %,
    16-halves 38 %, 4-adjacent groups 42 %, exact-then-one-rounding 45 %, exact 16- / 8-wide chunks 44 / 41 %.
  - Against the exact sum: 45 % bitwise, 39 % off by 1 ulp, 14 % by 2-8, 2.4 % by more. The errors are sign-symmetric.
  - The descriptor's relaxed-precision flag changes nothing (identical outputs).
  - N7-7b (CPU fit): block floating-point adder models (chunks of 4 / 8 / 16 / 32, 20-32 kept fraction bits after
    aligning to the chunk's largest exponent, truncate / round / floor, C first / last / inside the block) top out at
    45.6 %.
  - **So the unit's datapath is not a reproducible fp32 sequence.** Fable's hybrid (1-2-pair groups on the ALUs) cannot
    be bitwise to NAX6. It also makes a pair's bits depend on its group's size (companion rows), which breaks the row
    invariance SD == AR needs, so it is dead for any exact reference, not only for REF2-T.
- **The NAX road after tonight:**
  - The measured peak is 1.34x (M27-P), so NAX6's op set per group is ~4.6 us in its own form (0.70 G op/s with its
    epilogue).
  - Its kernel A runs ~7-8 us per group against ~4.2-5.0 us of bytes. ~57 % matrix-unit and ~53 % DRAM utilization:
    latency-bound with too little concurrency per core.
  - Two schedule rewrites (N7-2 pipelined, N7-5 loader / op simdgroups) were bitwise and slower. Threadgroup memory
    doesn't matter (N7-3), and NAX6's operand form runs at the register rate (N7-4).
  - Left exact to NAX6: the dispatch fusion (small in a dependent chain; to be measured quiet) and register trimming,
    if registers limit residency (untested).
  - Left as reference changes: the tier merge (B: -15-17 % of groups; the NAX reference needs certifying anyway) and
    the quality-defined int8 (C).
- **N7-8 (Mon 21:57, shared lock beside Fit12 at ~80 % GPU; scripts/n7_regs.py, data/n7_regs.txt): registers matter
  only at the edge.** Kernel A with extra live registers per thread: +16 +2 %, +32 +5 %, +64 +41 % (a cliff: residency
  halves or spills). Bits equal. So trimming registers gains a few % at most: not the main limiter. (c), A + B fused
  per group with completion counters, is the remaining exact candidate; its gain is A's last-wave tail and B's ramp.

## H-1 hybrid on the FAST track (Leap, pre-registered Mon 2026-11-02 22:0x EET; orchestrator 22:0x under Kamen's two-track rule; scripts/h1_hybrid.py, data/h1_hybrid.txt)

- **Why:** NAX6 pays its full 16-row op set per group (~11.8 us for A + B), and 57 % of a 16-row layer's groups hold one
  pair (20 % two). 3c computes a pair in ~5.7 us on the SIMD ALUs. Bits differ (1-8 ulp vs NAX6) and depend on group size
  (no row invariance), which the FAST track allows if quality holds. EvalSuite judges quality later.
- **Arms** (real layer-20 pools and routing, 4 traces, S 8 / 16 / 32; whole expert call with the LY = 8 combine):
  - NAX6, all pairs (the C1 call).
  - H1: one-pair groups -> 3c (tier_glu_pairs), the rest -> NAX6.
  - H2: groups of <= 2 pairs -> 3c.
  - Both halves are launched in one eval, so the GPU can overlap them. The grouping is on the host in this screen; the
    engine would need a GPU split of gpu_grp.
- **Lines:** speed only. PASS: H1 or H2 >= 15 % faster than NAX6 at S 16. KILL: < 5 %. Also reported: S 8 / 32, and how
  many outputs differ vs NAX6 (it should be the routed small groups' rows only).
- **Prediction:** H1 ~-25 % at S 16 (41 singleton groups x (11.8 - 5.7) us saved of ~0.85 ms), H2 ~-25 % too (2-pair
  groups tie); smaller at S 32 (fewer singletons), larger at S 8. PASS 60 %.
- **Result (Mon 22:03, shared lock, GPU 50-75 % free; data/h1_hybrid.txt): the pre-registered arms KILL (+1-3 %).
  The post-hoc compact arm reaches +10 % at S 16: BETWEEN band, not the verdict.**

  | S | NAX6 ms | H1 (3c via tier_glu_phase) | H2 | H1c (3c on a compact pair list) | H2c | pairs on 3c (H1 / H2) |
  |---|---|---|---|---|---|---|
  | 16 | 0.908 | +3.3 % | +1.5 % | **+10.4 %** | +6.7 % | 24 / 42 % |
  | 8 | 0.592 | +3.3 % | +2.5 % | **+15.0 %** | +12.1 % | 32 / 52 % |
  | 32 | 1.343 | +2.1 % | +0.7 % | +6.1 % | +3.9 % | 16 / 29 % |

  - tier_glu_phase launches 3c's full pair grid (25,600 kernel-A threadgroups, most exiting at once), which eats the
    gain. A compact pair list (tier_glu_pairs + the OOO split's combine) is the arm that works. In the engine it needs
    the split on the GPU (gpu_grp can emit the small groups' pairs), so the engine number would land between the two.
  - Singleton groups are 57 % of the groups but only 24 % of the pairs. Moving them off the matrix unit saves their op
    sets, but 3c's ~5.7 us per pair is not free, and the remaining NAX6 groups keep their cost. So ~10 %, not Fable's
    21-30 %.
  - Bits: 75-100 % of output rows differ from NAX6 (max |diff| / max |y| ~2e-2: the 3c vs NAX rounding of one
    expert's rows); quality is EvalSuite's on the FAST track.
  - FAST-track value: ~-4 ms per 16-row check (~10 % of the ~43 ms expert term) with a GPU split. Small next to the gate's
    union cut; worth a slot only if the FAST stack wants it.
- **N7-9 (Mon 22:09, shared lock, GPU ~94 % free; scripts/n7_gap.py, data/n7_gap.txt): the A + B fusion ceiling, in
  dependent chains.** S 16: call 0.849 ms = A 0.490 + B 0.231 + **0.128 outside them (15 %)**; the A -> B boundary
  alone +0.025. S 32: 1.385 = 0.851 + 0.410 + 0.124 (9 %); boundary +0.030. So fusing A into B buys ~0.03 ms. The
  real fixed cost is the four small dispatches (gpu_grp, rowsum64(x), rowsum64(h), combine) and their boundaries, ~0.10
  ms per call, ~4.8 ms per 48-layer check. That is the exact lever left on the NAX road (twice that in OOO's two phases).

## N7-10 NAX7-F: NAX6 in three dispatches (Leap, pre-registered Mon 2026-11-02 22:1x EET; scripts/tierglu_nax7.py tier_glu_nax7f, n7f_screen.py; data/n7f_screen.txt)

- **What:** the same per-output arithmetic as NAX6 + combine_phase, with no spin-waits:
  - K0: gpu_grp's grouping (threadgroup 0), rowsum64(x) (the others), and the zeroing of the counters, in one dispatch.
  - K1: NAX6's kernel A. Each threadgroup, after writing its h tile, fences device memory and bumps its group's
    counter; the last of the group's 20 threadgroups computes the group's h row sums (rowsum64's sequential order).
  - K2: NAX6's kernel B. After writing its y tile it fences and bumps a counter per (row, 64-column block) for each
    pair of its group; the 10th arrival does that row block's combine (combine_phase's LY = 8 order, TgMultiply).
- **Lines:** bits 96 / 96 (the combined output vs NAX6 + combine_phase), else BLOCKED. PASS: whole call >= 8 % faster
  at S 16. KILL: < 3 %.
- **Prediction:** bitwise 80 % (the fences / last-arriver visibility are the risk). Call -0.06..-0.09 ms at S 16
  (-7-10 %). PASS 50 %.
- **Result (bits: Mon 22:14 shared + 22:41 quiet; timing: quiet lock 22:41; data/n7f_screen.txt): bits PASS (96 / 96
  at 22:14, 64 / 64 at 22:41), but KILL: +1.2 % at S 16 (0.920 -> 0.909 ms), +5.7 % at S 8.**
  - The shared run at 22:14 (Breakthrough at 50 % GPU) read -4.6 % / +2.4 % with NAX6 at ~2x its quiet time. Its S 32
    timing then hung: 20 min blocked in MLX's wait for a GPU event, 0 % CPU. I stopped it by PID at 22:35; the GPU served
    the other jobs throughout. The cause is unknown: S 32's bits had passed, and the kernels have no spin-waits. S 32 was
    left out of the quiet run.
  - So N7-9's 0.128 ms "outside A and B" is not what dispatches cost: removing three dispatches and two small kernels
    returns ~0.01 ms. Most of it is the grouping and row-sum work itself (still in K0) and the fill / drain of each
    kernel, not the boundaries.
- **The exact NAX road is closed beyond NAX6 itself.** Five bitwise restructurings: pipelined (N7-2) +11 % slower,
  loader / op simdgroups (N7-5) +35 % slower, dispatch fusion (N7-10) +1 %. Residency, registers and operand form are
  not the limit (N7-3 / 4 / 8). The hybrid can't be bitwise (N7-7). What is left on the NAX road: the tier merge
  (reference change, with the NAX reference) and, on the FAST track, the compact hybrid (H-1: +10 % at S 16).

## G3 the row-interleaved GDN recurrence (Leap, pre-registered Mon 2026-11-02 22:4x EET; Fable B GDN_PASS.md G3; orchestrator 21:25, NewIdeas' order; scripts/gdnbody.py g3, gdnbody_test.py, gdnbody_time.py)

- **What:** GDNB's phase B steps each simdgroup's RPS state rows one after another (decay + dot, simd_sum, delta,
  update + dot, simd_sum, per row). G3 interchanges the loops: all RPS rows' decay + dot, then their RPS reductions,
  then the deltas, then the updates + dots, then the reductions, so RPS independent chains are in flight per step.
  Every element's operations and operands are unchanged, so it should be bitwise. Switch: gdn_body(..., g3=True), off
  by default (ref2's GDNB path never passes it).
- **Test:** gdnbody_test with G3 at NSG 8 and 16 (pieces, layer at S 16 / 8 / 1, 16-row == 16 one-row calls), then
  gdnbody_time's body chain at S 16 and S 1 (row term = body(S 16) - body(S 1)).
- **Lines (Fable's):** bits as REF2-T, else BLOCKED. PASS: the row term halves (NSG 8: 0.042 -> <= 0.021 ms per
  layer). KILL: it falls < 20 %.
- **Prediction:** bitwise 90 %. The row term falls ~30-50 % (the simd_sums' latency overlaps across rows, but at RPS
  16 the extra registers may cost residency). PASS 35 %, BETWEEN 45 %, KILL 20 %.
- **Result (Mon 22:42-22:43; bits: shared lock; timing: shared lock beside Fit12's engine build at ~80 % GPU, rough;
  data/gdnbody_test.txt, gdnbody_time.txt): bitwise everywhere, but a rough KILL: the row term falls only 4-8 %.**
  - Bits: G3 at NSG 8 and 16, every check == REF2-T: pieces (q, k, v, y, conv and recurrent state), the layer at S 16 /
    8 / 1, 16-row == 16 one-row calls. The default path (G3 off) re-checked: unchanged.
  - Body per layer (ms, 36 chained; the busy GPU inflates everything ~1.5x):

    | layout | S 16 | S 1 | row term (S 16 - S 1) |
    |---|---|---|---|
    | REF2-T chain | 0.136 | 0.051 | 0.085 |
    | GDNB NSG 8 | 0.119 | 0.050 | 0.069 |
    | **G3 NSG 8** | 0.115 | 0.049 | **0.066 (-4 %)** |
    | GDNB NSG 16 | 0.110 | 0.051 | 0.059 |
    | **G3 NSG 16** | 0.102 | 0.048 | **0.054 (-8 %)** |
    | G3 NSG 32 | 0.098 | 0.050 | 0.048 |

  - Interleaving the rows' chains barely moves the row term, so the serial simd_sum chain is not what costs. The
    compiler or the other resident simdgroups already hide its latency, and Fable B's waves model (waves x RPS x 0.175
    us) does not hold.
  - G3 stays in gdnbody.py as a bitwise option (off). If a quiet slot comes free I can confirm the number; the gap to
    the 20 % kill line is too large for the contention to explain.

## M27-C the int8 x int8 matrix-unit probe (Leap, pre-registered Mon 2026-11-02 22:4x EET; EXPERT_KERNEL.md design C's gate; orchestrator 22:5x; scripts/nax_int8_probe.py, data/nax_int8_probe.txt)

- **Why:** NAX6 is bound by its op set on padded 16-row tiles (N7-1..8). If matmul2d runs int8 x int8 -> int32 at
  ~2x the bf16 rate, the op set halves and the kernel turns byte-bound: Fable's design C, ~43 -> ~26 ms per 16-row
  check. FAST track: activations quantized per (row, 64-group) to int8; integer sums are order-free (row-invariant).
- **Probe:** M27-P's register-operand loop at 16 x 32 x 32 for operand types bf16 x bf16 -> fp32 (the baseline), int8
  x int8 -> int32, int8 x int8 -> fp32, uint8 x int8 -> int32, half x half -> fp32 (for reference); 1 / 2 / 4
  independent accumulators; threadgroups of 4 simdgroups; 1,280 simdgroups x 1,024 ops. A type pair that does not
  build means no such mode.
- **Lines (the orchestrator's):** PASS: an int8 mode at >= 1.6x bf16's best rate -> design C kernel + a Breakthrough
  map arm for flips. KILL: no int8 mode, or < 1.2x. BETWEEN otherwise.
- **Prediction:** an int8 mode builds (60 %); if it does, its rate is 1.0-2.0x bf16 (the M5's "4x AI compute" claim
  hints at int8 throughput, but MPP may route int8 through the same bf16 datapath). PASS 30 %, KILL 50 %.
- **Result (Mon 22:45-22:46, shared lock beside Fit12's engine build at ~80 % GPU, rough; data/nax_int8_probe.txt):
  PASS by the line on the register form; the threadgroup 4-bit form design C would use is 1.55x NAX6's.**
  - The first run used plain `char`, which MPP rejects (its operand list is uint8_t / int8_t / uint4b / int4b / float
    / half / bfloat; plain char is a distinct C++ type). With int8_t / uint8_t:

    | form (16 x 32 x 32) | G op/s | T MAC/s | vs bf16 |
    |---|---|---|---|
    | bf16 x bf16 -> fp32 (registers) | 0.899 | 14.7 | 1.00x |
    | **int8 x int8 -> int32 (registers)** | **1.559** | 25.6 | **1.73x** |
    | uint8 x uint8 -> int32 (registers) | 1.526 | 25.0 | 1.70x |
    | half x half -> fp32 (registers) | 0.929 | 15.2 | 1.03x |
    | bf16 x uint4 (threadgroup) -> fp32 = NAX6's form | 0.755 | 12.4 | 1.00x (of NAX6) |
    | **int8 x int4 (threadgroup) -> int32** | **1.169** | 19.2 | **1.55x of NAX6's form** |
    | uint8 x uint4 (threadgroup) -> int32 | 1.019 | 16.7 | 1.35x of NAX6's form |

  - The header also implements int8 x int4 / uint8 x uint4 with the 4-bit operand from device memory. More than 2
    chains is again slower.
  - For design C: the 3- / 4-bit codes go in as int4 (code - 8, the shift folded into the bias term) against int8
    activations per (row, 64-group). Integer sums are exact, so row-invariant. The op set per group falls from ~4.6
    to ~3.0 us, below the group's ~4.2-5.0 us of bytes. NAX6's overlap problem (N7-1) still applies, so the kernel
    gain is likely ~20-30 % rather than Fable's ~40 %.

## NAX-C design C on the FAST track (Leap, pre-registered Mon 2026-11-02 22:4x EET; EXPERT_KERNEL.md design C; M27-C PASS -> the orchestrator's next step; scripts/tierglu_naxc.py, naxc_screen.py; data/naxc_screen.txt)

- **What:** NAX6's kernels and grouping with the matrix ops in int8 x int4 -> int32:
  - activations quantized per (row, 64-group) to int8 (scale = max|x| / 127, round to nearest; a pass that replaces
    rowsum64), for x into kernel A and h into kernel B;
  - the 3- / 4-bit codes decoded as before and stored as int4 (code - 8: one XOR per word);
  - per group: cg = sum of x_q (q - 8) (exact int32), then
    acc += s_x (s (cg + 8 sum x_q) + b sum x_q), the same affine dequant as NAX6, on the dequantized activations.
  - Integer sums are exact, so a row's result does not depend on its companions (row-invariant).
- **Test:** real layer-20 pools and routing (4 traces, S 8 / 16 / 32):
  - quality: max and mean |y - y_ref| / RMS(y_ref) against an fp32 reference of the same weights (NAX6 as a second
    reference);
  - speed: the whole call (+ combine) in dependent chains, NAX-C vs NAX6.
- **Lines (FAST, speed):** PASS >= 15 % faster than NAX6 at S 16; KILL < 5 %. Quality is reported (EvalSuite / the
  Breakthrough map decide it later).
- **Prediction:** N7-1's knockouts put ~64 % of kernel A's time outside the ops. 1.55x faster ops therefore save
  ~0.27 x the op time: ~-12-16 % on the call. PASS 40 %. Mean error ~5e-3 x RMS (mission 27's 5.8e-3).
- **Result (Mon 22:5x, shared lock beside ojox-e8 at ~80 % GPU; data/naxc_screen.txt): KILL on speed. NAX-C = NAX6
  (S 16 -0.3 %, S 8 -1.0 %, S 32 -0.1 %), under both lines (mine < 5 %, the orchestrator's < 8 %).**
  - Quality vs fp64, |err| / RMS, one S 16 pass per trace:

    | trace | NAX-C mean / max | NAX6 mean / max | 3c (REF2-T's kernel) mean / max |
    |---|---|---|---|
    | code | 1.08e-2 / 7.5e-2 | 3.6e-3 / 2.5e-2 | 1.23e-2 / 9.6e-2 |
    | chat | 1.07e-2 / 7.5e-2 | 3.6e-3 / 3.2e-2 | 1.26e-2 / 9.2e-2 |
    | math | 1.06e-2 / 7.5e-2 | 3.6e-3 / 2.6e-2 | 1.25e-2 / 1.0e-1 |
    | bulgarian | 1.07e-2 / 8.7e-2 | 3.6e-3 / 3.4e-2 | 1.26e-2 / 1.05e-1 |

    int8 activations put design C at ~3x NAX6's error but slightly better than 3c, today's exact default (its fp32
    chains and bf16 intermediates sit 1.25e-2 from fp64).
  - **Why no speed:** the faster unit throughput (M27-C: 1.55x in this form) never shows. Each simdgroup waits on its
    own ops' latency (two dependent ops per quant group) and on its codes; the unit's throughput is not the binding
    term. Same lesson as N7-2 / N7-5: NAX6 is latency-bound per simdgroup with too little concurrent work per core.
  - What would attack that directly, not tried yet: loads that need no registers, i.e. Metal's simdgroup async copies
    from device into threadgroup memory several stages ahead (if the M5's MSL still supports them), so each simdgroup
    has several groups' codes in flight without the register cost that sank N7-2.

## N7-11 NAX6's split-K chunk counts at 16 rows (Leap, pre-registered Mon 2026-11-02 22:5x EET; FAST track, row-invariant; scripts/n7_ks.py, data/n7_ks.txt)

- **Why:** NAX6 is latency-bound per simdgroup (N7-1..10, NAX-C). Its chunk counts (kernel A: 4 K-chunks per
  projection, kernel B: 2) were chosen for 1-3-row calls (K-62j). More chunks = more, shorter simdgroups per group =
  more independent work in flight per core. The bits change with the chunk count (other chunk sums), but each count
  is a fixed structure, so rows stay invariant (a new NAX reference per count, FAST until certified).
- **Arms:** (ks, ks_b) = (4, 2) NAX6, (2, 1), (2, 2), (4, 1), (5, 2), (4, 5), (5, 5) (threadgroup memory caps ks at
  5); the whole call + combine, dependent chains of 16, real layer-20 routing, S 16 / 8 / 32, 4 traces x 3 passes.
- **Lines:** PASS: an arm >= 8 % faster than NAX6 at S 16. KILL: none >= 3 %.
- **Prediction:** (5, 5) or (5, 2) wins ~3-6 % at S 16 (more simdgroups, but the chunk-sum exchange grows). PASS 25 %.
- **Result (Mon 22:55-22:58, shared lock, GPU ~55 % free; data/n7_ks.txt): KILL. NAX6's own (4, 2) is the best at S
  16; every other count is within -0.2..-3.2 %.** Row invariance holds for every arm (row 0 alone == inside the
  16-row call).

  | (ks, ks_b) | S 16 | S 8 | S 32 |
  |---|---|---|---|
  | (4, 2) NAX6 | 1.182 ms | 0.579 | 1.465 |
  | (2, 1) | -1.0 % | -2.1 % | +1.7 % |
  | (2, 2) | -0.2 % | -0.9 % | +1.4 % |
  | (4, 1) | -1.2 % | -1.1 % | -0.0 % |
  | (5, 2) | -0.7 % | -2.4 % | -1.1 % |
  | (4, 5) | -3.2 % | +1.3 % | -2.9 % |
  | (5, 5) | -3.1 % | -1.2 % | -4.2 % |

  - More, shorter simdgroups don't buy overlap either. The expert kernel's per-group cost sits where NAX6 has it
    across every schedule, op type and split tried tonight. The levers left are fewer groups (tier merge, the gate's
    union cut) and fewer bytes, not the kernel's inner structure.

## D-COMMIT mission 29's design D screen (Leap, written and pre-registered Mon 2026-11-02 23:0x EET for Tue after 15:00, the one big job by day; ~/FableGod/DRAFT_HIDDEN.md design D; scripts/d_commit_screen.py, data/d_commit_screen.txt)

- **What:** hide the commit replay behind the next draft by submitting the draft on a second MLX GPU stream right after
  the check (acceptance on the GPU, R-11's G1). Exact: only the stream assignment changes. A same-state check drafts one
  block on both streams (tokens equal) before the timing.
- **Harness:** R-11's (CA's resident 12-layer slice + DFlash2 L1-20, 16-row checks, keep 10; ~20.5 GB active). Arms
  SB (the engine's cycle), G1, D; interleaved rounds, paired medians.
- **Lines:** PASS D >= 0.6 ms per cycle faster than G1 at 12 layers (~2.4 ms at 48: most of the commit's ~3 ms hidden).
  KILL < 0.2 ms.
- **Prediction:** exact 95 %. D -0.4..-0.8 ms per cycle at 12 layers: the commit replay (~0.75 ms at 12 layers) overlaps
  the draft if MLX's second stream runs concurrently on the GPU. PASS 45 %.
- **Result (Tue 15:00-15:01, share lock "MEM 21 GB" as the one big job; data/d_commit_screen.txt): KILL, exact.**
  - Exactness: the draft on the second stream == the default stream (tokens).
  - 5 rounds x 8 cycles per arm, none dropped; every commit landed; 20.5 GB active.
  - Per-cycle wall (median of rounds; 12 layers + the drafter): SB 50.56 ms, G1 51.77, D 51.66.
  - Paired: G1 - D **0.06 ms** per cycle (the KILL line is < 0.2; ~0.2 ms at 48 layers). SB - G1 -0.06, SB - D -1.10:
    all three arms sit within the ~1 ms noise of 5 rounds.
  - Reading: the commit replay is either already overlapped (the engine's SB submits it asynchronously), or MLX's
    second stream does not run concurrently with it on this GPU. Hiding the draft behind the commit by stream assignment
    buys nothing measurable. Prediction (-0.4..-0.8 ms) wrong.
- **Quiet confirmation of H-1c (FAST, the orchestrator's ask 23:0x), pre-registered for the same Tuesday slot:**
  the compact hybrid (one-pair groups on 3c via tier_glu_pairs, the rest on NAX6) vs NAX6 at S 16 / 8 / 32, quiet lock,
  dependent chains. Line: confirm >= +8 % at S 16 (the shared run read +10.4 %), else drop it from the FAST stack.

## NAX-REF decision brief for Kamen (Leap, Mon 2026-11-02 23:0x EET; the orchestrator's ask: three lines, put to Kamen Tue)

- **What a NAX reference (REF2-T with the experts on NAX6, ref2 CFG tier_nax_ooo = C1) buys:**
  - Speed: the expert term at 16 rows, NAX6 43.2 vs 3c 50.5 ms per check standalone (C2). Inside OOO's two phases the
    groups fragment: 1.26x of 3c instead of 1.75x standalone; BigMoE's live C1 A/B at C 256 runs in tonight's
    01:00-05:30 ticket.
  - On top, only with this reference: tier merge B (-15-17 % of the groups, ~-5 ms per check, 0.02 % of outputs move
    by one bf16 ulp vs NAX6).
  - Accuracy: NAX6 is ~3.5x closer to fp64 than 3c (mean |err| / RMS 3.6e-3 vs 1.25e-2, NAX-C table above), and it is
    row-invariant by construction (fixed split; INV checked at S 1 / 8 / 16 / 32).
- **What certifying it needs (the REF2 / REF2c path):** freeze the code (C1 + B if built); EvalSuite regenerates the
  210 core rows (math / IFEval / HumanEval+) under the new reference's AR; the live audit (spec loop == its own AR at C
  256); one same-scores test vs REF2-T; and, after REF2c-T's IFEval failure, the IFX re-test too. Roughly one night of
  the 45 GB route (regen + audit, ~5-7 h) plus an IFX slot.
- **The risk:** NAX's bits differ from 3c's in ~70 % of expert outputs (1-8 bf16 ulps). That is the scale at which
  REF2c-T failed IFEval (-8), even though NAX6 is the more accurate of the two, so a same-scores failure would be
  IFEval's noise floor, not worse arithmetic. The engine gain depends on OOO's phase sizes (1.26x, not 1.75x). The exact
  NAX road beyond NAX6 is closed (N7-2..11), so this reference buys NAX6 + tier merge and no more kernel headroom.

## H-1i the in-kernel hybrid, NAX6-H (Leap, pre-registered Mon 2026-11-02 23:0x EET; FAST; for BigMoE's fastBEh arm, switch by Tue ~08:00; scripts/tierglu_nax6h.py, h1i_screen.py; data/h1i_screen.txt)

- **What:** H-1c's idea in the engine's form. Inside NAX6's own kernels A and B, a group holding ONE pair (nq == 1, from
  NAX6's grouping) skips the 16-row matrix op set. Each lane decodes its column's 64 codes per quant group exactly as
  NAX6 does and takes an fp32 FMA dot with the pair's activations, then NAX6's affine epilogue and chunk sums.
  - No pair list, host sync, extra dispatch or combine; it runs unchanged in OOO's phases.
  - Duplicate (expert, tier) pairs from gated rows form 2+ pair groups and stay on the matrix path.
  - hybrid=False compiles NAX6 exactly.
- **Test:** bits (hybrid off == NAX6, 64 cases); quality vs fp64 (K-62g's deq) and the deviation vs NAX6 (mean / max,
  for Breakthrough's map); speed of the whole call + combine in dependent chains vs NAX6 and H-1c's compact split, at S
  16 / 8 / 32 / 4 / 2 / 1.
- **Lines (FAST speed):** PASS NAX6-H >= +8 % vs NAX6 at S 16. KILL < +3 %.
- **Prediction:** +10-20 % at S 16 (no separate 3c dispatches or combine; the ALU path per one-pair group ~2-3 us vs
  ~7 us on the matrix path), more at S <= 4 where most groups are singletons. Error vs fp64 between NAX6's and 3c's.
  PASS 65 %.
- **Result (Mon 23:06-23:07, shared lock beside ojox-e8 at ~80 % GPU, rough; data/h1i_screen.txt): PASS, +8.0 % at S
  16; faster than NAX6 at every width and than H-1c everywhere.**

  | S | NAX6 ms | NAX6-H | H-1c (compact split) |
  |---|---|---|---|
  | 16 | 0.940 | **0.864 (+8.0 %)** | 0.905 (+3.7 %) |
  | 8 | 0.623 | 0.558 (+10.4 %) | 0.574 (+7.9 %) |
  | 32 | 1.482 | 1.382 (+6.7 %) | 1.423 (+4.0 %) |
  | 4 | 0.414 | 0.374 (+9.8 %) | 0.421 (-1.6 %) |
  | 2 | 0.287 | 0.258 (+9.9 %) | 0.279 (+2.8 %) |
  | 1 | 0.200 | 0.169 (+15.7 %) | 0.182 (+9.3 %) |

  - Bits: hybrid off == NAX6, 64 / 64.
  - Quality vs fp64 (S 16, 4 traces): NAX6-H = NAX6 (mean 3.6e-3, max 2.6-3.1e-2); 3c 1.2-1.3e-2.
  - **Deviation vs NAX6 (for Breakthrough's map): 0-1 % of pair rows differ, mean 1e-12..1.7e-8, max 0..7e-3 x RMS.**
    The fp32 FMA dot and the matrix op agree after the bf16 roundings of h and y almost always, so NAX6-H is NAX6 to the
    last bit on ~99 % of pairs.
  - **Wired: ref2 CFG["nax_hybrid"] (off by default; COUNTS["nax_hybrid"])** on both C1 paths (_nax6_glu and
    _ooo_nax6's two phases). Check (scripts/nax_hybrid_check.py, data/nax_hybrid_check.txt): off == NAX6 + combine,
    on == NAX6-H + combine (S 16 and S 1), counted once per call; ref2.self_test PASS with it off. In both checked cases
    on == off bitwise.
  - The quiet confirmation (pre-registered under D-COMMIT) stays for Tuesday.
- **Quiet confirmation (Tue 15:01, quiet lock; data/h1i_screen.txt): PASS, +9.1 % at S 16** (the line: >= +8 %).

  | S | NAX6 ms | NAX6-H | H-1c (compact split) |
  |---|---|---|---|
  | 16 | 0.795 | **0.722 (+9.1 %)** | 0.707 (+11.0 %) |
  | 8 | 0.511 | 0.458 (+10.4 %) | 0.431 (+15.7 %) |
  | 32 | 1.197 | 1.104 (+7.8 %) | 1.119 (+6.5 %) |
  | 4 | 0.343 | 0.303 (+11.7 %) | 0.281 (+17.9 %) |
  | 2 | 0.235 | 0.203 (+13.8 %) | 0.178 (+24.4 %) |
  | 1 | 0.151 | 0.123 (+18.3 %) | 0.136 (+10.0 %) |

  - Bits: hybrid off == NAX6, 64 / 64. Quality = NAX6's: 0-1 % of pair rows differ from NAX6, max 7e-3 x RMS.
  - Quiet, H-1c's compact split reads +11.0 % at S 16 (+3.7 % in Monday's shared run). That also passes H-1c's own
    pre-registered line (>= +8 % at S 16, under D-COMMIT). But H-1c needs its pair split and separate 3c dispatches
    wired into OOO, while NAX6-H is already the ref2 switch (nax_hybrid) and the base of NAX6-NH. So NAX6-H stays the
    FAST stack's hybrid. H-1c's extra ~2 % at S 16 (more at S 2-8) is noted for later.

## N7-12 kernel-A register trim (Leap, pre-registered Mon 2026-11-02 23:1x EET; Fable C's audit flag via the orchestrator 23:0x; scripts/n7_trim.py, data/n7_trim.txt)

- **Why:** N7-8 only added registers (+16: +2 %, +32: +5 %, +64: +41 %). If residency is register-limited at ~1-2
  threadgroups per core, a trim that crosses the next residency step would put more independent groups on each core.
- **Arms (bitwise to NAX6 by construction: the same values stored, the same ops):** T1 the decode in two halves (3 code
  words + 4 + 4 nibble words live instead of 6 + 8 + 8); T2 = T1 + the A operand loaded straight into the cooperative
  tensor (no vec4 temporaries). Kernel A alone and the whole call, S 16 / 8 / 32, real layer-20 routing, bits vs NAX6.
- **Lines (the orchestrator's):** PASS >= +8 % at S 16. KILL < +3 %.
- **Prediction:** +1-3 % (the mirror of N7-8's +16 / +32 slope) unless a residency step is crossed. PASS 15 %.
- **D1 kernel A lite, priced instead of built (D1-A0, data/d1a_break.txt):**
  - In place, the norm pass costs 28.5 us per read; alone in a dependent chain it costs 9.8 us, against a 7.6 us floor
    for two trivial dependent kernels.
  - "Lite" (no normed write; normed formed on the fly from written x inv x gamma) adds ~4 ALU ops per A element
    inside S4's latency-bound 128-step walk, and the 11 column tiles each re-form the same tile. ~+5 us per read in S4
    against ~1-2 us of saved stores: likely negative.
  - Full kernel A has the same problem plus the rms prologue. Not built; the norm pass stays.
- **Result (Mon 23:11-23:13, shared lock beside ojox-e8 at ~80 % GPU, rough; data/n7_trim.txt): bits 24 / 24, KILL.**
  Neither trim helps: A + B at S 16 T1 -0.9 %, T2 -1.6 %; S 8 -1.3 / -0.8 %; S 32 -1.4 / -1.0 %. Kernel A alone is the
  same (-0.1..-2.3 %). The compiler had already kept those temporaries short-lived (no residency step crossed).
  Combined with N7-3 (threadgroup memory flat) and N7-8 (+16 / +32 registers only +2 / +5 %), residency is not the lever.
  The exact kernel-A work ends here.

## PURGE-1 purgeable elastic expert slabs, part (a) the toggle cost (Leap, pre-registered Mon 2026-11-02 23:2x EET; the cloud auditor's answer via the orchestrator 23:3x; native/purge_probe.swift, data/purge_probe.txt)

- **Idea:** extra expert-cache slots live in Metal buffers marked volatile (setPurgeableState), so macOS discards them
  under memory pressure instead of swapping. The loader marks a slab nonVolatile at the layer's gate (an "empty" answer
  means its experts are misses), then volatile again after the layer. The cache then tracks free RAM.
- **Part (a), now:** 64 shared buffers of 32 MB (2 GB, pages touched), no memory pressure. Per call: volatile ->
  nonVolatile and back, median / p90 over 20 rounds; whether any buffer comes back empty (it should not without
  pressure); a 32 MB read after the toggle (pages still resident).
- **Part (b), later** (only with Kamen's apps closed and no big job: 00:30-01:00 before BigMoE's ticket, else Tue after
  the combination): a memory balloon to check that the slabs empty before any swap-out (vm_stat). Stop at once if swap
  grows > 1 GB.
- **Lines (the orchestrator's):** PASS = slabs emptied before swap-outs and toggle <= 20 us. KILL = swap / compression
  first, buffers refuse to purge, or toggle > 100 us. Part (a) decides the toggle half.
- **Prediction:** a toggle is a kernel call on the buffer's VM object: ~5-30 us for 32 MB. PASS on (a) 60 %.
- **Result (a) (Mon 23:27, share lock, no memory pressure; data/purge_probe.txt): PASS on the toggle half.** 64 x 32 MB
  shared buffers, 20 rounds, two runs: -> volatile median 0.7-0.8 us (p90 1.7-1.8), -> nonVolatile median 0.2 us (p90
  0.2). No buffer came back empty (no pressure). A full page walk after the toggle took 51 us (pages still resident).
  Swap unchanged. Per layer, a gate toggle costs ~1 us: free next to the ~1.7 ms per layer of a check.
- **Part (b)** waits for 00:30-01:00 with Kamen's apps closed and nothing else running (or Tue after the combination).
  It is a memory balloon, so it runs only then.
- **Part (b) pre-registration (Mon 23:5x EET; native/purge_balloon.swift -> data/purge_balloon.txt):**
  - **What:** 64 x 32 MB shared buffers filled with pseudo-random bytes (incompressible, like expert records) and set
    volatile, plus a 65th buffer that only times the toggle. A toggle re-queues a buffer's purge token, so the 64 counted
    slabs are never toggled. First, 5 s idle to measure the system's own compression and swap-out rate. Then a balloon
    of pseudo-random anonymous memory grows in 256 MB steps. After each step it records: slabs empty; compressions and
    swap-outs over the idle rate; swap used; the kernel's pressure level; one timed toggle.
  - **Stops:** all slabs empty (+2 steps), swap used +1 GB, pressure critical, 120 s, or a 48 GB balloon. The balloon is
    freed at once.
  - **When:** only after 00:30. Conditions: no PAUSE / HOLD, lid open, Kamen's apps closed (Chrome, Telegram, the
    Claude app), no job running in peer ls, BigMoE not loading. It runs under a quiet lock, because the balloon presses
    on everyone's memory. Otherwise it waits for Tue night.
  - **Lines (the orchestrator's, made measurable):**
    - PASS = all 64 slabs empty while compressions over the idle rate stay <= 2 GB (the slabs' own size) and swap-outs
      over the idle rate stay <= 0.25 GB, with a median toggle under the balloon <= 20 us.
    - KILL = any of: >= 2 GB compressed or > 0.25 GB swapped out before the FIRST slab empties; slabs still non-empty
      at the stop; a median toggle > 100 us.
    - BETWEEN otherwise.
  - **Dry run (Mon 23:45, 0.5 GB balloon, no pressure):** the code paths work. The toggle took 4.0 us (both calls) and
    the idle rates were ~0 MB/s. Free memory reads 0.07 GB, because macOS keeps the rest as file cache. So the balloon
    first takes the file cache (clean pages dropped, not compressed) before any pressure starts.
  - **Prediction:** my reading of XNU's pageout scan is that it purges ripe volatile objects before it reclaims other
    pages, and from the warning level on it also purges unripe ones. Our tokens are new, so some compression may come
    first. PASS 55 %, BETWEEN 30 %, KILL 15 %.
  - **Tue 00:30 check: deferred to Tue night**, by the pre-registered condition. Chrome and Telegram were running (the
    screen was locked), and swap was already at 9.06 GB. Under real pressure Chrome may discard tabs, which could lose
    Kamen's state, so the balloon waits for a window with his apps closed.
  - **Result (Tue 15:05, quiet lock; the orchestrator's gates: Kamen's apps closed, HID idle 2 h 11 min, no process
    over 1 GB, a 5-minute hard stop; data/purge_balloon.txt): PASS.** The run took 21 s.
    - All 64 volatile slabs emptied at a 36.75-38.75 GB balloon, within 0.5 s, right after the file cache was taken
      (free 0.1 GB). At that point: compressions +367 MB over the idle rate (line <= 2 GB), swap-outs +0 (line <= 0.25
      GB), pressure still normal.
    - Toggle under the balloon: median 2.0 us (line <= 20), max 64 us over 157 steps.
    - After the slabs were gone, the next step compressed +28 GB in 2.7 s (pressure warning), and swap began (+196 MB).
      Swap ended at +314 MB after the stop (inside the +1 GB stop).
    - **What it means:** the order of reclaim is file cache, then volatile purgeable buffers, then compression, then
      swap. An expert cache in volatile slabs would give itself back before anything of Kamen's or the engine's gets
      compressed or swapped. That is the elastic cache the idea needs (the loader's toggles cost ~1-2 us).
  - **Accounting (NewIdeas 15:09 for vm_allocate purgeable memory; Leap 15:2x for MTLBuffers,
    native/purge_footprint_mtl.swift):** volatile memory is invisible to phys_footprint and to the lifetime peak
    (RUSAGE_INFO_V4).
    - A touched 256 MB shared MTLBuffer counts 259.7 MB non-volatile and 3.7 MB volatile.
    - A buffer written while volatile is never charged, and the lifetime peak stays at the non-volatile high-water mark.
    - **So a volatile expert tier must have its full size reported separately, and it counts in full for the 24 / 12 GB
      lines.** With per-layer pinning, the lifetime peak would otherwise see only about one pinned slab.
    - NewIdeas ranked PURGE-1 top of the exact 45 GB queue (RANKED row 33: C 300 / 320 without a free-memory night),
      routed to BigMoE for the engine tier, with Leap as mechanism owner.

## N7-13 a fragment-native code layout for the expert records (Leap, pre-registered Mon 2026-11-02 23:2x EET; the cloud auditor via the orchestrator 23:3x; scripts/tierglu_nax6n.py, n7_native.py; data/n7_native.txt)

- **Why:** each NAX6 lane reads its row's 24 B of codes per quant group at a 960 B stride (kernel A; 240 B in kernel
  B), plus its scale / bias at an 80 B stride and its LSB words at 320 B: every load instruction touches 32 lines.
  L1-26's native layout was bitwise and 1.15-1.22x on the dense side. N7-1's DIAG 3 (one line per load, no DRAM) gave
  -17 % on kernel A: a loose upper bound.
- **What:** one layer's records repacked (on the CPU) so that for each (expert, 32-row tile, quant group) the 32 rows'
  codes (768 B), LSB words (256 B), scales and biases (64 B each) are contiguous. NAX6's kernels with only the address
  arithmetic changed. The same values reach the same ops, so it is bitwise by construction.
- **Test:** bits vs NAX6 (y, 64 cases: 4 traces x S 8 / 16 x 8 passes); speed of the whole call + combine in dependent
  chains, S 16 / 8 (and 32), real layer-20 routing.
- **Lines (the orchestrator's):** PASS >= +6 % at S 16. KILL < +2 %.
- **Prediction:** +3-8 % on kernel A, +2-5 % on the call (the DRAM bytes do not change, only the line count per
  instruction). PASS 30 %.
- **Result, shared run (Mon 23:27-23:28, share lock beside Breakthrough at ~50 % GPU, rough; data/n7_native.txt):
  PASS, bitwise. +16.2 % at S 16, +16.6 % at S 8, +17.3 % at S 32.**
  - Bits: NAX6-N == NAX6, y, 64 / 64. The repack of one layer's pools takes 1.0 s on the CPU.

    | S | NAX6 ms | NAX6-N |
    |---|---|---|
    | 16 | 1.113 | 0.932 (+16.2 %) |
    | 8 | 0.737 | 0.615 (+16.6 %) |
    | 32 | 1.784 | 1.475 (+17.3 %) |

  - **This corrects tonight's "the kernel's inner structure is at its ceiling".** The scattered per-lane loads (32 lines
    per instruction) were a real limiter. N7-1's DIAG 3 (-17 %) was most of the way there: the gain is the line count
    per load, not the DRAM bytes.
  - It needs the records in the native layout, and 3c (REF2-T) reads the row-major one (its lanes split K within a
    row), so one record format cannot serve both kernels. That is BigMoE's integration question: a second format for
    the NAX preset, or a repack on load.
  - It stacks with H-1i (the ALU path reads the same codes), and it adds to the NAX reference's case: NAX6-N vs 3c at 16
    rows ~36 vs 50.5 ms per check standalone.
  - A quiet confirmation is queued (big gain).
- **Quiet confirmation (Mon 23:29-23:31, quiet lock): PASS, +18.7 % at S 16, bitwise 64 / 64.**

  | S | NAX6 ms | NAX6-N |
  |---|---|---|
  | 16 | 0.910 | **0.740 (+18.7 %)** |
  | 8 | 0.591 | 0.489 (+17.1 %) |
  | 32 | 1.424 | 1.165 (+18.2 %) |

  - On the NAX road this is the largest exact lever found: NAX6-N's expert call at 16 rows is ~0.74 ms per layer
    against 3c's ~0.91-1.0 (S11 / C2 terms), so ~1.25-1.35x 3c before tier merge or the hybrid.
  - Next steps (routing): the native layout under the hybrid (H-1i's ALU path reads the same codes); the record-format
    question for BigMoE (a second format for the NAX preset, or a GPU repack on load).

## N7-14 the stack NAX6-NH = native layout + in-kernel hybrid (Leap, pre-registered Mon 2026-11-02 23:3x EET; orchestrator 23:3x; FAST; scripts/tierglu_nax6h.py tier_glu_nax6nh, n7_stack.py; data/n7_stack.txt)

- **What:** H-1i's ALU path for one-pair groups on N7-13's native records (the same values, so NAX6-NH == NAX6-H
  bitwise, and NAX6-NH with the hybrid off == NAX6). Four arms: NAX6, NAX6-N, NAX6-H, NAX6-NH; the whole call +
  combine in dependent chains at S 16 / 8 / 32 / 1; bits checked as above.
- **Lines (the orchestrator's):** PASS NAX6-NH >= +22 % over NAX6 at S 16. KILL < +12 %. Shared lock; the quiet
  confirmation goes with Tuesday's design-D slot.
- **Prediction:** +24-27 % (N7-13's +18.7 % plus most of H-1i's +8 %; the ALU path also gets the coalesced loads).
  PASS 60 %.
- **Result (Mon 23:31-23:32, share lock, GPU otherwise ~free; data/n7_stack.txt): PASS, +23.3 % at S 16.**
  Bits: NAX6-NH with the hybrid off == NAX6, 48 / 48; NAX6-NH == NAX6-H, 48 / 48.

  | S | NAX6 ms | NAX6-N (exact) | NAX6-H (FAST) | **NAX6-NH (FAST)** |
  |---|---|---|---|---|
  | 16 | 0.836 | 0.691 (+17.3 %) | 0.764 (+8.6 %) | **0.642 (+23.3 %)** |
  | 8 | 0.517 | 0.423 (+18.2 %) | 0.462 (+10.7 %) | 0.393 (+23.9 %) |
  | 32 | 1.321 | 1.107 (+16.2 %) | 1.210 (+8.5 %) | 0.983 (+25.6 %) |
  | 1 | 0.161 | 0.139 (+13.2 %) | 0.134 (+16.7 %) | 0.118 (+26.4 %) |

  - The two levers add almost fully. NAX6-NH's expert call at 16 rows is ~0.64 ms per layer, ~1.4-1.5x 3c's.
  - On the NAX road the stack is: exact NAX6-N (bitwise to NAX6, needs native records), FAST NAX6-NH (99 % of pairs ==
    NAX6), and later tier merge B. The quiet confirmation goes with Tuesday's design-D slot.
- **Wired (Mon 23:3x): ref2 CFG["nax_native"]** (off by default; COUNTS["nax_native"]) selects NAX6-N, or NAX6-NH with
  nax_hybrid too, on C1's NAX paths. **The pools must be native** (tierglu_nax6n.repack); 3c can't read them. Check
  (scripts/nax_hybrid_check.py, data/nax_hybrid_check.txt): native pools + nax_native == NAX6 + combine (S 16 and S 1),
  + nax_hybrid == NAX6-H + combine, counted; ref2.self_test PASS with both off.

## N7-15 repack-at-load: the GPU repack of freshly loaded slots to the native layout (Leap, pre-registered Tue 2026-11-03 00:0x EET; the orchestrator 23:5x for BigMoE's load path; scripts/tierglu_nax6n.py repack_slots, n7_repack_check.py, n7_repack.py; data/n7_repack*.txt)

- **Why:** NAX6-N (+18.7 % quiet at S 16, bitwise == NAX6) needs native records. The disk can't hold a second 45 GB
  copy, and the CPU gather costs +130-180 us per demand burst (BigMoE's count). So slots are loaded row-major as today
  and rewritten on the GPU inside OOO's graph.
- **What (BigMoE's spec, agreed Tue 00:0x):**
  - Only in-service demand loads leave a slot row-major, with its int32 flag = 1. Everything else (prefetch, JOINT's
    sidecar reads, ensure()) is gathered native on the CPU, with flag 0. So phase 1 never meets a dirty slot.
  - Between the spins and phase 2, `repack_slots` rewrites in place the dirty slots named by the call's pairs (base w3
    / s3 / b3, sidecar l1 / b) and clears their flags. It runs as two launches:
    - a one-threadgroup claim: unique slots, once each, negative slots skipped;
    - a persistent grid that moves each 32-row tile through threadgroup memory in one pass. The permutation never
      leaves a tile.
  - It returns the slot arrays, so MLX orders phase 2 after the rewrite.
  - Wiring: ref2's `_ooo_nax6` and `_nax6_glu`, behind CFG nax_native and `ref2.native_flags(...)`. It goes in after
    BigMoE's night ticket, with ref2.self_test.
- **Tests:**
  - (1) n7_repack_check.py, all bitwise: claim semantics; the rewrite == repack(); every other slot untouched; flags;
    idempotence; duplicates; lazy chaining. End to end, NAX6-N after the repack == NAX6, both in one call and in OOO's
    two phases.
  - (2) n7_repack.py, speed: dependent chains of 16 S 16 calls on changing real routing windows. Arms: NAX6, NAX6-N,
    and repack + NAX6-N with n dirty slots per call; plus the repack alone.
- **Lines (the orchestrator's, made measurable):** PASS = repack-at-load keeps >= 80 % of NAX6-N's gain on a cold-cache
  S 16 check; KILL < 50 %.
  - "Cold-cache" is taken as the 45 GB default's miss rate: 3 base + 1 sidecar slots per layer call, ~192 per check.
    C 256 measured 171 per check with per-layer sizes and 219 uniform.
  - The curve up to 12 + 4 slots per call covers colder checks.
- **Not before** BigMoE's ticket ends (~07:00), under a share lock. Compile check done: both kernels build with
  `xcrun metal` (CPU only, Mon 23:58).
- **Prediction:**
  - Cost estimate: ~17 us per base slot (2 x 2.15 MB at ~250 GB/s), ~6 us per sidecar slot, and ~8 us per call for
    the two launches. So 3 + 1 slots cost ~65 us against NAX6-N's ~170 us gain per call: kept ~60 %.
  - Phase 2 then reads the just-written slots partly from cache, which may give some of that back.
  - BETWEEN 55 %, PASS 25 %, KILL 20 %.
  - If BETWEEN, the next lever is to repack each slot as it lands, inside BigMoE's spin kernel, hidden behind the
    remaining loads.
- **Result (Tue 01:51-01:54, quiet lock; data/n7_repack_check.txt, data/n7_repack.txt): pre-registered KILL, 47 %
  of the gain kept at 3 + 1 slots per call.** A post-hoc fix of the claim kernel brings it to 50 %, exactly on the line.
  - **Self-test: PASS**, all bitwise, after one fix.
    - The first run failed test D (two calls chained in one lazy graph): every slot named by both calls was rewritten
      twice.
    - Cause: mx.depends orders MLX's graph but sets no GPU barrier between the two claim kernels. A flag array routed
      through mx.depends is also a copy.
    - Fix: each claim takes the previous claim's token on the same flags as an input, which is a real buffer
      dependency. D then passed.
    - The real patterns passed both times: E (NAX6-N after the repack == NAX6, 12 / 12) and F (OOO's two phases + the
      repack == NAX6 + combine, 12 / 12).
  - **Speed, per S 16 call + combine.** NAX6 0.802 ms, NAX6-N 0.653 ms (+18.6 %, the N7-13 gain, 149 us). Gain kept:

    | dirty slots per call (base + sidecar) | 0 + 0 | 1 + 0 | 2 + 1 | **3 + 1** | 6 + 2 | 12 + 4 |
    |---|---|---|---|---|---|---|
    | pre-registered run | 89 % | 78 % | 59 % | **47 %** | 5 % | -84 % |
    | post-hoc, claim v2 | 95 % | 82 % | 63 % | **50 %** | 8 % | -81 % |

    - Claim v2 checks each slot's flag first, so clean pairs (almost all) stop at once, and scans for duplicates in
      threadgroup memory.
    - The repack alone costs ~18.5 us per base slot (2 x 2.15 MB at ~230 GB/s: DRAM speed, as estimated) and ~3.5-4.8 us
      per sidecar slot. Break-even is ~8 base slots per call. The fixed in-call overhead is ~7 us after claim v2 (~16
      before).
  - **What it means:**
    - At the 45 GB default's ~3.6 misses per layer call, repack-at-load keeps about half of NAX6-N's gain: ~3.5 of ~7 ms
      per check.
    - The per-slot cost is DRAM-bound on the GPU's critical path, so no kernel tuning recovers the rest.
    - The full gain needs one of two things:
      - (1) native records on disk: a one-way repack of the 45 GB route, which waits on Kamen's NAX-reference decision.
        This keeps all +18.6 % at zero load cost. **This result puts a number on that decision.**
      - (2) the repack hidden inside BigMoE's spin, each slot as it lands, behind the remaining loads. Its demand SSD
        time is ~49 ms per check, so the GPU is mostly waiting there anyway.
  - **Wiring (live in ref2, inert by default; ref2.self_test ALL PASS + inert check):**
    - native_flags() registry; _native_repack in _nax6_glu and in _ooo_nax6 before phase 2; CFG nax_repack_p1 off.
    - One fix after review: a layer registered without sidecar flags now passes its sidecar slots through. Before, it
      would have dropped them.

## B6 FATT16-S, the key-split FATT16 for long context (Leap, pre-registered Tue 2026-11-03 00:1x EET; Fable B's mission B6 via the orchestrator 00:0x, ~/FableGod/fableb/LONG_CONTEXT.md section 5; FAST-K; scripts/fatt16s.py, b6_fatt_split.py; data/b6_fatt_split.txt)

- **Why (Fable B):** attention grows ~1.37 ms per 1K keys per check (R-81), so long chats lose 15-40 %. R-66's FATT16
  is batch- and N-invariant but latency-bound: one simdgroup per head walks every 32-key block.
- **What:** fatt16s = FATT16's inner loop unchanged, plus three changes.
  - **(a) A KS-way split** (KS 4) of each 16-row tile's walk over simdgroups. The partials (m, l, O) merge in a fixed
    order (((0 + 1) + 2) + 3) through threadgroup memory, with the standard rescale (an empty partial gets factor 0).
    **One deviation from section 5:** simdgroup s takes the absolute blocks b = s, s + 4, ... (interleaved), not a
    contiguous quarter. Contiguous quarters of the N keys move when N grows, so INV test (c) (N + 37 / + 300 masked
    keys) would fail by construction.
  - **(b) Blocks with no visible key for any of the 16 rows are skipped** (simd_any over the tile's mask entries).
    This is exact: such a block leaves (m, l, O) unchanged anyway.
  - **(c) 16 or 8 real rows.** With 8, two query heads of the same kv head are stacked in one 16-row tile.
  - K / V are read through their strides: the KV cache's slices are not copied (a copy would be ~20 MB, ~0.1 ms per
    call at 10K).
  - Compile check done (xcrun metal, CPU only, Tue 00:1x).
- **Test (b6_fatt_split.py):**
  - Accuracy vs fp64.
  - INV (a)-(d) as R-66 at N 2K / 10K / 20K with R 16 and R 8, plus (e): 8 rows == the first half of a 16-row call.
  - SDPA-level ms: vs the 12 chunks, MLX's one call and fatt16.
  - Then the real layer 3 in R-72 / R-81's harness: 16-row calls, 8 chained with trim, median of 20. Arms: ref2t,
    single, fatt16, fs4 (pre-registered), fs2 / fs8 (post-hoc). 8-row calls at 10K. y vs ref2t as a sanity check, and a
    count of which SDPA each verify call took.
- **Lines (Fable B's):** PASS = INV PASS and fs4 >= 3x vs REF2-T at 10K (R-81's 2.03 ms -> <= 0.68 ms per call).
  BETWEEN 1.5-3x. KILL < 1.5x or INV FAIL.
  - R-81 timed synced single calls; this harness chains 8. So the verdict's factor uses ref2t measured here by the same
    method, and the factor vs R-81's number is reported beside it.
- **Prediction (mine):**
  - The SDPA itself gets much faster: ~5-8x vs the 12 chunks at 10K (4x the parallelism, ~25 % of blocks skipped).
  - The layer call's other work (projections, the indexer's scoring over every key, mask build, RoPE, o_proj, cache
    writes; maybe ~0.5-0.8 ms) dilutes that. fs4 at the layer level is ~2-2.7x at 10K.
  - INV PASS 90 %; PASS 25 %, BETWEEN 60 %, KILL 15 %.
- **When:** after N7-15's GPU runs (after BigMoE's ticket, ~07:00), under a share lock, ~10 min.
- **Result (Tue 01:47, quiet lock; data/b6_fatt_split.txt): BETWEEN.** fs4 is 1.70x vs ref2t at 10K with INV PASS
  (1.78x vs R-81's number). The pre-registered key split is NOT what helps.
  - Accuracy vs fp64: fatt16s mean 1.75e-3 (fatt16 1.78e-3, MLX's one call 2.29e-3), max 1.25e-2.
  - INV: PASS everywhere. Tests (a)-(d) at 2K / 10K / 20K with R 16 and R 8, plus (e) for R 8.
  - SDPA level (synthetic, the indexer's mask shape), ms per call:

    | N | 12 chunks | MLX one call | fatt16 | fs4 | fs2 | fs8 |
    |---|---|---|---|---|---|---|
    | 2K | 0.349 | 0.106 | 0.114 | 0.135 | 0.120 | 0.219 |
    | 10K | 1.065 | 0.486 | 0.523 | 0.517 | 0.535 | 0.652 |
    | 20K | 1.534 | 1.153 | 1.040 | 0.991 | 1.064 | 1.200 |

    R 8 at 10K: 0.311 ms, 1.66x faster than R 16 (two heads share each K / V load).
  - Layer level (layer 3, 8 calls chained, median of 20), ms per 16-row call:

    | T0 | ref2t | single | fatt16 | fs4 | fs2 | fs8 | fs4 vs ref2t (vs R-81) | fs4 vs single |
    |---|---|---|---|---|---|---|---|---|
    | 2K | 0.846 | 0.484 | 1.056 | 0.479 | 0.524 | 0.548 | 1.77x (2.19x) | 1.01x |
    | 10K | 1.936 | 1.305 | 4.623 | 1.141 | 1.163 | 1.147 | **1.70x** (1.78x) | 1.14x |
    | 20K | 2.838 | 2.342 | 9.159 | 1.851 | 2.336 | 1.957 | 1.53x (1.67x) | 1.26x |

    - 8-row layer calls at 10K: fs4 1.233 ms vs its 16-row call 1.141, so no gain at the layer level (ref2t at 8 rows
      1.922).
    - y vs ref2t, mean |diff| / RMS: 2.2-2.4e-3 for the FATT arms (single 1e-6). The counter confirms every verify call
      of the FATT arms took the FATT kernel.
  - **What it means:**
    - The key split adds nothing: fs2 / fs4 / fs8 match fatt16 at the SDPA level. R-66's "one latency-bound walk" is not
      the limit at long context.
    - ~~The limit is the K / V re-read per query head~~ **CORRECTED Tue 15:1x (B6-G):** the R 8 form's 1.66x came from
      computing half the query rows (12 tiles instead of 24), not from shared K / V loads. B6-G then loaded each K / V
      block once for all 12 heads and was SLOWER (0.84x). K / V re-reads are not the limit.
    - ~~Next lever: GQA packing~~ (B6-G: KILL).
    - At the layer level, fs4 beats ref2t and single because of the strided K / V reads: unsplit fatt16 copies the
      cache slices and pays 4.6 ms at 10K.
    - The 3x line is out of reach at the layer level. The call's other work at 10K is ~0.6 ms (fs4's 1.141 minus its
      SDPA's ~0.52). Even a free SDPA would give only ~3.1x vs ref2t.
    - As a FAST-K replacement for att_single (REF2c), fs4 is +14 % at 10K and +26 % at 20K on the attention layer, and
      equal at 2K.
  - **Prediction check:** I predicted the SDPA at 5-8x vs the 12 chunks; it was 2.06x, so that was wrong. I predicted
    2-2.7x at the layer level; it was 1.70x. BETWEEN, as predicted (60 %).

## B6-G FATT16-G, GQA packing for long-context attention (Leap, pre-registered Tue 2026-11-03 15:1x EET; B6's next lever, routed by the orchestrator 15:0x; FAST-K; scripts/fatt16g.py, b6g_screen.py; data/b6g_screen.txt)

- **Why:** B6 showed fatt16s bound by K / V re-reads. The 12 query heads of a kv head each read the same 32-key blocks;
  stacking 2 heads gave 1.66x. Published fix (TensorRT-LLM's XQA, FlashInfer's GQA head-group fusion): load each K / V
  block once and serve all of that kv head's query heads from it.
- **What:**
  - A threadgroup = (kv head, head group of HG tiles, key split s), one simdgroup per 16-row tile.
  - Each absolute 32-key block b = s, s + KS, ... is loaded once into threadgroup memory: K + V = 32 KB, the M5 Pro's
    limit (queried: 32768 B). FATT16's QK^T / online softmax / PV then run unchanged from it for every tile.
  - Blocks that no row sees are skipped.
  - Partials (O, m, l) per (split, head, row) go to a second kernel, which folds them in split order and normalizes.
    INV holds by construction (absolute b mod KS, fixed fold order).
  - Compile check done (xcrun metal, CPU only).
- **Test (b6g_screen.py, share lock, < 5 GB):**
  - (1) SDPA level, synthetic with the indexer's mask shape: accuracy vs fp64; INV (a)-(e) at 2K / 10K / 20K with R 16
    and R 8 for the primary config; INV (a) and (c) at 10K for every other config.
  - ms per call vs fs4 (and the 12 chunks, MLX's one call). Configs (HG, KS) = (12, 8) primary, plus (12, 4), (12, 16),
    (6, 8), (4, 6). R 8 at 10K.
  - (2) Layer level, R-72's harness: ref2t, single, fs4, fg (12, 8), and the best SDPA config if it differs (post-hoc).
- **Lines:** PASS = INV PASS and fg (12, 8) >= 2.0x faster than fs4 at 10K at the SDPA level (<= ~0.26 ms per call).
  KILL < 1.3x or INV FAIL. BETWEEN otherwise.
- **Prediction:**
  - K / V traffic drops 12x. The SDPA becomes compute-bound: ~1.5 G MAC per call at 10K (with skipping), ~0.11 ms at 14
    T MAC/s, plus softmax, the load barriers and the merge's ~10 us.
  - So ~0.15-0.25 ms at 10K, 2-3.5x vs fs4. Layer level at 10K: 1.14 -> ~0.8-0.85 ms (~2.3x vs ref2t).
  - INV PASS 90 %; PASS 55 %, BETWEEN 30 %, KILL 15 %.
- **Result (Tue 15:09-15:12, share lock, Mac otherwise idle; data/b6g_screen.txt): KILL.** fg (12, 8) is 0.84x vs fs4
  at 10K at the SDPA level, i.e. slower, in every config. INV PASS everywhere.
  - Accuracy vs fp64 = fs4's (same per-row arithmetic). INV (a)-(e) PASS at 2K / 10K / 20K, R 16 and R 8. (a) and (c)
    PASS for every config.
  - SDPA ms per call:

    | N | 12 chunks | one call | fs4 | fg (12, 8) | (12, 4) | (12, 16) | (6, 8) | (4, 6) |
    |---|---|---|---|---|---|---|---|---|
    | 2K | 0.349 | 0.108 | 0.137 | 0.171 | 0.170 | 0.200 | 0.194 | 0.183 |
    | 10K | 1.060 | 0.487 | 0.514 | **0.609** | 0.670 | 0.629 | 0.660 | 0.695 |
    | 20K | 1.531 | 1.166 | 0.983 | 1.158 | 1.294 | 1.174 | 1.253 | 1.345 |

    R 8 at 10K: 0.431 ms, 1.41x faster than its R 16 (half the rows).
  - Layer level (layer 3), ms per 16-row call: ref2t 0.736 / 1.909 / 2.792, single 0.483 / 1.312 / 2.345, fs4 0.493 /
    1.104 / 1.839, fg (12, 8) 0.532 / 1.240 / 2.120 at 2K / 10K / 20K. fg is 0.87-0.93x of fs4. fs4 reproduced B6.
  - **What it means:**
    - Staging K / V once per threadgroup costs more than the L2 re-reads it saves: two barriers per block, a load
      not overlapped with compute, and 32 KB of threadgroup memory, so one threadgroup per core.
    - With fs2 = fs4 = fs8 (B6) as well, the limit of FATT16's walk is neither the K / V traffic nor the simdgroup count.
      Most likely it is each simdgroup's serial chain per block (operand loads -> 16 QK MMAs -> softmax -> 16 PV MMAs).
      NAX6 showed the same "op.run blocks" pattern; the M5's matrix units reach ~21 % of peak here.
    - Finding the real limit needs a profile (Metal GPU counters), not another layout guess.
    - At the layer level, the call's other work (~0.6 ms at 10K: projections, the indexer, mask build, ~80-110 kernel
      launches) is now as large as fs4's SDPA. It is the next attention lever.
  - **Prediction check:** I predicted 2-3.5x faster (PASS 55 %); it was 0.84x. Wrong, because of the mistaken B6
    attribution above.

## H-1p the H-1c port into NAX6-H: an ILP ALU dot (Leap, pre-registered Tue 2026-11-03 15:3x EET; the orchestrator's routing 15:1x; FAST; scripts/tierglu_nax6h.py ilp=True, h1p_screen.py; data/h1p_screen.txt)

- **Why:** H-1i's quiet confirmation put H-1c's compact split at +11.0 % vs NAX6-H's +9.1 % at S 16, and further ahead at
  S 2-8. H-1c itself needs a GPU pair split, separate 3c dispatches and a combine to run sync-free, and its screen
  precomputed the split on the host.
  - The likely difference is in the singleton path. NAX6-H's lane computes one output column as ONE serial FMA chain:
    64 per quant group, each waiting on the last. 3c's lanes split K into short independent chains and reduce at the
    end.
- **What:** tier_glu_nax6h / tier_glu_nax6nh(..., ilp=True) (template ILP_C, off by default). The ALU dot becomes 8
  independent chains (one per decoded word, 8 FMAs each), summed in a fixed tree. Nothing else changes.
  - Still row-invariant per pair, and still FAST.
  - ilp=False compiles the shipped NAX6-H's dot unchanged, so BigMoE's fastBEh arm is untouched.
- **Test (h1p_screen.py, shared, small):**
  - Bits: ilp=False == the shipped NAX6-H (the pre-change module) and hybrid off == NAX6, on 4 traces x S 1 / 8 / 16 /
    32.
  - Quality vs fp64, and the deviation vs NAX6 (ilp on).
  - Speed of the whole call + combine in dependent chains at S 16 / 8 / 32 / 4 / 2 / 1: NAX6, NAX6-H, NAX6-H ilp, H-1c.
- **Lines (FAST speed):** PASS = NAX6-H ilp >= +2.0 % faster than NAX6-H at S 16 (H-1c's quiet lead was +2.1 %), with
  the bits checks passing and quality = NAX6's. KILL < +0.5 %.
- **Prediction:** if the serial chain bounds the singleton path, 8x ILP speeds that part up a lot: ~+2-4 % at S 16
  (singletons are part of the work) and ~+8-15 % at S 2-4. PASS 55 %.
- **Result (Tue 15:13-15:14, share lock, Mac otherwise idle; data/h1p_screen.txt): KILL, +0.1 % vs NAX6-H at S 16.**
  - Bits: hybrid off == NAX6, 64 / 64. ilp=False == the shipped NAX6-H (git cf507df, kernels renamed), 64 / 64.
  - Quality with ilp on = NAX6's (0-1 % of pair rows differ, max 3e-6 x RMS).
  - Speed vs NAX6-H: S 16 +0.1 %, S 8 +0.2, S 32 +0.3, S 4 +0.1, S 2 0.0, S 1 -0.1. H-1c in the same run: S 16 0.703
    vs NAX6-H 0.717; S 8 0.429 vs 0.454; S 2 0.179 vs 0.203.
  - **What it means:** the serial FMA chain does not limit NAX6-H's singleton path (fast-math reassociation or occupancy
    already hides it). H-1c's lead comes from elsewhere. Candidates: 3c's mapping (per pair 160 + 320 small
    threadgroups, lanes splitting K), and its screen's host-precomputed split.
    - The remaining gap is ~14 us per layer call at S 16 (~0.7 ms per check) and ~20-25 us at S 4-8 (chat-sized
      checks, ~1 ms per check).
    - A second port would be a mapping change for singleton groups. Not started: diminishing returns next to the
      attention layer's ~7 ms per check of other work at 10K.
  - The ilp code was committed as tested, then reverted: the shipped NAX6-H stays the plain version.

## ATT-PARTS where a long-context attention call's time goes (Leap, Tue 2026-11-03 15:1x EET; the orchestrator's routing 15:1x; scripts/att_parts.py, data/att_parts.txt; shared, rough)

- **Method:** knockouts in R-72 / B6's harness: layer 3's real weights, 16-row calls, fs4 as the SDPA. Each was run
  with the live K-73 prep fusions (att_rope + idx_prep) on and off. The arms:
  - fs4: the whole call;
  - stub: the SDPA replaced by zeros that still depend on q / k / v / mask, so everything else runs;
  - stub-nosel: the stub with the indexer's selection off.
- **Chained, 8 calls per eval (B6's method), K-73 on, ms per call:**

  | keys | whole | SDPA | selection + mask | rest |
  |---|---|---|---|---|
  | 2K | 0.428 | 0.123 | 0.003 | 0.302 |
  | 10K | 1.070 | 0.501 | 0.088 | 0.481 |
  | 20K | 2.036 | ~1.0-1.27 | 0.109 | 0.657 |

- **Unchained, each call evaluated on its own, K-73 on, ms per call:**

  | keys | whole | SDPA | selection + mask | rest |
  |---|---|---|---|---|
  | 2K | 0.811 | 0.291 | 0.001 | 0.519 |
  | 10K | 1.668 | 1.022 | 0.120 | 0.526 |
  | 20K | 2.681 | 1.990 | 0.160 | 0.531 |

- **Finding 1, a harness artifact:**
  - The chained "rest" grows with N (0.30 -> 0.66 ms); unchained it does not (0.52-0.53). Chaining 8 cache updates in
    one graph defeats MLX's in-place slice update: each call's mlx_lm KVCache write finds the cache still referenced by
    the previous call's attention, so it copies the whole cache, O(N).
  - The live engine updates each layer's cache once per check, after the previous check was evaluated, so it should not
    pay this.
  - So B6 / B6-G's layer-level times include an O(N) copy in every arm, which compresses their factors toward 1, and
    my "~0.6 ms of other work at 10K" was inflated.
- **Finding 2:** unchained single calls inflate the SDPA (1.02 vs 0.50 ms at 10K), likely from GPU clocks dropping
  between synced calls or cold caches. The live check keeps the GPU busy, so the chained SDPA numbers are the
  representative ones.
- **Live estimate per 10K call:** SDPA ~0.50 + selection ~0.09-0.12 + a fixed ~0.30 (projections, RoPE, cache writes,
  launches; constant in N) = ~0.9 ms. K-73's prep fusions save ~0.04-0.06 ms per call.
  - The cost that grows with context is almost all the SDPA walk: ~0.047 ms per 1K keys per call, ~0.57 ms per 1K
    keys per check over 12 layers. The selection adds ~0.13 ms per 1K keys per check.
  - So the next long-context lever is the SDPA walk's efficiency (fs4 runs at ~21 % of matrix peak), which needs the
    GPU-counter profile, not the non-SDPA work.

## B6-D fs4 without the register spill: output dims split across simdgroups (Leap, pre-registered Tue 2026-11-03 15:2x EET; from the fs4 GPU profile the orchestrator routed 15:1x; FAST-K; scripts/fatt16d.py, b6d_screen.py; data/b6d_screen.txt)

- **The profile (Tue 15:19-15:21, three xctrace captures under quiet locks; temp files deleted by exact path):**
  - With --attach, the 'Metal GPU Counters' instrument records no counters. With --launch and GPU + Metal GPU Counters,
    the counter tables are still empty (0 rows), but the trace carries the GPU intervals (2,136 rows), the shader list
    (fatt16s present) and **graphics-compiler-spill-events: 1,660 rows, one per command buffer, 544 bytes each.**
  - The workload's command buffers hold fatt16s and a few small MLX copies, so the spill is fatt16s's.
  - 544 B = 136 fp32 values, close to the 128 fp32 accumulators per lane (8 output tiles x 16) that fs4 / FATT16 keep
    beside the S tile, x[16] and the operands. A spilling accumulator makes every block's rescale (o *= f) and PV
    accumulation go through stack memory.
  - That fits everything seen so far: ~21 % of the matrix peak, no gain from more simdgroups (B6) and none from fewer
    K / V reads (B6-G).
- **What (fatt16d):** fs4 with each simdgroup holding HALF the output dims: 4 tiles, 64 accumulators per lane.
  - The two halves of a (head, key split) each recompute S = QK^T (16 MMAs, duplicated) and the identical online softmax.
    Each does its 4 PV tiles, and the per-half partials merge in the same fixed order.
  - Every output element sees the same MMA ops in the same order as in fs4, so **bitwise == fs4 is expected** (checked
    directly; INV as B6 if not).
- **Lines:** PASS = bits == fs4 (or INV PASS) and >= 1.5x faster than fs4 at 10K at the SDPA level. KILL < 1.15x.
- **Prediction:** if the spill dominates, 1.5-2.5x despite the duplicated QK. PASS 50 %.
- **Bits (Tue 15:24, share lock beside ojox-99, correctness only): fatt16d == fs4 bitwise** at N 2K / 10K / 20K, R 16
  and R 8, all outputs. fs4's INV properties carry over as they are. The speed part waits for a quiet lock: MTP Head's
  shared walk ran until ~15:48, and the peer hook blocked timing beside it.
- **Speed (Tue 15:33, quiet lock): KILL, 0.87x vs fs4 at 10K** (slower; bitwise == fs4).
  - ms per call: 10K fs4 0.512, fd KS 4 0.592, KS 2 0.525, KS 8 0.645. 20K fs4 0.983, fd 1.170 (0.84x). R 8 at 10K: fs4
    0.309, fd 0.285 (1.08x).
  - The 2K line (fs4 0.243 vs fd 0.139, "1.75x") is a first-arm artifact: fs4 was measured first in a fresh process
    and normally reads 0.135-0.137.
  - **What it means:** halving the accumulators per lane does not pay for the duplicated QK MMAs. Either the 544 B spill
    is not what limits fs4, or this split recovers less than it adds.
  - Three restructurings of fs4's walk (B6 key split, B6-G GQA staging, B6-D output split) all failed to beat it. Its
    per-block chain is ~32 MMAs whose operands are assembled lane by lane (device vec4 loads -> 16 cooperative-tensor
    element moves per MMA), at ~4x below the per-core matrix rate.
  - The remaining idea (new, untested, a larger rewrite): feed the MMAs from MPP tensor views, so operands go straight
    from memory or threadgroup memory without per-element assembly, as NAX6 does with its B operand.
  - Parked unless routed. fs4 stays the FAST-K attention: +14 % vs att_single at 10K, +26 % at 20K on the layer.

## NATIVE-3C the safety net for the native-layout disk conversion (Leap, pre-registered Tue 2026-11-03 15:4x EET; the orchestrator's routing 15:3x for Kamen's pending NAX6-N disk decision; scripts/native_sample.py, tierglu_3cn.py, native3c_screen.py; data/native3c_screen.txt)

- **Question:** if the 45 GB route's records were converted to NAX6-N's native layout, can 3c (REF2-T's exact
  reference kernel) still serve experts from them bit for bit?
- **Expectation:** stock 3c cannot read them; its lanes split K within a row, so on native records it would read
  garbage. The fallback is **3c-N**: 3c's kernels A / B with only the address arithmetic swapped to the native layout.
  - The swap is applied to `_A` / `_B` before the BC (combine) and AP / BP (OOO phase) variants are derived. Those are
    the kernels REF2-T uses: tier_glu, tier_glu with scores, tier_glu_phase.
  - The same values reach the same ops in the same order, so 3c-N == 3c bitwise by construction. It will be slower:
    lanes reading different groups land 768 B apart instead of 6.
- **Test (small jobs, < 5 GB at a time, share lock):**
  - (1) Convert layers 3, 20 and 44 of records3 into a scratch dir (data/native_sample/, ~4.5 GB): the same record
    layout and piece offsets, each piece permuted by native_map. The real records are not touched.
  - (2) Pools from the sample files (the disk format end to end) vs pools from records3. Then on real routing, S 1 / 8
    / 16, for each layer, bitwise:
    - 3c-N(native) vs 3c(row-major): tier_glu, tier_glu + scores, and tier_glu_phase's two phases + combine_phase;
    - NAX6-N(native) vs NAX6(row-major).
  - (3) Speed on layer 20 (dependent chains): 3c vs 3c-N, NAX6 vs NAX6-N, at S 16 / 8 / 1.
- **Lines:** PASS (the fallback exists) = every comparison bitwise, every layer and width. FAIL = any mismatch. Then the
  fallback must be an adapter or dual files: +70.4 GB for a second records3, below the 110 GB floor at today's ~174 GB
  free. Speed is reported, with no line.
- **Prediction:** bitwise 95 %. 3c-N 10-30 % slower than 3c (scattered loads). NAX6-N +18.6 % as before.
- **Result (Tue 15:38, share lock, Mac otherwise idle; data/native3c_screen.txt): PASS, the fallback exists.**
  - Sample (layers 3 / 20 / 44, 3 x 1.50 GB in data/native_sample/, written in 6.7 s): every piece of every expert ==
    native_map(records3), for all three layers.
  - Bitwise, per layer (4 traces x S 1 / 8 / 16 x 3 windows = 36 calls each): 3c-N(native) == 3c(row-major) for tier_glu,
    tier_glu + scores (BC) and tier_glu_phase x 2 + combine_phase. NAX6-N(native) == NAX6(row-major). 36 / 36 in every
    case, on all three layers.
  - Speed on layer 20 (chained):

    | S | 3c | 3c-N | NAX6 | NAX6-N |
    |---|---|---|---|---|
    | 16 | 1.394 | 1.458 (-4.3 %) | 0.824 | 0.680 (+21.1 %) |
    | 8 | 0.698 | 0.750 (-7.0 %) | 0.520 | 0.426 (+22.1 %) |
    | 1 | 0.095 | 0.096 (-0.7 %) | 0.156 | 0.135 (+15.6 %) |

  - Peak RSS was 11.1 GB, not the "< 5 GB" pre-registered: the memory-mapped files (~3 GB) plus transient numpy copies
    plus ~3 GB of pools. No other job was running.
  - **For Kamen's decision:**
    - A one-way conversion of the records to the native layout keeps the exact reference: REF2-T's 3c paths run as 3c-N
      on the converted records, bit for bit, at a 1-7 % cost on the expert call.
    - No second copy is needed (a dual-file fallback would be +70.4 GB, which the disk floor rules out).
  - **Still needed before converting:**
    1. A ref2 switch that routes REF2-T's 3c calls to 3c-N when the records are native. Small, like nax_native.
    2. Every other reader of records3 must read the new layout. In particular, BigMoE's prefill read_rebuilt (CPU,
       records -> MLX 4-bit rows) needs the inverse gather. Other tools reading records3 directly (e.g. Breakthrough /
       EvalSuite scripts) need it too.
    3. 3c's pair kernels (AS / BS) and BigMoE's LP1 patch: same swap, not yet built.
    4. records3m (M01's packed fields) is a separate format for BigMoE.
- **Conversion prep (Tue 15:42, share lock; the orchestrator 15:4x; data/native3c_check.txt): both pieces built,
  default off, N3C check PASS.**
  - tierglu_3cn now also has the pair kernels (tier_glu_pairs_n, tier_glu_split_n). It re-derives its native sources
    from tierglu's current ones on every call: if BigMoE's LP1 patch rewrites them, 3c-N follows. Kernel names carry a
    source hash.
  - ref2 CFG["native_3c"] (off) sends tier_call's 3c branch and _tier_slots to _tier_glu_3cn, and OOO to _ooo_3cn
    (BigMoE's _ooo_call step for step). A guard raises on native_3c + tier_nax4, and on native_3c + tier_nax_ooo
    without nax_native.
  - Bitwise on layers 20 / 44 (24 windows each, S 1 / 8 / 16), 24 / 24 for every item:
    - pairs and split;
    - ref2's selector with the switch on, with and without scores;
    - _ooo_3cn vs BigMoE's original _ooo_call, with a stand-in gate and random residency.
  - COUNTS native_3c = 144. Switch off = BigMoE's own 3c. Guards raise. ref2.self_test PASS. Peak RSS 4.4 GB.

## DISK-PLAN converting the 45 GB route's records to the native layout (Leap, Tue 2026-11-03 15:5x EET; the orchestrator's ask 15:4x; written for Kamen's decision; nothing converted)

- **What it buys:**
  - NAX6-N reads the experts +16-22 % faster than NAX6, with zero load-time cost; the at-load GPU repack (N7-15)
    keeps only ~half.
  - The exact reference survives: REF2-T runs as 3c-N on the converted records, bit for bit, at 1-7 % on the 3c expert
    call (NATIVE-3C, N3C).
- **The store:** records3 = 48 files x 1.50 GB = 72.1 GB, the 45 GB route's default. records3m (68.0 GB,
  Breakthrough's 12 GB line) stays as it is. Free disk today: 182.5 GB; the floor is 110 GB.
- **Options:**
  - **A, in place, layer by layer (recommended).** scripts/convert_records_native.py records3 --yes. Per layer:
    permute, write a temp file, fsync, re-read and compare bitwise, atomic rename, mark the layer's layout in
    index.json.
    - Extra disk: one layer (1.5 GB) at a time.
    - Time: ~2.4 s per layer measured, so **~2 min for all 48 layers.**
    - Resumable (marked layers are skipped) and reversible (--inverse, ~2 min, lossless).
    - It refuses to run if free disk would fall under 112 GB, or if any process has a records3 file open.
    - Tested on a scratch copy of layer 20: the converted copy == the native sample, and the inverse == the original,
      both bitwise.
  - **B, a temporary second copy, then a swap:** +72.1 GB for the duration, leaving ~110 GB free, i.e. at the stop line.
    Rejected.
  - **C, keeping both layouts:** +72.1 GB permanently. Rejected for the same reason.
  - **D, no conversion:** at-load repack (N7-15 GPU repack, ~50 % of the gain at the default's miss rate) or BigMoE's
    CPU gather.
- **Before A runs:**
  1. The engine switches: ref2 CFG native_3c for REF2-T's 3c calls (built, bitwise, off); nax_native (+ nax_hybrid)
     for the NAX paths (built, off).
  2. BigMoE's ext:
     - read_rebuilt (prefill, records -> MLX 4-bit rows on the CPU) needs the inverse permutation;
     - pool loads need nothing (native file -> native pool);
     - a guard: refuse row-major readers (stock 3c, NAX6, tier_nax4, TIER_GLU=3b, read_rebuilt without the inverse)
       on layers marked native;
     - the LP1 patch: 3c-N re-derives itself after it.
  3. The other readers: 205 files in other sessions reference records3 (BigMoE 50 including the engine, Breakthrough
     51, MoonShots 45, Fit12 39, TrainLab 11, FableGod 8, SpecDecode 1), plus 32 of Leap's. List:
     data/records3_readers.txt.
     - After A, a script that reads rows directly gets permuted data silently, unless it checks index.json's per-layer
       "layout".
     - Needed: a heads-up to every session, and the active consumers checked or switched to the inverse view (the
       inverse of tierglu_nax6n.native_map, as in convert_records_native.py's permute).
  4. A smoke test after conversion: BigMoE's residency check, plus a short REF2-T decode compared token for token
     with the pre-conversion reference.

## WHOLE-CHIP CPU (SME / NEON) and Neural Engine beside the GPU's expert kernel (Leap, pre-registered Tue 2026-11-03 17:1x EET; BREAKTHROUGH PUSH bet "use the whole chip", the orchestrator's routing 17:0x; ~/FableGod/push/BREAKTHROUGH_PUSH.md; scripts/wc_screen.py, native/wc_cpu.c, scripts/wc_ane.py; data/wc_screen.txt)
- **Question:** the CPU's matrix units and the Neural Engine sit idle while the GPU checks. Can either take a share of a
  16-row check's routed experts at the same time and add >= 20 % expert throughput while the GPU slows < 5 %?
- **Why it might work [K]:** at S 16 NAX6 holds ~171-175 GB/s, 60-66 % of the 258-287 GB/s the M5 Pro streams (C2,
  DAY9); NAX6-N is ~+21 % on top (~210 GB/s). So ~80-110 GB/s of read bandwidth is unused during the expert kernel.
  ANE-1 [K]: the GPU's expert proxy slowed x1.070 with the Neural Engine drafting back to back (shared GPU).
  ANE-2 [K]: the Neural Engine runs int8 weights at ~79 G weights/s; int4 blockwise falls back to the CPU.
- **Prior art (tags):**
  - Published: HeteroLLM (arXiv 2501.14794) splits tensors across a mobile GPU and NPU at once and reports that each
    unit alone under-uses the SoC's bandwidth. NPUMoE (arXiv 2604.18788) runs MoE prefill on Apple's NPU with static
    capacity tiers and grouped experts, keeping dynamic ops on the CPU / GPU. llama.cpp + Arm KleidiAI run int4 GEMV /
    GEMM on SME2 (CPU only). llama.cpp's tensor overrides, kTransformers and Fiddler put MoE experts on the CPU beside
    a discrete GPU.
  - Adapted here (untested on Apple unified memory): split ONE layer's routed experts between the GPU and the CPU (or
    the Neural Engine) at the same time, during a 16-row verify check, using the bandwidth the GPU leaves.
- **Arms (layer 20, real routing from expA traces, S 16, code + chat windows):**
  - G, the GPU alone: NAX6-N on data/native_sample (the push's planned layout; the stricter reference, since it uses
    the most bandwidth), dependent chains of 16 calls; the windows' unique-expert bytes at the PLAN 4433333333 tiers
    give its GB/s and experts/ms.
  - CPU, at most 4 threads (this session's CPU rule; a production engine could use more, noted as [P] where it matters):
    - C-BW: pure streaming reads of a private 2 GB buffer, 1 / 2 / 4 threads. A bound: no CPU kernel streams more.
    - C-Q4: a 4-bit expert kernel on the same windows' row-per-expert pattern (gate + up, SiLU, down, scores),
      int8 dot products (i8mm USDOT), activations Q8 per 64-group, fp16 scales / biases; synthetic 4-bit experts of
      the real shapes (speed does not depend on the values). FAST-only numerics (not bitwise to 3c / NAX6). Correctness
      check: cos >= 0.999 vs a float64 reference of the same synthetic experts.
    - C-SME: Accelerate's cblas_sgemm (SME) on the dense 16-row expert shape, fp32 weights: SME's dense rate on our
      shape, measured alone only.
  - A, the Neural Engine: one Core ML call computing 32 experts' gate / up / down for all 16 rows, int8 per channel
    (ANE-2's encoding), placement checked with MLComputePlan.
- **Measure:** each unit alone; then the GPU chain in phases alternating "GPU alone" with "GPU + one unit running
  back to back"; the GPU's time per call gives its slowdown s, the unit's own log gives its rate R_u' during the phase.
- **Effective gain** of a split that gives the unit its best share (time-balanced): gain = R_u' / R_g + 1 / (1 + s) - 1,
  with R in experts per ms of the same windows (for the Neural Engine, which computes all 16 rows of an expert it
  holds, experts per ms of its 32-expert call).
- **Lines (the orchestrator's suggestion, adopted):**
  - PASS: one unit's gain >= +20 % with s < 5 %, measured in the same phase.
  - KILL: every unit's gain < +10 %, or s >= 5 % in every phase where a unit's gain would reach +10 %.
  - BETWEEN: otherwise.
  - C-BW is a bound, not a kernel: if even its gain at s < 5 % stays under +20 %, no CPU kernel can pass.
- **Not covered by a PASS (would be the next step):** the Neural Engine runs only compiled weights, so it could hold a
  fixed expert subset only as an int8 copy (memory at 45 / 24 GB), with a per-layer handoff (48 per check); a CPU
  share needs a CPU-side 4-bit layout or kernel for records3's 3 + 1-bit pieces, and a same-scores test for its Q8
  numerics.
- **Prediction [G]:** G ~0.65-0.75 ms per call at ~190-220 GB/s. C-BW 4 threads alone ~90-130 GB/s; beside G the GPU
  slows 10-25 % at full CPU streaming. C-Q4 4 threads ~30-60 GB/s-equivalent, gain +10-25 % at s 3-10 %. The Neural
  Engine ~16 experts/ms (gain ~+20-25 % before its slowdown), s ~5-7 %. Verdict odds: BETWEEN 55 %, KILL 30 %, PASS 15 %.
- **Budget:** <= 10 min of runs, by day as a small job: share lock, MEM ~4 GB, after the GPU is free (one GPU job at a
  time by day). Swap guard +2 GB.
- **Results (Tue 17:18-17:21, share lock, GPU otherwise idle, Kamen on PowerPoint; data/wc_screen.txt):**
  - Alone:
    | unit | rate alone |
    |---|---|
    | C-BW 1 / 2 / 4 threads | 68.5 / 129.8 / 220.6 GB/s |
    | C-Q4, 4 threads | 22.3 experts/ms (61.5 GB/s of 4-bit expert bytes); check cos 0.99990 (min row 0.99990) |
    | C-SME (Accelerate sgemm, fp32 weights) 1 / 4 threads | 2.2 / 5.2 experts/ms (352 / 821 GFLOP/s; 44 / 103 GB/s of fp32 weights) |
    | ANE a8 (int8 per channel) | 14.4 experts/ms |
    | **ANE a4 (4-bit LUT per 16-channel group)** | **26.2 experts/ms** (1.25 ms per 32-expert call, 126 G weights/s; MLComputePlan: all 96 linears + SiLU / mul / add on the Neural Engine) |
  - G alone: NAX6-N 0.761 ms per call (mean of the alone phases), 81.1 experts/ms, 190 GB/s of tier bytes, 62.1 unique
    experts per 16-row window.
  - Beside G (s per phase vs the two neighbouring alone phases; gain = R_u' / R_g + 1 / (1 + s) - 1):
    | unit | phase 1: s, gain | phase 2: s, gain | mean |
    |---|---|---|---|
    | **ANE a4** | +6.4 %, +24.7 % (24.8 experts/ms) | **+2.4 %, +23.0 %** (20.6 experts/ms) | s +4.4 %, gain **+23.8 %** |
    | C-Q4, 4 threads | +4.9 %, +16.0 % (16.7 experts/ms) | +0.7 %, +16.5 % (13.9) | s +2.8 %, gain +16.3 % |
    | ANE a8 | +4.6 %, +12.6 % (13.8) | - | |
    | C-BW 1 / 2 / 4 (bounds) | +0.9 % at 42.6 GB/s (bound +21.5 %); +3.7 % at 71.9 GB/s (bound +34.3 %); +22.8 % at 106.4 GB/s | | |
  - **Verdict on the pre-registered line: PASS for the Neural Engine with 4-bit LUT weights**, at the line's edge: one
    of its two phases met both clauses (s +2.4 %, gain +23.0 %); the other had gain +24.7 % but s +6.4 %; mean s +4.4 %.
    The CPU kernel is BETWEEN (+16 % at s <= 4.9 %); ANE int8 BETWEEN (+12.6 %). Prediction was BETWEEN 55 / KILL 30 /
    PASS 15 %, so the Neural Engine came out better than I expected: LUT4 runs natively and twice as fast as int8.
  - **Confound:** G's alone time drifted up through the run, 0.684 -> 0.837 ms (+22 %, monotonic), most likely heat
    from the arms (C-BW 4 threads at 220 GB/s, sgemm) at day fan settings. s is taken against the neighbouring alone
    phases, but a sustained load's thermal cost is not in it. A quiet re-run with longer phases (3 x 20 s each) is needed
    before building anything.
  - **Bandwidth headroom is real [K]:** 2 streaming threads took 72 GB/s beside G for s +3.7 %. My C-Q4 is compute-bound
    (~15 GB/s per thread), so a better CPU kernel (SME2 int8 dot, or 6 P-cores in an engine that owns the Mac) has room
    up to the ~+34 % bound [P]. SME through Accelerate on fp32 weights is useless here (5 experts/ms).
  - **Structural cost for the Neural Engine (the next question, not in the line):**
    - It runs compiled weights only, so it can hold a FIXED set of experts per layer, and each call computes its whole
      set for all 16 rows.
    - Measured on the traces [K]: a hot set taken from OTHER domains covers only 3-16 % of a window's experts at 64 per
      layer. A session-adaptive set (the same text's first half -> its second half) does much better: at 16 experts
      per layer, a window uses 52-89 % of the set (mean ~71 %), = 10-25 % of the window's experts; at 32, 47-81 % (mean
      ~63 %), = 18-46 %.
    - So the useful Neural Engine rate is ~0.7 x 25 = ~18 experts/ms, gain [P] ~+18 % before handoffs, ~+9 % halved.
      Each layer needs a GPU -> ANE -> GPU handoff (48 per check, ~0.1-0.2 ms each if not hidden), and the hot set has
      to be recompiled per session (~2 s per layer model on the CPU). It is FAST-only (LUT4 per 16 channels is not our
      affine 3 / 4-bit tiers, and the Neural Engine computes in fp16). Memory is roughly neutral if the LUT4 copies
      replace those experts' GPU pool slots.
  - **Also unchecked:** a4 re-runs one 79 MB model, so on-chip caching on the Neural Engine side could flatter it; the
    re-run should alternate several models with different weights, as 48 layer models would.
  - **Next, if routed:** (1) the quiet sustained re-run (longer phases, ANE a4 + C-Q4 together too); (2) the per-layer
    handoff cost: a native GPU -> ANE -> GPU round trip on MTLSharedEvent + Core ML with zero-copy inputs (the decisive
    cost for any per-layer split); (3) a CPU kernel on the real 3 + 1-bit pool format, reading the GPU's slots in place.

## P11 11-bit packed scales inside 3c-N and NAX6-N (Leap, pre-registered Tue 2026-11-03 17:2x EET; Kamen approved the combined conversion, the orchestrator's order 17:3x; MoonShots M-27 / M-27b; format pinned with BigMoE 17:2x; scripts/p11.py, tierglu_3cn.py, tierglu_nax6n.py, p11_check.py, convert_records_p11.py; data/p11_check.txt)
- **Format (scripts/p11.py, the reference):** per s3 / b3 / b piece, native order, then 11-bit codes
  sign << 10 | (exp - base) << 7 | mantissa7 (base = the piece's minimum exponent; no zero code), 32 codes per 11 u32
  words LSB-first, base in word 8800, 8804 words = 35,216 B. Record "native+p11": MSB 2,064,384 + LSB 720,896 =
  2,785,280 B (today 2,932,736). Reference self-test on layer 20 [K]: row-major -> native+p11 -> row-major == the file
  for all 512 records, the native pieces == data/native_sample's.
- **Kernels:** pools whose s3 / b3 / b are uint32 [C, 8804] switch 3c-N (all seven kinds) and NAX6-N to P11P reads:
  a view of the expert's piece whose [] returns as_type<float>(bf16 bits << 16), i.e. exactly (float) of the bf16
  value. bf16 pools take the unchanged path (inert by default).
- **Line:** PASS = every compared output bitwise equal to the same kernel on bf16 native pools, on two full layers (20,
  44) with every expert read at a 4-bit rank and at 3-bit ranks (all s3 / b3 / b pieces), through 3c-N's tier_glu /
  scores / phases / pairs / split, ref2's native_3c selector and OOO path, and NAX6-N; plus ref2.self_test. FAIL = any
  mismatch. Speed (rough, no line): p11 vs bf16 on layer 20, S 16.
- **Prediction [G]:** PASS (the reads change only where the floats come from). Speed: 3c-N p11 +5-20 % slower (one
  11-bit decode per scale read, 3c is ALU-bound), NAX6-N +2-8 %.
- **Results (Tue 17:29-17:35; share lock; data/p11_check.txt, p11_check_run{,2,3}.log):**
  - **PASS, every comparison bitwise**, three times: v1, v2 and v3 below. On layers 20 and 44, all 512 experts were read
    at a 4-bit rank and at 3-bit ranks; 16 / 16 for each of tier_glu_n, + scores, phases, pairs, split, ref2
    native_3c tier_glu, ref2 native_3c OOO, and NAX6-N. ref2.self_test PASS.
  - Speed of the p11 read (layer 20, S 16, rough):
    | version | 3c-N bf16 -> p11 | NAX6-N bf16 -> p11 |
    |---|---|---|
    | v1: every s3[gi] decoded in place | 1.597 -> 2.716 ms (**+70 %**) | 0.782 -> 0.833 (+6.4 %) |
    | v2: each simdgroup decodes its 4 rows x groups once into threadgroup memory | 1.644 -> 1.950 (+18.6 %) | 0.685 -> 0.737 (+7.6 %) |
    | **v3: v2 + a lane keeps one row's bit offset, two-word reads, no branch (NAX6-N: P11R)** | **1.597 -> 1.615 (+1.2 %)** | **0.684 -> 0.699 (+2.2 %)** |
  - My prediction (+5-20 % / +2-8 %) was right only after v3. v1's in-place decode doubles 3c's ALU work per scale
    read: 3c is ALU-bound, and a lane reads 2 scales per 16 weights.
- **ref2 per-layer routing (BigMoE's proposal 17:3x, agreed; scripts/layout_router_check.py PASS):**
  - _tier_glu_sel / _ooo_sel route by t["layout"] or bigmoe.tier._LAYOUT[0]: 0 -> BigMoE's 3c, 1 / 2 -> 3c-N (p11 from
    the pools' dtype).
  - Without a router, CFG native_3c decides as before, so this is inert until BigMoE's patch lands (Wed after FLOOR).
  - Guards: native_3c on layout 0 raises; layout 1 / 2 with tier_nax4, or with tier_nax_ooo without nax_native, raises.
  - Plumbing: _tier_slots sets _LAYOUT[0]; apply() sets LAYOUT_ROUTER = True; _native_repack skips p11 pools.
  - ref2.self_test PASS.
  - Fix after BigMoE's review (17:4x): with REF2 removed (r57_screen toggles it around drafts), a converted layer used
    to fall through to BigMoE's row-major 3c silently. Now layout 1 / 2 takes 3c-N whether REF2 is on or off;
    layout_router_check.py covers it (PASS), self_test PASS. The commit for EvalSuite's snapshot is the one after
    2b3fc63.
- **Combined tool (scripts/convert_records_p11.py) on MY scratch copy of records3 L00 / L20 / L47 (data/scratch_p11):**
  - Read-only check: 0 pieces fail to pack (2 s).
  - md5 of the 3 files: 5 s.
  - Forward: 3.5-3.7 s per layer, temp + fsync + re-read + inverse compare included. **The files == BigMoE's
    independent reference converter's (~/BigMoE/scratch/r3p11), cmp identical; the index entries match.**
  - --inverse: 3.5 s per layer; **the md5s match the originals, the files cmp equal to records3, and the index entries
    are restored exactly** (a native+p11 entry keeps its row-major entry under "rowmajor" for this).
  - So for all 48 layers: ~3 min, plus ~80 s of md5.

### DISK-PLAN addendum: the combined format (native + p11), Kamen approved Tue 17:2x (not before Thu 15:00, after RELAX; 1 h heads-up)
- One in-place pass: scripts/convert_records_p11.py records3 --md5 data/records3_md5_rowmajor.txt --yes (it supersedes
  convert_records_native.py for this store).
  - Per layer: native order + 11-bit s3 / b3 / b pieces; 1.50 -> 1.43 GB per layer; the store shrinks 72.1 -> 68.5 GB.
  - Peak extra disk: one temp layer (1.43-1.50 GB). It refuses under 112 GB free, while lsof shows a records3 file open,
    or if any piece does not pack (the read-only check covers all 48 layers before anything is written).
  - --inverse restores the exact original bytes; checked by md5 on the scratch copy.
- Kernels: 3c-N and NAX6-N read the p11 pools natively (bitwise, +1-2 %). REF2-T's exact path keeps its bits through
  ref2's native_3c / per-layer router.
- BigMoE's side: the readers are done and bitwise (761cb3d: read_rebuilt_layout, the pool sizes from the index); the
  router plumbing is applied Wed after FLOOR.
- Still to do before the real run: the heads-up to the sessions that read records3 (data/records3_readers.txt), and
  BigMoE's end-to-end decode on a part-converted scratch store.

## WHOLE-CHIP-2 the handoff cost and the sustained re-run (Leap, pre-registered Tue 2026-11-03 17:4x EET; the orchestrator's order 17:3x after WHOLE-CHIP's edge PASS; native/wc_handoff.swift, scripts/wc_screen.py; runs at night or with Kamen away, never while he is on the Mac)
- **Why:** WHOLE-CHIP's edge PASS (the Neural Engine with LUT4 weights, +23.8 % expert throughput at s +4.4 %) is worth
  ~+18 % [P] after the fixed-set loss. A per-layer split needs a GPU -> ANE -> GPU round trip in each of the 48 layers of
  a check. The orchestrator: "if 48 handoffs per check eat the gain, we stop there".
- **H, the handoff (native Swift: Metal + Core ML, no Python in the loop):**
  - Setup: one Metal command buffer per check of 48 "layers". Each layer: a GPU kernel writes x (16 x 2560 fp16) into
    a shared buffer; the command buffer signals an MTLSharedEvent; the GPU then runs its own share; it waits on the
    event's next value; a combine kernel reads the Neural Engine's output.
  - A CPU thread waits for each signal (spin on signaledValue), runs the Core ML predict, and signals back. The predict
    uses the a4 expert model (K = 8 / 16 experts), with x and the output as MLMultiArrays over the Metal buffers'
    memory (zero copy, output backings).
  - Arms per layer:
    - (a) GPU only: the share kernel sized to NAX6-N's full layer (~0.68 ms, a bandwidth-bound read at its rate).
    - (b) the split: the GPU's share sized to its part + the ANE's K experts in parallel.
    - (c) handoff only: empty GPU work + an ANE call of K experts, to isolate the round trip.
    - (d) the ANE call alone from Swift, same IO.
  - Overhead per layer = (c) - (d): the event round trip + predict overhead beyond its compute. Hidden-ness: (b)
    against max(the GPU part, (c)).
- **Lines (per layer; 48 per check):**
  - KILL: the round-trip overhead (c) - (d) >= 135 us. 48 x 135 us = ~6.5 ms, all of the [P] gain at 16 rows.
  - PASS: <= 50 us (<= 2.4 ms per check, so >= +12 % net remains [P]) AND (b) <= 0.9 x (a) at the best K.
  - BETWEEN otherwise.
- **S, the sustained re-run (wc_screen.py, quiet):**
  - G alone, then G + ANE a4, then G + C-Q4, then G + both; each 20 s, alternated 3 times, after a 60 s warm-up to
    steady temperature.
  - The ANE arm cycles 4 different 32-expert models, so no weights are reused within 4 calls, against on-chip caching.
  - Lines as WHOLE-CHIP (PASS: gain >= +20 % with s < 5 %, now as the 3-phase mean); plus "both": the sum's gain and s.
- **Prediction [G]:**
  - H: (d) ~0.65 ms at K 16; overhead ~60-150 us (Core ML's predict path ~50-100 us + two event hops ~10-30 us each),
    so BETWEEN-to-KILL.
  - S: the ANE a4 holds +18-24 % at s 4-7 % sustained, so BETWEEN / edge. Both together: s >= 8 %.
- **Budget:** H ~3 min, S ~6 min, each under a quiet lock at night (after 00:30) or with Kamen away; MEM ~4 GB.
- **Build + functional smokes (Tue 17:4x-17:58, short share locks; NOT the measurement):**
  - native/wc_handoff.swift (+ wc_bridge.h) and wc_ane.py build4 (a4 models of K 4 / 8 / 12 / 16, and four 32-expert
    models with different weights for S).
  - Found while building, [K]: **a running GPU kernel never sees a CPU store to shared memory on this M5 Pro**
    (native/wc_spintest.swift). Volatile loads, relaxed-atomic loads, atomic RMW, and a write-combined buffer all spin to
    their cap (4 M iterations, 140-400 ms); a store made before the kernel starts is seen at once. So CPU -> GPU must be
    an event the CPU signals; a GPU spin on a CPU flag is out (the first smoke with it hung and was stopped by PID).
  - Protocols: E1, a GPU-signalled event in the buffer, and E2, BigMoE's post kernel + a CPU spin on shared memory + a
    CPU-signalled wait, both cost ~135-155 us per round trip when nothing hides it.
  - With E1, the GPU's own share did not overlap the Neural Engine call (smoke 1: the split 2.2x slower). With E2 it
    does.
  - A Neural Engine call costs ~130 us fixed + ~33 us per expert (K 4: ~0.24 ms, K 16: ~0.65-0.71 ms), so even K 8
    takes about as long as the GPU's whole layer.
  - Smoke 3 (stand-in at 0.70 ms): best split K 8 at B / A = 0.934; sync overhead ~104-137 us per layer. Smoke 6
    (stand-in reading at a full 288 GB/s): B / A 0.99-1.04. So the stand-in is throttled to NAX6-N's ~200 GB/s for the
    run.
  - S's code paths smoke (2 s phases): a4x4, which cycles four different-weight models, slowed the GPU +11.2 % at
    26.2 experts/ms (gain +18.5 %). The single-model a4 was +2.4 / +6.4 % in WHOLE-CHIP, so on-chip reuse may have
    flattered it. A4x4 + C-Q4 together: s +34 %, gain +22 %. A hint only; the quiet run decides.
  - Armed: scripts/run_wc2_night.sh (attached; no PAUSE / HOLD, then a quiet lock in the FIFO; H then S; swap guard;
    unlock). Logs: data/wc2_night.log, wc2_handoff.txt, wc2_sustained.log.
  - Re-armed 18:0x on the orchestrator's timing fix. The ticket is queued no earlier than 00:44, because Fit12's quiet
    slot is 00:30-00:43:30 and it drops a grant after 00:35:30. The cap is a HARD ~10 min: H at most 150 s, S at most
    420 s, each killed at its cap, lock 11 min.
  - **Deviation from the S pre-registration (forced by the cap):** 12 s phases instead of 20, and a 45 s warm-up
    instead of 60; still 3 rounds of {a4x4, q4, a4x4 + q4}, each between alone phases.
> (a quote from the project owner, kept private)
    overnight"):** the night runner was stopped (19:00). The run goes now (`WC2_NOW=1`), by day, under a quiet lock
    with Kamen on the Mac (PowerPoint, Chrome, Spotify). The order became M-28 -> Fit12 -> WC2: Fit12 was granted at
    19:00:53 while no ticket of mine was in yet.
    - scripts/wc2_load.py records the background load beside H and S: per app, CPU from ps time deltas and GPU from
      ioreg AGXDeviceUserClient accumulatedGPUTime; the runner's tree is kept apart. It also logs GPU utilization,
      swap and available memory, plus a 10 s baseline before H. Outputs: data/wc2_load.jsonl, data/wc2_marks.txt,
      data/wc2_load_summary.txt.
    - Lines and caps are unchanged. If the noise makes S unreadable, S is re-run alone at night (the orchestrator).
- **RESULT (Tue 19:08:56-19:22:57, quiet lock, by day with Kamen on the Mac; data/wc2_handoff.txt,
  data/wc2_sustained.log, data/wc2_load_summary.txt): H = KILL at the best split (BETWEEN at the other K); S =
  UNREADABLE (two heat freezes). The handoff eats the gain, so the bet stops, as agreed with the orchestrator.**
  - **H [K]** (19:09:06-19:09:35, no freeze; per layer, 20 checks of 48 layers; stand-in GPU layer 0.700 ms at 206 GB/s;
    window 62 experts, hot-set use 0.71):

    | K | C handoff | D ANE alone | **C - D** (the line) | C - predict inside | B split | **B / A** |
    |---|---|---|---|---|---|---|
    | 4 | 331.3 us | 244.5 | 86.8 us | 102.6 | 720.8 | 0.980 |
    | 8 | 464.6 | 386.3 | **78.4 us** | 104.0 | 700.7 | 0.953 |
    | 12 | 659.7 | 512.6 | **147.1 us** | 132.2 | 694.9 | **0.945** (best) |
    | 16 | 773.1 | 657.6 | 115.5 us | 129.9 | 725.7 | 0.987 |

    - A, the GPU-only layer: 735.2 us. The round trip alone: E1 (GPU-signalled event) 151.8 us, E2 (BigMoE's post
      kernel + a CPU-signalled wait) 128.7 us.
    - Against the lines: no K is near PASS (the overhead is never <= 50 us; B / A never <= 0.9). At the best split
      (K 12, 0.945) the overhead is 147 us >= 135: **KILL**. K 4 / 8 / 16 are BETWEEN (78-116 us).
    - What it means: the best split is only 5.5 % faster per layer (+5.8 % throughput; K 8: +4.9 %), against the
      ~+18 % [P] without handoffs. 48 handoffs cost ~3.8-7.1 ms per check, 11-20 % of the 35.3 ms GPU check, so they
      eat ~2/3-3/4 of the gain. The split's unhidden cost is ~33 us (K 8) to ~61 us (K 12) over the ideal max(GPU share,
      ANE call).
    - The pre-registered prediction held: H BETWEEN-to-KILL; (d) at K 16 0.658 ms (pred. ~0.65); overhead 78-147 us
      (pred. 60-150).
  - **S: UNREADABLE, not a result.** The heat watcher (fanwatch, the 5000 rpm day cap) froze the research jobs, my
    runner included, twice:
    - 19:11:13-19:15:57 (284 s by the sampler's gap) and 19:18:38-19:22:01 (204 s); 488 s of S's 801 s were frozen.
    - The chip went HOT (thermal state 1) 37 s into S (19:10:12).
    - The "none" phases spread 0.68-1.25 ms per call (a 1.8x range), and one q4 phase ran 0 s.
    - The printed "PASS ['a4x4']" (s -7.5 %, gain +41 %) is an artefact of the freezes: a negative slowdown is noise.
    - Exit 145 = 128 + 17, consistent with SIGSTOP (17 on macOS).
  - **Background load (scripts/wc2_load.py; CPU in cores, GPU in % of wall time, my runner's tree apart):**
    - 10 s baseline: 1.50 cores, GPU 7.2 % (Spotify 3.5 %, WindowServer 1.8, Terminal 1.7). The "claude" CPU
      (0.21-0.34 cores) is the Claude CLIs themselves.
    - During H: 1.22 cores, GPU 3.0 % (max 5.6).
    - During S: 1.50 cores (max 2.85), GPU 8.2 % (max 33; WindowServer 5.2 %).
    - Swap +0, available >= 29 GB.
    - So H's noise is small against its 80-150 us effects; S's problem is heat, not the apps.
  - **Cap overrun (my bug, fixed):** run_capped counted its own loop seconds, and the freeze stopped the runner too, so
    S ran 801 s wall against the 420 s cap and outlived the 11-min lock (19:19) by ~4 min. No grant was waiting or
    made in that time (turns.jsonl), and PAUSE was set by the heat. run_capped now caps on the wall clock ($SECONDS;
    tested), and the load summary flags freezes ("FROZEN ... s").
  - [K] By day under the 5000 rpm fan cap, a sustained GPU + ANE + 4-core CPU load is frozen within ~1.5 min, so
    sustained co-run numbers need the night (fans uncapped).
  - **Decision:** per the orchestrator ("if 48 handoffs per check eat the gain, we stop there"), the per-layer
    GPU -> ANE split is stopped. S's night re-run is moot for this bet and is not armed; it is one command if wanted.
    The one lever left [untested, not pursued without routing]: an ANE dispatch far cheaper than Core ML's ~130 us
    fixed cost per call (a private-API path), which would need <= ~50 us per round trip to matter.

## W1-PICKUP the release -> GPU pickup of a loader call (Leap, pre-registered Tue 2026-11-03 19:3x EET; the orchestrator's routing 19:2x for the WALL 1 SPRINT, BigMoE leads; native/w1_pickup.swift)
- **Why:**
  - Of the ~0.28 ms per loader call, ~96 us sits outside the I/O path [K M-28]: the last record's copy ~53 us, the
    release ~8, the pickup 0-40, the plan ~10-30 (Fable A's WALL1_D1_SPEC).
  - The pickup is GPU-side. OOO waits for the loader with a one-thread kernel (ext bigmoe_spin) that polls FRESH line
    pairs: 128 pairs per (layer, pass) region, 192 ints each at MAXK 160. It idles 32 x 1.1^p FMA iterations between
    polls, because a running kernel never sees a CPU store on a line it has already read [K WHOLE-CHIP-2].
  - The release writes the stamped slots into all 128 pairs (~82 KB), then the tags. Event mode (an MTLSharedEvent per
    layer, encodeWait in the next command buffer) is the engine's other path.
  - [K SC5]: the pickup adds +35-40 us after 0.6-1 ms stalls and +130 after 2 ms.
- **Prior art:**
  - MLX 0.32.2 (BigMoE's venv) has a Fence "fast mode" (MLX_METAL_FAST_SYNCH=1, Metal 3.2+): one-thread kernels
    fence_wait / fence_update over a `volatile coherent(system) device uint*`, with seq_cst `thread_scope_system`
    fences under `#pragma METAL internals`. They are compiled in mlx.metallib.
  - My wc_spintest never tried coherent(system).
  - An MLX issue reports a deadlock with fast synch in a distributed pipeline, so correctness gets its own check here.
- **Harness.** Each trial is one command buffer: an m1 marker (BigMoE's post pattern: a device atomic store + fence),
  then the wait, then an m2 marker. The CPU (user-interactive QoS) sees m1, waits w, releases (timed), then times until
  it sees m2. Arms:
  - **A, today:** bigmoe_spin verbatim (G1 = G2 = 1.1, KNEE 0, PAIRS 128, RS 192, n = 160 stamped slots, a 128-region
    ring like the engine's) with the engine's release (all pairs, a fence, then the tags). The kernel also bumps
    status[32 + got], S0's planned histogram, so each trial's pair index maps to its pickup time.
  - **B, coherent:** one thread polls ONE row, the same lines every poll, with coherent(system) loads and system
    fences. The CPU writes the row once (stamped words, a release fence, the tags). Same decode and stamp check as A,
    with an iteration cap (a timeout is counted).
  - **M:** MLX's own compiled fence_wait from mlx.metallib on a timestamp word. It runs only if B has no timeout,
    because it has no cap.
  - **C, event:** end the encoder -> encodeWaitForEvent(ev, seq) -> the next encoder; the release is
    setSignaledValue(seq) (BigMoE's event mode).
  - **N, null:** m1 -> m2 with no wait (the markers' floor).
  - w in {-1 (released before the commit: a hidden burst), 0, 50, 200, 600, 1000, 2000} us; 300 trials per (arm, w),
    interleaved in a shuffled order each round.
- **Metrics:**
  - release = the CPU time of the release writes or signal;
  - pickup = t(m2 seen) - t(release done), median and p90;
  - overshoot(w) = pickup(w) - pickup(-1);
  - correctness: every trial's decoded slots equal the released ones, and no timeouts.
- **Lines** (B is the fastest-correct-form candidate; C is recorded):
  - **PASS:** B is correct on every trial AND its pickup median is <= 10 us and p90 <= 20 us at w = 0.6, 1 and 2 ms
    AND B's release + pickup is <= A's - 25 us (median) at w = 1 ms. Then BigMoE gets a drop-in (kernel + release
    writer) behind a flag, for an engine A/B with S0.
  - **KILL:** B sees no release (timeouts) or any wrong slot, or it beats A by < 10 us at w = 1 ms. Then today's spin
    stays, and the pickup is not where the 96 us is.
  - **BETWEEN** otherwise.
- **Prediction [G]:**
  - A: overshoot ~5-10 % of w (+30-60 us at 0.6-1 ms, +100-150 at 2 ms); its floor (w = -1) 5-15 us over N; release
    3-8 us.
  - B: works, pickup 1-6 us flat, release < 1 us -> PASS. M about equal to B.
  - C: 40-120 us flat (the command processor's wake), release 2-10 us.
- **The engine prize [P]:**
  - ~23 exposed bursts per check x ~40 us + 83 releases x ~5 us is ~1.3 ms per check at 45 GB (~+1.3 % tok/s, ~+0.7 %
    halved). Small alone.
  - A ~us pickup is what S7b (a per-expert release while the burst lands) and lever 2's in-flight forecast waits need.
- **The hook with BigMoE's S0 (Wed):**
  - (1) A's p -> time table on this Mac, so S0's status[32 + p] histogram reads as a pickup bound without the
    BIGMOE_DELAY_US calibration.
  - (2) If PASS, a drop-in coherent spin + one-row release behind a flag (BigMoE's edit, in its repo). S0's t_rel then
    bounds release + pickup directly.
- **Budget:** ~1 h of build and smokes (share lock, tiny). The measurement is ~1 min under a short quiet lock (one GPU
  thread + one CPU core, no heat). By day: short bursts only.
- **Smoke findings (Tue 19:37-19:39, share lock, 3-5 trials per cell; NOT the measurement):**
  - [K] The m1 marker written BigMoE's post way (device atomic stores + a device-scope fence) reached the CPU only
    ~30 ms after the commit, while the spin kernel ran on and touched almost no memory. With no wait (N), both markers
    appeared together at the end of the command buffer (~140 us). As a system-scope store (MLX's fence_update), the
    marker leg is ~3 us. The measurement therefore uses system-scope markers.
  - Today's spin (A): release ~9.7 us (82 KB into 128 pairs). Pickup 47 us at w = 0 (pair ~3), 85 us at w = 600
    (pair ~40).
  - B (coherent spin): correct on every trial, also on one REUSED row (B1). Release 0.2 us. Pickup 10-15 us, polls
    ~1.06 us apart.
  - C (event): release 2-2.5 us, pickup 17 us at w = 0 but **368 us at w = 600**.
- **Addendum, pre-registered before it runs (Tue 19:4x): P, the post's visibility to the CPU (the GPU -> CPU half;
  the "see" stage, ~0.005 ms [G] in the D1 spec).**
  - Per trial: m1 (system scope) -> a post kernel writing a stamped row the way BigMoE's post does (device scope) ->
    [P1 only: a 64 MB read kernel, the hit work's traffic] -> a hold (B's spin, released by the CPU once it sees the
    post, or after 5 ms) -> m2.
  - The CPU times m1 seen -> the post's tag seen. Arms: P0 (device-scope post, the GPU idle after it), P1 (device
    scope + 64 MB of traffic), PS (the post's tags at system scope, idle after).
  - **Line:** if P1's or P0's median visibility delay is >= 20 us over PS's, the engine's post should publish its tags
    at system scope. That is a kernel-only change, exact (the same words), and S0 can time it with a system-scope stamp
    word in the post. Under 5 us: the post is not part of the 96 us.
  - **Prediction [G]:** P0 is visible only at a flush point (ms, or the command buffer's end, capped at 5 ms here); P1
    becomes visible when the traffic evicts the line (tens to a few hundred us); PS is ~3 us.
- **RESULT (Tue 19:41:29-19:41:39 and 19:43:38-19:43:42, quiet locks; 300 trials per cell, interleaved;
  data/w1_pickup.txt, data/w1_pickup_P.txt): PASS. The coherent(system) spin (B) is correct on all 4,200 trials,
  the reused row (B1) included. Release + pickup goes from ~23-80 us today to ~3-4 us, flat in the wait.**

  | arm (median us) | release | pickup w -1 | w 0 | w 50 | w 200 | w 600 | w 1000 | w 2000 |
  |---|---|---|---|---|---|---|---|---|
  | A today (spin + 128-pair release) | 6.4-7.9 | 16.7 | 15.7 | 16.2 | 28.9 | 17.6 (p90 67) | **73.5** | 67.2 |
  | B coherent spin, ring of rows | 0.17-0.21 | 3.9 | 3.7 | 2.9 | 2.9 | 2.9 | **2.9** | 2.9 |
  | B1 coherent spin, ONE reused row | 0.17 | 3.8 | 3.6 | 2.8 | 2.8 | 2.8 | 2.8 | 2.8 |
  | M MLX's own fence_wait (flag only) | 0.21-0.25 | 2.0 | 1.8 | 1.2 | 1.2 | 1.2 | 1.2 | 1.2 |
  | C MTLSharedEvent | 0.9-2.3 | 2.2 | 12.1 | 96.0 | 76.6 | 78.3 | 77.2 | **360.5** |
  | N markers alone | | | 1.0 | | | | | |

  - **Today's spin, A, read as the engine pays it:**
    - Its poll schedule on this Mac (the time from the spin's start to pickup at pair p): p0 17 us, p6 22, p26 73,
      p41 236, p52 623, p58 1080, p65 2074 us. The gap between polls is ~10 % of the elapsed wait.
    - A burst landing at a random moment pays half a gap on top of the ~17 us floor: ~+29 us at 0.6 ms, ~+54 at
      1 ms, ~+104 at 2 ms. That matches [K SC5]. The fixed-w medians above sample single phases of this schedule
      (w 600 falls just before p52).
    - Even a hidden burst pays the floor: 6 fresh-line misses + the decode = 16.7 us, against B's 3.9.
  - **C:** the event's wake costs 77-96 us after >= 50 us of waiting and 360 us after 2 ms, so it is not a
    candidate.
  - **Against the lines:**
    - B is correct on every trial.
    - B's pickup median is 2.9 us and its p90 3.1 at w = 0.6, 1 and 2 ms (lines: <= 10 and <= 20).
    - At w = 1 ms, B's release + pickup is 3.1 us against A's 80.0 (line: >= 25 us better).
    - **PASS.**
  - **The prediction held,** except two cells: C was worse at 2 ms (360 us, pred. 40-120), and A's floor was
    15.7 us over N (pred. 5-15).
- **P, the post's visibility to the CPU (all line tags, the engine loader's criterion; P2 / P1 = 8 / 64 MB of reads
  after the post):**
  - P0, device scope, idle GPU: 293/300 not all visible within 5 ms; the few that were, at ~1.8 ms.
  - P2, device scope + 8 MB: 20.0 us (p90 26.9).
  - P1, device scope + 64 MB: 23.0 us (p90 26.8).
  - PS, system scope, idle GPU: **7.4 us** (p90 7.6), 0 bad.
  - Device-scope rows: 0.7-1.7 % of trials still had stale words after all tags, the engine's rare stamp re-reads
    (b45s: ~50 of ~80k posts).
  - **What it means:**
    - A device-scope post reaches the CPU only when something after it flushes or evicts its lines: ~20 us after
      the next real dispatch, not ~5 us (the D1 spec's [G]). It does not arrive at all while the GPU only spins.
  - **The P line is met only through P0, the idle case.** With traffic after the post, system scope saves 13-16 us,
    under the 20 us line. Worth taking as insurance against a post followed by little GPU work, and lane-parallel
    writes may bring PS's 7.4 us to ~2 us [G].
- **The engine prize [P], 45 GB C 256: ~23 exposed bursts and 83 calls per check:**
  - Release -6.3 us per call.
  - Pickup -13 us at the floor and ~-40 more on an exposed burst.
  - Post -13-16 us on exposed calls.
  - In all ~1.7-2.0 ms per check, ~+2 % tok/s at the bound (~+1 % halved): exact, kernel + release-writer only.
  - The bigger value: a ~3 us pickup makes per-expert releases (S7b) and lever 2's in-flight forecast waits cheap.
    Each would cost 17-100 us with today's spin.
- **Caveats:**
  - This is a harness, not the engine: MLX's encoder and buffers, OOO's concurrent work beside the spin, and the
    real wait distribution all differ. The engine A/B decides.
  - `coherent(system)` and system-scope fences need `#pragma METAL internals : enable` (undocumented; MLX uses it
    in production behind MLX_METAL_FAST_SYNCH). So the old path stays as the fallback behind a flag.
- **Hook for BigMoE's S0 (Wed), proposed:**
  - (1) The p -> time table above, so the status[32 + p] histogram reads as time.
  - (2) Drop-ins with reference code in native/w1_pickup.swift: BIGMOE_SPIN_COHERENT (w1_coh_spin: one simdgroup
    polls ONE row; the release writes one row, not 128 pairs) and BIGMOE_POST_SYS (w1_post_sys).
  - (3) The engine A/B (default vs spin-coherent vs + post-sys, traced). Line, proposed: stall_report's cadence a
    minus burst a falls >= 0.03 ms AND tokens identical.
- **Handoff (Tue 19:4x-19:5x; BigMoE 19:4x: "W1-PICKUP joins Wed's build", plan 42939ed):** patches/w1_pickup.
  - w1_pickup.patch against BigMoE's ext at ec08cec, two flags, both default OFF:
    - BIGMOE_SPIN_COHERENT: a one-simdgroup coherent spin on pair 0, and the release writes pair 0 only.
    - BIGMOE_POST_SYS: post_ids through a system-scope pointer, with macros that preprocess to today's statements.
  - Checks: it applies cleanly and passes c++ -fsyntax-only with BigMoE's flags. test/kcheck (the exact kernel texts,
    on the GPU) PASS:
    - the spin's slots exact over 400 trials, the -1 / -2 codes included;
    - the fast path, a late-fixed stale word (polled again) and the timeout all as today;
    - post_ids: the patched default and POST_SYS write bitwise-identical rows and dep to today's kernel.
  - Base and sidecar spins never share a row: the region is keyed by (seq, L), and the sidecar has its own L.
  - BigMoE puts them into ext/build_w1 on Wed. Thu's traced arms add + SPIN_COHERENT / + POST_SYS. BigMoE splits
    tonight's stall by zero-hit layers, to see whether the post's late "see" is the bigger lever.

## KV8 8-bit K / V as a long-context SPEED lever at 45 GB (Leap, pre-registered Tue 2026-11-03 21:3x EET; the orchestrator 21:2x after NewIdeas' Strata prior-art pass; FAST-Q; scripts/fatt16q.py, kv8_screen.py, kv8_layout_probe.py; data/kv8_screen.txt)
- **Why:**
  - Strata (a llama.cpp fork tuned for this model) keeps K / V in int8. At ~10K keys attention adds ~23.5 ms per
    check (FIXED-PART), and agent turns live there.
  - TRIED has KV8 only as a 12 GB memory lever (KILL on footprint) and as a parked napkin (+20-45 % at 100K).
- **Prior art and priors:**
  - llama.cpp's q8_0 KV cache and its flash-attention dequant; MLX-LM's QuantizedKVCache (kv_bits 8, group 64, two
    quantized_matmuls); KIVI / KVQuant (per-channel K); SageAttention (int8 QK^T with K smoothing: subtracting the
    channel mean shifts each row's scores by a constant, which softmax cancels exactly).
  - Fable B's LONG_CONTEXT F2 says "~0 speed": the K / V bytes are ~1 of 24 ms at 10K, and the dequant adds work to a
    latency-bound kernel.
  - My B6-G found that K / V re-reads are not fs4's limit; its per-block serial chain is.
  - The one mechanism by which 8 bits can cut THIS cost (compute) is int8 matrix ops. M27-C [K]: MPP matmul2d does
    int8 x int8 -> int32 at 1.73x bf16 (25.6 vs 14.7 T MAC/s).
  - Layout [K] (kv8_layout_probe.py, 21:31): MPP's get_multidimensional_index confirms fs4's hard-coded bf16 layout.
    The int8 (16, 32, 32) tiles follow it extended to 32 dims, and the int32 score tile has exactly fs4's float layout,
    so fs4's online softmax runs on it unchanged.
- **Arms** (SDPA level, R 16, B6's synthetic setup with the indexer's mask, N 2K / 10K / 20K):
  - mlx1: MLX's one bf16 call.
  - mlxq8: MLX-LM's quantized SDPA path (8 bits, group 64), the published MLX form.
  - fs4: bf16.
  - **q8s, the storage form** (llama.cpp / Strata): int8 K / V (smoothed; per-token per-head scales) converted to bf16
    for fs4's MMAs. K's scale multiplies the scores; V's scale folds into P; V's mean is added at the output.
  - **q8m, the MMA form** (SageAttention-style): Q in int8 per (head, row), int8 x int8 -> int32 QK^T on the (16, 32,
    32) descriptor (8 runs per block instead of 16). The scores are float(acc) x s_q x s_k x scale; PV as in q8s.
  - Quantization of K / V happens on the cache write, so it is not timed. Q's quantization in q8m is timed.
- **Error:**
  - Every arm's output vs fp64 attention on the bf16 inputs (mean |diff| / RMS, and the max), and vs fs4.
  - At N 10K also on REAL-weight K / V: layer 3's projections, k_norm and RoPE over random hidden states, B6's layer
    harness, captured in the verify call.
  - Logits: kernel-independent (a fake-quant engine run of the same int8 K / V). Run only if the speed line passes;
    EvalSuite judges quality after that.
- **Lines** (ms per call, 8 chained, median of 20, a quiet lock; per check = 12 layers x the difference):
  - **PASS:** the best 8-bit fs4 form is >= 1.25x faster than fs4 at 10K AND >= 1.2x at 20K, with a mean error
    <= 1e-2 vs fp64.
  - **KILL:** both forms < 1.10x at 10K, or the error is > 1e-2.
  - **BETWEEN** otherwise.
- **Prediction:**
  - q8s 0.90-1.05x (the converts add work to a chain-bound loop; the bytes were never the limit).
  - q8m 1.05-1.30x (QK^T is about half the chain, run at 1.7x; softmax and PV remain).
  - Errors: q8s ~3-6e-3, q8m ~5-9e-3, mlxq8 ~3-5e-3.
  - mlxq8 slower than mlx1 (two dense quantized matmuls plus a softmax over all keys).
  - Verdict: PASS 10 %, BETWEEN 50 %, KILL 40 %.
  - Even a PASS is ~1-1.5 ms per check at 10K on top of fs4. The big long-context lever stays F1 itself: fs4's
    adoption, ~23.5 -> ~13 ms at 10K.
- **Budget:** ~1 h build; the screen ~3 min (< 1 GB). Timing under a short quiet lock; the error parts can share.
- **RESULT (Tue 21:36 accuracy on a share lock, 21:36:58-21:37:01 timing on a quiet lock; data/kv8_screen.txt,
  data/kv8_runner.log): BETWEEN.** The best 8-bit form (q8m) is 1.15x vs fs4 at 10K and 1.17x at 20K, so -0.8 /
  -1.7 ms per check on top of fs4. The error clause holds on synthetic and real-weight K / V, and FAILS on a stress
  case with outlier K channels.

  | N, ms per call (R 16) | mlx1 | mlxq8 (MLX-LM) | fs4 | q8s (storage) | q8m (int8 MMA) |
  |---|---|---|---|---|---|
  | 2K | 0.144 | 0.471 | 0.138 | 0.138 (1.00x) | 0.147 (0.94x) |
  | 10K | 0.485 | 1.016 | 0.515 | 0.464 (1.11x) | **0.450 (1.15x)** |
  | 20K | 1.146 | 1.834 | 0.987 | 0.893 (1.11x) | **0.846 (1.17x)** |

  | mean (max) \|diff\| / RMS vs fp64 | mlx1 | mlxq8 | fs4 | q8s | q8m |
  |---|---|---|---|---|---|
  | synthetic, 2K / 10K / 20K | 2.3e-3 | 6.6-7.6e-3 | 1.75e-3 | 6.9e-3 | 7.7-7.8e-3 |
  | outlier K, 10K (8 channels x 6, offsets) | 3.4e-3 | **1.32e-2** | 1.7e-3 | **1.51e-2** (0.31) | **1.60e-2** (0.26) |
  | real-weight K / V, 10K (layer 3) | 1.6e-3 | 4.7e-3 | 5.7e-4 | 4.4e-3 | 5.0e-3 |

  - fs4 reproduces B6 (0.515 / 0.987 ms; 1.75e-3).
  - The real-weight K is flat (channel RMS max / median 1.1) because the hidden states are random. Real text can
    give K outliers (massive activations; RoPE turns a shared bias into a rotating one that the mean subtraction
    removes only in part), so the stress case is the open risk.
  - **Harness bug, fixed:** MLX-LM's `queries *= scale` rebinds the CALLER's array in MLX. The first accuracy run
    (21:35) therefore fed fs4 / q8s / q8m a q scaled by 1/16 and is VOID; it is marked in the data file. The arm
    now uses a verbatim copy with `queries = queries * scale`.
  - **What it means:**
    - 8 bits buy ~11 % from the bytes (q8s) and ~15 % with int8 QK^T (q8m) at 10-20K, nothing at 2K. Both are below
      the 1.25x line. QK^T is only part of fs4's per-block chain, and softmax and PV stay.
    - MLX-LM's published int8 path is 2-3x SLOWER than MLX's bf16 call (0.31-0.62x). Two dense quantized matmuls
      over every key, so in MLX it is a memory saver only.
    - Per check at 45 GB: -0.8 ms at 10K and -1.7 at 20K, ~1-2 % of the cycle, on top of fs4. The long-context
      lever stays F1 itself: fs4 in place of REF2-T's chunk set, ~23.5 -> ~13 ms per check at 10K (B6), FAST-K
      certification.
  - **The prediction held:** q8s 0.90-1.05x predicted, 1.00-1.11x measured (slightly better at long N); q8m
    1.05-1.30x predicted, 1.15-1.17x measured; mlxq8 slower than mlx1; BETWEEN.
  - **Not run, by the line:** the logit-level check (it was tied to PASS).
  - **What would move it:** per-channel K handling (KIVI-style channel scales folded into Q, or a SmoothQuant-style
    migration) for the outlier case, then real-text K / V, then EvalSuite. That only pays if ~1 ms per check at
    10-20K matters; not started without routing.
- **Decision (the orchestrator 21:4x): KV8 stops at BETWEEN.** ~1 ms per check at 10K isn't worth the outlier-K risk.
  F1 (fs4 in place of REF2-T's chunk set, ~23.5 -> ~13 ms per check at 10K) is the long-context lever. NewIdeas
  queues its FAST-K certification with EvalSuite after Thursday's RELAX.

## F1-SWITCH fs4 in REF2's QSA verify calls, a default-off flag (Leap, pre-registered Tue 2026-11-03 22:3x EET; the orchestrator 22:3x, "tonight"; EvalSuite's PREREG_f1 63cd47e needs it; scripts/ref2.py, f1_switch_smoke.py; data/f1_switch_smoke.txt)
- **What:** `ref2.CFG["fatt_verify"]`, default False, also settable by `REF2_FATT_VERIFY=1`.
  - It is read PER CALL in ref2's `_verify_sdpa`, the hook on MTPLX's `Q._verify_sdpa`, so a variant's ref2_cfg can
    set it (EvalSuite's ask 1).
  - When on, a bool-mask verify call with 16 or 8 rows runs fs4 (fatt16s, KS 4) and skips REF2-T's key buckets and
    class re-runs: fs4 is N-invariant by construction (B6 INV (c)).
  - Everything else (prefill chunks, other masks, other row counts, REF2 off) takes today's path unchanged.
  - REF2's `_att` pads every attention call under 16 rows to 16, so 8-row checks and the AR's 1-row steps reach it as
    16-row calls and take fs4 too. That is what makes SD == AR hold under F1.
  - Per-process counts (EvalSuite's ask 2, its gate 2): COUNTS["fatt_verify"] (through fs4), COUNTS["verify_old"]
    (every other verify call) and COUNTS["verify_old_eligible"] (16- / 8-row bool-mask calls that took the old path:
    must be 0 with the flag on).
  - fs4 is imported lazily on the first switched call. The lazy modules for snapshots are scripts/fatt16s.py and
    scripts/fatt16.py.
  - Atomic write; ref2.self_test after the edit (ref2 is live-imported).
- **Smoke (f1_switch_smoke.py; layer 3's real weights; B6's layer harness, 1,024-row prefill chunks of random hidden
  states; T0 2K and 10K):**
  - (1) self_test PASS, and fatt16s not imported after apply() with the flag off.
  - (2) Flag off == the pre-switch ref2 (data/ref2_before_f1.py), BITWISE: y of a 16-row and an 8-row verify call,
    in two subprocesses, compared bit for bit.
  - (3) The switch takes effect. Flag on: every verify call of the checks and AR steps goes through fs4 and
    verify_old_eligible stays 0. Flag off: none does. y(on) vs y(off) mean |diff| / RMS ~2e-3 (B6: 2.2-2.4e-3).
  - (4) **SD == AR at the layer level, bitwise.** A check of 16 (and of 8) real rows at positions T0..T0+R-1 against
    REF2's AR: each row run alone (padded to 16 by `_att`) on the cache grown one token at a time. Row r's output must
    equal the check's row r bit for bit. Run with the flag on (F1) and off (REF2-T's own property, the control).
  - (5) Speed sanity: ms per 16-row layer call at 10K, flag on vs off (B6: fs4 1.141, ref2t 1.936).
- **Line:**
  - **PASS** = (1)-(4) all hold, and (5) is within +-15 % of B6's fs4 number.
  - Anything else is FAIL: fixed before the switch goes to EvalSuite. If (4)'s control fails, the test is wrong,
    not the switch.
  - The model-level token gates (flag off == REF2-T tokens; live == AR) are EvalSuite's step 0 on its harness.
- **Prediction:** PASS (85 %). The likeliest snag is (4): any other component of the layer that is not
  row-invariant would show in the control as well.
- **RESULT (Tue 22:27, share lock beside Breakthrough's run at ~50 % GPU; data/f1_switch_smoke.txt): (1)-(4) PASS.
  (5) failed under the shared GPU and is being re-run under a quiet lock (scripts/run_f1_speed.sh -> data/f1_runner.log).**
  - (1) self_test PASS. With the flag off, apply() leaves fatt16s unloaded.
  - (2) Flag off == the pre-switch ref2, BITWISE, for 16- and 8-row verify calls at 2K and 10K (each version in its
    own subprocess).
  - (3) The counts are exact.
    - Flag on: the check and its R AR steps are R + 1 calls through fs4, verify_old_eligible 0.
    - Flag off: 0 through fs4.
    - y(on) vs y(off): 1.9-2.3e-3 (B6: 2.2-2.4e-3).
  - (4) **SD == AR at the layer level, BITWISE, with the flag on:** 16/16 and 8/8 at 2K and 10K. REF2-T's own
    control (flag off) holds too, 16/16 and 8/8.
  - (5) Shared: flag off 2.584 ms (B6 ref2t 1.936), on 1.711 (B6 fs4 1.141), 1.51x. Both arms are 33-50 % over
    B6's quiet numbers, i.e. the other run's load. The quiet A-B-A-B re-run decides (5).
  - Handed to EvalSuite: commit, flag, counts keys, lazy modules.

## S7B-PROBE can the expert kernel start on each expert as it lands? (Leap, pre-registered Tue 2026-11-03 22:3x EET; Wednesday's kernel package, the orchestrator 22:3x "prepare your kernel tests"; WALL1_D1_SPEC S7b "per-expert release, GPU-side with Leap"; native/s7b_probe.swift)
- **Why:**
  - Today the GPU waits for a whole burst's release, then runs phase 2. The last record's copy (~53 us) and much of a
    burst's SSD bytes sit on the GPU's clock.
  - With W1-PICKUP's ~3 us coherent pickup, a per-expert release becomes affordable: a phase-2 threadgroup would start
    on expert e as soon as e lands.
  - Two facts decide whether that is safe and fast, and the first also bears on today's engine's correctness.
- **A, bandwidth:** a 256 MB shared buffer streamed by many threadgroups (uint4, summed so the loads stay live) through
  (i) plain device loads, (ii) `volatile coherent(system)` loads, (iii) `coherent(device)` loads (if it builds).
  GB/s from the GPU timestamps, median of 10.
- **B, stale data:** a 2 MB "slot" the GPU has already read (pattern P0) is rewritten by the CPU (P1), then released
  through a coherent flag. The GPU re-reads it in four ways, counting the words still P0:
  - B1: plain loads in the SAME kernel, after its coherent wait;
  - B2: plain loads in the NEXT dispatch of the same encoder (today's engine: spin kernel -> phase 2);
  - B3: as B2, with a buffer memory barrier between;
  - B4: as B1, with coherent(system) loads.
  - 20 trials each.
- **Lines:**
  - A: coherent streaming >= 0.8x plain = the per-expert path can read weights coherently; < 0.5x = it cannot.
  - B: any stale word in B2 or B3 is a correctness finding for TODAY's engine; it goes to BigMoE at once with the
    case.
  - B1 stale + B4 clean + A >= 0.8x -> C uses coherent weight loads. B1 clean -> plain loads after a coherent wait are
    enough.
- **C (Wed morning, after A / B):** 8 experts x 2.1 MB land 50-100 us apart (a CPU memcpy plus a per-expert release).
  A phase-2 stand-in (each expert's threadgroups read the record and do NAX6-like work, ~10 us) waits per expert, vs
  waiting for all.
  - Line: >= 30 % of the burst's land-to-done span hidden, with results bitwise equal to wait-all.
- **Prediction [G]:**
  - A: coherent ~0.6-0.9x plain (uncached, but streaming hides latency).
  - B1 stale (the same kernel keeps its L1 / L2 lines). B2 / B3 clean, because dispatch boundaries invalidate. If B2
    is stale, today's engine relies on luck.
  - C: ~30-50 % of the span hidden.
- **(5) quiet re-run (Tue 22:31:51, quiet lock, after a heat pause cleared; data/f1_runner.log): PASS.**
  - Flag off 2.047 ms (B6 ref2t 1.936); flag on 1.179 ms (B6 fs4 1.141, +3.3 %); 1.74x. A-B-A-B; the passes agree
    within 2 %.
  - **F1-SWITCH: PASS, (1)-(5).** The switch is ready for EvalSuite's step 0 (commit 0002e2e; the runner fixes come
    with this entry).
- **RESULT A + B (Tue 22:32:53, share lock beside Breakthrough's run at ~50 % GPU; data/s7b_probe.txt):**
  - **A, PASS:** 256 MB streamed at 219.7 GB/s with plain loads, 239.1 with coherent(system) (1.09x), 212.7 with
    coherent(device). A first run (22:32:37) gave 209.5 / 194.3 / 163.9 (0.93x). Coherent(system) loads stream at
    plain speed, so the per-expert path can read weights coherently.
  - **B, a correctness finding [K]:**

    | case (2 MB slot, 20 trials) | stale words per trial | trials with any |
    |---|---|---|
    | B0 control: first read after the rewrite | 0 | 0 / 20 |
    | B1 plain loads, same kernel | 524,288 (all) | 20 / 20 |
    | **B2 plain loads, NEXT dispatch (spin -> phase 2)** | **505,808-524,288** | **20 / 20** |
    | B3 = B2 + memoryBarrier(.buffers) | 0-524,288 | 16 / 20 |
    | B4 coherent(system) loads, same kernel | 0 | 0 / 20 |

    - Inside one command buffer, a CPU rewrite of memory the GPU read earlier IS NOT SEEN by later dispatches while
      the GPU still caches those lines. A buffer memory barrier does not fix it; coherent(system) loads do, at no
      bandwidth cost (A).
    - Today's engine: the loader rewrites slots while a check's command buffer runs, and phase 2 reads them with plain
      loads. At 45 GB steady state a slot is reused about a check later, after GBs of traffic have evicted its lines,
      which fits the bitwise runs.
    - **The hazard:** any path that rewrites a slot the GPU read recently in the same command buffer (two passes per
      check over the same pools, tier / ghost copies, small pools with fast reuse, S7b-like reuse) would read the OLD
      weights silently. Slot data carries no stamps; the routing rows and tables do.
    - Sent to BigMoE and the orchestrator at once (the pre-registered rule).
  - **The prediction held:** B1 stale, B4 clean. **It did not hold:** B2 / B3 were predicted clean.
  - **For S7b:** C reads the weights with coherent(system) loads (B4 clean, A at plain speed). C runs Wednesday
    morning.
- **B5 and a re-run with per-trial patterns (Tue 22:35:54, share lock; data/s7b_probe2.txt). Patterns now differ per
  trial, so an earlier trial's lines cannot pass as fresh.**
  - A: plain 281.6, coherent(system) 284.5 (1.01x), coherent(device) 276.8 GB/s; the GPU was freer this time.
  - B0 0 / 20; B1 stale 19 / 20; **B2 stale 20 / 20**; **B3 (barrier) stale 20 / 20 (all words)**; B4 0 / 20.
  - **B5, the slot last read by an EARLIER (completed) command buffer and rewritten while the NEXT one runs: 0 stale,
    0 / 20.** A command-buffer boundary between the GPU's last read and the rewrite is enough.
  - So BigMoE's reading of today's engine holds: a victim's last read sits in an earlier command buffer.
  - The hazard is exactly a read, then a CPU rewrite, then a re-read inside ONE command buffer. The fix for such paths
    is coherent(system) slot reads (free: 1.01x) or a command-buffer boundary.
  - BigMoE runs (c) tonight at C 160 (every token vs REF2-T). (a), default-off coherent slot reads for 3c-N / NAX6-N
    (and 3c if MPP allows) with a bandwidth check on the real kernels, is Leap's for Wednesday.
- **B6 (Tue 22:37:27, share lock beside Breakthrough's run; data/s7b_probe3.txt): the B2 pattern with the re-read
  through a PLAIN kernel parameter cast to `volatile coherent(system)` inside the body (all an MLX metal_kernel can do;
  the cast also compiles in MLX): 0 stale, 0 / 20.**
  - So BigMoE's ask (a) can be built in MLX kernels by casting the pool pointers.
  - Caution: B6's trial took 379 us against ~165-185 for the other cases. Its scalar strided coherent loads through the
    cast look ~2x slower on the 2 MB read. Probe A's coherent loads were uint4 and ran at plain speed.
  - The real expert kernels' speed with cast-coherent pool reads decides the cost (Wednesday, (a)).
  - The same run repeated B2 (stale 17 / 20) and B3 (stale 19 / 20); B0, B4 and B5 stayed 0 / 20.

## COH coherent(system) slot reads in the expert kernels, default off (Leap, pre-registered Tue 2026-11-03 22:4x EET; BigMoE's ask (a) after S7B-PROBE, 22:3x; scripts/tierglu_coh.py, coh_check.py; data/coh_check.txt)
- **Why:**
  - S7B-PROBE B: inside ONE command buffer, a CPU rewrite of a slot the GPU read earlier is not seen by later
    dispatches. Coherent(system) reads fix it (B4, and B6 through a cast).
  - Today's engine is safe by command-buffer boundaries (B5). Per-expert release (S7b), lever 2's in-flight forecast
    reads and two-pass checks at 12 GB break that argument, so BigMoE wants a default-off safety net.
- **What (tierglu_coh.py, built from the LIVE sources by string substitution, so no live module changes tonight):**
  - NAX6-N's kernels A / B and 3c-N's kernels with the pool pointers (codes, LSB words, scales, biases; never the
    activations or slot ids) cast to coherent(system).
  - Two forms: **cn** (`coherent(system) const device`, non-volatile, so the compiler may still vectorize the 6-word
    code loads) and **cv** (`volatile coherent(system)`, B6's proven form). Both use the same arithmetic, so they should
    be bitwise == the plain kernels.
- **Test (coh_check.py, real layer-20 pools from data/native_sample, P11's full-layer routing):**
  - (1) Bits: every output == the plain kernel, NAX6-N and 3c-N, S 16.
  - (2) Speed: ms per call (8 chained, median of 20), plain vs cn vs cv, S 16.
  - (3) Freshness in the kernel itself: one slot's records rewritten by the CPU between two calls in ONE command
    buffer. Plain must show the hazard (stale output), cn / cv the fresh output. Only if (1) and (2) allow.
- **Lines:** PASS = (1) bitwise AND the better form costs <= 3 % per call on both kernels. KILL = not bitwise, or
  > 15 % on either. BETWEEN otherwise (then the safety net costs that much, and BigMoE weighs it per path).
- **Prediction [G]:** cn ~0-4 % (probe A: coherent streaming at plain speed); cv 10-30 % (B6: scalar volatile cast
  loads ~2x on the probe). PASS 55 % (via cn), BETWEEN 35 %, KILL 10 %.
- **RESULT (Tue 22:44, share lock beside Breakthrough's run, rough speed; data/coh_check.txt, data/s7b_probe4.txt):
  KILL by the line. Per kernel: NAX6-N's safety net is free, 3c-N's costs +26 %.**
  - Building it took two fixes. My SYS block must come AFTER NAX6's MPP header; placed before it, `coherent(system)`
    is an "invalid argument". 3c-N needed a coherent overload of the 3c header's tg_codes3, which reads the codes
    through `row` and `src`.
  - (1) Bits: every window, both kernels, both forms bitwise == plain.
  - (2) ms per call (layer 20, S 16, 4 code windows):

    | kernel | plain | cn | cv |
    |---|---|---|---|
    | NAX6-N | 0.666 | 0.671 (+0.7 %) | 0.698 (+4.8 %) |
    | 3c-N | 1.410 | 1.781 (+26.3 %) | 1.808 (+28.2 %) |

  - Freshness of the non-volatile form [K]: S7B-PROBE B7 (the B2 pattern through a non-volatile coherent(system)
    cast) read 0 stale in 20 / 20. So **cn is a real safety net**; cv is not needed. That run repeated B2 / B3 stale
    20 / 20 and B4-B6 at 0, and B6 ran at 167 us, so the earlier 379 us was noise.
  - Why 3c-N pays: its codes come in as 3-6 single-byte loads per lane (tg_codes3), and coherent byte loads are
    slow; NAX6-N loads 6 words. A tg_codes3 that loads covering 32-bit words and extracts the bytes could close most
    of that [G], not built.
  - For BigMoE: on the FAST NAX6-N paths, cn is the safety net (+0.7 %). On the exact 3c-N path, a command-buffer
    boundary is the cheaper fix (B5) unless the word-load rewrite pays off. Integration (a default-off ref2 flag that
    swaps the kernels) waits for BigMoE's pick of paths.
  - **The prediction held in part:** cn ~0-4 % on NAX6-N held; cv 10-30 % was wrong for NAX6-N (+4.8 %); 3c-N's
    +26 % was not foreseen.

## P11PAD the "native+p11+pad" records format: zero-copy-aligned pieces (Leap, Tue 2026-11-03 22:4x-22:5x EET; BigMoE's proposal 22:4x after MoonShots' M-29; an engineering change gated by bitwise / md5 checks, no prediction; scripts/p11.py, tierglu_3cn.py, tierglu_nax6n.py, convert_records_p11.py, p11pad_check.py; data/p11pad_check.txt)
- **Why:**
  - M-29 (MoonShots, f1a98b3): reading records straight into the slots saves ~173 us per 2-record call, but only if
    every piece is 4 KB-aligned in the file and in the pool (today's unaligned direct path is +843 us).
  - The converted layout can carry that alignment for free.
- **The format:**
  - MSB s3 / b3 pieces zero-padded from 8804 to 9216 words (36,864 B = 9 x 4 KB), so msb_sz stays 2,064,384.
  - LSB part ordered l1_g, l1_u, l1_d, then b_g, b_u, b_d (b unpadded), so lsb_sz stays 720,896 and the l1 planes
    sit at 4 KB multiples.
  - The index keeps the logical piece order with these physical offsets; the base word stays at 11 * NB11 = 8800.
- **What changed:**
  - p11.layout_p11(r, pad) / to_p11(raw, r, pad); pieces() takes the new layout name.
  - 3c-N and NAX6-N read the pieces' stride from the pools' shapes: PWM_C (MSB s3 / b3 / b2) and PWL_C (LSB b),
    template constants, in place of PW11's constant. The asserts relax to >= piece_words and a multiple of 4.
  - convert_records_p11.py --pad. Forward runs from row-major only; --inverse takes either p11 layout.
  - Unpadded pools get today's constants, so the old instantiations are unchanged.
- **Checks:**
  - **(A) Layout, layers 20 and 44 (P11PAD check, candidates loaded by path before the live modules):** sizes kept;
    every MSB piece and l1 plane 4 KB-aligned; padded records -> row-major bitwise; each packed piece == the
    unpadded one + zero padding.
  - **(B) Kernels, 128 experts x 2 layers, every expert at 4- and 3-bit ranks:** tier_glu_n (+ scores), phases,
    pairs, split and NAX6-N on PADDED p11 pools == UNPADDED == bf16, bitwise (4 / 4 calls each).
  - **(C)** ref2.self_test PASS with the candidates, and again after the atomic install (backups
    data/{p11,tierglu_3cn,tierglu_nax6n}_before_pad.py).
  - **Tool, on scratch copies of L00 / L20 / L47:**
    - --pad forward: layouts correct and aligned, the row-major entry kept, the md5 list == the P11 test's
      committed one.
    - --inverse: 3 / 3 md5s match the originals.
    - The plain (unpadded) forward / inverse also still 3 / 3.
    - The scratch copies were deleted by exact path.
  - **A bug the tool's own verify caught, fixed:** the inverse's back-transform check rebuilt the unpadded layout,
    so it refused, and the layer stayed untouched.
- **Not done here:** a cmp against BigMoE's independent reference (r3p11 regenerated in the padded layout); BigMoE's
  call.

## F1-R8 fs4's 8-row form for checks and AR steps with <= 8 real rows (Leap, pre-registered Tue 2026-11-03 22:5x EET; extends F1-SWITCH; FAST-K, bitwise == F1 by INV (e); scripts/ref2.py, f1r8_check.py; data/f1r8_check.txt)
- **Why:**
  - REF2's `_att` pads every call under 16 rows to 16, so chat's 8-row checks (BigMoE's t 0.3 chat setting) and every
    AR step reach fs4 as 16-row calls, and fs4 computes 16 rows whose padding outputs are thrown away.
  - fs4's R 8 form stacks two heads of a kv head in one tile and computes only 8 rows. B6: 0.311 vs 0.517 ms per call
    at 10K (1.66x).
  - B6's INV (e) PASS says R 8's rows == the first 8 rows of the R 16 call, bit for bit.
  - B6's "no gain at the layer level" for 8-row calls ran THROUGH the padding, i.e. R 16.
- **What:**
  - `_att` records the call's real row count, and CFG["fatt_r8"] (default False, read per call; only with
    fatt_verify) routes a padded verify call with <= 8 real rows to fatt16s at R 8 on q's first 8 rows.
  - The output's padding rows (discarded by `_att`) are zeros. COUNTS["fatt_r8"].
- **Test (f1r8_check.py, B6 / F1's layer harness, layer 3, T0 2K and 10K):**
  - (1) Bits: 8-row checks and 1-row AR steps give real-row outputs with fatt_r8 on == off, bitwise; SD (16-row
    check) == AR with fatt_r8 on, bitwise.
  - (2) Counts: every <= 8-row call takes R 8.
  - (3) Speed: ms per 8-row layer call at 10K and 2K, on vs off (B6's method, a quiet lock).
- **Lines:** PASS = (1) and (2) hold and the 8-row layer call is >= 1.15x faster at 10K. KILL = any bit differs or
  < 1.05x. BETWEEN otherwise.
- **Prediction:** bits PASS (INV (e)). Speed at 10K: the SDPA's ~0.2 ms saved of a ~1.1 ms layer call is ~1.2x; at
  2K, ~1.05x. PASS 60 %.
- **What it would be worth [P]:** -2.4 ms per 8-row chat check at 10K on top of F1 (12 layers x 0.2), and EvalSuite's
  AR harness (every step padded) runs faster.
- **Cross-check (Tue 22:5x): my tool's --pad output == BigMoE's independent reference (~/BigMoE/scratch/r3p11pad),
  byte for byte:** cmp identical, md5s equal for L00 / L20 / L47 (98fcba62..., 1f375d00..., f770f6d1...), and the
  index entries match on layout, sizes and piece offsets. BigMoE's reader (test_layout.py) PASSes on them too. My
  scratch copies were deleted by exact path.
- **(1) + (2) (Tue 22:57, share lock beside Breakthrough's run; data/f1r8_check.txt): PASS.**
  - At T0 2K and 10K, with fatt_verify on, fatt_r8 on == off bitwise for an 8-row check and for 16 one-row AR steps
    (16 / 16).
  - SD (a 16-row check, fs4 R 16) == the AR steps with fatt_r8 on: 16 / 16.
  - Counts: on {r8 17, fs4 18} (the 8-row check + 16 AR steps took R 8, the 16-row check did not); off {r8 0, fs4 18}.
  - ref2.self_test PASS after the edit (atomic; backup data/ref2_before_r8.py; both flags default off).
  - (3) speed is queued under a quiet lock (scripts/run_f1r8_speed.sh -> data/f1r8_runner.log).
- **(3) speed (Tue 22:58:24, quiet lock; data/f1r8_runner.log): PASS, 1.22x at 10K.**
  - ms per 8-row layer call, A-B-A-B: 10K 1.402 -> 1.151 (1.22x); 2K 0.505 -> 0.480 (1.05x). No heat pause during
    the run.
  - **F1-R8: PASS.** -3.0 ms per 8-row chat check at 10K (12 layers x 0.25), on top of F1, bitwise == F1. The flag
    stays default off; it can be switched on wherever F1 is (EvalSuite's AR harness pads every step, so it gains too).
  - **The prediction held:** ~1.2x at 10K and ~1.05x at 2K, as predicted.

## DENSE-N (Leap, pre-registered Wed 2026-11-04 13:5x; BUILD_DAY_1104 #1, the dense pass; the orchestrator's optional quiet slot before 14:40)
- **Why:** #1 asks for the dense pass ~32 -> ~15-18 ms; FAST wants "a floor-level 16-row 4-bit dense GEMM". REF2's 4-bit
  dense set (N >= 1024) runs qmm16c at ks 1, nt 1 (ref2._qmm). L1-26 / ab28 (Oct 24) found the fragment-native layout
  (qmm26.qmm16n) bitwise == qmm16c and x1.15-1.22 faster at (2, 2) / (1, 1) on three shapes, but it was never wired in.
  The dense weights are resident, so the repack is a one-time cost at load (no SSD or store change; RAM +~1.5 GB while
  prefill keeps MLX's layout).
- **Prior art:** L1-26 / ab28 (ours); Marlin / FLUTE-style offline interleaved int4 layouts.
- **Test** (scripts/dn_check.py, run_dn_check.sh; quiet slot, MEM <= 2.5 GB, ~2 min): random 4-bit codes at the
  125B's REF2 dense shapes and per-check counts (qkv 36, z 36, out_proj + o_proj 48, q_proj 12, shared down 48; lm_head
  and R-83's qkv | z concat beside the set), one distinct matrix per call:
  - (1) bits: qmm16n at (1, 1) / (1, 2) / (1, 4) == qmm16c (1, 1); qmm16c (1, 2) / (1, 4) == (1, 1) (nt only groups
    tiles, so exact-safe); the FAST arm nF22 == qmm16c (2, 2) (its own numerics);
  - (2) speed: ab28's interleaved A/B (shuffled rounds x median of 7, paired ratios vs c11), ms per check.
- **Lines:** PASS = (1) AND the best exact native arm (one nt for the set) saves >= 1.0 ms per check vs c11; KILL = not
  bitwise, or < 0.4 ms; else BETWEEN. FAST reported against the floor (280 GB/s), no line.
- **Prediction:** bitwise everywhere; c11 at ~60-70 % of the floor; native (1, 2) ~80 %; saving 1.2-2.0 ms per check
  (the set's floor ~5.5 ms). 65 % PASS, 25 % BETWEEN, 10 % KILL.
- **Result (Wed 14:04:00-14:04:17, quiet lock, 17 s; data/dn_runner.log): PASS.**
  - **Bits:** on every shape, qmm16n at (1, 1) / (1, 2) / (1, 4) == qmm16c (1, 1), and qmm16c (1, 2) / (1, 4) ==
    (1, 1). nF22 == qmm16c (2, 2), and (2, 2) != (1, 1), as expected.
  - ms per 16-row check, medians of 10 shuffled rounds (paired ratio vs c11):

    | shape (calls) | c11 (REF2 today) | n12 (native, exact) | FAST nF22 | floor |
    |---|---|---|---|---|
    | in_proj_qkv (36) | 2.727 (70 %) | 2.275 (83 %), x1.20 | 2.311 | 1.896 |
    | in_proj_z (36) | 1.804 (63 %) | 1.508 (75 %), x1.19 | 1.507 | 1.138 |
    | out_proj + o_proj (48) | 2.505 (61 %) | 2.015 (75 %), x1.24 | 1.968 | 1.517 |
    | q_proj (12) | 1.252 (61 %) | 1.046 (72 %), x1.19 | 1.019 | 0.758 |
    | shared down (48) | 0.567 (28 %) | 0.550 (29 %), x1.03 | 0.556 | 0.158 |
    | **the set** | **8.85** | **7.39: -1.46 ms, x1.20, 74 % of the floor** | 7.36 | 5.47 |
    | beside: R-83 qkv \| z concat (36) | 4.980 | n11 3.698, x1.33 | 3.679 | 3.033 |
    | beside: lm_head (1) | 1.868 | n14 1.579, x1.18 | 1.626 | 1.277 |

  - qmm16c's nt alone is worth nothing (c14 -0.04 ms). The layout is the gain. nt 1 / 2 / 4 are within 0.07 ms.
  - FAST: ks 2 adds nothing over the exact native arm (7.36 vs 7.39). At 74 % of the floor, a "floor-level" FAST GEMM
    has ~1.9 ms left on this set. Shared down (28 % of its floor: 48 latency-bound calls at K 640) and the small MLX
    shapes (kern23: ~3.3 ms against a 0.4 ms floor) are the inefficient part, not the big GEMMs.
  - **The prediction held:** bitwise; c11 61-70 % of the floor; native 72-83 %; saving 1.46 ms in the 1.2-2.0 band.
  - Worth: -1.46 ms per check (-2.7 ms with R-83's concat on, -0.3 more if the head takes qmm16c), exact. RAM +~1.5 GB
    for the native copies, because prefill keeps MLX's layout.
  - Next: the wiring is staged (scripts/ref2_dn_patch.py, CFG dense_native, off). Apply it, run ref2.self_test and
    dn_wire_check.py, then hand BigMoE the env var (REF2_DENSE_NATIVE=1).

## R-78b: the drafter's R-78 forms at block rows other than 16 (Leap, Wed 2026-11-04 14:1x; the orchestrator's route after MTP Head's DRAFT17)
- **Why:** draft_fused's forms ran only at exactly 16 rows. At 17 / 32 rows, form C fell back to a per-call weight copy:
  +8.6 / +9.0 ms per drafter pass (TrainLab REPORT 'DRAFT17'). DRAFT17 (NewIdeas: code's bonus row, x0.67-0.71 code
  bytes at 24 GB [P]) needs 17-row passes.
- **Line:** open only the row counts where both forms are bitwise == nn.Linear's x @ W.T, so the drafts stay bitwise.
- **Result (scripts/r78b_rows.py + r78b_installed.py; data/r78b_rows.txt; share lock, ~1 min):**
  - Raw forms on the nine projection shapes (q, MLP gate / up / down = C; k, v, o, both conv kernel projections =
    B), 4 inputs per (shape, rows): **both forms are bitwise at every row count 16-64 and at 80 / 96 / 128 / 200 /
    256 / 512 / 1000. They differ at 2-15** (MLX's small-M path; form B is also bitwise at 1). That matches R-78's 8 and
    15 DIFFERS.
  - Installed drafter (all 45 patched projections of the 5 layers, through their classes): rows 8 (fallback), 16,
    17, 32 (forms) all bitwise == x @ W.T, 180 / 180 calls.
- **Change (draft_fused.py):** FORM_ROWS default = 16..64 (was {16}); DRAFT_FORM_ROWS adds rows (never below 16). 16-row
  passes are unchanged. 17 / 32-row passes lose the copy fallback, bitwise, ~0 ms vs +8.6 / +9.0 (DRAFT17's R17g /
  R32g). BigMoE's default drafts 16-row blocks only (0 of 308 agent checks ran 32 rows), so its runs are unchanged.

## HC8-1P: the FAST hyper-connection read, 8-bit mixers in one pass over the streams (Leap, pre-registered Wed 2026-11-04 14:2x; BUILD_DAY_1104 #1 FAST; routed: after the _rows16 fix)
- **Why:** the HC read is the dense pass's biggest excess. It runs ~9.9 ms GPU per check (97 reads) against a 4.5 ms bf16
  byte floor. The exact road ends near ~6.5 (D1 + D5, HC_READ.md). FAST may change numerics; the gate is EvalSuite
  same-scores. HC8 (8-bit mixers) halves the bytes, and its quality is already measured: the map's hc8w arm flipped
  1.44 % of rows (0.00 % of sure rows); EvalSuite decision 4 PASS with a lean; adopted on the 12 GB line.
- **Design (scripts/hc8r.py), 3 dispatches per read, the streams read once:**
  - K1: per (32-column tile, stream-aligned K partition) simdgroup:
    - the pending write h' = h + bo x inj (the reference's bf16 ops; tile 0 stores it);
    - A = bf16(h' * w_norm) (the norm weight before the GEMM, 1 / rms after it: D2's fold without a weight change);
    - a code MMA on the 8-bit codes, y = s * P + b * X per 64-group;
    - fp32 partials, plus per-partition sums of h'^2.
  - K2: rms per (row, stream), c = sum_p partial_p / rms (fixed order), mix1 = silu(c / 4), inject = 2 sigmoid(c / 4).
  - K3: up = mix1 @ W_up8^T (code MMA, K 320) with the tail fused in: mixed = mean_s sigmoid(up_s) * h'_s / rms_s * w_s,
    fp32, one bf16 rounding.
  - Weights: the HC8 recipe (mx.quantize 8-bit g64 of input_mix_weight_down / up); block_inject stays bf16.
- **Prior art:** HC8 (Breakthrough round 28; the map's hc8w); D2's row-invariant one-pass read (HC_READ.md); RMSNorm
  weight folding into the next linear (QuaRot / SpinQuant); split-K GEMM with a fixed-order reduction. New here: the
  8-bit HC read with the norm moved after the GEMM, at 16 rows on the M5's matrix units.
- **Test** (scripts/hc8r_screen.py; real HC weights of all 96 mixers; share lock first for correctness, then a quiet
  timing):
  - (1) correctness: vs an fp32 MLX emulation of the same FAST math with the same 8-bit weights, every output within
    bf16 grain (rel rms <= 4e-3); written == the exact read's written, bitwise (the same pending-write ops).
  - (2) deviation vs the exact read (teacher-forced, per read): rel rms of mixed / inject, reported next to HC8-eager's
    (MLX's own 8-bit path, the map's numerics).
  - (3) ms per 96-read chain vs today's S4 + R-82 and kernel B, arms interleaved, median of 40.
- **Lines** (on the chain, 96 reads): PASS = (1) AND <= 5.0 ms (from ~9.5-9.8); KILL = (1) fails after one fix, or
  > 7.0 ms (no better than 8-bit bytes alone); BETWEEN otherwise.
- **Prediction:** ~3.5-4.5 ms per chain (K1 ~13-15 us, K2 ~3, K3 ~14-16, 2 boundaries). (2): mixed rel rms ~1.2-1.5x
  HC8-eager's. PASS 55 %, BETWEEN 35 %, KILL 10 %.
- **Result (Wed 14:19-14:20; data/hc8r_screen.txt, hc8r_runner.log): BETWEEN on the pre-registered line. Correct and
  as accurate as HC8 itself; 6.13 ms per chain.**
  - (1) Correctness (share lock), first build:
    - read() vs emulate(): mixed rel rms max 1.48e-3 (median 1.07e-3); inject max 2.3e-4.
    - written == emulate's and == the exact read's, bitwise on all 96 reads.
  - (2) Deviation vs the exact read, teacher-forced, median [max] of 96:
    - mixed: HC8-1P 6.69e-3 [1.47e-2] vs HC8-eager (MLX's own 8-bit path, the map's numerics) 6.70e-3 [1.44e-2]. The
      one-pass order adds nothing beyond the 8-bit weights, so the hc8w quality evidence carries over.
    - inject: 3.5e-3 (~1 bf16 ulp; HC8-eager 0).
  - (3) Quiet, ms per 96-read chain, median of 40:

    | arm | ms |
    |---|---|
    | S4g (S4 + R-82) | 9.44 |
    | B (kernel B) | 8.50 |
    | HC8-1P NP 8 | 8.37 |
    | **HC8-1P NP 16** | **6.13: -3.30 vs S4g, -2.37 vs B** |

  - **Verdict: BETWEEN** (PASS line 5.0, KILL line 7.0). The prediction (3.5-4.5) was too optimistic.
  - NP 8 -> 16 is worth 2.2 ms, so K1 is latency-bound: each 64-group iteration waits for its loads, and there is no
    prefetch. Next, post-hoc and labelled so: a per-kernel breakdown, then K1 / K3 with the next group's loads issued
    ahead (S4's GPF), vector A loads, and NP 20 / 40.
- **Post-hoc (Wed 14:21-14:27; labelled post-hoc, not the verdict):**
  - Per-kernel breakdown at NP 16 v1 (share lock): K1 38.5 us per read, K2 2.8, K3 22.4. K1 was the problem: all 11
    column tiles rebuilt the same A tile from scalar 2-byte loads. NP 20 / 40 helped little (34.5 / 33.5 us).
  - **v2** (scripts/hc8r.py `v=2`; numerics unchanged, written bitwise 96 / 96):
    - K1: one threadgroup per partition builds A once per 64-group (uint4 loads, threadgroup memory, double-buffered,
      one barrier); the 11 tile simdgroups' codes are prefetched one group ahead.
    - K3: mix1 and its group sums staged once per threadgroup; codes prefetched; the tail's loads issued before the GEMM.
    - **Quiet: NP 20 v2 5.22 ms per chain vs S4g 9.67 / B 8.73 in the same run (NP 16 v2 5.52, NP 40 v2 5.40, v1 NP 16
      6.38). That's -4.45 ms vs S4g, -3.5 vs kernel B, x0.54.**
  - v3 (scales / biases prefetched too): 5.40 vs S4g 9.58 (x0.56), no gain (register pressure). Reverted to v2.
  - Where it stands: ~54 us per read against a weight floor of ~26 us (6.9 MB) + 2 boundaries. Left: K2 folded into K1
    (last-threadgroup reduction), A loads one group ahead. Next is the ref2 FAST switch, so BigMoE can run an arm.
- **Wired (Wed 14:30, ref2 live, default OFF; hc8_wire_check.py PASS on the candidate and live; DENSE-N's check still PASS;
  self_test PASS):**
  - ref2 CFG["hc8_fast"] / REF2_HC8_FAST=1 (+ hc8_np / REF2_HC8_NP, default 20): a <= 16-row HC read of a GatedResidual
    with 8-bit mixers runs hc8r.read v2 at 16 padded rows.
  - Without the switch, 8-bit mixers take MTPLX's eager chain; bf16 mixers stay on the exact S4 path, bitwise.
  - The FAST arm's model change: hc8r.hc8_install(model) (the HC8 recipe; ple=False by default). It frees the bf16
    mixers (keep=False), so it saves ~0.6 GB of RAM at 45 GB as well as ~4.3 ms per check.
  - Checks (real weights, layers 0-3):
    - on == hc8r.read bitwise; 1- and 8-row calls == the 16-row call's rows (SD == AR);
    - off -> eager, written equal, FAST vs eager rel rms <= 5e-3;
    - bf16 mixers on == off bitwise.

## DENSE-N-RAM: DENSE-N with one copy of the weights (Leap, pre-registered Wed 2026-11-04 14:4x; BigMoE 14:3x: "your RAM-neutral DENSE-N form would be welcome" before its next 45 GB slot)
- **Why:** at 45 GB RAM binds. DENSE-N kept MLX's layout for prefill (+1.5 GB = ~11 cache slots), so BigMoE runs it at C 213
  vs C 224. BigMoE's default has K83_EXACT (R-83's qkv | z concat), so DENSE-N is worth ~-2.0 ms per check there.
- **Design (ref2):** dense_native_convert(model, head=False) replaces each eligible 4-bit dense weight in place by its
  native copy, 4-D on purpose so that any direct reader fails loudly (one copy; the originals are freed).
  - _qmm: a converted weight takes qmm16n at <= 16 rows (bitwise == qmm16c). Otherwise (prefill, or ref2 off) it is
    rebuilt in MLX's layout per call (_dn_unpack: the exact inverse) for MLX's own kernel: stock bits.
  - R-83's _InProjExact unpacks converted modules at construction and converts its concat; its prefill path slices the
    unpacked form.
- **Test** (scripts/dn_ram_check.py, share lock):
  - (a) unpack(repack(w)) == w bitwise on the dense shapes;
  - (b) converted vs unconverted modules: outputs bitwise at rows 1 / 8 / 16 (ref2 on) and 17 / 32 / 256 (prefill),
    and with ref2 off;
  - (c) R-83 built from converted modules == unconverted, on- and off-path;
  - (d) active memory after convert vs before (one copy);
  - (e) ref2.self_test.
- **Lines:** PASS = all bitwise AND memory within +-2 % of before; KILL = any bit difference that survives one fix.
- **Prediction:** PASS. The risk is code outside ref2 reading a converted weight directly. It fails loudly; BigMoE's smoke
  run would show it.
- **Result (Wed 14:40, share lock; data/dn_ram_check.txt; on the candidate, then live): PASS.**
  - (a) unpack(repack(w)) == w bitwise on 7 shapes (incl. R-83's concat and the lm_head).
  - (d) 8 weights converted (98 MB; the stand-in's 5 modules + the GDN's qkv / z + R-83's concat). Active memory
    0.513 -> 0.513 GB (+0.00 %): one copy. The lm_head stays in MLX's layout. MLX's kernel on a converted weight
    raises.
  - (b) After convert == before, bitwise: ref2 on at rows 1 / 8 / 16 / 17 / 32 / 256, ref2 off at 16 / 64 (17 native
    calls, 25 unpacked).
  - (c) R-83 matches: converted by convert() when built before, converted at construction when built after, both ==
    unconverted, on- and off-path.
  - (e) dense_native_unconvert restores weights and outputs bitwise. ref2.self_test PASS. DENSE-N's and HC8-1P's
    wiring checks still PASS.
  - **Use:** after load, ref2.apply(); ref2.dense_native_convert(model) (head=False). That gives DENSE-N at no RAM cost
    (BigMoE's default with R-83: ~-2.0 ms per check). BigMoE's prefill evaluates per layer, so the unpacked transients
    stay at ~one layer.

## NATIVE-FRI: native expert records in the engine, the Friday arms (Leap, pre-registered Wed 2026-11-04 14:5x; BUILD_DAY #3; the orchestrator's option A: no loader patch, Friday's in-place conversion instead)
- **What changes Friday:** the 00:30 in-place conversion (native+p11+pad; reversible, ~2 min each way, no second copy).
  BigMoE's layout patch then reads converted records. ref2 routes 3c calls to 3c-N (bitwise == 3c, 4-7 % slower on the
  expert call, NATIVE-3C) and NAX calls to NAX6-N (bitwise == NAX6, +21 % at S 16) with no new engine code.
- **Track:** NAX's bits differ from 3c's, so NAX6-N is a FAST lever until Kamen certifies a NAX reference (the
  orchestrator asked him 14:5x: EvalSuite same scores on math / code / instructions, ~1 night, ~+12-14 % EXACT).
- **Arms** (BigMoE's harness, the 45 GB default plus the arm's switch; C 224 everywhere = equal RAM, since the padded
  records keep recsz; 14 prompts; A-B-B-A within each store phase):
  - Phase R (row-major store, before the conversion): D = today's default (3c, REF2-T); N = R57_NAX=1 (NAX6, C1).
  - Phase C (converted store): Dc = default (3c-N); Nc = R57_NAX=1 (NAX6-N, nax_native from the pools' dtype).
- **Exactness, token for token on every prompt:** Dc == D (== REF2-T's rows; by construction 3c-N == 3c);
  Nc == N (by construction NAX6-N == NAX6). A mismatch is a bug, not a verdict.
- **Lines (geo steady tok/s):**
  - #3's FAST value = Nc vs D: PASS >= +6 %, KILL < +2 %, BETWEEN in between.
  - Diagnostics, no lines: Nc vs N (the native gain on the NAX path; [P] ~+5-8 % from the expert term's share); Dc vs D
    (the conversion's net for EXACT: 3c-N's -1..-7 % on the expert call against p11's -10.9 % bytes and pad's zero-copy
    reads); N vs D (C1, if it never ran).
- **Adoption:** Nc via EvalSuite same-scores (FAST, the 11-04 rule), or EXACT if the NAX reference is certified. The
  conversion stays either way (Kamen-approved); Dc is the EXACT default after it.
- **Prediction:** Nc vs D +8-14 %; Nc vs N +5-8 %; Dc vs D -2..+3 % (kernel cost against byte gains). PASS 55 %.
- **Step 1 result (Wed 14:49, share lock beside Breakthrough's GPU run; correctness only; data/nat_engine_check.txt):
  PASS.**
  - BigMoE's live package (load_index / make_pools / make_sidecar / ext.ensure; the lever-1 ext with
    set_direct_split; layouts native / native+p11 / native+p11+pad) loaded the same 64 experts of layers 20 and 47
    from my scratch store (APFS clones converted by convert_records_p11.py --pad) and from today's records3. The slot
    assignment was identical.
  - On 9 routings per layer (S 16 / 8 / 1 x 3):
    - 3c-N(converted) == 3c(row-major), BigMoE's tier_glu, REF2-T's kernel: 9 / 9 for tier_glu, with scores, and
      phases + combine;
    - NAX6-N(converted) == NAX6(row-major): 9 / 9.
  - BigMoE's own scripts/test_layout_engine.py covers the bytes (every slot == its record's part; the prefill rebuild
    == records3's). So Friday's converted store runs today's exact default as 3c-N, bitwise, and the NAX arm as NAX6-N,
    bitwise.
  - BigMoE 14:5x: **C1 never ran** (the DAY9 night ticket was dropped at its 05:30 NOTAFTER; only smokes ran). So
    Friday's phase-R NAX arm (N) is the first real C1, and the NATIVE-FRI predictions rest on kernel numbers only.
- **AMENDMENT (Thu 2026-11-05 19:0x, before arming; the orchestrator 15:5x asked whether NATIVE-FRI is still needed):**
  - Why the design changes:
    - BigMoE converted records3 in place on Thu 10:28 (native+p11+pad), so the row-major phase (D, N) can no longer run.
    - Exactness is already covered: EvalSuite's NAX6 Part 2 PASS 14:47 (NAX6-N on the converted store: live loop == AR
      36 / 36, AR == Part 1's row-major NAX6 rows) and its 10:50 bridge (ref2tnaxn == ref2tnax on 3 rows; check 2 for
      the 3c-N path). That is Nc == N, and Dc == D through 3c-N == 3c.
> (a quote from the project owner, kept private)
      and BigMoE's Friday ticket times no NAX arm (its fexr is 3c-N, timing unused).
  - **Phase C only:** Dc / Nc / Nc / Dc on the converted store.
    - Dc = the 45 GB default (3c-N, OOO two phases); Nc = + R57_NAX=1 (NAX6-N, nax_native).
    - 14 ref14 prompts (dkv_thu.sh's keys), C 224 (else 208 / 192 by NO-SWAP), C45_ALARM 900.
    - BigMoE's c45 runner and a frozen copy of its screen with OUT -> ~/Leap/data/natfri/r57; scripts/natfri_c.sh.
    - It runs in a quiet turn after BigMoE's Friday ticket, ~30 min (4 arms x ~7 min, from BigMoE's dkvt_224: 411 s).
  - **Lines (amended):** #3's value = Nc vs Dc, pair-mean geo steady tok/s (tpsw) over the prompts both passes kept:
    PASS >= +6 %, KILL < +2 %, BETWEEN in between.
    - D became Dc because Dc is the default Nc would replace. The old diagnostic Dc vs D was -2..+3 %.
  - **Tokens:** every Dc prompt == REF2-T (ref14 gate+t, prefix), else a bug. Nc pass 1 == pass 2 (NAX determinism),
    else VOID.
  - **VOID:** fewer than 12 prompts in both passes of either arm.
  - **Prediction:** Nc vs Dc +8-14 % (the old Nc vs D prediction; 3c-N is 4-7 % slower than 3c on the expert call, against
    p11 / pad gains). PASS 55 %.
- **v4 (post-hoc, Wed 14:5x; bitwise == v2 by construction and on 96 / 96 reads: mixed, written, inject):**
  - v2 breakdown at NP 20 (share lock): K1 29.4 us per read, K2 2.6, K3 22.7.
  - Changes: K1's A builders load the next group's h / w_norm / bo one group ahead. K2 runs one threadgroup per row and
    also emits mix1's per-(row, 64-group) sums in v2's order (k = 0..63). K3 drops its serial sums and one barrier.
  - Quiet: 4.97 ms per 96-read chain vs v2 5.27 (same run), S4g 9.77, B 8.64. That's -4.80 vs S4g, -3.67 vs B, x0.51.
  - The screen script printed "PASS" against the <= 5.0 line, but v4 is post-hoc; the pre-registered verdict stays
    BETWEEN.
  - ref2 CFG hc8_v / REF2_HC8_V now defaults to 4. It is bitwise == v2, so EvalSuite's c805fd0 (v2) certification
    carries over. hc8_wire_check / dn_ram_check / dn_wire_check PASS after the change.
- **Prep found and fixed (Wed 14:5x): NAX6-NH had no p11 path.** Friday's FAST arm (R57_NAX=1 + R57_H1C=1 ->
  ref2 nax_native + nax_hybrid -> tier_glu_nax6nh) would have read the converted store's 11-bit scale pieces as bf16
  (N taken from s3's shape = 9216 words): silently wrong outputs.
  - Fix (tierglu_nax6h.py): _native_hybrid(p11=True) splices the hybrid's ALU path into NAX6-N's p11 core (P11R views
    return the same floats), and tier_glu_nax6nh takes the p11 kernels and strides for uint32 pools, as tier_glu_nax6n
    does.
  - The bf16 sources are byte-identical to before (checked), so live users are unchanged. Backup
    data/tierglu_nax6h_before_p11.py.
  - P11PAD check with NAX6-NH added (layers 20 / 44, 128 experts): NAX6-NH on padded p11 == unpadded p11 == bf16
    native pools, 4 / 4 / 4 per layer. All other kernels still PASS; ref2.self_test PASS.
- **AMENDMENT 2 (Thu 2026-11-05 23:4x, before it ran; the orchestrator 23:4x: NO-SWAP ~33-38 GB tonight).** Phase C
  gets BigMoE's lower fri_night rungs, so it runs instead of waiting out WAITMEM 44:
  - Ladder: 224:44.3 / 208:42.2 / 192:40.0 / 176:37.9 / 160:35.7 / 144:33.6 / 128:31.4.
    - NO-SWAP needed = lifetime peak + ~4 GB, at 0.134 GB per C slot.
    - Measured at C 160: exact 31.7 GB, qall 31.2 GB.
  - turn45: PEAK 28, WAITMEM 32 (was 40 / 44). The arms, the lines and the order are unchanged.
  - All four arms share one C, so the Nc / Dc pair stays like for like. Exact tokens don't depend on C, and the report
    labels C.
  - FLAT-2 gets the same ladder via scripts/flat2_ladder.sh, which passes FLAT2_C to zero's unchanged run_flat2.sh.
    - FLAT does not depend on C (zero's INSTALL).
    - The chosen C goes to data/flat2_ladder.log.
- **RUN 1 (Fri 2026-11-06 01:26-01:33): VOID, the swap guard.**
  - Granted 01:26:35 with NO-SWAP 44.8 GB, so the ladder took C 224 (needs 44.3): 0.5 GB to spare.
  - Swap grew 8.0 -> 10.9 GB in the first arm (nfdc_224). The runner's guard stopped it at 01:33:14 (rc 143) and ran no
    other arm. The report: 0 prompts in every arm -> VOID (data/natfri/natfri_report.txt, c45.log).
  - FLAT-2 then ran at C 224 with 48.2 GB (3.9 GB to spare) with no swap growth. So the ladder's ~4 GB headroom is not
    enough at the top rung; the re-run keeps >= 3 GB beyond the ladder's need.
  - No verdict yet; the re-run is placed by the orchestrator.
- **RUN 2 (Fri 2026-11-06 02:58-03:22, C 192, quiet; after MTP Head's L1-20, with the QOM-SPLIT-1C check first in the same turn): PASS.**
  - NO-SWAP 44.4 GB with margin 3 -> C 192 (needs 40.0). Every arm: lifetime peak 38.1 GB. Swap flat (10.73 -> 10.12 GB).
  - Arms: nfdc 272 s, nfnc 411 s, nfncb 263 s, nfdcb 452 s. Arms 2 and 4 followed a 10 s SSD cool-down (68-69 C > 66 C).
  - Tokens: both Dc passes == REF2-T on 14 / 14 prompts. Nc pass 1 == pass 2 on 14 / 14. Not a bug, not VOID: 13 prompts
    were kept in both passes, and VOID is < 12.
  - **Steady tok/s (tpsw, pair means, geo over 13 prompts): Dc 46.75, Nc 49.86 -> Nc / Dc = 1.0665 (+6.7 %). PASS
    (>= +6 %).**
    - It is under the predicted +8-14 % and 0.7 points over the line.
    - The order effect was Dc -2.1 % and Nc +0.2 % (pass 2 / pass 1); the A-B-B-A pair means cancel linear drift.
  - Per domain Nc / Dc: humaneval +3.6 % (4), math500 +6.9 % (3), gsm8k +7.1 % (1), ifeval +13.8 % (2), multi +8.0 % (2),
    agent +1.5 % (1).
> (a quote from the project owner, kept private)
    switch to make.
  - Data: data/natfri/natfri_report.txt, c45.log, natfri.log, nf*_192.txt, r57/; the frozen screen and runner are in snap/.

## OOO-1C: one-call OOO on the NAX path (Leap, pre-registered Wed 2026-11-04 15:2x; BUILD_DAY #5, the orchestrator 15:0x; BigMoE 15:1x: "build it as a patch in your copies")
- **Why:** OOO pays two expert calls per MoE layer: phase 1 on the resident pairs, the loader's spin, then phase 2 on the
  late ones, then combine_phase. Fable C priced the second call's fixed costs at -3.7 ms per check [P]. Kamen approved
  the NAX reference 15:0x (pending EvalSuite), so the exact default's OOO becomes ref2._ooo_nax6. It is built there first.
- **Design (scripts/ooo1c.py; no ext or bigmoe/tier.py change):**
  - post as today. One NAX6 call: its grouping (gpu_grp_1c) keys resident pairs as NAX6 does and late pairs by expert
    id, sorted after every resident key, so late groups come last in the grid.
  - Kernel A resolves a late group's slots inside the kernel: thread 0 polls the release region (BigMoE's coherent-spin
    layout: line 0's tag == seq, written last; the pair's word stamped with seq) with coherent(system) loads and
    system-scope fences. It writes the resolved slots for kernel B, which is NAX6's B unchanged. Then the same
    combine_phase.
  - The other dispatches go: phase 2's grouping + A + B, and the two spin kernels.
- **Exactness:** a pair's bits do not depend on its group (INV), so the result == today's two phases == NAX6 on the
  released slots, bitwise.
- **Test** (scripts/ooo1c_check.py; real records3 pools of L20 / L44; 16 rows, top-10; ~70 % base / ~80 % sidecar
  resident; LP1 -2 sidecars):
  - (1) region released before the call: one-call == NAX6 on the final slots == today's two phases, bitwise; resolved
    slots == the final slots; no timeout;
  - (2) a late release (CPU writes 20 ms after the launch, the loader's order): the same bits;
  - (3) ms per MoE layer call in 48-call chains, two phases (+ 2 stand-in spin dispatches) vs one call.
- **Lines:** (1) / (2) bitwise, else BLOCKED (one fix). Speed (3): PASS >= 0.06 ms per layer (>= ~2.9 ms per check at
  48 MoE layers); KILL < 0.02 ms per layer; BETWEEN otherwise. The engine number comes from BigMoE's slot later.
- **Prediction:** bitwise; -0.04..-0.08 ms per layer (a grouping kernel + NAX6's A / B fixed costs + 2 spins) ->
  -2..-3.8 ms per check. PASS 50 %, BETWEEN 40 %, KILL 10 %.
- **Result (Wed 15:08, share lock; data/ooo1c_check.txt): correctness PASS; speed BETWEEN (rough).**
  - (1) Released before the call: one-call == NAX6 on the final slots == today's two phases, bitwise. Resolved slots ==
    the final slots, including LP1 -2 sidecars. No timeout. 4 calls on L20 / L44.
  - (2) Late release: the CPU wrote the release 20 ms after the launch. The kernel waited in place (calls 22.4-22.9 ms)
    and gave the same bits, with no timeout. The in-kernel spin works with live CPU writes.
  - (3) 48-call chains, interleaved, median of 15: two phases (+ 2 stand-in spin dispatches) 1.699 ms per layer call,
    one call 1.642. That's -0.056 ms per layer = **-2.7 ms per check**, just under the 0.06 PASS line: BETWEEN, under a
    share lock. The engine number comes from BigMoE's slot (it includes the real spins and the loader's timing).
- **3c form + timing at a realistic late fraction (Wed 15:10-15:12, share lock; post-hoc conditions, labelled so):**
  - The 3c form (today's exact default: kernel A takes its pair from a resident-first order and resolves late slots in
    place; 3c's fused-combine kernel B reads them) == tier_glu(scores) == BigMoE's two phases + combine_phase, bitwise,
    including a live 20 ms late release.
  - At the pre-registered 70 / 80 % resident split the share-lock timing was noise-level: NAX -0.056 then +0.041 ms per
    layer in two runs; 3c +0.007. Each late pair's ~160 threadgroups pay system-scope fences, and 35 % of pairs were
    late.
  - At the engine's late fraction (96 / 97 % resident; BigMoE's ~3.6 misses per layer at C 256) and 40 interleaved
    rounds: NAX 2.244 -> 2.159 ms per layer (-0.085 = **-4.1 ms per check**); 3c 1.495 -> 1.429 (-0.066 = **-3.2 ms
    per check**). That's in line with Fable C's -3.7 [P].
- **Wired (15:14, ref2 live, default OFF): CFG["ooo_1c"] / REF2_OOO_1C=1.**
  - BigMoE's 3c path (ref2._ooo_sel, REF2 on, not native) and C1's NAX path (_ooo_nax6, not native / hybrid / repack)
    take the one call when the gate has release regions (spin mode); otherwise they fall back to the old route.
  - No ext or bigmoe/tier.py change.
  - ooo1c_wire_check PASS (candidate + live): 3c == tier_glu(scores); NAX == NAX6 + combine_phase; off and event mode
    take the old route. DENSE-N(-RAM) / HC8-1P checks still PASS; self_test PASS. Backup data/ref2_before_ooo1c.py.
- **U3 (t3) form (Wed 15:1x; BigMoE's question for tonight's all-in FAST arm).** ref2.apply() runs after r57_screen
  installs _ooo_call_t3, so _ooo_sel wraps it. Without this fix, ooo_1c would have ignored t["nolsb"] (the sure rows'
  sidecars at tier 4). Now:
  - the 3c one-call path mirrors _ooo_call_t3: the dup-id sidecar post, and sidecar slot -2 for the sure rows;
  - in the kernels, a sidecar slot -2 is final (no poll; LP1 -> tier 3), and only -1 waits;
  - the NAX path declines nolsb.
  - Checks:
    - ooo1c_check (1) / (2) still PASS for NAX and 3c;
    - ooo1c_wire_check (f): U3 sure rows through ref2 == tier_glu(scores) with sidecar -2 == _ooo_call_t3's two
      phases, bitwise, with BigMoE's LP1 patch on;
    - (a)-(e) PASS.
  - Backups data/ooo1c_before_t3.py, data/ref2_before_ooo1c_t3.py. Gap told to BigMoE, not changed: with R57_NAX=1,
    _ooo_nax6 ignores nolsb even with ooo_1c off.

## MTP-STEP-0: what's inside a ~2.2 ms MTP chain step (Leap, Wed 2026-11-04 15:2x; the orchestrator's route after MTP Head's RERANK-0: the MTP-assisted pick keeps x1.23 on code but its 15 sequential steps at ~2.2 ms net x0.92; it pays below ~1.4 ms per step at a 90 ms cycle, clearly at ~0.5)
- **Bytes estimate:** a step is the MTP layer (3 HC reads ~39 MB bf16, attention projections ~25 MB, MoE top-10 ~30 MB,
  fc_hidden / fc_embedding) plus the 4-bit lm_head, 248,320 x 2,560 = 358 MB (~75 % of the step's bytes).
- **Measured (scripts/mtp_head_time.py, data/mtp_step0.txt; share lock beside BigMoE's 80 % GPU job, rough; real
  lm_head weights; 1 row; 64-call dependent chains):**
  - full head (MLX quantized_matmul + argmax): **1.73 ms** (207 GB/s); qmm16c's 16-row tile 2.66 (wrong shape for 1
    row).
  - screened to 32,768 vocab rows: **0.24 ms** (+ a logsumexp for the pick's p: 0.24).
  - a gathered 2,048-row set (candidates / session tokens): **0.024 ms**.
- **Reading:** the lm_head is ~1.7 of the ~2.2 ms. With a screened head (an FR-Spec-style top-32K + the session's tokens +
  DFlash2's candidates) a step is ~0.75 ms; with candidates only (~2K rows) ~0.5 ms, i.e. the MTP layer alone. Below
  that, the layer itself has to shrink: HC8 on its 3 reads, fewer dispatches (a compiled step), a FAST MoE.
- **Quality is MTP Head's replay:** the pick needs the MTP's top-1 and its p. A screened head changes both when the true
  top-1 falls outside the set. The kept-per-check effect must be measured before any build.
- **Native forms for Friday's converted store (Wed 15:2x; scripts/ooo1c_native_check.py; ref2 live):**
  - The same splices on NAX6-N's kernel A (bf16 native and p11 sources) and on 3c-N's (native and native + p11);
    each family's own kernel B (NAX6-N's B; 3c-N's fused-combine bc) reads the resolved slots.
  - On L20 converted by p11.to_p11(pad=True), bf16-native and native+p11+pad pools, ~70 / 80 % resident: NAX6-N one
    call == NAX6 and 3c-N one call == 3c on the row-major pools, bitwise, in 3 released calls and a live 20 ms late
    release. Slots resolved; no timeout.
  - Through ref2 (layout 2 records): 3c-N via _ooo_sel and NAX6-N via tier_nax_ooo + nax_native, == the row-major
    references, bitwise.
  - ref2 now gives the native layers the one call too. Layers with N7-15's registered dirty flags keep the two phases.
    The row-major / DENSE-N-RAM / HC8-1P checks still PASS. Backup data/ref2_before_ooo1c_native.py.
- **MTP-STEP-1 (Wed 15:26, share lock; scripts/screened_head.py, screened_head_check.py): the screened head as a drop-in
  piece.**
  - ScreenedHead(lm_head, frspec_rank[:32768], extra_cap 4096): the base rows gathered once into one contiguous 4-bit
    matrix, plus up to 4,096 session-token rows added as they appear (add(ids)). It returns (top-1 id, p over the
    subset) with no host sync.
  - The subset's logits == the full head's at those ids, bitwise. Top-1 == the full head's whenever that token is in
    the subset. p >= the full p, inflated by the missing mass: MTP Head's replay recalibrates the pick threshold.
  - ms per call (1 row, chains, same run): full head 1.257, screened 0.171, screened + 2,048 session rows 0.217
    (x5.8-7.4). With today's step ~2.2 ms (head 1.26-1.73 of it), a screened step is ~0.6-0.9 ms [P]. The MTP layer's
    own share is next to time.
  - Quality gate: MTP Head's MTPSCREEN replay (the pick's kept per check with the screened head vs full 11.70 vs DFlash2
    alone 9.54).
  - **MTPSCREEN (MTP Head, Wed 15:3x, TrainLab b36b990): the screen PASSES; the pick is parked on the layer's cost.**
    - Corrected baseline: the x1.23 / 11.70 was the MTPPICK lead withdrawn on 11-03 (off-by-one alignment). Aligned, on
      HPD-0's 666 code checks: DFlash2 9.57, MTP b5 alone 9.97, the pick with the full head 10.51 (x1.098), union oracle
      11.35.
    - Kept per check by head: a (this head, FR-Spec 32K + session) 10.51; b (DFlash2's 16 + session) 10.53; b0 (the 16
      alone) 10.40. b keeps 102 % of the full head's gain. The subset's inflated p needed no threshold shift.
    - Step at 1 row (mtp_forward + argmax / softmax max + one sync, 400-pair cache, shared GPU): full 2.571 ms, the 32K head
      1.353, a 16-column stand-in 1.280. So the layer + mixer is ~1.28 ms, not my ~0.5 [P]. The head part is 1.29 ms
      (matches 1.257 here).
    - The pick (+0.94 kept per check) costs 15 x 1.35 = ~20 ms per check: x0.90 at a 90 ms cycle. It breaks even near
      ~0.6 ms per step, so the layer would have to be ~2x faster. Parked behind tier merge B (orchestrator 15:3x).
    - Where the head pays today: R57_CHATSW's MTP chain drafts with exactly these steps. It halves each step, and drafts
      change only where the full top-1 is outside the set (0.3 % of depths with set a). The output stays exact (drafts
      are verified), so it is a speed lever for the chat switch, offered to BigMoE.

## SMALL-N: the small 4-bit calls on the native kernel (Leap, pre-registered Wed 2026-11-04 15:3x; the FAST small-shape bundle, the orchestrator's fallback item)
- **What:** the 4-bit g64 calls that still run MLX's 16-row kernel (N < 1024 or N % 64 != 0) in BigMoE's default: R-74's
  shared gate | up | gate concat (N 1281, x48), the QSA k / v projections (N 512, x24), R-83's b | a (N 96, x36). On
  qmm16n with zero rows padded to a multiple of 32 (FAST: other bits than MLX's call; more accurate, like qmm16c).
- **Screen** (scripts/small_n_screen.py, ~1 min, share lock): ms per check, MLX's 16-row call vs qmm16n padded, random
  4-bit weights at the shapes and counts, interleaved, 15 rounds.
- **Line:** build the ref2 switch if the bundle saves >= 0.5 ms per check; KILL < 0.2.
- **Prediction:** 0.4-0.9 ms (MLX's small calls are launch-bound, ~10-25 us each).
- **Result (Wed 15:28, share lock; data/small_n_screen.txt): BUILD.** ms per check, MLX's 16-row call -> qmm16n padded:
  - shared gate | up | gate (R-74, N 1281 -> 1312, x48): 1.737 -> 0.885;
  - k / v (N 512, x24): 0.638 -> 0.422;
  - b | a (R-83, N 96, x36): 0.568 -> 0.490;
  - **bundle 2.94 -> 1.80 ms per check (-1.15), FAST.** The prediction (0.4-0.9) was low.
- **Wired (15:30; ref2 + glue_fused live, default OFF): CFG["small_native"] / REF2_SMALL_NATIVE=1.**
  - Marked resident weights get a zero-padded native copy once and run on qmm16n: QuantizedLinear calls, R-83's
    concats, and R-74's concat, which glue_fused now marks for ref2 (marking only, inert while off).
  - sn_wire_check PASS (candidate + live): calls taken and row-invariant; error vs fp32 2.12e-3 (MLX's call 2.13e-3);
    R-74 output vs the MLX path rel 1.6e-4; switch off == today bitwise. The other wiring checks still PASS; self_test
    PASS. Backups data/ref2_before_sn.py, data/glue_fused_before_sn.py.

## TIER-MERGE: one pair group per expert on the NAX path (Leap, Wed 2026-11-04 15:4x; the orchestrator's GO 15:3x; FableGod EXPERT_KERNEL.md design B)
Pre-registered 15:45, before any code ran.
- **Why:** NAX6 groups pairs by (expert, tier). An expert picked at rank 0-1 (4-bit) by some rows and at rank 2-9 (3-bit)
  by others gets two groups, so its codes are loaded and decoded twice, there are two epilogues, and two threadgroup rows
  per output tile. At S 16 that is 3,680 groups per check vs 3,112 distinct experts (-15 %; EXPERT_KERNEL.md section 5).
- **Two forms (scripts/tierglu_merge.py; grouping key = slot | tier-2 flag | the slot's sidecar slot):**
  - MERGE-X (mode 1, EXACT): one code load and decode per merged group feeds both nibble sets. The 3-bit rows take the
    op on the 3-bit nibbles, the 4-bit rows the op on the 4-bit nibbles (the same op count as NAX6's two groups), with
    NAX6's epilogue per row. Every pair's bits == NAX6 by construction: INV (an op row's result does not depend on the
    other rows), the same split-K chunks, the same epilogue expression with per-row operands.
  - MERGE-B (mode 2, a reference change): one op on the 3-bit nibbles for all rows. A 4-bit row adds L = x . l (its
    LSB plane, a masked fp32 sum on the SIMD ALUs) as P4' = 2 P3 + L, then NAX6's 4-bit epilogue. Where every fp32 sum
    is exact, P4' == P4 and the bits equal NAX6's. Elsewhere they differ by rounding.
- **Predictions (k62f harness: layer 20's real records, real routing, 4 traces, S 8 / 16 / 32, share lock, rough):**
  - MERGE-X == NAX6 bitwise on every pass and trace (kill line: any differing bit = a bug to fix, never a FAST
    variant).
  - MERGE-X kernel time at S 16 -4..-8 % vs NAX6. It saves the non-op share (~2.8 of 10.15 us) of the ~15 % merged
    groups. Kill as a speed lever below -2 %.
  - MERGE-B: differing bits only on 4-bit pairs, all <= 1 ulp, on <= 0.05 % of their bf16 outputs. Kernel time at S 16
    -8..-15 % vs NAX6 if the ALU term overlaps the ops (doc's bound -14 %). Kill (stays unbuilt in ref2) unless >= 3 %
    faster than MERGE-X.
  - Both: INV (a pass's row alone == inside the pass, bitwise); two phases + combine == one call.
- **Result (Wed 15:48 + 15:52, share lock beside BigMoE's 80 % GPU run; scripts/merge_check.py, data/merge_check_L20.txt):
  KILL, all three forms. Merging is exact but slower.**
  - Groups: -12..-19 % as predicted (S 16 code 82.4 -> 70.2).
  - Exactness: MERGE-X and MERGE-X2 == NAX6 bitwise on every pass (4 traces x 3 widths x 8 passes, random and real tap
    activations). INV and the phase form PASS for all forms. MERGE-B differs on 0.013 % of the 4-bit pairs' bf16
    outputs (none on 3-bit pairs), max 262 ulp (a near-zero output).
  - ms per call vs NAX6 (run 1 / run 2, same harness, interleaved):

    | form | S 8 | S 16 | S 32 |
    |---|---|---|---|
    | MERGE-X | +18..+22 % / +24..+25 % | +13..+21 % / +23..+27 % | +18..+26 % / +26..+29 % |
    | MERGE-B | +22..+40 % | +30..+46 % | +33..+54 % |
    | MERGE-X2 (NAX6's core verbatim for one-tier groups) | +9..+14 % | +11..+13 % | +14..+17 % |

  - Why [P]: the call is latency-bound per threadgroup, not op- or byte-bound. A merged group with both tiers runs its two
    passes (or its LSB sums) in sequence inside one threadgroup; NAX6 ran them as two threadgroups in parallel. The 15 %
    fewer groups do not shorten the critical path, and every added instruction in the hot loop costs: MERGE-X's per-row
    selects and second op tensor cost +25 % on every group. EXPERT_KERNEL.md design B priced the merge as fewer op
    groups at a fixed per-group cost; measured, the per-group cost is the chain's latency.
  - Prior evidence I should have checked first: my own S11 (Sun 23:37: a grouped 3c that decodes once per (expert, tier)
    chunk was bitwise but slower, 0.714 vs 0.597 ms per layer) and L1-24 (Sat: expert time grows with pairs, not
    distinct experts). Same lesson on the NAX path.
  - Kept: scripts/tierglu_merge.py (exact merged forms, for the record). No ref2 switch. Lead for the kernel track: the
    NAX path looks occupancy / latency-bound (20 KB of threadgroup memory per threadgroup in kernel A). An occupancy
    probe (more threadgroup memory -> fewer resident threadgroups -> how much slower) would say whether a smaller
    footprint pays.

## NAX-OCC: is the NAX6 call bound by threadgroup residency? (Leap, Wed 2026-11-04 15:55; after TIER-MERGE's KILL)
Pre-registered 15:56; the run started 15:55, before any number was seen.
- **Probe (scripts/nax_occ_probe.py, data/nax_occ_probe.txt):** layer 20's real records + routing, code / chat, S 8 / 16 /
  32. Kernel A's threadgroup memory is padded 20 -> 24 / 28 / 32 KB and kernel B's 10 -> 20 / 32 KB, with the same
  arithmetic (bits checked == NAX6). Separately, two independent calls on two GPU streams vs back to back.
- **Prediction:** A at 32 KB +0..+5 % (residency is not bound by threadgroup memory at 20 KB). Two streams 15-30 % below
  back to back at S 8-16 (the call leaves capacity idle).
- **Kill line:** A at 32 KB < +5 % slower -> no footprint lever (a smaller footprint would not pay). Two streams < 10 %
  better -> no overlap lever.
- **Result (15:55, share lock, GPU otherwise free; data/nax_occ_probe.txt): KILL, both levers.**
  - Bits == NAX6 for every padded arm.
  - Kernel A 20 -> 24 / 28 / 32 KB: -0.4..-1.7 % at every S and trace. A is not bound by its threadgroup memory, so a
    smaller footprint would not pay.
  - Kernel B 10 -> 20 KB: +0.0..+0.3 %; 10 -> 32 KB: +7.9..+12.5 %. B needs >= 3 resident threadgroups per core and
    has them at 10 KB.
  - Two calls on two GPU streams vs one stream: -7.7 / -13.1 % at S 8, -4.7 / -5.8 % at S 16, -1.2 / -2.4 % at S 32
    (code / chat). The call leaves little capacity idle at S >= 16. (The "back to back" arm is void: it evaluated two
    calls, not eight.)
  - Reading with TIER-MERGE [P]: the GPU is busy during the call, but the work isn't op- or byte-bound (matrix unit
    ~57 %, DRAM ~53 % per EXPERT_KERNEL.md). Instructions in the hot loop are what cost: MERGE-X's per-row selects
    alone added +25 %. The lever left in NAX6 / NAX6-N is fewer instructions per stage (decode, epilogue, A
    marshalling), kept bitwise.

## NAX-ABL: where NAX6-N's call time goes, by ablation (Leap, Wed 2026-11-04 16:1x; after TIER-MERGE / NAX-OCC)
Pre-registered before the run.
- **Probe (scripts/nax_ablate.py, data/nax_ablate.txt):** NAX6-N on layer 20's records repacked native, real routing
  (code / chat), S 8 / 16 / 32. Arms (timing only; ablated arms compute garbage): no-dec (raw words as nibbles),
  no-epi (acc += P), no-op (no matrix op; A loads die with it), no-A (A from constants), no-codes (code words from
  constants), skeleton (no-dec + no-op + no-epi + no-codes).
- **Prediction at S 16 (share of the full call saved by removing the part):** decode 20-35 %, op 20-30 %, epilogue
  8-15 %, code loads 10-20 % (native layout), A loads 5-10 %, skeleton 15-25 % of the call left.
- **Use:** the biggest bitwise-reducible part gets the next build (a leaner decode, epilogue or A path with the same
  arithmetic). Kill line: no part above 10 % except the op -> NAX6-N's hot loop has nothing left to cut, and the expert
  kernel lever is NAX7's dispatch / byte work, not instructions.
- **Result (15:58, share lock beside Fit12's CPU job; data/nax_ablate.txt; full arm == tier_glu_nax6n bitwise):**

  | removed | S 8 code / chat | S 16 code / chat | S 32 code / chat |
  |---|---|---|---|
  | decode | -6.1 / -9.8 % | -9.8 / -10.1 % | -10.2 / -10.3 % |
  | epilogue | -10.4 / -9.9 % | -11.2 / -11.0 % | -11.2 / -11.3 % |
  | matrix op (+ A marshalling) | -16.9 / -27.4 % | -27.5 / -24.4 % | -30.2 / -28.8 % |
  | A loads | -2.6 / -3.8 % | -4.2 / -4.0 % | -4.4 / -4.5 % |
  | code loads | -9.3 / -7.0 % | -6.3 / -7.0 % | -5.7 / -6.4 % |
  | all but the skeleton | -77.7 / -80.0 % | -82.9 / -82.5 % | -84.3 / -83.8 % |

  - The parts are super-additive: they sum to ~59 % alone but ~83 % together, so they mostly serialize. That fits
    TIER-MERGE and NAX-OCC.
  - Verdict BETWEEN on the pre-registered rule: the epilogue (11 %) and decode (10 %) sit at the 10 % line. The op is
    the largest part.
  - The skeleton's 17 % includes empty threadgroups: the grid's y is P (the most groups possible) because the group
    count is only known on the GPU, and at S 16 about half the rows return at once.
  - Next micro-screen: vectorized epilogue loads (bitwise), and a tight grid (y = the real group count; measurement
    only) to price the empty launches.

## NAX-MICRO: two cheap cuts in NAX6-N's call (Leap, Wed 2026-11-04 16:1x; from NAX-ABL)
Pre-registered before the run (scripts/nax_micro.py, data/nax_micro.txt; layer 20 native, code / chat, S 8 / 16 / 32).
- epi-v4: the epilogue's scale / bias reads as four float4 loads per stage instead of per-element loads. The
  arithmetic is the same, so bitwise. Prediction -1..-4 % (the compiler may already merge them). Kill below -2 %.
- tight grid: the launch grid's y = the real group count instead of P (measurement only: it needs the count on the
  host). Prediction -3..-8 % at S 16 (about half the rows return at once). Kill below -3 %. If it pays, the build is
  a persistent / grid-stride form that needs no host count.
- **Result (16:00, share lock; data/nax_micro.txt; both arms == tier_glu_nax6n bitwise): KILL, both.**
  - epi-v4: -0.6..+0.4 %. The compiler already merges those loads.
  - tight grid: -2.3 / -3.0 % at S 16 (code / chat), -1.9..-4.3 % elsewhere. The arm also skipped the per-call grouping
    kernel (its grouping was shared), which is about that much (~18 us at S 16), so empty rows cost ~nothing.
  - With NAX-ABL / NAX-OCC: NAX6-N's hot loop has no cheap cut left. The 16-row op padding is fixed (no 8-row shape,
    K-62c). What's left for the expert kernel is structural (NAX7's fewer dispatches) or bytes.

## BP-0: DFlash2's block pass by part (Leap, Wed 2026-11-04 16:1x; the orchestrator's route 15:5x: the drafter is 10.7 ms of a 100.2 ms code check in BigMoE's all-in C 160 arm, 6.3 with D7b quiet; byte floor ~1-3 ms)
Pre-registered before the run (scripts/bp_profile.py, data/bp_profile.txt; R-79's loader: DFlash2 + l120_ft bound to
the real 4-bit embed / lm_head, ~3.5 GB; share lock, rough).
- Arms: bf16 + R-76 / R-78 forms (K83, the engine's exact drafter) and D7b (layers 4-bit). Parts timed alone:
  hidden_states, compute_logits (the 4-bit lm_head at 16 rows, 248K vocab), the two-stage top-16, the selector's
  hidden_projection, chain_fused. Plus the head alone: full vs screened (FR-Spec top-32K + 2K session rows).
- **Prediction (bf16):** propose ~9-11 ms; hidden_states ~6-7, compute_logits ~1.6-2.0, top-16 ~0.3-0.6, projection
  + chain ~0.5. Screened head ~0.25-0.35 ms vs ~1.6-2.0 full.
- **Use:** the drafter's outputs are verified, so any change that keeps acceptance is allowed (drafts bitwise where the
  form allows, acceptance unchanged on replays where it doesn't). The biggest part with a known cut gets built first:
  likely the screened draft head (~-1.5 ms per check if acceptance holds), then the layers' bytes (8-bit layers: half
  of bf16's bytes, nearly lossless, unlike D7b's 4-bit -1.3 % code kept).
- **Result (16:06, share lock, GPU ~free; data/bp_profile.txt; 2 interleaved rounds):**

  | arm | propose | hidden_states | compute_logits | top-16 | projection | chain |
  |---|---|---|---|---|---|---|
  | bf16 + forms (K83) | 10.21-10.25 ms | 7.74-7.85 | 2.50-2.52 | 0.09-0.10 | 0.20 | 0.24-0.25 |
  | D7b (4-bit layers, MLX's own qmm: no REF2 here) | 9.76-9.84 | 7.31-7.35 | 2.51-2.53 | 0.09-0.10 | 0.20-0.21 | 0.24-0.26 |

  - The head alone at 16 rows: full 248K vocab 2.149 ms, screened (FR-Spec 32K + 2K session rows) 0.454 ms.
  - Reading [P]: hidden_states is ~76 % of the propose. Its bf16 bytes are ~0.85 GB (floor ~3.4 ms at 250 GB/s), so
    it runs at ~2.3x its byte floor. The head is ~25 %, and a screened head cuts ~1.7 ms of it. Top-16, projection and
    chain are ~0.55 ms together. D7b's 6.3 ms (R-85b) came with REF2's qmm16c inside the propose.
  - Plan: (1) the screened draft head (FAST-safe, acceptance replay); (2) the layers' bytes: 8-bit layers on a 16-row
    kernel (~half bf16's bytes, near lossless) or D7b with qmm16c; (3) what's left of hidden_states (attention, convs,
    norms).

## BP-1: DFlash2's draft head screened to FR-Spec 32K + session tokens (Leap, Wed 2026-11-04 16:2x; from BP-0)
Pre-registered before the run (scripts/bp_screen_check.py, data/bp_screen_check.txt).
- **What:** head2s's compute_logits over ScreenedHead's rows (FR-Spec top-32K + the window's prompt ids + each check's
  committed ids, cap 4096) instead of the full 248K vocab; the same two-stage top-16; subset indices mapped back to
  token ids. Drafts are verified, so the output stays exact.
- **Replay:** MTP Head's D7B-1 / DA-1 engine-order walk (copied): the bf16 drafter with K83's wiring, the 24 code / math
  windows of ~/Leap/data/dfl2d_*, cut t 0.3, paired per window.
- **Prediction:** kept per check x0.995-1.000 on code and math (FR-Spec covers ~98.3 % of code tokens, and the session
  rows cover most of the rest). Propose -1.5..-1.8 ms.
- **Line:** adopt (offer BigMoE an R57 switch) at kept >= x0.995 on both. KILL below x0.981 (at a 90 ms cycle, -1.7 ms is
  worth ~1.9 % of kept). In between, report it as a trade.
- **Result (16:08, share lock; data/bp_screen_check.txt; 52 s, peak 1.9 GB): BETWEEN, a trade.**
  - Code (24 windows): kept 9.248 -> 9.183, x0.9930 (90 % CI x0.987-x0.998; lower in 4, higher in 1). Propose 9.92 ->
    8.13 ms.
  - Math (60 windows): kept 6.671 -> 6.595, x0.9885 (CI x0.983-x0.993; lower in 21, higher in 6). Propose 9.60 -> 7.70
    ms.
  - Net at a 100 ms cycle: code +1.1 %, math +0.75 %; at 40 ms, code +3.8 % [P]. The misses are true tokens outside
    FR-Spec 32K + session.
  - Next, BP-1b: a 64K base (more coverage, head ~0.75 ms) to clear x0.995.
- **BP-1b pre-registered (16:1x):** base = FR-Spec top-65,536 + session (cap 4096). Prediction: kept >= x0.997 on code
  and math, propose -1.3..-1.5 ms. Same lines.
- **BP-1b result (16:09): BETWEEN again.** Code x0.9944 (CI x0.989-x0.999; lower in 3, higher in 1), math x0.9940 (CI
  x0.991-x0.997). Propose -1.51 ms (code 10.02 -> 8.51, math 9.67 -> 8.16).
  - Doubling the base recovers only part of the loss. A changed candidate set also moves the selector's softmax
    confidences, and so the t 0.3 cut, even where the true token is in the set [P].
  - Net [P]: at a 100 ms code cycle 32K +1.1 % / 64K +0.95 %; at 40 ms +3.8 % / +3.2 %. Math +0.85 % / +1.0 % at 100 ms.
  - Verdict: an exact-safe speed trade, not a clean PASS. Offered to BigMoE as an R57 switch with both bases; their
    steady tok/s A/B decides.
- **BP-1c pre-registered (17:0x):** base = FR-Spec top-98,304 + session (SVH's base: on the stored streams no generated
  token falls outside it on code, 0.007 % on math). Prediction: kept x0.998-1.002 on both (candidate sets change only
  rarely). Line: >= x0.997 on both -> build the fast form (qmm16c over the subset in the drafter's head, ~-0.9 ms per
  propose vs the full head on qmm16c). Else KILL.
- **BP-1c result (16:53, share lock): PASS.** Code kept 9.248 -> 9.248, **x1.0000** (no window changed); math 6.671 ->
  6.668, **x0.9995** (1 window lower). Propose with MLX's kernels 9.95 -> 8.68 (code), 9.51 -> 8.35 (math).
- **BP-9 pre-registered (17:0x; scripts/bp_sub_check.py, d7b_nax.DraftSubset / screened_head2s):** the fast form on
  qmm16c with D8-NAX (the exact default's drafter candidate): full head on qmm16c vs subset on qmm16c, the walk
  feeding session ids. Prediction: kept x0.999-1.001; propose -0.8..-1.0 ms (~6.8 -> ~5.9). Line: kept >= x0.998 and
  >= 0.6 ms faster -> offer BigMoE.
- **BP-9 result (16:55, share lock, GPU ~free; data/bp_sub_check.txt): PASS.** Propose D8 + head on qmm16c 7.13 / 6.83 /
  6.76 -> D8 + subset on qmm16c **5.84 / 5.84 / 5.77 ms (-0.99)**. Kept: code 9.290 -> 9.290 (x1.0000, no window
  changed), math 6.675 -> 6.671 (x0.9995, one window). The exact default's drafter: bf16 + forms 10.4 -> D8-NAX 7.5 ->
  + head on qmm16c 6.8 -> + subset **5.8 ms**: -4.6 ms per code check at unchanged acceptance.

## BP-2: the drafter layers' bytes (Leap, Wed 2026-11-04 16:1x; from BP-0: hidden_states 7.8 ms, ~2.1x the bf16 floor)
- **BP-2a pre-registered (scripts/bp_q8_bench.py, data/bp_q8_bench.txt): the drafter's projections at their shapes**
  (q 4096x2560, k / v 1024x2560, o 2560x4096, gate / up 7680x2560, down 2560x7680 at 16 rows; fc 2560x12800 at 8 rows):
  bf16 nn.Linear vs R-78's x @ Wt vs MLX quantized_matmul 8-bit / 4-bit g64 vs qmm16c (4-bit, 16 rows).
  - Prediction: bf16 projections ~5-5.5 ms per propose. MLX 8-bit at 60-75 % of its byte floor, ~2.5-3 ms. qmm16c 4-bit
    ~1.5 ms.
  - Line: MLX 8-bit <= 0.6x bf16 -> an 8-bit drafter needs no new kernel (next: its acceptance replay, D7B-1's walk with
    bits 8). Else build a 16-row 8-bit NAX kernel (MPP's bf16 x uint8 from threadgroup memory).

## BP-3: DFlash2's hidden_states by ablation (Leap, Wed 2026-11-04 16:1x; BP-0: 7.8 of a 10.2 ms propose)
Pre-registered before the run (scripts/bp_hs_ablate.py, data/bp_hs_ablate.txt; bf16, plain modules; timing only).
- Arms: no-proj (every nn.Linear a same-shape fill), no-sdpa (attention returns the queries), no-conv (the grouped
  dynamic convs are identities), no-norm, skeleton (all four).
- Prediction (share of hidden_states saved): projections ~45-50 % (BP-2a: ~3.8 ms bf16), attention ~15-20 %, convs
  ~8-12 %, norms ~5 %, skeleton ~15-20 % left.
- Use: with BP-2a (MLX's quantized_matmul at 16 rows is slow; qmm16c 1.46 ms for all 4-bit projections), it picks
  the build: the projections' bytes (an 8-bit NAX kernel, or D7b on qmm16c) vs the non-projection work.
- **Result (16:11, share lock; data/bp_hs_ablate.txt): projections 74 %.** hidden_states 8.08 ms: no-proj 2.14 (-74 %),
  no-sdpa 6.26 (-23 %), no-conv 7.10 (-12 %), no-norm 8.00 (-1 %), skeleton 0.58 (-93 %). The projections (k / v twice,
  fc, the conv kernel projections) are ~5.9 ms; attention ~1.8; convs ~1.0.

## BP-4: D7b on the NAX kernel (Leap, Wed 2026-11-04 16:2x)
- **Finding that motivates it:** r57_screen calls ref2.remove() before every propose. So the D7b drafter (R57_D7B=code,
  in BigMoE's all-in FAST code preset) runs MLX's 16-row 4-bit quantized_matmul, which BP-2a timed no faster than
  bf16; BP-0's D7b propose is 9.8 vs bf16's 10.2 ms. The preset likely pays D7b's -1.3 % code kept (D7B-1) for almost
  no speed. R-85b's 6.3 ms D7b propose had REF2's qmm16c inside.
- **Build (scripts/d7b_nax.py):** install(draft) turns the D7b drafter's eligible 4-bit QuantizedLinears into NaxQL:
  <= 16-row bf16 calls run qmm16c (padded rows), whatever REF2's state. Same 4-bit weights; only the fp32
  accumulation order differs from MLX's kernel.
- **Pre-registered (scripts/bp_d7b_nax_check.py, data/bp_d7b_nax_check.txt):** D7b-NAX propose 6.0-7.0 ms (D7b MLX
  ~9.8, bf16 + forms ~10.2). Kept per check x0.995-1.005 vs D7b-MLX on code and math (engine-order walk).
- **Line:** propose <= 7.5 ms with kept within +-0.5 % of D7b-MLX -> offer it to BigMoE as the D7b drafter's kernel
  (R57_D7B arms). Otherwise KILL.
- **Result (16:14, share lock; data/bp_d7b_nax_check.txt; 79 s, peak 2.6 GB): PASS.**
  - ms per propose (3 interleaved rounds): bf16 + forms 10.16-10.25; D7b (MLX qmm) 9.30-9.40; **D7b-NAX 7.21-7.43** (-2.05
    ms vs D7b, -2.9 vs bf16).
  - Kept per check (engine-order walk):
    - code: bf16 9.248, D7b 9.130, D7b-NAX 9.091. D7b-NAX / D7b x0.9957 (90 % CI x0.989-x1.003).
    - math: bf16 6.671, D7b 6.524, D7b-NAX 6.536. D7b-NAX / D7b x1.0019 (CI x0.998-x1.006).
  - 45 NaxQL modules (9 per layer: q k v o gate up down + 2 conv kernel projections); 870 MLX calls left (the
    1,000-row context prefills).
  - Verdict: the FAST code preset's R57_D7B=code drafter gets ~-2 ms per code check at the same acceptance. D7b still
    costs -1.3..-1.7 % code kept vs bf16 (D7B-1); the 4-bit-aware fine-tune is the fix for that.
- **BP-4n pre-registered (16:2x; scripts/bp_d7b_naxn_check.py):** the same on DENSE-N's layout
  (d7b_nax.install(native=True): qmm16n on a native copy, +~0.26 GB). Prediction: bitwise == D7b-NAX (per module and
  whole proposes: DENSE-N's ks-1 guarantee), propose -0.2..-0.5 ms vs D7b-NAX.

## BP-5: an 8-bit 16-row NAX kernel for the drafter (Leap, Wed 2026-11-04 16:2x; for the exact default's bf16 drafter)
Pre-registered before any run (scripts/qmm16c8.py, qmm16c8_check.py; data/qmm16c8_check.txt).
- **Why:** the bf16 drafter's projections are ~5.9 of its 8.1 ms hidden_states (BP-3). 4-bit (D7b) costs 1.3-1.7 %
  code kept; 8-bit RTN g64 should be near lossless. MLX's 8-bit quantized_matmul at 16 rows moves only ~95 GB/s
  (BP-2a), so it needs its own kernel.
- **Kernel:** qmm16c's v3 code MMA with byte codes. B = bf16(code), exact; per output and 64-group y += s P + b X; rows
  padded to 16, so row-invariant by construction.
- **Prediction:** error vs fp32 <= MLX 8-bit's. 1 / 8 rows == the 16-row call's rows, bitwise. 150-230 GB/s at the
  drafter's shapes (best ks per shape), i.e. ~1.6-2.4x faster than bf16 there.
- **Line:** >= 1.4x faster than bf16 over the drafter's projection set -> wire an 8-bit drafter (NaxQL8) and run
  BP-1's acceptance walk vs bf16. Below 1.2x -> KILL.
- **Result (16:18, share lock beside BigMoE's 80 % GPU run; data/qmm16c8_check.txt): PASS.**
  - Error vs fp32 2.32-2.35e-3, = MLX 8-bit's (2.32-2.33e-3), better on down (MLX 3.62e-3). ks 2 / 4 give the same
    error. 1 / 8 rows == the 16-row call's rows, bitwise.
  - Best us per call (bf16 in brackets; 20 repeats per eval, so the smaller weights sit warm in cache):
    q 32.0 (49.5), k / v 18.6 (76.7), o 35.4 (48.9), gate / up 55.6 (140.7), down 52.0 (231.3), fc 89.1 (423.6).
  - Over the drafter's set: ~1.6 vs ~5.0 ms, ~3x on paper. The real propose decides (BP-6).
- **BP-4n (16:19, data/bp_d7b_naxn_check.txt): KILL.** D7b on DENSE-N's layout (qmm16n) is bitwise == D7b-NAX (every module
  at 16 / 8 rows; 24 whole proposes), but slower: 8.27-8.32 vs 7.76-7.94 ms (BigMoE at 80 % GPU beside it).
- **BP-6 pre-registered (16:2x; scripts/bp_d8_check.py, d7b_nax.apply8):** the exact default's drafter at 8 bits. The
  layers' Linears + fc are 8-bit RTN g64 on qmm16c8 (fixed ks per shape). Prediction: propose 10.2 -> 6.5-7.5 ms; kept
  x0.997-1.003 vs bf16 + forms on code and math. Line: kept >= x0.997 on both and propose <= 8 ms -> PASS, offer
  BigMoE an exact-default drafter switch. Otherwise report.
- **BP-6 result (16:20, share lock beside BigMoE's GPU run; data/bp_d8_check.txt; 73 s, peak 3.8 GB): PASS.**
  - ms per propose (3 interleaved rounds): bf16 + forms 10.28 / 10.46 / 10.53; **D8-NAX 8.06 / 7.36 / 7.48**; D7b-NAX
    7.69 / 7.13 / 7.18.
  - Kept per check vs bf16 + forms:
    - code: 9.248 -> 9.277, **x1.0031** (90 % CI x0.992-x1.014; lower in 3, higher in 1 of 24). D7b-NAX x0.9830.
    - math: 6.671 -> 6.678, **x1.0009** (CI x0.996-x1.006; lower in 11, higher in 12 of 60). D7b-NAX x0.9797.
  - 46 NaxQL8 modules (the 45 layer Linears + fc). The drafter's RAM halves (~1.0 -> ~0.5 GB).
  - Verdict: the exact default's drafter -2.9 ms per check at unchanged acceptance. It also beats D7b-NAX for the FAST
    preset: ~0.3 ms slower, without D7b's -1.7..-2 % kept.

## BP-7: the draft head on qmm16c (Leap, Wed 2026-11-04 16:2x; with BP-6's D8-NAX drafter)
Pre-registered before the run (scripts/bp_head_check.py, data/bp_head_check.txt).
- **What:** head2s's compute_logits is the target's 4-bit lm_head at 16 rows, on MLX's kernel in the engine's propose
  (REF2 is removed there): 2.5 ms in BP-0. Here it runs on qmm16c (same weights, own accumulation order), then the same
  output multiplier / softcap and top-16.
- **Prediction:** propose -0.5..-0.9 ms vs D8-NAX. Kept x0.997-1.003 (only accumulation order differs).
- **Line:** kept within +-0.3 % with propose >= 0.4 ms faster -> PASS (goes with D8-NAX's switch). Otherwise KILL.
- **Result (16:23, share lock beside BigMoE's 80 % GPU run; data/bp_head_check.txt): KILL as measured; one quiet
  re-measure allowed.**
  - Kept unchanged: code x1.0015 (CI x0.998-x1.006), math x0.9995 (CI x0.999-x1.000).
  - Propose: D8-NAX 8.31 / 8.61 / 8.58 vs + head on qmm16c 9.70 / 8.25 / 8.19 ms. Median -0.33 ms, below the 0.4 line,
    in noisy rounds. The bytes (0.36 GB at ~240 GB/s vs MLX's ~167) predict ~-0.6 ms.
- **Quiet re-measure (16:3x, share lock, GPU otherwise free): PASS.** Propose D8-NAX 7.67 / 7.33 / 7.37 vs + head on
  qmm16c 6.81 / 6.85 / 6.80 ms: **-0.56 ms** median. Kept unchanged (same as above). With D8-NAX, the exact drafter
  goes 10.4 -> ~6.8 ms. Drop-in: d7b_nax.head_logits(draft, hid) replaces draft.compute_logits in head2s. It falls
  back to compute_logits for a non-4-bit / converted head or > 16 rows.

## BP-8: the drafter's attention core (Leap, Wed 2026-11-04 16:2x; BP-3: SDPA ~1.8 ms of hidden_states)
- **BP-8a pre-registered (scripts/bp_sdpa_bench.py, data/bp_sdpa_bench.txt):** MLX's SDPA at the drafter's shape (16
  queries, 32 / 8 heads x 128, ctx 1,000 / 2,047 + 16 keys): bool mask (today) vs additive bf16 mask vs none vs causal,
  and plain matmuls + softmax. Prediction: the bool mask takes a slow path (~300+ us), mask None / causal ~40-80 us.
  If so, the build is a mask the fast path accepts, or an in-kernel window (a 16-query sliding-window attention
  kernel). Drafts must stay bitwise where the form allows, or keep acceptance.
- **BP-8a result (16:24; data/bp_sdpa_bench.txt): no SDPA lever.** us per call at ctx 1,000 / 2,047:

  | form | ctx 1,000 | ctx 2,047 |
  |---|---|---|
  | bool mask (today) | 76.4 | 140.2 |
  | additive mask | 73.0 | 119.5 |
  | none | 65.1 | 110.4 |
  | causal | 61.2 | 109.5 |
  | plain fp32 | 571 | 1,273 |

  - The additive mask is bitwise == the bool mask and 3-15 % faster: ~-0.1 ms per propose at a full window.
  - SDPA itself is ~0.4-0.7 ms per propose. BP-3's no-sdpa arm (-1.8 ms) also dropped work that became dead with it:
    the K / V concats, the block rows' k / v projections, their norms and RoPE.
  - What's left in attention is small: concat-free K / V (~0.2 ms), the additive mask (~0.1 ms).

## MTP-STEP-2: the chat MTP step by piece, standalone (Leap, Wed 2026-11-04 16:3x; the orchestrator's next item after the block pass)
Pre-registered before the run (scripts/mtp_step_profile.py, data/mtp_step_profile.txt).
- **Harness:** the MTP head built as qwen4_exp attach_mtp builds it (Qwen4ExpMTP from mtp.safetensors, 1.67 GB),
  the real 4-bit embed and the screened head; a 400-position QSA cache; no trunk.
- **Prediction:** step ~1.3-1.5 ms (MTPSCREEN: 1.353). Pieces: MoE block ~0.4-0.5 ms, QSA attention ~0.2-0.3, each HC
  read ~0.1-0.15, fc prep ~0.1, mixer ~0.1, screened head ~0.2. The pieces sum to more than the step (sync per eval).
- **Use:** the drafter's step is exact-safe territory (verified drafts), so FAST kernels may go in where acceptance
  holds. The largest piece gets the first build; the target is ~0.6 ms per step (MTP pick breaks even, chat -~3 ms
  per check).
- **Result (16:28-16:31, share lock, GPU ~free; data/mtp_step_profile.txt; standalone MTP head 1.67 GB, 13 quantized
  modules: MoE 4-bit g32 / g64 and 8-bit g64 parts):**
  - (1) The step as the chain runs it: **1.145 ms** median (p10 1.122, p90 1.312), vs MTPSCREEN's 1.353 in the engine
    on a shared GPU.
  - (2) Pieces alone (us, 20 per eval): fc prep 76; attn HC read 50; mlp HC read 56; MoE block 165; residual write 16;
    mixer 48; screened head + argmax / p 191. QSA attention (one call per eval, with sync) 406.
  - (3) Ablation inside the step (full 1.330 ms this run):

    | removed | ms saved |
    |---|---|
    | attention | 0.43 (-32 %) |
    | MoE | 0.35 (-26 %) |
    | head + mixer | 0.33 (-25 %) |
    | mixer alone | 0.09 (-7 %) |
    | fc prep | 0.13 (-10 %) |

    - The "no-hc" arm (-57 %) is VOID: with a constant write, attention and the MoE go unconsumed and MLX skips them.
    - HC reads + write by difference ~0.1 ms (pieces alone 122 us).
  - Reading [P]: ~0.6 ms per step needs most of these. Candidate cuts:
    - attention: the QSA decode path at 1 row is ~0.43 ms over a 400-key cache; below ~2,051 keys the indexer keeps
      every key, so a dense SDPA is the same selection;
    - MoE: REF2's expert kernels scoped to the step, ~-0.2;
    - fc + mixer: 8-bit, ~-0.1;
    - a 16K screened base: ~-0.1, a trade.

## MTP-STEP-3: mtplx's own fused decode paths for the MTP step (Leap, Wed 2026-11-04 16:3x)
Pre-registered before the run (mtp_step_profile.py STEP_ONLY=1, one process per setting; data/mtp_step_profile.txt).
- **Why:** mtplx gates fused decode kernels behind env flags that default OFF (MTPLX_FUSED_MOE_DECODE, MTPLX_FUSED_HC /
  _V3, MTPLX_FUSED_QSA_INDEXER). BigMoE sets none of them, so the chat MTP step (MTP-STEP-2: 1.145 ms) runs the
  unfused paths.
- **Screen:** the step median and a hash of 8 chained steps' outputs per setting (default, each flag alone, all).
- **Prediction:** MoE-decode and HC fusion -0.1..-0.3 ms together. Bits may differ from the default (fp32 order).
  Drafts are verified, so acceptance is what counts.
- **Line:** >= -0.2 ms together -> the engine sets them for the MTP chain only (if they're read per call) and
  acceptance is checked. Else KILL.
- **Result (16:31-16:34, share lock; seeded reruns): KILL.**
  - Step ms: default 1.168 / 1.207. HC + HC_V3 2.602 (other bits). All four 2.622 (bits = HC's). QSA_INDEXER alone
    1.433 / 1.159 (default's bits). MOE_DECODE alone 1.288 (first, unseeded screen).
  - MOE_DECODE and QSA_INDEXER are bitwise == the unfused paths and no faster. The HC fusion is 2.2x slower at 1 row.
- **MTP-STEP-4 (16:35): REF2's kernels during the step: KILL.** Step 2.007 / 2.212 ms vs 1.202 / 1.375 without. REF2
  pads to 16 rows (qmm16c 416 calls, MLX's 16-row call 728, padded Linear 208, HC read 208); MLX's own 1-row paths
  are faster at 1 row.
- **Floor [P]:** the step reads ~170 MB: screened head 47, bf16 fc / HC / mixer ~65, attention ~30, MoE ~30. That's ~0.68 ms
  at 250 GB/s vs 1.15 measured (1.7x). ~0.6 ms per step needs byte cuts (8-bit fc / HC / mixer -32 MB, a 16K head
  -24 MB) and new 1-row kernels (fused MoE GEMV, 1-row HC read, QSA decode). Several builds for ~-3 ms per chat check
  (~8 %). Parked unless the orchestrator wants the multi-kernel effort.

## MTP-STEP-5: the chat MTP chain in chunks on the GPU (Leap, Wed 2026-11-04 16:3x)
Pre-registered before the run (mtp_step_profile.py CHUNK_TEST=1).
- **Idea:** R57_CHATSW's chain syncs to the host after every step (int(tk.item()) feeds the next step). On the GPU, the
  token can stay an array (embedding of take(subset ids, argmax)), so 2 / 4 / 8 steps go per eval. Each step's ops are
  unchanged, so the tokens are the same; only the host round trips go. The early-exit cut is applied after the chunk
  (steps past the cut are wasted GPU work).
- **Prediction:** chunk 4 saves 0.15-0.35 ms per step of the 1.15 (host sync + graph latency). Tokens identical.
- **Line:** >= 0.15 ms per step at chunk 4 with identical tokens -> offer it (with the screened head) as the chat
  chain's form. Else KILL.
- **Result (16:3x, share lock): KILL on the economics.** ms per step: chunk 1 1.185, chunk 2 1.104, chunk 4 1.036, chunk
  8 0.998; tokens identical. Chunk 4 saves 0.149 ms per step, at the line, but the chain's early exit wastes up to
  chunk - 1 steps (~1 ms each) past the cut. That cancels the saving at typical lengths. The chain is mostly GPU-bound
  (~1.0 ms per step).

## D5: the MLP write as the next HC read's pending write (Leap, Wed 2026-11-04 16:4x; HC_READ.md D5; the orchestrator's go)
Pre-registered before the run (scripts/ref2_d5_patch.py, d5_check.py; data/d5_check.txt).
- **What (EXACT):** ref2's write (_hc_write = the patched Q._hyper_residual_write) stashes (output, hyper, block_out,
  inject) in one carry slot. An injecting HC read whose input IS that output reads (hyper, pending) instead and folds
  the write into its norm pass (leapk hc_read / HC8: the same bf16 ops). The write's own kernel runs only for other
  consumers (r57's taps, the mid-loop async_eval). CFG hc_pend_mlp / REF2_HC_PEND=1, default off.
- **Prediction:** bitwise == off on every read (mixed, written, inject) and through a 4-layer real stack; one fold per
  layer boundary. Saving ~5-15 us per boundary (one launch + one re-read of the stream). About -0.3..-0.5 ms per check
  across ~36 foldable boundaries (taps and mid-loop evals keep their writes).
- **Line:** any differing bit = a bug (no FAST variant). PASS = bitwise + saves >= 4 us per boundary in the stack ->
  BigMoE gets REF2_HC_PEND=1 for its exact arms.
- **Result (16:40-16:43, share lock; data/d5_check.txt): exact, but KILL as a speed lever.**
  - (a) 48 / 48 HC reads bitwise (layers 3 / 20 / 44, attention and MLP side, 4 scales, R-73's hc_write off and on),
    fold counted. (b) The HC8 path is bitwise too.
  - (c) Real 4-layer stack (layers 4-7: GDN x 3 + attention, stub MoE): bitwise, 3 / 3 folds. (d) self_test PASS.
  - Speed: stack 2.583 vs 2.586 ms. Isolated write + next read per boundary: 85.4 -> 83.0 us with hc_write off (-2.4),
    84.0 -> 84.3 with it on (+0.3). The read itself is ~82 us; REF2's compiled write is already one cheap kernel, so
    there's little to fold away. HC_READ.md's -0.5 ms assumed a costlier write.
  - ref2 keeps the switch (default off, inert, documented). Not offered to BigMoE.

## SVH: the screened verifier head (Leap, Wed 2026-11-04 16:5x; FAST; the orchestrator's dense-diet item (a))
Pre-registered before the run (scripts/ref2_sh_patch.py, sh_wire_check.py; data/sh_wire_check.txt).
- **What:** ref2 CFG screened_head / REF2_SCREENED_HEAD=1 (default off). The registered trunk lm_head's <= 16-row calls
  (REF2 on: the verify pass) run qmm16c over FR-Spec's top-98,304 ids + up to 4,096 session ids (the engine adds the
  prompt and each check's committed ids). The logits are scattered into a full-vocabulary -inf array, so the engine's
  interface is unchanged. Inside the subset they equal REF2's full head bitwise (qmm16c columns are N-independent);
  a row whose true argmax is outside the subset changes (FAST). The drafter's propose and the MTP chain run with REF2
  removed and see the plain head.
- **Deviation estimate (stored token streams, 20.7K generated tokens; out-of-set = rows that can change):** 96K +
  session: code 0.000 %, math 0.007 % (1 / 14,320). 96K alone: 0.078 % / 0.042 %. 64K + session: 0.094 % / 0.293 %.
  32K + session: 0.235 % / 0.761 %.
- **Prediction:** head at 16 rows ~1.6 -> ~0.6-0.7 ms (147 MB vs 360 MB), -0.9..-1.0 ms per check. Wiring bitwise
  inside the subset.
- **Line:** wiring PASS and >= 0.7 ms saved per head call -> offer to BigMoE's FAST arms + EvalSuite's next FAST batch
  (the map's deviation and same-scores decide). Else KILL.
- **Result (16:44, share lock, GPU ~free; data/sh_wire_check.txt): PASS.**
  - (a) Base 98,304 rows: subset logits == REF2's full head bitwise; the rest -inf; argmax == full wherever the full
    one is in the subset (12 / 16 random rows); counted.
  - (b) 300 session ids: bitwise == full at those ids; reset drops them.
  - (c) Switch off == REF2's head, uncounted; with REF2 removed, the plain head.
  - (d) self_test PASS.
  - (e) ms per 16-row head call: REF2 full **1.582 -> screened 0.795** (base + 2,045 session rows): **-0.79 ms per
    check**.
  - Offered to BigMoE's FAST arms (install + session-id feed) and EvalSuite's next FAST batch.

## GDNB-2 + the orchestrator's option 3 (Leap, Wed 2026-11-04 17:0x): is there an exact lever left in GDN / attention?
- **From the record, no profile needed:**
  - GDN (Mon GDNB quiet timing): a layer is ~0.338 ms at 16 rows. The body (prework + recurrence + gated norm) is
    ~0.08 ms (~2.9 ms per check), already fused (GDNX / R-61). The other ~0.26 ms is the in / out projections, since
    moved to ~74 % of their floor by DENSE-N-RAM. There is no ~6 ms GDN body.
  - Attention (R-72): ~0.6 ms per 16-row call with K-73's two fusions (81-87 kernels). Projections ~0.15 ms. The rest
    is REF2's exact verify SDPA: MLX's own 16-query SDPA in each row's AR class (R-40 3m), an extra call when rows
    straddle a class, bucket pads, cache updates, mask build, norms, gate. A bitwise replacement means replaying MLX's
    NAX SDPA kernel's arithmetic: large and uncertain. att_single (-0.2 ms / call) is REF2c, not bitwise.
- **GDNB-2 pre-registered (scripts/gdnb_wire_check.py, data/gdnb_wire_check.txt):** ref2's gdn_body (exact, built Mon,
  never bundled) re-checked under today's config (R-83's fused in_proj, DENSE-N-RAM, verify scope).
  - Prediction: bitwise at S 16 / 8 / 1 for NSG 8 and 16. -0.010..-0.017 ms per layer (Mon: -0.012 NSG 8, -0.017
    NSG 16), i.e. -0.4..-0.6 ms per check.
  - Line: bitwise and <= -0.008 ms per layer -> offer BigMoE REF2 gdn_body (+ GDNB_NSG=16) for the exact bundle. Else
    KILL.
- **GDNB-2 result (16:50, share lock, GPU ~free; data/gdnb_wire_check.txt): PASS.**
  - Real layer 4 (GDN), R-83's fused in_proj, DENSE-N-RAM (3 weights, 32 MB, in place), verify scope.
  - GDNB NSG 8 and 16 == off, bitwise (y, conv state, recurrent state) at S 16 / 8 / 1; taken every time. self_test
    PASS.
  - ms per GDN layer at S 16 (36 chained): off 0.3708, NSG 8 0.3605 (-0.0103), **NSG 16 0.3512 (-0.0196): -0.71 ms per
    check**.
  - Offered to BigMoE for the exact bundle: ref2.CFG["gdn_body"] = True + GDNB_NSG=16 (bitwise == REF2-T, so their
    token check is the only gate).

## DENSE-3: the dense projections at 3 bits (MSB-3), quality first (Leap, Wed 2026-11-04 16:5x; the orchestrator's dense diet (b), FAST)
- **Tools (scripts/dense3.py, dense3_check.py; data/dense3_check.txt):** to3(w, s, b) gives MLX's own 3-bit arrays
  (q3 = q4 >> 1, s3 = 2 s4, b3 = bf16(b4 + s4 / 2)), so MLX's bits=3 kernel serves prefill. sim(model) puts the same
  values in today's 4-bit format (codes & 0xEEEEEEEE, biases bf16(b + s / 2)), so a quality screen needs no new kernel.
- **Check (16:52, share lock): PASS on the tools.** MLX bits=3 dequant of to3 == s3 q3 + b3 bitwise; sim == to3
  bitwise (GDN in_proj_qkv / out_proj of layer 4, attention q_proj of layer 7).
- **Quality signal: poor.** 3-bit vs the 4-bit weights: relative rms 0.171-0.179 on the weights and 0.169-0.181 on the
  outputs, for every projection. That's roughly double the 4-bit grid's own noise, on weights every token reads at
  every layer.
- **Prediction:** the map's deviation is far above the FAST line. The expert MSB-3 tiers work because experts are
  router-weighted and averaged over 10; the dense projections aren't.
- **Gate:** NewIdeas' map arm with dense3.sim (no kernel). The 3-bit 16-row kernel is built only if it passes. Bytes
  per check -0.34 GB (GDN + attention), ~-1.2..-1.8 ms if it ever passes.

## BP-10: what's left in the 5.8 ms exact-safe propose (Leap, Wed 2026-11-04 17:0x)
Pre-registered before the run (scripts/bp_d8_ablate.py, data/bp_d8_ablate.txt): D8-NAX + the screened head on qmm16c,
ablated per piece. The stubs consume their inputs, so nothing upstream goes dead (unlike BP-3's no-sdpa).
- Prediction (ms saved of ~5.8): projections 1.5-2.0, SDPA alone 0.4-0.7, convs (with their projections) 0.6-0.9, norms
  0.1, head 0.5-0.7, chain 0.2-0.3, skeleton ~1.
- Use: the largest exact-safe piece gets the next build.
- **Result (16:57 / 16:59, two runs, share lock, GPU ~free; data/bp_d8_ablate.txt):** propose 5.41 / 5.36 ms.

  | removed | ms saved (run 1 / run 2) |
  |---|---|
  | projections (8-bit layers + fc + conv projections) | 2.10 / 1.90 |
  | head (screened, qmm16c) | 0.85 / 0.78 |
  | SDPA alone | 0.70 / 0.62 |
  | convs (with their projections) | 0.46 / 0.35 |
  | norms | 0.05 / -0.03 |
  | chain | -0.04 / -0.12 |
  | all of them (skeleton left) | 3.19 / 3.20 (skeleton 2.23 / 2.16) |

  - BP-10b: the CPU graph build is only 0.38 ms; eval + read-back 5.01 ms. So the ~2.2 ms skeleton is GPU glue: a few
    hundred small kernels (embed, concats, cache updates, RoPE, residuals, transposes, masks), sync included.
  - Left on the drafter (exact-safe, each small): glue fusion ~-0.5..-1, SDPA (additive mask, concat-free K / V)
    ~-0.3, convs ~-0.2. The projections are near the 8-bit byte floor (~0.49 GB, ~1.7-1.9 ms).

## D3-0: is the exact HC read's mix-down byte-bound? (Leap, Wed 2026-11-04 17:1x; HC_READ.md D3, the lossless pack)
Pre-registered before the run (scripts/hc_bw_probe.py, data/hc_bw_probe.txt).
- **Why:** D3 puts MoonShots' lossless exponent-offset pack (~12.4 bits per weight, -22 % bytes, bit-identical decode)
  into the 16-row HC read's loaders: exact. It pays only if the read is bound by its weight bytes. My D1-A0 note
  called S4's walk latency-bound; the read moves only ~160 GB/s.
- **Probe:** leapk's S4 (merged down + inject, N 324 x K 10,240 bf16, 6.6 MB) over 48 distinct weight copies vs the same
  kernel with every tile's B rows pointed at tile 0's (~1/11 of the DRAM bytes).
- **Prediction:** byte-bound share 20-40 % of S4's time. D3's ceiling (-22 % of that share) ~-0.2..-0.5 ms per check
  for the down GEMM.
- **Line:** D3 is built only if its ceiling over both GEMMs is >= 0.5 ms per check. Else KILL D3 without a build.
- **Result (16:59, share lock, GPU ~free; data/hc_bw_probe.txt): BETWEEN.** S4 38.9 us per call (171 GB/s; 48 distinct
  6.6 MB weights) vs 18.7 us with the B rows cached: the byte-bound share is 20.2 us (52 %).
  - D3's ceiling on the down GEMM is -22 % of that: 4.4 us per read, 0.43 ms per check. With the up GEMM (hc_b's
    kernel B, the same bytes) ~0.85 ms per check if the decode were free.
  - The decode adds ALU work in loaders that are already ~half latency-bound, so realistic is ~-0.4 ms per check.
  - It needs custom fragment loaders for MLX's NAXTile (S4) and kernel B. Parked behind larger levers; not built.

## QOM-OOO: one-call OOO aware of QOM's -2 base slots (Leap, Wed 2026-11-04 17:1x; the orchestrator's item 1 for 23:00)
Pre-registered before the run (scripts/ooo1c.py qom forms, ref2 _ooo_1c_call; ooo1c_qom_check.py; data/ooo1c_qom_check.txt).
- **Why:** BigMoE left REF2_OOO_1C out of tonight's all-in arms. Its one-call path ignored QOM's optional uses (posted with
  bit 21, not read on a miss: slot -2), so ~-3.2 ms per FAST check was missing.
- **What:** with t["qom"] on the row-major 3c path, ref2 posts the optional uses with bit 21. Kernel A (QOM form)
  treats a late base slot resolving to -2 as a dropped use: it reads its row's rank-0 slot (polled if late), as the two
  phases' where(kq, sm, sm[:, :1]). It writes keep = 0, and kernel B (bc_qom) takes score 0 for it. Breakthrough's
  rule keeps rank < k0, union members and residents; a -2 can only be an unread optional use, so the drops are the
  same pairs. NAX / native / no-t3 QOM calls keep the two phases.
- **Prediction:** bitwise == the QOM two phases (released before and late), and the non-QOM one-call unchanged.
- **Line:** any differing bit = a bug. PASS -> BigMoE gets REF2_OOO_1C back in the QOM arms (their smoke decides).
- **Result (17:07-17:08, share lock; data/ooo1c_qom_check.txt, ooo1c_wire_check.txt): PASS.**
  - Kernel level (real records3 layers 20 / 44): one-call (qom) == the QOM two phases bitwise in 6 released-before
    calls (5-10 sure rows, 34-68 optional uses, 11-25 dropped; 138 dropped in all) and 2 late releases (~21 ms, no
    timeouts). The non-QOM one-call is unchanged.
  - ref2 wiring (g): QOM through ref2._ooo_sel == the QOM two phases bitwise (49 optional uses, 4 dropped). The post
    carries bit 21 on exactly the optional uses; t["qom"] is consumed. (a)-(f) and self_test still PASS.
  - BigMoE put REF2_OOO_1C=1 into QA / Q8 / QF (their assert became an ooo_1c_qom guard). Their 17:11 smoke of
    tonight's configs: qallf rc 0, ooo_1c_qom 192; the exact lever-2 arm 48 / 48 == REF2-T.
- **Native form (17:15, share lock; data/ooo1c_qom_check.txt): PASS.** The QOM form on 3c-N (tier_glu_3cn_1c(qom=True):
  the same kernel-A splice on 3c-N's source + bc_qom on 3c-N's bc). On layer 20 converted to bf16-native and
  native+p11+pad pools it is == the row-major QOM two phases, bitwise, in 6 scenarios (94 drops). ref2 routes
  QOM + native to it. QOM is refused loudly where only a two-phase path would serve it (NAX; native with ooo_1c off
  or declined). Wiring (a)-(h) PASS.

## BP-11: drafter glue (Leap, Wed 2026-11-04 17:0x-17:1x; the orchestrator's item 3)
- **BP-11a census (data/bp_census.txt):** the exact-safe propose (D8-NAX + screened head) launches 234 kernels (205 in
  hidden_states): CustomKernel 79, Concatenate 36, RMSNorm 27, Gather 26, RoPE 15, Full 13, Add 11, Multiply 5, SDPA 5,
  the compiled SiLU-mul 5, others 12. 22 of them (11 Full + 11 Concatenate) are qmm16c8's padding of the 8-row
  context calls (k / v x 10, fc).
- **BP-11b (qmm16c8 row bound M_C, no padding pass):** rows < M bitwise == the padded call (6 shapes x M 8 / 1 / 16 x ks
  1 / 2; data/qmm16c8_check.txt).
  - Propose A/B, interleaved: row bound 5.99 vs padded 5.83 ms. INCONCLUSIVE: BigMoE's 17:11 smoke had the GPU ~70 %
    busy beside it, and the gap is within that noise. Kept (bitwise, 22 fewer kernels); re-time when the GPU is free.
- Left: q / k norm + RoPE fusion (~15 kernels), concat-free K / V (~10 concats + ~42 MB of copies), the remaining
  concats / gathers. ~0.3-0.6 ms in all; timing needs a free GPU.

## SESSION-VOCAB: the r57 hook harness for SVH + BP-9 (Leap, Wed 2026-11-04 17:2x; the orchestrator's route (b))
- **What:** one object (scripts/session_vocab.py, SessionVocab) feeds the session's token ids to both screened heads:
  SVH (ref2 CFG screened_head, the verifier's head; FAST) and BP-9 (d7b_nax.DraftSubset, the drafter's head;
  exact-safe). The proposed r57 diff (data/r57_session_vocab.diff, made on a copy of BigMoE's current r57_screen.py;
  their repo untouched) has 4 hooks: SV built after the HEADC block (R57_SVH / R57_DSUB / R57_SV_CHECK, all off by
  default); head2s delegates to SV when R57_DSUB=1 (supersedes R57_HEADC); SV.new_prompt(ids) before prefill;
  SV.committed([tok]) after the first token and SV.committed(new) after `out += new`. head2s is the only drafter-head
  call site in r57 (draft17 has none).
- **Line:** (a)-(c) and (e) true; B0 == A in every window; B / A within x0.995-x1.005 (pre-registered).
- **Result (17:24-17:25, share lock, 62 s, mlx peak 2.35 GB; data/session_vocab_check.txt): PASS.**
  - (a) both subsets hold the same extras, in the same order, after a prompt, a commit, an empty commit, the 4,096 cap,
    past the cap, and a new prompt (which drops the last one's).
  - (b) SV.head2s == screened_head2s bitwise. The drafter's head sees P = 15 rows per 16-position block (positions
    1..15), so R57_D17 gives 16 rows, not 17 (corrected d7b_nax's comment). 16 rows: row 15 == row 3's bits; 18 rows
    (the generic > 16 chunk path) == 16 + 2-row calls. Every candidate is inside the subset and finite.
  - (c) REF2 on: the trunk head at 16 rows is screened (finite on exactly the 98,308 subset ids, counted once);
    17 rows take the full head; REF2 removed: the plain head, uncounted.
  - (d) BP-9's walk: kept per check code 9.290 / math 6.671 for A (BP-9's own feeding), B0 (SessionVocab, A's
    feeding: == A in all 84 windows) and B (r57's hook order incl. the first token: x1.0000 both). Check mode: the
    full head's top-1 fell outside the subset in 2 of 42,480 draft rows. committed() costs 0.014 ms per check (1 of
    2,832 checks gathered new rows: almost every token is in the 98,304 base).
  - (e) ref2.self_test PASS with SVH on; the switch restored off.
- **Hand-off:** BigMoE applies the diff, then smokes one prompt with R57_DSUB=1 R57_SVH=1 R57_SV_CHECK=1 and prints
  SV.report(): expect svh_calls ~ checks + 1, head2s == proposes, chk_out ~0, dsub_extra == svh_extra.

## DSDPA: split-K attention for the drafter's block (Leap, Wed 2026-11-04 17:3x; drafter glue)
- **Prior art:** Flash-Decoding (split the keys over threadgroups, merge with log-sum-exp; Dao et al. 2023), published.
  Here it targets the drafter's 16-row GQA block: q 32 heads x 16 rows, K / V 8 heads x (ctx_len + 16), head_dim 128.
  The mask is the cache rows inside the 2,048 window | the whole block. Overlap ideas (PEARL / CPUMTP) are KILLs
  owned elsewhere.
- **DSDPA-0 probe (17:30, share lock; data/dsdpa_probe.txt):** MLX's SDPA, 5 layers: 0.46 ms at ctx_len 1,008 and
  0.83 ms at 2,054 (~64 GB/s for 8.5 MB of K / V per layer: latency-bound).
  - An additive mask is **bitwise == the bool mask** and 4 % / 16 % faster.
  - Two 8-row calls: 3x slower. GQA-packed matmuls: 1.1-1.2x slower.
- **DSDPA-1 v1 (scripts/dsdpa.py):**
  - Kernel A: one threadgroup per (kv head, key chunk), 4 simdgroups = the group's 4 q heads. S = QK^T and O += PV on
    MPP matmul2d 16x32x16 with qmm16c's fragment maps. P's row-sum reduction is xor 1 / 8, and P's C fragment *is* the
    next A fragment (the K permutation is free), so there is no shuffle.
  - fp32 online softmax; P in bf16, summed from the rounded values. K / V rows come from the cache and block buffers,
    so there is no concat. In-kernel mask.
  - Kernel B merges the chunks and writes o_proj's [1, 16, 4,096] directly.
- **Line (pre-registered):** <= 4 ulps everywhere; best nc <= 1/2 of the MLX path (concat + bool SDPA + transpose) at
  ctx 2,054 -> walk.
- **Result v1 (17:36, share lock; data/dsdpa_check.txt):**
  - Accuracy PASS: <= 1.00 ulp at ctx_len 1 .. 2,062 (s 1 / 3). Adversarial window rows: 1.00 ulp (a leaking mask:
    464 ulps).
  - Timing MISSED the line:

    | arm (ms per 5 layers) | ctx 1,008 | ctx 2,054 |
    |---|---|---|
    | MLX path (bool mask) | 0.648 | 1.175 |
    | MLX path (additive mask) | 0.618 | 1.031 |
    | dsdpa nc 4 (best) | 0.444 (x0.69) | 0.655 (x0.56) |
    | dsdpa nc 8 | 0.469 | 0.709 |
    | dsdpa nc 16 | 0.575 | 0.783 |

  - Read: ~4.7 us per 32-key step, and more chunks make it slower. Each step waits ~16 times on scattered device loads
    (latency), with low occupancy (~190 registers per thread). Next: v2 stages each K / V tile in threadgroup memory
    with coalesced 16-byte loads by all 128 threads.
- **Result v2 (17:39, same share lock; data/dsdpa_check.txt, second block).** Recorded and committed at 23:3x: it was
  left out when MISSION 33 started at 17:4x. The code is unchanged since the run (mtime 17:39:13, run 17:39:37).
  - v2 stages each 32-key K / V tile in threadgroup memory (coalesced loads by all 128 threads); it is attend's
    default.
  - The run was more contended: the MLX baseline was 1.5x slower at ctx 1,008 than 3 minutes earlier.
  - Accuracy PASS: <= 1.00 ulp at every ctx_len / s; adversarial window rows 1.00 ulp.
  - Timing still MISSED the line, by a hair at 2,054:

    | arm (ms per 5 layers) | ctx 1,008 | ctx 2,054 |
    |---|---|---|
    | MLX path (bool mask) | 0.991 | 1.252 |
    | MLX path (additive mask) | 0.923 | 1.101 |
    | v1 nc 4 | 0.680 | 0.704 |
    | v2 nc 2 (best) | 0.627 (x0.63) | 0.634 (x0.51) |
    | v2 nc 4 | 0.655 | 0.670 |
    | v2 nc 8 | 0.686 | 0.709 |
    | v2 nc 16 | 0.749 | 0.749 |

  - Read: v2 is 8-10 % faster than v1 in the same run.
  - x0.51 against the x0.50 line comes from a contended run; only a quiet re-time would settle it.
  - PARKED. DKV (PASS, handed to BigMoE) already removes the concat copy that is part of this baseline, so DSDPA's
    gain over DKV + MLX's SDPA is smaller: roughly 0.2 ms per propose at 2,054 (DSDPA-0's SDPA alone 0.83 ms vs v2's
    0.63 ms; different runs, an estimate).

## DKV: a copy-free drafter window cache (Leap, Wed 2026-11-04 17:4x; drafter glue, exact)
Pre-registered before the run (scripts/dkv.py, scripts/dkv_check.py, data/dkv_check.txt).
- **Why:** each propose copies every layer's whole K / V window twice: mlx_lm's RotatingKVCache._update_concat
  (temporal order + trim + append = a concat), then the attention's concatenate([keys, block keys]). DSDPA-0 / -1:
  the second copy alone costs ~0.19 / 0.35 ms per propose at ctx 1,008 / 2,054.
- **What:** LinRotCache (a RotatingKVCache subclass) keeps the window in temporal order in a preallocated linear
  buffer.
  - Multi-token updates: trim the front exactly as _update_concat does, then write [context rows, block rows] in one
    in-place slice update. Old views are dropped first, so MLX donates the buffer. SDPA reads the view [a, b + L):
    the same rows in the same order as the reference's concat.
  - 1-token updates (A6 dwin's first propose; k = 0 checks) run the base class's own code. The next multi-token update
    adopts its temporal order (one copy, as the reference's).
  - Buffer end: compaction, one copy per ~(cap - window) / S proposes.
  - DF._trim_recent_cache trims the window in place.
  - Optional: the additive mask (bitwise == bool, DSDPA-0).
- **Prediction:** bitwise == the reference everywhere (it presents the same values in the same order). -0.3 ms per
  propose at 1,000 context rows; -0.5..-0.7 ms at 2,047 rows; +0.03..0.14 with the additive mask.
- **Line:** any differing bit = a bug. Hand to BigMoE as an exact lever if dkv <= ref - 0.25 ms at 2,047 rows; else
  KILL.

## MISSION 33 (K90): the expert kernel to ~90 % of streaming (Leap builds, Fable A thinks; Wed 2026-11-04 18:0x-23:00)
Kamen approved ~17:4x; brief ~/FableGod/MISSION_33_KERNEL90.md; spec ~/FableGod/KERNEL90.md section 1 (arms 0-8, lines
pre-registered by Fable A). Harness: scripts/k90.py (layer 20's real records3 + the native repack, real routing traces
code / chat / math / bulgarian, B = record bytes counted from the routing + x / h / y), scripts/k90k.py (the arms'
kernels), timing = the whole call + combine in dependent chains of 16, interleaved, medians of 15.
- **Arms 0-3 (18:11, share lock, GPU contended by Breakthrough's map batch: relative numbers; clean repeat after 19:05;
  scripts/k90_arms.py, data/k90_arm012.txt):**
  - Bits: NAX6-N == NAX6 96 / 96. NAX-CN == NAX-C (row-major) 96 / 96: NAX-CN = NAX-C's core with NAX6-N's 9 address
    swaps, an exact port. NAX-CN INV 96 / 96.
  - Arm 0 STREAM = NAX6-N's kernels A and B with only their loads (codes, LSB words, scales, biases; the same prologue,
    grid and load instructions; xor-folded). R = 215-230 GB/s, contended.
  - Shares of R: NAX6-N 73-77 %, NAX6 58-60 %, NAX6-NH 73-80 %, NAX-CN 70-76 %. At S16, NAX6-N runs 11.7-12.1 us per
    group vs STREAM ~9.0.
  - **Arm 3 NAX-CN (int8 x int4 on native): KILL** (line PASS >= +8 %, KILL < +3 % time saved at S 16). Time saved vs
    NAX6-N: S16 -0.2 / -3.3 % (code / chat; mean -1.7 %), S8 -4.6 / -4.2 %, S32 -2.1 / -2.1 %. The int8 op rate does not
    bind on native either, so arm 5 (NAX-CH) is dropped.
- **Arm 4 NAX8-P1, persistent kernels A + B (18:16, share lock, contended; scripts/k90_p1.py, data/k90_p1.txt): KILL.**
  - Bits: 96 / 96 == NAX6 at RES 3 / 4; every other RES pair 6 / 6.
  - Every config is SLOWER than NAX6-N, monotonic in residency. Time saved at S16 (code / chat): RES_A / RES_B 2/3
    -44 / -56 %; 3/4 -21 / -24 %; 4/6 -0.6 / -9.9 % (best; mean -5.2 %, line KILL > -2 %). S8 and S32 the same shape.
  - Read: the normal grid's launch hides each threadgroup's serialized item fetch + prologue chain (4-5 dependent
    loads) behind other resident threadgroups. A few looping threadgroups per core cannot.
  - Fix on the way: MLX binds inputs of < 8 elements in the constant address space (no atomics there), so the counters
    are a 32-element array.
- **Probes 0b / 0c (Fable A 18:1x; 18:20, share lock, contended; scripts/k90_probe0.py, data/k90_probe0.txt):**
  - 0b: STREAM with U = 2 / 4 quant groups of loads in flight per simdgroup vs U = 1. At S16: u2 -1.6 / +3.2 % (code /
    chat; mean +0.8 %, line arm 9 if >= +8 %); u4 -4.6 / +0.4 %. S8 / S32: -2..-10 %. Not concurrency-bound: **no arm 9.**
  - 0c: persistent STREAM (P1's loop around the loads-only body) at RES 3/4, 4/6, 8/12 vs the plain grid: best -9.5..+4.0
    %. No structural headroom on the load side.
  - Extension (not pre-registered): P1 at RES 6/8 and 8/12 is still slower than NAX6-N (-1.5..-11.5 %).
  - STREAM's own R (grouping inside the call here): 204-240 GB/s, contended. Arms 6 (P2, needs arm 4 <= -4 %) and 9 are
    off.
- **Arm 7 NAX8-GU (gate + up per simdgroup; kernel A, 4 simdgroups = the K chunks, the A operand marshalled once per op
  pair; 18:25, share lock, contended; scripts/k90_arm.py gu, data/k90_arm_gu.txt): KILL.**
  - Bits 96 / 96 == NAX6.
  - -23..-31 % slower at every width (S16: -23.1 / -29.6 %, mean -26.3 %; line PASS >= 4 % saved, KILL < 1 %).
  - Halving the simdgroups doubles each one's serial decode and op chain. The marshalling saving is small beside it,
    consistent with the call being latency-bound per simdgroup.
- **Arm 8b (kernel A's left operand as a tensor<device T> view of the check's 16 x rows, an x row -> pair map for the
  stores; S <= 16; scripts/k90_arm.py 8b, data/k90_arm_8b.txt): KILL.**
  - Bits 64 / 64 == NAX6 (S 8 / 16): the MPP API takes a device-tensor left operand, and INV holds.
  - -1.0..-15.1 % (S16 -1.3 / -1.0 %, mean -1.1 %). The tensor-operand load path is no cheaper than the register fill.
- **Probe MUL-first (both kernels: the first op of each quant group in mode::multiply, no cg zero-writes; 18:26, share
  lock, contended; scripts/k90_arm.py mf, data/k90_arm_mf.txt): KILL.**
  - Bits 96 / 96 == NAX6: the unit's multiply mode equals accumulating onto 0, bitwise.
  - -4.0..-11.7 % slower (S16 -5.7 / -4.0 %, mean -4.9 %).
- **Arm 8a skipped (reasoned):** a threadgroup A tile shared by the gate and up simdgroups of a K chunk needs a
  threadgroup barrier per quant group (writer -> readers; Metal has no pairwise simdgroup barrier). That is N7-5's
  lockstep, a KILL, and 8b showed that a tensor-operand A path is no faster than the register fill.
- **QUIET set (18:29:54-18:30:59, a queued quiet lock in a gap between map arms, arranged by the orchestrator; data/
  bw_probe_k90.txt, data/k90_arm012.txt "QUIET", data/k90_quiet.log):**
  - bw_probe: the 4 GB read kernel 285-286 GB/s (best 292); 1-2 GB 270-277.
  - **R (STREAM, NAX6-N's own loads) 287-305 GB/s.** S16: 296 (code) / 295 (chat). This is >= bw_probe's machine rate:
    NAX6-N's gathered native access pattern streams at the full rate. Its B / t can exceed DRAM slightly, because B counts
    each record once while 4-bit and 3-bit groups re-read shared pieces from cache.
  - Shares of R, quiet:

    | arm | S8 | S16 | S32 | note |
    |---|---|---|---|---|
    | NAX6-N (exact) | 72 % | 74 % (218-219 GB/s) | 70-72 % | S16 = 77 % of bw_probe's 285 |
    | NAX6-NH (FAST) | 76-77 % | 79 / 81 % (234-238 GB/s) | 76-79 % | |
    | NAX6 (row-major) | 56-57 % | 58-59 % | 55-57 % | |
    | NAX-CN | 69-70 % | 72 % | 68-71 % | KILL confirmed quiet, -2.4 % |

  - S16 per group: NAX6-N 9.1-9.2 us, STREAM ~6.8 us.
  - Read: the load side runs at the machine's rate; the whole gap to 90 % is exposed compute on top of the loads.
- **Decode LUT (bitwise by construction: spread3's two 12-bit halves are independent, so spread3(x) = T[x & 0xFFF] |
  T[x >> 12] << 16; QUIET; scripts/k90_arm.py lut / lutd): KILL, both.**
  - Bits 96 / 96 each.
  - lutd (the table as a device input, via L1): -1.3 / -0.8 % at S16.
  - lut (per-threadgroup 8 KB table, filled at start): -14.5 / -15.6 % (the fill + bank conflicts).
- **Reusable fact (MUL-first):** matmul2d mode::multiply == mode::multiply_accumulate onto a zeroed destination,
  bitwise (96 / 96).
- **Probe PF-6 (the next quant group's six code words into registers before the current decode, distance 1, nothing
  else moved; 18:32, share lock, contended; scripts/k90_arm.py pf6, data/k90_arm_pf6.txt): KILL.** Bits 96 / 96.
  S16 +0.6 / -0.5 % (mean +0.0 %; line PASS >= 5 %, KILL < 1 %); S8 -4.2 / -3.8 %, S32 -2.7 / -4.1 %. The compiler's
  schedule already overlaps the code loads.
- **Mission 33 builder summary (18:33):** 10 bitwise variants of NAX6-N's loop and structure, all slower or equal:
  persistent (P1), gate + up per simdgroup (GU), a device-tensor A (8b), MUL-first, two decode LUTs, prefetch (PF-6), plus
  the int8 form (NAX-CN) and the load probes 0b / 0c. The load side streams at the machine's rate (R 287-305 GB/s vs
  bw_probe 285). NAX6-N stays the exact kernel at 70-74 % of R (S16 77 % of bw_probe), and NAX6-NH (FAST) is at 76-81 %.
  Fable A's verdict: ~/FableGod/KERNEL90.md.
- **Result (19:03, share lock, GPU ~3-6 % free beside Breakthrough's map batch; data/dkv_check.txt): exactness PASS,
  timing INCONCLUSIVE.**
  - (a) Stress on one real attention layer: 726 steps (6 prompts, 63 trims), outputs != reference in 0. Every path ran:
    fast 622, slow (1-token) 104, adopt 93, compact 1, trim 56.
    - Deviation from the pre-registration: r57's start offsets (0 for short prompts). The first run's random preset
      offsets drove the REFERENCE into its own crash: a 1-token update cannot grow a non-full buffer past the window,
      the a6w_160 case r57 avoids.
  - (b) BP-9's walk: per-propose hashes bitwise == the reference for dkv AND dkv + additive mask (code 686 proposes,
    kept 9.290; math 2,146, kept 6.671).
  - (c) ms per propose, median of 3 rounds (single rounds swing 7-16 ms under the contention):

    | context rows | ref | dkv | dkv + add | mins (ref / dkv / dkv + add) |
    |---|---|---|---|---|
    | 1,000 | 10.41 | 8.13 (-2.28) | 8.79 | 7.56 / 6.66 / 7.43 |
    | 2,047 | 8.94 | 8.70 (-0.24) | 11.95 | 8.32 / 7.59 / 7.17 |

    The line is -0.25 ms at 2,047 rows. Re-time on a free GPU before handing it to BigMoE.

## TRELLIS FULL-STORE PREP (Leap, Wed 2026-11-04 20:0x-; the orchestrator 20:0x, Kamen wants today's usage used)
Goal: be ready if tonight's screen 7 passes (NewIdeas' corrected estimate: code 1.13-1.21 % for the mix, line 1.34 %).
(1) a Metal TRL encoder for all 24,576 experts (the 2.40 b mix first, then TRL-3), bit for bit == Fit12's C encoder on
layer 18 (Fit12's store = the reference); size the 48-layer Hessian capture. (2) the decode path for serving: k 3 and
the engine wiring are new; pre-register the speed line. No full-store disk writes (31-36 GB is Kamen's call after the
screen); GPU tests small, on share locks.
- **OUTCOME (Wed 23:3x): screen 7 KILLed the mix and the trellis family STOPS, so no full store gets built.** From the
  owners' records (~/TRIED/newideas.md):
  - S7-TRL (NewIdeas): in-frame flips code / IFEval / multi 1.45 / 9.54 / 10.44 %. Pre-registered verdict: code over
    its line, chat KILL x2; re-derived by Breakthrough 8a473b0.
  - TRL-JUDGE-MAP (NewIdeas 22:49, on Fit12's routed judge): the inputs' distribution does not explain the miss
    (whole-expert error ratio <= 2.0 on every domain), so the family stops by the pre-registered rule.
  - Kept as tools: the bitwise Metal encoder, TRL-NAX (0.87x NAX6-N), and the trl-v1 store + runbook.
- **Prior art:** QTIP (Tseng et al., NeurIPS 2024) encodes with a Viterbi per tile on CUDA GPUs and decodes with
  computed codes (1MAD / 3INST) in fused matvec kernels. ik_llama.cpp's IQ*_KT have a Metal decode port. Not found: a
  Metal Viterbi encoder, or a trellis decode grouped per expert inside a routed MoE GLU. Tag: published method; the
  Metal encoder and the grouped decode are new builds.

### (1) Encoder, pre-registered 20:1x
- **Bit-for-bit target:** fit_trellis.dylib's Viterbi. Its arithmetic as compiled (otool: no fadd at all, 4 fmadd +
  5 fmla): t = 0 cost = d * d; t > 0 cost = fma(d, d, best[j]); min over the 2^k predecessors with strict < (lowest h
  wins ties); final state = the lowest among the minima. LUT3 has 3,726 distinct values in 4,096 states, so ties are
  real; the kernel copies the tie rules.
- **Line:** every tile == the C output bitwise (random, Laplace, tiny-scale, all-zero, constant and exact-LUT tiles,
  k 2 and 3), and Fit12's pipeline with only the Viterbi swapped reproduces layer 18's store bit for bit (all 1,536
  8-bit views). Anything else is a bug.

### (1) Encoder result (20:16-20:4x; share locks, GPU shared with MTP Head's P-CAP at 90 %)
- **Viterbi kernel (scripts/trl_metal.py):** PASS. 0 of ~2,000 tiles differ from Fit12's C at k 2 and at k 3,
  including the tie-heavy cases (data/trl_metal_test.txt). Valid trellis paths; the bitstream packs and unpacks
  exactly. NMSE on N(0, 1): 0.0710 at k 2 (Fit12's B3 C test: 0.0711), 0.0193 at k 3.
- **Speed (data/trl_metal_bench.txt; GPU shared):** 194 k tiles / s at k 2 and 246 k tiles / s at k 3 in calls of
  2,560 tiles. Fit12's C on 4 cores: ~1.5 k tiles / s (4.2 s per 6,400-tile matrix). So ~130x.
- **Layer 18 end to end (scripts/trl_l18_check.py):** Fit12's own code (modules imported read-only, CPU default
  device, 4 BLAS threads as its runner), with fit_b3.vit swapped for the Metal Viterbi.
  - H_x recomputed == Fit12's cached H_x, bitwise.
  - Expert 0's down view differed at first. Cause: e0 / e1 came from the 14:22 smoke, before fit_trl_store.py's
    14:3x change of H_h from fp64 to fp32. With H_h in fp64, e0 matches 3 / 3 (e1 matches either way). The
    reference store is mixed on that point; harmless.

### (2) Decode path, pre-registered 20:3x (before any kernel exists)
- **Kernel (scripts/tierglu_trl.py, TRL-G):** a trellis-decode GLU grouped per distinct expert, like NAX's grouping
  (gpu_grp; no tiers in the mix).
  - Lane = output row. Each lane walks its row's 16 x 16 tiles, extracts each 12-bit state from a 64-bit window
    (k 2: words r, r + 1; k 3: words 3r / 2 .. + 2), computes 3INST, and serves up to 4 rows of the group per pass.
  - Rotations: x' = H512(s_L x) once per token; h' = H128(s_e h) per pair (mx.hadamard_transform).
  - Store as trl_encode.py writes it: k 2 tiles in 17 words, k 3 in 25, bf16 row scale; the decode multiplies by
    inst3(s) / RMS3.
  - The format and the kernel need no LUT.
- **Correctness:** real layer-18 codes (trl_encode.py, experts encoded in-process) vs a float64 numpy forward of the
  same codes (dense Q = LUT3[s] x sc, the same rotations). Line: max |dy| <= 1e-2 of max |y| per call (fp32 accumulation;
  bf16 x).
- **Speed line:** TRL-G (mix, rotations and combine included) vs NAX6-N (layer 18 records3, native, plan 4433333333),
  per S 16 call on layer 18's code / chat routing (traces125). Resident pools (random streams for TRL; the decode cost
  does not depend on the values). Dependent chains of 16, interleaved medians (k90's method), S 8 reported too.
  - PASS <= 1.00x NAX6-N: TRL serves at no kernel cost. Its byte saving is free at 24 / 12 GB, and 45 GB is not slowed.
  - BETWEEN 1.00-1.25x: a gain only where SSD bytes dominate (24 / 12 GB). NewIdeas prices it.
  - KILL > 1.25x: at 24 GB the decode would eat what the store saves. Assumes ~12 MB of SSD saved per kept token
    (NewIdeas' replays: mix 7.2 vs MSB-3 FAST 19.5 MB), ~1.1 ms at ~11 GB/s, against ~4.5 ms of expert GLU per kept
    token at NAX6-N speed (0.65 ms per call x 48 layers / ~7 kept per check), so break-even is ~1.24x.
- **Prediction:** 0.8-1.0x. B10(b)'s per-pair 3INST kernel decoded ~713 G weights / s with one FMA per weight. Grouped,
  each weight is decoded once per distinct expert (~62 per S 16 call, ~2.6 rows each): 305 M decodes ~0.43 ms + the
  extra FMAs ~0.1 ms, vs NAX6-N 0.65 ms (quiet).

### (2) Decode correctness (20:28-20:31; data/tierglu_trl_test.txt)
- TRL-G's fp32 output per pair == the float64 numpy forward of the same streams within 4-7e-7 of max |y| (rms 4e-7).
  The reference uses TM.unpack_states, a bit-level unpack independent of the kernel's 64-bit windows. Covered: k 2 / 3
  (the mix), k 3 / 3 (TRL-3) and k 2 / 2, with 1, 3 and 16 tokens. In the 16-token case experts are shared, giving groups
  of up to 16 rows (4 passes). PASS.
- Deviation from the pre-registered tolerance: I set "<= 1e-2 of max |y|" on the engine path (bf16 per-pair y + bf16
  combine_phase, as NAX6-N's harness). That path reads 0.6-1.24e-2 against float64, i.e. bf16 rounding (2^-9 per
  rounding, summed over 10 ranks), and it fails the 1e-2 line in 4 of 9 cases. The line was mis-set for a bf16 output.
  The decode is judged on the fp32 path (5e-7), which separates a decode bug (O(1)) from output rounding.

### (1b) The 48-layer capture, sized (20:3x; from Fit12's B10-r capture: scripts/b10r.sh + fit_b12.py b10_cap, read only)
- **What:** the MoE input x at every layer for the fit rows R0..R7 (8 parts x 4,096 tokens: math / code / instr /
  multi / wiki / know / agent, round-robin; fit_b10r_text.py). R8 / R9 (validation, for the damping search) and U
  are not needed with fixed damping.
- **Disk:** 4,096 x 2,560 x 2 B = 21 MB per part per layer, so 168 MB per layer and **8.05 GB for 48 layers** (+31 MB
  of ids). Temporary: delete after the encode.
- **GPU:** one prefill of 32,768 tokens; the capture adds only copies. B10-r captured 11 parts (45,056 tokens) at 9
  layers in 140 s (Oct 25), so ~100-150 s at 48 layers (Fit12: ~5 min with margin). It needs a quiet lock (the
  engine's prefill), with b10r.sh's SSD (<= 66 C to start, stop at 73 C) and swap guards. Command: F12_CAP_LAYERS =
  0..47, F12_CAP_TEXTS = R0..R7.
- **RAM:** the capture engine (fit_b12.py: F12_V3F store, BIGMOE_JOINT) + ~1 GB of x per part before writing.
- **H_x per layer:** X^T X in fp64 (429 GFLOP) + inverse + Cholesky (2,560), ~3-5 s per layer on 4 cores, so
  ~3-4 min for 48 layers.
- **Privacy note:** the "agent" domain (1 of 7) is Kamen's own Claude Code sessions. The codes carry only a
  Hessian-weighted imprint of them, but if a TRL store is ever published, that calibration choice is Kamen's call.

### (2b) Engine wiring (plan; nothing in BigMoE's repo is touched)
- **Store (per layer):** one file of 512 fixed-size records. Mix record:
  - gate words 435,200 B + up 435,200 B (40 x 160 tiles x 17 words);
  - down 640,000 B (160 x 40 x 25 words);
  - bf16 scales 7,680 B; down signs 640 B (int8; or regenerate from seed 1000 L + e);
  - total 1,518,720 B, padded to 16 KB pages: 1,523,712 B. **37.5 GB for 48 layers.** TRL-3: 1,928,320 B ->
    47.5 GB. (Fit12's tight-bit sizes 36.2 / 46.7 GB do not pad tiles to words; the kernel's aligned 64-bit windows
    need the padding, +2.9 %.)
- **index.json:** per layer E, recsz, piece offsets, k per projection, layout "trl-mix-v1"; the reader refuses any other
  layout.
- **Pools (per slot):** wg / wu [C, 40, 160, 17], wd [C, 160, 40, 25] u32; scales; down signs. One pread per expert,
  as records3.
- **Call:** tier_glu_trl(x, P, slot) in place of tier_glu_nax6n for TRL layers. The x' rotation (per layer) and the h'
  rotation (per pair) live inside it; the combine is unchanged. A new CFG switch defaults OFF. A held ref2 diff (like
  r57_dkv.diff) comes after screen 7 and the speed verdict.
- **Amendment 20:4x (before any timing):** a second kernel-A variant, A2. One lane computes the gate row and the same
  up row, sharing the x' loads; SwiGLU stays in registers; bitwise == A in every test case. The S 16 verdict uses the
  better of A and A2 per trace (both defined before measuring). TRL-3 is reported, not judged.
- **Converted-store reader (20:4x):** trl_encode.read_record reads a native+p11+pad layer through BigMoE's
  ext.record_rowmajor and parses it as fit_q.read does. 8 / 8 experts of BigMoE's padded scratch copy (layer 20) ==
  fit_q.read on today's records3. So the encoder runs on the store before or after Thursday's conversion.

### (1) Encoder result, final (20:4x)
- **Layer 18, all 1,536 matrices, bit for bit == Fit12's store:** PASS. Experts 2-511: 1,530 / 1,530
  (data/trl_l18_check_2_511.log, 0 DIFF; two heat pauses on the way). Experts 0-1: 6 / 6 with H_h in fp64
  (the smoke's code).
- **TRL-3 (k 3 everywhere; Fit12 has no TRL-3 files):** the Metal path == Fit12's own C path in-process, 6 / 6 matrices
  (experts 5-6; TRL_FORM=trl3 TRL_VS_C=1; the log line's "== Fit12's store" means the C path in this mode). In-sample
  error vs MSB-3: gate 0.33-0.36x, up 0.34-0.37x, down 0.50-0.54x.
- **Batched encoder (scripts/trl_encode.py):** --check decodes the PACKED CODES (unpack -> LUT3 x bf16 scale ->
  Q R^T -> the 8-bit view), and 48 / 48 views == Fit12's store. So the store format round-trips exactly.
  - Speed without the check: 0.642 s per expert (32 experts, batch 16, one process, 4 BLAS threads). Per expert:
    prep (read + H_h + W R) 0.26 s, BlockLDLQ feedback 0.35 s. Viterbi wait 0.0 s: the GPU part is hidden; the CPU
    fp64 code that keeps the codes bitwise sets the pace.
  - **Full store (24,576 experts):** 4.4 h on one 4-thread process; ~1.1-1.5 h with 3-4 processes (12-16 cores, a
    night slot). Plus the capture (~2-5 min GPU, quiet) and H_x (~4 min CPU).
  - A FAST variant (feedback and H_h in fp32 on the GPU) would take ~0.1 s per expert (~40 min), but it is not
    bitwise == the screened encoder. Not built: the bitwise one fits a night.
- **Disk (df 20:4x):** 126 GB free; the floor is 110-112 GB (Thursday's conversion needs 112). The mix store (37.5 GB)
  needs ~25 GB freed first; TRL-3 (47.5 GB) ~35 GB.
- **Layer 30, H_x computed by the encoder (no cache; 1.2 s):** 48 / 48 views of experts 100-115 == Fit12's L30 store,
  bit for bit. So the per-layer path the full run uses is covered as well as the cached one.

### (2c) Store format + the serving chain on real codes (20:4x; scripts/trl_store.py)
- **Format trl-v1:** one file per layer of fixed records (16 KB-aligned) + index.json. Pieces: gate / up / down words,
  bf16 row scales, the expert's int8 h-signs. Layer constants: k per projection, x-sign seed L, Hadamard 512 / 128.
  Mix record: 1,518,720 B payload, 1,523,712 B on disk.
- **Test (PASS):** 16 real L18 experts, through encode (the bitwise pipeline) -> write_layer -> load_pools -> TRL-G
  (fp32). Against the dense float64 path W = Q R^T (y = W_d (silu(W_g x) * W_u x), fit_b3.rot's R) on real calibration
  inputs: max |dy| / max |y| = 6.1e-7. Against the float64 forward of the loaded pools: 4.8e-7. So the signs, the
  Hadamard order, the code format, the scale semantics and the kernel agree end to end. (The 24 MB test layer is in
  data/trl_store_test/, not committed.)
- **Amendment 20:5x (before any timing):** rows per pass became a template parameter. Layer 18's traces at S 16 have
  76-82 distinct experts per call, with 25-30 % of pairs in groups larger than 4, so 4 rows per pass re-decodes +9-11 %
  and 8 rows +1 %. Third variant: A2 at 8 rows (bitwise == A). The verdict takes the best of A / A2 / A2-rp8 per trace.

### (2) Decode speed RESULT (20:53-20:55; queued quiet lock, 18 s of GPU; data/trl_bench.txt): **KILL (1.41x)**
| per call, layer 18 | NAX6-N | TRL mix A | A2 | A2 rp8 | TRL-3 (A) |
|---|---|---|---|---|---|
| code S 16 | 0.809 ms | 1.314 (1.62x) | 1.460 (1.80x) | 1.833 (2.27x) | 1.484 (1.83x) |
| chat S 16 | 1.068 ms | 1.278 (1.20x) | 1.476 (1.38x) | 1.948 (1.82x) | 1.439 (1.35x) |
| code S 8 | 0.524 ms | 0.691 (1.32x) | 0.744 (1.42x) | 0.983 (1.88x) | 0.775 (1.48x) |
| chat S 8 | 0.620 ms | 0.731 (1.18x) | 0.821 (1.32x) | 1.064 (1.72x) | 0.854 (1.38x) |
- **Verdict (S 16, best variant per trace = A, mean of code / chat): 1.41x, KILL** (line: PASS <= 1.00x, KILL > 1.25x).
  Prediction 0.8-1.0x: wrong.
- **Geometry sweep (informational, same quiet lock; data/trl_sweep.txt):** 8 settings of threadgroup size and rows
  per pass, all 1.44-1.59x on code and 1.13-1.23x on chat. The best (2 / 4 simdgroups, 4 rows) is ~1.29x. Geometry is
  not the problem.
- **Read:** TRL-G is near-flat across traces (1.28-1.31 ms) and decodes only ~330 G weights / s (~400 M decodes per
  S 16 call incl. re-decodes). B10(b)'s per-pair TA reached 713 G / s. The difference is the mapping:
  - TA: lane = tile column, word-interleaved layout (coalesced word loads), each lane's 16 x values loaded once and
    reused across RA output rows.
  - TRL-G: lane = output row, a broadcast float4 x load per (row, tile column), two 68 B word segments per simdgroup.
  - A2 and 8 rows per pass are slower: more registers per lane, lower occupancy.
- **Next (a new bet, not run):** TRL-TA, a grouped kernel on TA's mapping. Lane = tile column, words interleaved
  per row block (a store-layout change: trl-v2, the encoder repacks), each group row's x segment kept in registers
  (rp x 16), one decode per (output row, tile column) served to rp rows.
  - Prediction: ~600-700 G decodes / s, i.e. ~0.6-0.7 ms per S 16 call, ~0.7-0.85x NAX6-N.
  - Same line as here.
- **What this means for the store:** quality (screen 7) and bytes are unchanged. The serving kernel as built would
  cost ~0.3-0.5 ms per layer call more than NAX6-N at S 16, which at 24 GB eats more than the ~12 MB per kept token of
  SSD it saves. The store is not blocked: the encoder is ready, and the kernel needs the TA-style rewrite.

### (2d) TRL-TA, pre-registered 20:5x (correctness before any timing: PASS)
- **Kernel (tierglu_trl.tier_glu_trl_ta):** B10(b)'s TA / TB mapping, grouped per expert. Layout v2 per projection:
  [rb][word][tile column], so the lanes' word loads are contiguous (the encoder's tiles transpose to it; to_v2).
  - A: lane = tile columns l, l + 32, ...; each group row's 16 x' values live in registers and serve RA = 4 output
    rows.
  - B: TB's (row, tile column) items over the lanes; RP rows per pass.
- **Correctness:** fp32 == float64 reference within 2-4e-7, all forms, 1-16 rows, RP 2 / 4.
- **Arms:** RP 2 / 3 / 4 at B10's geometry (RA 4, 8 simdgroups; RB 4, 8 simdgroups). Verdict = the best per trace.
- **Line:** unchanged (S 16 vs NAX6-N, PASS <= 1.00x, BETWEEN <= 1.25x, KILL above).
- **Prediction:** 0.75-0.9x (B10's TA decoded 713 G weights / s at one row; RP x the FMAs and RA-fold x reuse).

### (2e) TRL-TA RESULT (21:00; quiet lock, 20 s of GPU): **KILL (1.61x)**
- Per S 16 call: code NAX6-N 0.728 ms vs TA RP 2 1.300 (1.78x), RP 3 1.656, RP 4 1.785. Chat 0.991 vs 1.429 (1.44x).
  S 8: 1.37-1.53x. TRL-3 TA RP 3: 1.83-2.38x. Prediction 0.75-0.9x: wrong.
- More rows per pass are slower (RP 2 < 3 < 4). The x' rows in registers (RP x 16) cost occupancy more than they save.
  The mapping was not TRL-G's problem: both ALU kernels spend ~8-10 ops per weight on the 3INST decode plus every FMA.
  NAX6-N's dequant is ~1 op per weight (spread3 into 4-bit nibbles), and its FMAs run on the matrix units.

### (2f) TRL-NAX, pre-registered 21:0x (correctness first: PASS)
- **Kernel (tierglu_trl.tier_glu_trl_nax):** NAX6's MPP structure with the trellis decode.
  - Each lane decodes its output row's next 32 weights (two tiles) as half into a 32 x 32 threadgroup tile.
  - matmul2d (16 x 32 x 32, half x half -> fp32) multiplies all <= 16 rows of a group at once, so each weight is
    decoded once per group with no re-decodes, and the FMAs run on the matrix units.
  - The row scale applies once at the end; K is split over KS simdgroups (A) / KSB (B), with partial sums in
    threadgroup memory. Layout v1 (the encoder's).
- **Correctness:** fp32 out vs the float64 reference: max 3.7-6.8e-4, rms ~5.3e-4, all forms, 1-16 rows. That is half
  rounding of the A (x', h') and B (weights) operands, ~4.9e-4 each; the line was <= 5e-3.
- **Arms:** KS 4 (8 simdgroups per A threadgroup) and KS 2. Verdict = the better per trace. Line unchanged.
- **Prediction:** 0.8-1.0x. The decode stays ~8 ALU ops per weight, but there are no re-decodes and no ALU FMAs:
  ~400 M decodes x 8 ops at ~6 T ops / s ~0.55 ms, plus tile traffic.

### (2f) TRL-NAX RESULT (21:13-21:14; quiet lock, 12 s of GPU; data/trl_bench.txt): **PASS (0.87x NAX6-N)**
| per call, layer 18 | NAX6-N | TRL-G (A) | TRL-NAX KS 4 | TRL-NAX KS 2 | TRL-3 NAX KS 4 |
|---|---|---|---|---|---|
| code S 16 | 0.722 ms | 1.222 (1.69x) | 0.645 (0.89x) | **0.640 (0.89x)** | 0.756 (1.05x) |
| chat S 16 | 0.946 ms | 1.171 (1.24x) | 0.818 (0.86x) | **0.810 (0.86x)** | 0.966 (1.02x) |
| code S 8 | 0.460 ms | 0.632 (1.38x) | 0.422 (0.92x) | 0.424 (0.92x) | 0.493 (1.07x) |
| chat S 8 | 0.547 ms | 0.683 (1.25x) | 0.498 (0.91x) | 0.499 (0.91x) | 0.585 (1.07x) |
- **Verdict (S 16, the better KS per trace, mean of code / chat): 0.87x, PASS** (line <= 1.00x). Prediction 0.8-1.0x:
  right. TRL-G in the same run: 1.24-1.69x, so the lock was not unusually kind.
- **What it means:** the 2.40 b mix (1.518 MB per expert vs records3's 2.150 MB MSB-3 + 0.768 MB LSB) serves its
  expert GLU ~13 % faster than today's NAX6-N per layer call, at 30-48 % fewer bytes per expert. TRL-3 (3.07 b,
  1.928 MB, held-out quality near MSB-3's) runs at ~parity (1.02-1.07x). The decode path is no longer the reason to
  hold the store; quality (screen 7) and disk are.
- **Why it works:** each weight is decoded once per group (groups <= 16 rows, one MPP pass); the matrix units do the
  FMAs; the row scale applies once. The ALU kernels lost because they paid ~8-10 ops per weight-row product.
- **Caveats:**
  - Kernel-level, resident pools (random streams; decode cost does not depend on the values), one layer's routing.
  - FAST numerics: x', h' and the weights enter the MPP as half (~5e-4 relative, small next to TRL's own error).
  - Not in the engine yet. Wiring (CFG switch, pools, reader from the trl-v1 store) is the next build; a held ref2
    diff after screen 7.
- **TRL-NAX on real codes (21:17, trl_store.py test):** 16 real L18 experts through the trl-v1 store. Per pair against
  the dense float64 path W = Q R^T: max |dy| = 7.7e-4 of the pair's max |y| (half operands). PASS (<= 5e-3).
- **Runbook:** data/TRL_FULLSTORE_RUNBOOK.md covers disk, capture, the 4-process encode with --store (trl-v1 layers
  written in place, published by rename), checks, serving, rollback.

### DKV re-time (Wed 21:18; queued quiet lock, 6 s of GPU; data/dkv_check.txt, DKV_ONLY_C=1): **PASS**
- ms per propose (BP-0's cycle, median of 3 interleaved rounds; the rounds agree within ~0.2 ms):
  - 2,047 context rows: ref 7.41, **dkv 6.93 (-0.48)**, dkv+add 6.41 (-1.00);
  - 1,000 rows: ref 7.17, dkv 6.38 (-0.79), dkv+add 6.15 (-1.02).
- **Line (pre-registered 18:0x): dkv <= ref - 0.25 ms at 2,047 rows -> PASS.** Exactness: (a) / (b) bitwise PASS at
  19:0x (2f726fb), unchanged code.
- **Next:** hand data/r57_dkv.diff (R57_DKV / R57_DKV_ADD, both default OFF) to BigMoE as an exact drafter lever.
  The additive-mask variant adds another -0.5 ms at the same bits.

### L41 for screen 7 (the orchestrator 21:18 / 21:20): finish Fit12's layer 41 with the batched encoder
- **Pre-check (21:19):** experts 10 / 50 / 90 re-encoded from Fit12's inputs (b10rcap, its L41 _hx.npz). All 9 files
  are byte-identical (cmp) to Fit12's store, and the views match (--check 9 / 9).
- **Plan:** scripts/trl41_finish.sh N writes experts N..511 in Fit12's format into ~/Leap/data/trl41_for_fit12;
  Fit12 copies them in with cp -n. It runs after P-CAP's resume ends (the orchestrator: NO-SWAP margin >= 8 GB; P-CAP
  takes 36 GB) at ~1 s per expert. s7_trl's last start is 22:53.
- Note: my short quiet locks at 21:13 / 21:18 reset P-CAP's 3-minute clear-poll start rule (no other quiet lock).
- **L41 RESULT (21:34:38-21:40:20, share lock beside P-CAP's walk):** experts 146..511 (366) in 341 s, 0.93 s each
  including Fit12's 8-bit views. With Fit12's 0..147, all 1,536 / 1,536 files exist (layer 41 WHOLE).
  - In-run overlap e146 / e147: 6 / 6 byte-identical to Fit12's.
  - The memory / swap guard never fired; free memory stayed 34-44 GB with the walk up.
  - Handed to Fit12 21:40. Fit12 checks, then copies via temp names and renames, then announces the whole store
    (Breakthrough's dm_t3.trl_gate reads Fit12's store). Log: data/trl41_finish.log.
- **23:3x:** s7_trl has been read (OUTCOME above), and Fit12 had asked me to keep my copy until then.
  - Check: Fit12's L41 has 1,536 / 1,536 files; my 1,107 match by name and size.
  - The staging dir ~/Leap/data/trl41_for_fit12 (1.8 GB) is deleted at the first moment with no held quiet lock and no
    PAUSE / HOLD. The script re-checks and cmp's 7 files first (EvalSuite's RELAX held quiet until 00:32; heat PAUSE
    at 23:31).
  - **DELETED Thu 05:02:00** at the first such gap. The re-check was clean: Fit12's L41 1,536 / 1,536, 0 name / size
    mismatches, 7 / 7 sampled files identical (cmp). 132 GiB free afterwards.

## FAST-NAX: the FAST path on NAX6-N, QOM + U3 on NAX6-N's one-call (Leap, pre-registered Thu 2026-11-05 19:0x; the orchestrator 15:5x / 18:5x item 2)
- **Why:** on the converted store both the exact default and FAST qall (fri_night.sh's QA) run 3c-N, and NAX6-N / NH
  never engage:
  - qall sets no R57_NAX;
  - ref2._ooo_sel refuses QOM on the NAX path;
  - OOO-1C skips nax_hybrid.
  - Experts per 16-row check, quiet: 3c 50.5 ms vs NAX6 43.2 (C2, Mon). NAX6-N ~33-36 and NAX6-NH ~31-33 (KERNEL90's
    re-base). 3c-N is 4-7 % slower than 3c.
  - The orchestrator 15:5x (EXPERT_KERNEL line 67): group cost is fixed whatever the rows, so the per-group kernel cost
    is FAST's lever, not the rows.
- **Prior art:** none needed. This wires our own kernels together (tag: integration).
- **Build:** ooo1c.py + ref2, default OFF. The new CFG nax_1cq / REF2_NAX_1CQ=1 works together with ooo_1c and
  tier_nax_ooo + nax_native.
  - Grouping: late pairs are keyed by expert id, tier AND sidecar state, so a late group is uniform. A sidecar of -2 is
    a U3 sure row's final tier 3.
  - Kernel A head:
    - a late base slot of -1 is polled;
    - -2 drops the whole group: A writes rs_m = -2 and returns. In QOM, -2 is an optional-only expert the loader did
      not read (decided per expert in ext ensure(), so the drop is uniform per group);
    - each pair writes its OWN sidecar entry, and only for its 4-bit ranks.
  - Kernel B: a dropped group writes zeros for its rows, so combine_phase adds nothing for it.
  - ref2: _ooo_1c_call takes U3 (nolsb) and QOM on the native NAX path. If the one-call declines, QOM raises (as on
    3c-N).
- **Found while reading (fixed in the new head):** the existing NAX one-call's kernel A writes rs_l for every pair of a
  group, using the first pair's tier.
  - A resident group can mix a 4-bit pair with a -2 sidecar (LP1 or U3) and rank >= 2 pairs. Then the write lands on
    the next row's rank-0 entry, or past the buffer for the last row.
  - That path is default-off and unused (no arm runs R57_NAX with REF2_OOO_1C). Its own fix comes after the check
    shows it.
- **Correctness line** (GPU after Fri 00:15; scripts/ooo1cq_check.py; the converted records3's L20 / L44 pools through
  BigMoE's loader, read-only):
  - The reference: NAX6-N on the final slots (tier_glu_nax6n, per-pair tiers, U3 sure rows at tier 3) + combine_phase,
    with the dropped pairs at weight 0.
  - The one-call must be bitwise == it in each case:
    (a) released before the call;
    (b) a late release;
    (c) U3 sure rows and QOM drops together.
  - Any differing bit = a bug.
- **Speed line** (kernel level, share lock, rough): ms per MoE layer call in 48-call chains, NAX6-N 1C-Q vs 3c-N 1C QOM
  on the same calls.
  - PASS (worth an engine arm) <= 0.85x; KILL > 0.95x; BETWEEN otherwise.
  - Prediction 0.65-0.80x (C2 / KERNEL90).
  - Caveat: the OOO-1C harness (Wed 15:1x, contended) measured NAX6's two phases at 1.5x 3c's time per layer. This test
    settles that contradiction first.
- **Engine line** (BigMoE's arm, after the checks): qall + R57_NAX=1 + REF2_NAX_1CQ=1 vs qall, pair-mean geo steady
  tok/s on fri_night's 12 HumanEval. PASS >= +8 %, KILL < +3 %. Adoption via EvalSuite same-scores (FAST).
- **Next if the engine line passes:** NAX6-NH inside the one-call (the hybrid's ALU path for one-pair groups).
- **RESULT Thu 23:22-23:2x (share lock, MLX 0.32.2): correctness PASS, speed PASS (0.835x).** Data:
  data/ooo1cq_check.txt, data/fastnax_gpu.log.
  - (a) released before the call. 8 calls on L20 / L44, with 40 dropped pairs and 48 U3 sure rows: the one-call-Q +
    combine is bitwise == NAX6-N on the final slots + combine (drops at weight 0). rs_m / rs_l are right, dropped rows
    are 0, no timeouts. PASS.
  - (b) a late release (20 ms): PASS.
  - (c) is covered inside both, since every call mixes U3 sure rows and drops.
  - (e) ms per MoE layer call (48-call chains, median of 15, ~96 % resident): NAX6-N 1C-Q + combine 1.242 vs 3c-N 1C
    QOM 1.487 -> 0.835x. PASS against the <= 0.85x line, though a little short of the 0.65-0.80x prediction.
    - This settles Wednesday's contradiction: per call, NAX6-N's one-call is faster than 3c-N's. Wednesday's 1.5x timed
      NAX6's two phases under contention.
  - (d) diagnostic, the existing ooo1c NAX6-N one-call on the same U3 scenarios: 5 of 8 calls wrong (3 outputs
    differ, 5 rs_l wrong). This confirms the side find.
    - That path stays off and unused.
    - R57_NAX + REF2_OOO_1C without REF2_NAX_1CQ stays unsafe (BigMoE's note).
    - With REF2_NAX_1CQ=1, every NAX one-call takes the new head.
  - Harness slip: fastnax_gpu.sh called ref2.self_test() without ref2.apply() (KeyError 'qmm').
    - So its guard restored the pre-19:1x ref2 at 23:23:01.
    - I re-applied it from git (md5 eda11b8b) after a proper self_test (apply() first): all five checks True, the
      switch off by default.
    - The script is fixed.
  - Next: BigMoE's engine arm (the line above), then NAX6-NH inside the one-call, then the ooo1c head fix.

## QOM-SPLIT-1C: R57_QOM_SPLIT on ref2's one-call (Leap, idea Fri 2026-11-06 01:3x; BigMoE's find, the orchestrator deciding)
- The find (BigMoE, Fri 01:2x): with REF2_OOO_1C=1, R57_QOM_SPLIT does nothing.
  - ref2's one-call (QOM-OOO, Wed 17:1x) drops an optional use by the loader's answer (-2 = not resident).
  - It never calls Breakthrough's qom_keep_mx, the only place the split acts.
  - So Thu night's split arms ran plain qall. FLAT-2's BASE (R57_QOM_SPLIT=0.9 + REF2_OOO_1C=1) measures qall's flat cost.
- Why it is simple:
  - qom_keep_mx keeps a use if its rank < k0, OR it is in the block's loads, OR (it is resident AND the row is not strict).
  - The screen's optional uses are exactly the third case: opt = sure rows & ~(ranks 0-1 | the block's union).
  - A strict row (sure, cf >= T) keeps none of them, whatever the loader says.
  - So ref2 can drop opt & strict before the post: the row's rank-0 id, weight 0, as the two phases do with a dropped use.
  - Other rows keep the bit-21 post. No kernel change.
- Plan if the orchestrator says go:
  - A ref2 switch, off by default: T from R57_QOM_SPLIT, cf from t["qom"][2].
  - GPU check per call against BigMoE's two phases with the split (REF2_OOO_1C off), same loader state, bitwise.
  - The one difference: strict rows' dropped uses no longer touch the loader, so cache histories can drift. That is fine on the
    FAST track, which is certified under >= 2 histories.
- The orchestrator 01:3x: go, built that way, including the post-drop counter BigMoE needs for C3-R8 / "groups computed".
- RESULT (Fri 2026-11-06 02:58, first in Leap's NATIVE-FRI quiet turn): PASS, and LIVE. scripts/ref2.py = md5 eb43fcc2 (was FAST-NAX eda11b8b), with the split off by default.
  - Check (scripts/qomsplit_check.py drives `_ooo_1c_call` itself; layers 20 / 44, 16 rows, K 10, T 0.9, 8 sure rows, 4 of them strict):
    - (s) split on == Breakthrough's qom_keep_mx(split=0.9), bit for bit, on 3c-N 1C QOM and NAX6-N 1C-Q, including late releases.
      - 416 split drops in all, 281 of them on resident experts (uses the split removes that plain qall would keep).
      - Posted ids and bit-21 flags were correct, with no timeouts.
    - (o) split off == live FAST-NAX bitwise, with and without QOM, and with the counter on.
    - The counters matched the reference's keep mask in every call.
    - ref2.self_test passed after the install.
  - Data: data/qomsplit_check.txt and data/qomsplit_check.log.
  - API: R57_QOM_SPLIT=T now also acts on the one-call path. REF2_QOM_SPLIT overrides it; "" or "off" means off.
  - Counter: ref2.QCNT_1C = [] (None = off). Each one-call appends int32 [kept, dropped, groups, split_drops], with:
    - groups = distinct base slots after the drop over the call's rows (the union per check block);
    - ref2.COUNTS["qom_split"] = the calls where the split acted.
  - BigMoE was told at 03:0x. Their qall / s99 / s90 chunk re-arms on the one-call path after NATIVE-FRI and must end by 10:15.

## FLAT-2 RESULT: zero's fixed-cost measurement (Leap ran it, Fri 2026-11-06 01:33-01:37, C 224, quiet lock): KILL
- Lines (zero, LOSSY350_ZERO.md section 4): FLAT = the steady 2-row cycle minus 2 x the row sweep's warm slope.
  PASS <= 30 ms, BETWEEN 30-36, KILL > 36. 350 tok/s itself needs <= 25 at K 12.
- fri arm (the Friday FAST all-in: QOM + D17 + D7B code + HC8 + DNR + HCB + small native + D7BNAX + OOO-1C + split 0.9):
  - FLAT 53.74 ms -> KILL.
  - Cycle median 59.76 ms (IQR 59.30-60.17): draft 9.10, check + commit submit 50.60 (graph build 44.74, eval wait 5.59).
  - Row slope 3.01 ms per row (warm intercept 36.46 ms); 885 expert groups per 2-row check.
- all arm (+ DKV + DKV_ADD + GDNB): FLAT 55.20 ms -> KILL (cycle 61.22: graph build 46.09, eval wait 5.68).
- Warm 2-row check, synced split (fri; 111.1 ms synced, one sync per piece):
  - GDN 15.65, HC reads 10.96, attention 8.58, router + shared 5.74, experts 17.35, MoE tail 3.45, HC write 1.05,
    PLE 0.70, embed + norm + head 2.27, layer glue -6.05; pure CPU graph build 13.35.
  - Per added row (16 vs 2 rows) only the experts grow (+3.27 ms per row); every other piece is flat.
- Reading:
  - The fixed cost is ~29 ms above the 350 line and ~24 ms above PASS. It is spread over GDN, HC reads, attention,
    router + shared and the experts' per-check part, not one piece.
  - The check's submit (44.7 ms) is mostly per-layer work exposed by syncs; pure Python graph build is ~13 ms.
  - So fixed-cost work is large. In order: lazy commit (Fable B), the HC reads, GDN, and fewer syncs per layer.
- Caveats:
  - The split (R57_QOM_SPLIT=0.9) was inert in both arms under REF2_OOO_1C (QOM-SPLIT-1C), so this is qall's flat cost.
  - The memory was fine: NO-SWAP 48.2 GB at the start, swap flat at 11.1 GB through both arms.
- Data: data/flat2/ (screens, json, logs; copied from ~/FableGod/zero/build_1105/out), data/flat2_ladder.log.

## FLAT-CUT PLAN: the pieces, expected ms, lines (Leap, pre-registered Fri 2026-11-06 03:3x; the orchestrator 01:4x asked for it)
- Base: FLAT 53.74 ms (FLAT-2, fri arm) = draft 9.10 + check/commit 44.6 (warm, 2 rows, no SSD wait).
  - Every line below is on FLAT: zero's FLAT-2 probe, the fri config plus the piece, quiet, C labelled, against a base on
    the same night.
- What the records already settle (prior art, mine):
  - The per-layer id syncs are gone: GATE16 / R-55 (adopted 10-31), and ref2's post/spin resolves resident slots on the
    GPU.
  - The Python graph build already overlaps the GPU: B5b, async_eval every 4 layers (adopted 10-25, -17.5 ms).
  - B5a (10-25): a native loop with no Python build at all gained only ~3.4 ms more than B5b.
  - Today the submit is 44.7 ms against 13.35 of pure Python, so the CPU mostly waits on the GPU.
  - The flat cost is GPU time: ~1,000 small dependent kernels at 2-16 rows. The dense pass moves ~2.7 GB in ~44 ms (~61
    GB/s; zero's bus check) against ~250 possible. It is bubble-bound, not byte-bound.
- Step 0 (first, ~15 min, one 45 GB job): a GPU timeline of FLAT-2's warm 2-row check, with B5's xctrace method.
  - It gives each piece's kernel time against its gaps, and re-prices pieces 2-5 before they are built.
  - It is a measurement, so it has no line.
- Pieces (expected FLAT ms; line: PASS / KILL; exact pieces must be bitwise, FAST pieces need EvalSuite):
  1. Lazy GDN commit (Fable B's patch, ready): -0.6..-1.4. R-21: the replay moves 220 MB at ~190 GB/s.
     Line: Fable B's, i.e. bitwise; >= 0.8 ms per partial-accept cycle and <= 0.8 more per full accept = PASS; < 0.4 = KILL.
  2. HC reads (96 per check; v4's kernels are 4.97 ms per quiet chain, the piece is 10.96 in the loop): one dispatch per
     read, fused into the consuming norm / in_proj. -2..-5. Line: PASS >= 2.0, KILL < 1.0.
  3. GDN (36 layers, 15.65 ms; the body is only 0.082 ms per layer, so the rest is projections and small ops at 2
     rows): conv, gates, body and gated norm fused around the projection GEMVs, 2-3 dispatches per layer. -3..-6.
     Line: PASS >= 2.5, KILL < 1.0.
  4. "Fewer per-layer syncs" = the experts' fixed part: ~10 ms per 2-row check beyond the row cost, i.e. post, kernel
     A / B and combine per layer. Merge them into <= 2 dispatches per layer. -2..-5. Line: PASS >= 2.0, KILL < 1.0.
  5. CPU graph build 13.35 ms: compile with B5c's form (mx.compile per 4-layer run) or cache. 0..-3.
     - Run it only if step 0 shows >= 3 ms of GPU idle caused by the CPU. It is already overlapped (B5b), and B5a
       caps a full native loop at ~3.4 ms.
     - Line: PASS >= 2.0, KILL < 1.0.
  6. Ready elsewhere: FAST-tax att_single (Fable B's r57 patch, BigMoE's A-B-A): -2.4 [K R-62]. Fable B's line:
     PASS >= 2.0, KILL < 1.0.
  - Not on the list, but it is the second-largest flat term: the draft (9.1 ms, serial today). Drafting on a second queue
    inside the check's bubbles, or off the GPU (Mirror-SD, PARKED), would give -3..-7. The drafter's owners decide; I
    can build the queue part.
- Sums:
  - Pieces 1-6: -12..-23 -> FLAT ~31-42 ms.
  - With the draft: -15..-30 -> ~24-39.
- Program lines:
  - FLAT <= 37 ms (250 tok/s at K 12 with e <= 0.9) = PASS. <= 26 = the 330 line.
  - FLAT > 45 with pieces 1-4 in = KILL of the incremental route. Then propose the structural step: the dense pass as a
    few fused kernels per layer (megakernel-style; prior art: Hazy Research's Llama-1B megakernel and CMU's Mirage MPK,
    both 2025). Its floor at ~250 GB/s is ~11-14 ms of dense time.
- The e side (corrects my 01:5x note):
  - FLAT-2's 3.0 ms per row is the warm all-resident slope, with no drops.
  - On real QOM blocks the expert term is ~1.94 ms per row (FAR), ~2.2 per kept.
  - So 250 at K 12 needs FLAT <= ~21 without the split form and <= ~37 with it. The split form = QOM-SPLIT-1C (live
    02:58; BigMoE's A/B tonight) plus NH-1C, for NAX6-NH's 8.4 us per tile.
  - Both cuts are still needed.
- Order: lazy commit, then step 0, then pieces 4 and 3, then 2, then 5 only if step 0 says so. NH-1C goes in parallel,
  after Saturday's FAST-NAX arm.
- **AMENDMENT (Fri 03:5x, before anything ran; the orchestrator approved the plan at 03:4x with owner records to fold in):**
  - Step 0's method changes. xctrace is excluded: on 11-02 it wrote GPU counters to the per-user temp dir at ~120 MB/s,
    even with only the GPU instrument (CA runs 1-2, two disk ACTs), against the orchestrator's 3 GB cap. MLX exposes no GPU
    timestamps, and a .gputrace is excluded too.
  - New step 0: a one-layer piece bench with real weights at 2 and 16 rows, MEM <= 4 GB, share lock. Per piece: synced ms,
    R unsynced repeats (GPU-side ms), dispatch count, dispatch-gap calibration and bytes, giving in-kernel ms, gap ms and
    achieved GB/s. Scale by layer count and check against FLAT-2's synced pieces. The 04:10 quiet slot was given back.
  - Owner records folded in:
    - Piece 2: start from BigMoE's BIGMOE_HC_VREAD (bigmoe/fuse.py, 3 exact dispatches instead of ~11, ADOPTED in the 64
      GB era, off in R57_PRESET=45). The 'Fused HC block' idea was KILLed as its duplicate.
    - Piece 3: GDNB KILL (the body is 0.082 ms per layer) and 'R3 more GDN fusion' KILL (projections at the bandwidth
      floor). Piece 3's -3..-6 must show up in step 0 as conv / gate / norm time or gaps, else it is priced out.
    - Launch gaps (~1,000 x 1.3-5.9 us = ~1.5-6 ms) cannot make 44 ms alone, so step 0 must split in-kernel time from
      gaps.
    - Any megakernel proposal must answer the M5 KILL of grid-barrier persistent kernels (a barrier costs 1.97-2.47 us
      against 1.3-1.6 us per dependent dispatch) and MoonShots' M6b (per-layer fusion, PARKED).
    - The draft counts at -1.4..-1.7 ms (MTP Head's DRAFTOFF estimate: draft-ahead at a 34-43 % hit rate, which needs >=
      4-5 ms of GPU idle). Off-GPU drafting was KILLed 3 times (A7, M-19 CPUDRAFT, MEND-b). So 'with the draft' is ~29-41
      ms, not 24-39.
- **STEP 0, PART A (Fri 03:5x; from FLAT-2's own synced records, no new run; data/flat2/flat2_{fri,all}_224_flat2.json
  'seg'):**
  - Method: each piece's synced eval minus the calibrated round trip (sync_ms 0.133) = the piece's GPU time. 4 probe
    checks per arm, at 2 rows and 16 rows.
  - fri arm, GPU ms per check (2 rows / 16 rows):
    - GDN 13.2 / 12.2 (36 layers, 0.37 ms each); experts 11.1 / 57.0; HC reads 8.6 / 8.1 (hc_a 3.8, hc_m 4.8);
    - attention 6.4 / 6.5 (12 layers); router + shared 5.7 / 4.5; MoE tail 1.1; HC write 0.8; PLE 0.65; final HC 0.13.
    - Sum at 2 rows: 47.8 ms. The 'all' arm agrees to within 1-4 %.
  - **The check is GPU-bound.**
    - The GPU sum (47.8, plus head and embed) is about the whole unsynced check: submit 44.7 + eval wait 5.6 = 50.3.
    - So the CPU's 13.35 ms graph build is hidden behind GPU work, as B5a found. Piece 5 prices at ~0 and drops out
      unless the bench says otherwise.
    - This corrects FLAT-2's 'Reading' bullet "mostly per-layer work exposed by syncs": the unsynced loop has no
      per-layer syncs left.
  - **Experts' fixed GPU part: 11.1 - 2 x 3.27 = ~4.6 ms**, not ~10. The other ~6.3 ms of FLAT-2's 17.35 was CPU build,
    which is hidden. Piece 4 re-prices to -1..-2.5.
  - HC reads: 8.6 ms in this form against 4.97 ms for v4's quiet back-to-back chain. Piece 2 stays -1..-3; the bench
    decides.
  - Every dense piece is flat from 2 to 16 rows.
    - Launch gaps are ~1.3-1.6 us per dependent dispatch (M5, TRIED), only ~1.5-3 ms over ~1,000-2,000 dispatches. So
      most of the ~45 ms is in-kernel time at low efficiency: ZERO's 2.7 GB of dense bytes is ~60 GB/s, not ~250.
    - Step 0 part B (the bench) must measure per piece: kernel count, bytes, in-kernel ms against a size-matched floor,
      and the per-kernel minimum time.
  - Re-priced program, before the bench:
    - lazy -0.6..-1.4, HC -1..-3, GDN -1..-3, experts fixed -1..-2.5, FAST-tax -2.4, CPU build ~0 -> -6..-12
      -> FLAT ~42-48;
    - with the draft (-1.4..-1.7): ~40-46.
    - The incremental route likely misses PASS (<= 37) and may hit its KILL (> 45 with 1-4 in). The structural step
      (fewer, larger kernels per layer: the per-kernel minimum cost x ~1,000 kernels) is probably the real route. It has
      to answer the M5 grid-barrier KILL and M6b.

## STEP 0 PART B RESULT: what the 45 GB FAST check's kernels cost (Leap, Fri 2026-11-06 07:50-07:55, quiet lock, C 224)
- Why: Kamen (07:2x, via the orchestrator) made the dense-kernel rebuild the top priority. Part B gives the numbers it
  needs: per-kernel time in a dependent chain, bytes, GB/s, the cost of one sync point, kernel counts, the experts at 2 and
  16 rows. Fable B's DENSE_REBUILD v1 (section 6) asked for the same three things.
- Method (scripts/stepb/, commit 6de31d2 + this one): zero's FLAT-2 harness on BigMoE's r57_screen.py HEAD, unchanged except
  the probe. Two arms in one quiet lock: fri (today's FAST all-in without the split) and nax (+ FAST-NAX: R57_NAX=1
  REF2_NAX_1CQ=1). Two probes per arm (checks 8 and 24), at 2 and 16 rows.
  - stepb_k.py wraps every kernel-launching mx call, metal_kernel and compile before mlx loads; off, it passes through.
  - Chained replay: the engine's own calls of one kind (same kernel and shapes), across the layers in order (cold
    weights), each waiting on the last through a 3-kernel link. Per call = (chain - the same links alone) / n.
  - t_sync: a dependent chain of 200 kernels that move almost no bytes, at 32 to 655,360 threads.
  - Graph counts: mx.export_to_dot of each piece (one GDN and one attention layer) before its eval.
  - Checks: kernel-level forwards gave logits == the live check bitwise (12 of 12 at 2 rows). The replay repeats across the
    two arms' processes to within ~5 %. The per-call synced timings (one eval per call) are NOT usable: they swing 2-3x
    between processes (GPU clock between tiny command buffers). Only the chains, the piece split and the counts are used.
- **t_sync, one dependent sync point (us): 1.24 (32 threads), 1.40 (1 x 1,024), 1.55 (16 x 1,024), 1.56 (20 x 1,024),
  1.86 (80 x 256), 4.13 (512 x 256, 1 MB moved).** So one wide wait costs ~1.6 us, not the 12-20 us of DENSE_REBUILD's
  fit. ~1,000-4,000 dispatches per check are worth ~2-6 ms, not 20.
- **Kernels in a dependent chain (us per call at 2 / 16 rows; MB per call; GB/s; calls per check):**
  - GDN in_proj qkv|z (leap_qmm26_d, 512 tiles, ks 1): 113 / 115; 23.6 MB; 210; 36.
  - **out_proj (GDN) and o_proj (attention), 80 tiles, K 6,144, ks 1: 112 / 112; 8.9 MB; 81; 48.** Floor at 220 GB/s
    ~40 us.
  - **GDN a|b, 3 tiles, ks 1: 45 / 46; 0.22 MB; 5; 36.**
  - attention q, 384 tiles: 75 / 74; 17.7 MB; 243-247; 12. **attention k and v, 16 tiles each: 48 / 49; 0.8 MB; 18; 24.**
  - **shared gate|up, 41 tiles: 48 / 47; 2.0 MB; 42; 48.** Shared down (K 640): 13.7 / 13.6; 1.0 MB; 75; 48.
  - router logits (8-bit MLX qmm): 18.6; 1.46 MB; 80; 48. Attention's 8-bit 640 x 2,560: 21.9; 1.8 MB; 84; 12.
  - SDPA, head dim 256: 144 calls per check (12 head chunks per attention layer, each a Matmul / Softmax / Matmul
    fallback): 21.9 / 21.8 each, 20 GB/s.
  - HC8: k1c 22-26 (4.6 MB, 177-213 GB/s), k3c 27-29 (3.8 MB, 137-143), k2b 9.4-10.5 (0.45 MB, 44-49); 96 each.
  - gated_delta_step: 25-27 at 2 rows (6.2 MB, 239-250 GB/s), 57.5 at 16 (119 GB/s).
  - Glue: the 2 -> 16-row pads (Full + Concatenate, ~300 per 2-row check) 1-3 us each; sigmoid, compute_g, swiglu,
    moe_tail, prework, gated norm 1-5 us each.
  - Fit t = t_sync + bytes / B over the 4-bit kinds: B ~295 GB/s with a 33 us intercept. The intercept is the narrow
    kinds, not a sync cost: their time does not move with bytes.
- **Reading (the dense part).** The wide kernels already stream near the floor (in_proj 210, attention q 245, HC k1c ~200
  GB/s). The time is lost in narrow GEMVs. ref2 runs every native call at ks 1 (ref2.py 366 / 503 / 512, "bitwise == qmm16c
  (ks 1)"). One simdgroup per 32-column tile then walks the whole K. With 3-80 tiles the GPU is mostly empty, and the call
  costs ~1.1-1.2 us per K group whatever its bytes.
  - Per 2-row check:

    | piece | today | at 220 GB/s | saving | calls |
    |---|---|---|---|---|
    | out_proj / o_proj | 5.4 ms | ~1.9 | -3.5 | 48 |
    | a\|b | 1.6 | ~0.1 | -1.5 | 36 |
    | k\|v | 1.2 | ~0.2 | -1.0 | 24 |
    | shared gate\|up | 2.3 | ~0.5 | -1.8 | 48 |
    | shared down + 8-bit router | 1.5 | ~0.6 | -0.9 | |
    | SDPA chunks (fallback) | 3.2 | | -2.4 (R-62's att_single, Fable B's FASTTAX patch is ready) | |
    | HC k2b | 1.0 | | -0.7 | |
    | 2 -> 16-row pads | ~0.6 | | -0.5 | |
  - **Sum: ~-12 ms of the ~36 ms dense part (2-row check), with today's kernel set re-shaped, no megakernel.**
  - Replay chain sums per layer agree with part A: GDN ~0.31 ms + glue vs 0.37; MoE side ~0.10 vs 0.12; HC read ~0.06
    vs 0.09.
- **Kernel counts per layer (graph primitives that launch, 2 / 16 rows):**
  - GDN-type layer 64 / 44: HC read 6 / 4, GDN 14 / 8, HC read 9 / 3, router + shared 11 / 5, experts 22 / 22, tail 1,
    HC write 1.
  - Attention-type layer 131 / 116: the attention itself 82 / 81, mostly the 12-chunk SDPA fallback.
  - So ~3,900 / ~3,000 per check. The 2-row extra (~20 per layer) is the pads.
  - The experts' 22 = 3 kernels + ~19 tiny id / mask ops (BitwiseOr, Select, PostIds, ...): ~1.5 ms per check at
    ~1.6 us each.
- **Experts (one unit per layer; bytes = distinct experts x the MSB-3 record + ranks 0-1 x LSB, an upper bound: QOM drops
  are not taken off):**
  - fri (3c-N one-call): ~17 ms at 2 rows, **~60 ms at 16 rows (8.4 GB, 139 GB/s), +3.25 ms per row** (part A: 57, +3.27).
  - **nax (FAST-NAX, NAX6-N's one-call): ~14 ms at 2 rows, ~25 ms at 16 rows (8.5 GB upper bound), +0.98 ms per row.**
  - So FAST-NAX already meets the "experts 57 -> ~25 ms at 16 rows" target on all-resident blocks. That holds unless real
    blocks differ (misses, real QOM blocks), which BigMoE's FAST-NAX engine A/B (Saturday) decides.
  - The faster-kernel vs fewer-bytes question reads: 3c-N's kernel is the loss (~2x the read floor at 16 rows), NAX6-N
    is not (its time is at or under the upper-bound bytes' floor).
- **Against DENSE_REBUILD's rule (section 6).** t_sync(wide) ~1.6 us (< 6), B in flight 210-247 GB/s on the wide 4-bit
  kinds. So the count is not the lever: each removed wait is worth ~1.6 us. W and D still pay, through what they do to
  the narrow GEMVs: W folds a|b into in_proj's stream, D splits out_proj's K 8 ways. But the same gain comes from K-split
  and merged narrow projections in today's kernels, at far lower risk. The waits are worth ~1.5-6 ms per check over all
  kinds, a later step.
- Program, re-priced (dense part of a 2-row check, today ~36 ms): shape fixes -12 -> ~24 ms. Every kernel at 220 GB/s
  with today's counts -> ~16-17. 12 ms needs ~250 GB/s average and ~1/3 of today's kernel count.
- Data: data/stepb/ (reports stepb_{fri,nax}_224_report.txt; out/: json, screens, logs, dot_*/ graphs).
- **DN-KS, pre-registered Fri 08:0x (the first build; FAST, off by default).**
  - What: ref2's native dense calls pick ks per shape (REF2_DN_KS=1): the smallest ks (dividing K / 64, ks <= 16 for
    threadgroup memory) that gives >= ~320 simdgroups. Bits change in the K order, so FAST: EvalSuite's same-scores
    gate.
  - Step 1 (cheap, share lock): the chained microbench of qmm16n at the 5 narrow kinds' real shapes, ks 1 vs the
    candidates.
    - PASS: >= 5.0 ms per 2-row check saved over the 5 kinds.
    - KILL: < 2.5 ms. Prediction: -6..-8 ms.
    - Bits: max relative error vs ks 1 <= 2e-3 per projection.
  - Step 2: the FLAT A/B on the engine (quiet). PASS: FLAT -4.0 ms or more. KILL: < 2.0.
- **DN-KS step 1 RESULT (Fri 08:01, share lock, scripts/dnks_bench.py -> data/dnks_bench.txt): PASS, -7.74 ms per 2-row check.**
  - Round-robin over ks, 9 rounds, cold copies. The ks 1 baselines match part B's engine chains: out_proj 115 vs 112,
    a|b 47 vs 45, k / v 47 vs 48, gate|up 48 vs 47, in_proj 120 vs 113, attention q 75 vs 75.
  - The first, non-interleaved pass (07:59) was noisy (baselines up to 3.7x high); it is kept in the file but not used.
  - Results:

    | kind | ks 1 | best | at ks | GB/s | per check |
    |---|---|---|---|---|---|
    | out_proj / o_proj | 114.8 us | 41.4 us | 6 | 214 | -3.5 ms |
    | a\|b | 47.3 | 10.4 | 10 | | -1.3 |
    | k / v | 47.4 | 9.4 | 10 | | -0.9 |
    | shared gate\|up | 48.4 | 14.7 | 8 | 129 | -1.6 |
    | shared down | 16.7 | 9.1 | 5 | | -0.4 |
  - Controls: in_proj and attention q gain <= 0.55 ms in all, so they stay at ks 1 (rule: >= 320 simdgroups at ks 1).
  - Bits at the chosen ks: max relative error <= 2.0e-3 (most 0 or < 5e-6). Some other ks values flip one bf16 ulp
    (3.8e-3).
- **LAZY GDN COMMIT RESULT (Fable B's patch, ported onto ref2 eb43fcc2; Fri 08:01, share lock; data/gdnb_lazy_check.txt):
  BETWEEN.**
  - Bits: (1)-(4) all bitwise (lazy off == today; the replay == MTPLX's commit at keep 1 / 2 / 9 / 15 / 16, NSG 8 / 16,
    G3 off / on; a 40-event protocol stream), and ref2.self_test PASS.
  - Time (36 chained layers, S 16, median of 9; by day under a share lock):
    - keep 9: -0.56 ms per cycle (line PASS >= 0.8, KILL < 0.4);
    - keep 16: +0.93 ms (line <= 0.8);
    - net at 24 % full accepts: -0.20 ms per check.
  - Exact, small: BigMoE's call whether to stack it.
- **INSTALLED Fri 08:03:37: ref2.py 1851aa5c (eb43fcc2 + the lazy port + DN-KS), gdnbody.py 2ac884e8 (+ gdn_body_lazy).**
  - Both switches are off by default (REF2_GDN_LAZY, REF2_DN_KS); the live self_test passes.
  - Off, every path is the eb43fcc2 code, since _dn_ks returns 1. Backups: data/dnks/ref2_before_dnks.py,
    gdnbody_before_lazy.py.
  - DN-KS through ref2's own hook (dense_native, small_native; data/dnks/ref2_cand.py test 08:02): the 5 narrow kinds
    take their ks with error <= 6e-8; in_proj stays at ks 1.
  - The DENSE-N-RAM path (_dnr_qmm) uses the same _dn_ks; the engine A/B confirms it by COUNTS dn_ks.
- **DN-KS step 2 (pre-registered above):** FLAT A/B on zero's FLAT-2 harness (scripts/stepb/run_stepb.sh), arms fri, friks,
  fri2 (A-B-A, one quiet lock, C by the ladder). PASS: FLAT -4.0 ms or more against the mean of the two A arms. KILL:
  < 2.0. VOID if the A arms differ by > 2.0.
- **DN-KS step 2 RESULT (Fri 08:11-08:15, quiet lock, C 224; A-B-A on zero's FLAT-2 harness, 5 measured passes per arm; data/stepb/out/stepb_{fri,friks,fri2}_224_ab_*): PASS.**
  - FLAT: fri 55.17, fri + REF2_DN_KS 51.05, fri 55.30 ms. That is -4.19 ms against the A mean; the A arms agree within
    0.13 ms (line: PASS -4.0 or more).
  - Caveat: friks had 86 steady zero-miss cycles, under FLAT-2's own 100 minimum, so its verdict line reads INVALID. Its
    all-checks check + commit per measured pass agrees anyway: 47.6-48.2 vs 51.5-52.7 ms, about -4.0 ms.
  - DN-KS engaged (COUNTS dn_ks 89,760 in friks, 0 in fri / fri2). The probe's warm 2-row check alone: 43.88 / 41.28 /
    43.80 ms (-2.6).
  - The engine gain is about half the chained bench's -7.7: in the check some narrow calls (k / v, a|b beside in_proj)
    already overlapped other work. The chains serialize them.
  - Determinism (08:2x, share lock): every DN-KS shape at its ks, 59 repeats + 30 interleaved with other work, at 2 and 16
    rows: 0 differ. The warm repeats in friks were bitwise == the live check (4 of 4). Its 'last forward == live' was 0
    of 2 (fri / fri2: 2 of 2); that is the probe's last re-run after the sweeps and the drafter probe, not
    kernel noise. It is noted and does not gate this result.
  - FAST: adoption is EvalSuite's same-scores gate. BigMoE stacks REF2_DN_KS=1 tonight (their message 08:0x).
- **DRAFTER PROFILE (STEPB_DRAFT=1, same run; propose re-run on a copy of the drafter cache, chained replay of its calls):**
  - propose ~8.8 ms warm (walls 8.74-8.84; the loop's draft 9.1-9.2).
  - With R57_D17 the draft block has 17 rows, one more than d7b_nax's NaxQL takes (<= 16 -> qmm16c). So every block
    projection falls back to MLX's quantized_matmul at 52-57 GB/s:
    - gate|up 2560 -> 7680: 219 us each, x10;
    - down 7680 -> 2560: 219 us, x5;
    - q 2560 -> 4096 and o 4096 -> 2560: 107 us each, x5 each;
    - k / v 2560 -> 1024: 33 us, x10;
    - 2560 -> 640: 22 us, x10.
  - The context-row k / v (2 rows) run qmm16c (36 us). SDPA is 64 us x5.
  - Replayed kinds sum to 5.9 of the 8.8 ms.
  - So the drafter has the dense pass's own problem: a wrong kernel for the shape, not the bytes (~53 MB per propose:
    ~0.25 ms at 220 GB/s).
- **D17-NAX, pre-registered Fri 08:2x (the drafter's 17-32-row calls on qmm16c; drafts are verified, so the output stays
  exact):**
  - NaxQL takes 17-32 rows as two 16-row qmm16c halves, with per-shape ks for the narrow shapes. Switch D7BNAX_32=1, off
    by default.
  - Expected -1.5..-2.5 ms per propose (weights read twice unless the halves share the SLC).
  - Line: drafter probe walls, A/B in one quiet lock. PASS: propose -1.5 ms or more. KILL: < 0.5. Kept per check within
    1 % on the engine's own blocks (BigMoE's A/B).
- **D17-NAX bench (Fri 08:19, share lock beside BigMoE's shared run at 100 % GPU, so both forms are inflated;
  scripts/d17nax_bench.py -> data/d17nax_bench.txt).** 17-row calls on the drafter's shapes, MLX vs NaxQL halves,
  round-robin, cold copies (us; per-propose saving in ms):

  | shape | ks | MLX | halves | per propose |
  |---|---|---|---|---|
  | gate\|up | 2 | 322.7 | 141.8 | -1.81 |
  | down | 4 | 412.2 | 143.7 | -1.34 |
  | q | | 131.1 | 72.7 | -0.29 |
  | o | | 123.9 | 106.1 | -0.09 |
  | k\|v | | 24.9 | 30.1 | +0.05 |
  | 2560 -> 640 | | 14.7 | 33.2 | +0.19 |

  - Net -3.30 ms per propose. Bits vs MLX <= 6.4e-3 (qmm16c's own numerics; drafts only).
  - So the candidate takes halves only for weights >= 4 MB. Candidate data/d17nax/d7b_nax_cand.py; the live d7b_nax.py is
    unchanged.
  - The engine A/B loads it through stepb's preload (STEPB_D7BNAX), arms friks / friks32 / friks2.
- **D17-NAX engine A/B RESULT (Fri 08:22-08:26, quiet lock, C 224, A-B-A on the DN-KS base; data/stepb/out/stepb_{friks,friks32,friks2}_224_d17_*): PASS on time.**
  - Drafter probe propose walls (warm): friks 8.74-8.88, friks32 6.63-6.85, friks2 8.78-8.89 ms, so -2.1 ms per propose
    (line -1.5).
  - The loop's own draft per cycle: 9.10-9.15 / 7.08-7.13 / 9.11-9.15 ms (-2.0).
  - FLAT: 50.36 / 48.94 / 50.92 ms (-1.70 vs the A mean). All three arms carry FLAT-2's INVALID label (86 zero-miss cycles
    each, the same in every arm, so the comparison holds).
  - The halves path: 60 qmm16c calls per propose (10 before), 81 concatenates (31). Replayed kinds sum 3.3-3.5 ms (5.9
    before). A true 32-row kernel (one weight read, no concats) is the next -0.5..-1 ms.
  - Kept: not measured here (2-row checks); the line's kept clause (within 1 %) waits for BigMoE's A/B on real blocks.
  - Installed Fri 08:26:28: d7b_nax.py dd9066d3 (switch D7BNAX_32=1, off by default; off = the old calls). Backup
    data/d17nax/d7b_nax_before.py (376ceee4).
- Combined so far on a 2-row FLAT: fri 55.2 -> + DN-KS 51.0..50.6 -> + D17-NAX 48.9 ms (-6.3 ms, ~11 %).
- **SDPA-1 pre-registered Fri 08:3x: the 12-chunk SDPA fallback (head dim 256; 144 calls per check, ~3.1 ms chained) vs
  the two single-call forms ref2 already has, on the new base.**
  - Base: DN-KS + D7BNAX_32. Arms: b; batt (R-62 att_single, set after import ref2 by stepb); bfs4 (F1's fatt_verify +
    F1-R8, REF2_FATT_VERIFY=1 REF2_FATT_R8=1); b2. One quiet lock, C by the ladder, 5 measured passes each.
  - Line per form, FLAT vs the mean of b / b2: PASS -1.5 ms or more, KILL < 0.5. VOID if b / b2 differ by > 1.5.
  - Both forms are FAST (bits differ from REF2-T past 512 keys / F1's own reference). The better PASS goes into the stack;
    EvalSuite's gate decides adoption.
- **SDPA-1 RESULT (Fri 08:39-08:44, quiet lock, C 224; data/stepb/out/stepb_{b,batt,bfs4,b2}_224_sdpa_*): both single-call
  forms PASS.**
  - FLAT: b 48.22, b2 48.57 (A mean 48.40, within 0.35); **batt (att_single) 44.34 = -4.06; bfs4 (fatt_verify + F1-R8)
    44.22 = -4.18** (line -1.5).
  - check + commit per pass: 46.9-48.0 vs batt 43.4-44.1 and bfs4 43.2-43.8. The warm 2-row check alone: 39.05 / 40.08 vs
    36.56 / 36.35.
  - Engaged: att_single 5,280 calls; fatt_verify 5,280 + fatt_r8 5,112.
  - The gain is about 1.7x R-62's -2.4 estimate, as part B's chain (3.1 ms) suggested.
  - Pick: they are within noise of each other. att_single is bitwise == REF2-T below 512 keys (R-62), so it is exact for
    short contexts and FAST past them. fs4 is FAST everywhere but already in EvalSuite's F1 track. The stack's call
    (EvalSuite's gate).
- **Running total on the 2-row FLAT (fri, C 224): 55.2 -> DN-KS 50.6 -> + D17-NAX 48.4-48.9 -> + single-call SDPA 44.2-44.3
  ms: -11 ms (-20 %), all installed off by default or already in ref2.**
- **GDN-CHUNK GPU check (Fable B's row-parallel delta step, kernels/gdn_chunk.py; Fri 08:45-08:48, share lock beside
  EvalSuite's shared run; scripts/gdn_chunk_check.py -> data/gdn_chunk_check.txt): KILL.**
  - Bits fine: y max relative error <= 6.9e-4, 0.01-0.03 % of bf16 outputs differ, state <= 2.5e-7, at S 1-16.
  - Time, chained over 36 layers:
    - NSG 16: 121.7 us at S 16 vs gated_delta_step 86.3; 83.3 vs 53.7 at S 2.
    - NSG 32: 128.9 vs 64.5 at S 16; 40.4 vs 33.5 at S 2.
  - So 1.4-2x slower; line PASS <= 40 us at S 16, KILL > 50.
  - Why: one threadgroup per value head (48 threadgroups) against the reference's 1,536. Phase 3's forward substitution
    runs on 128 threads per head, 16 dependent steps.
  - Redesign note for Fable B: the substitution and the state update are independent per state row dv. So a threadgroup
    per (head, dv block) can recompute the head's small k.k / k.q tables and own its rows, giving ~768+ threadgroups.
- **HC-K2C GPU check (Fable B's kernels/hc_k2c.py vs hc8r kernel 7 k2b; Fri 08:46, share lock; scripts/hc_k2c_check.py ->
  data/hc_k2c_check.txt): KILL.**
  - Bits fine (mix1 <= 2.9e-5, xsum <= 3.4e-6).
  - Chained over 96 calls: k2b 4.46 us, k2c 15.21 us (line PASS <= 6, KILL > 8).
  - k2b alone chains at ~4.5 us. Part B's ~10 us per k2b in the engine replay includes its wait on k1c's partials, so
    this item is worth at most ~0.5 ms.
- **GDN-CHUNK v2 (Fable B's gdn_chunk2.py: TAB + MAIN, RB state rows per threadgroup; Fri 08:50, share lock with ~5 % GPU
  free beside EvalSuite): KILL.**
  - Bits as v1 (fine).
  - S 16 chained over 36: RB 8 (768 threadgroups) 149.2 vs 91.0 us; RB 4 (1,536) 183.0 vs 93.0. S 2: 47.4 / 50.2 vs
    37.2 / 39.3.
  - Fable B's own rule: dead for the check. The kept-prefix state formula stays a candidate for the commit replay.
- **WIDE-NT bench (Fri 08:47, share lock, contended; scripts/wide_bench.py -> data/wide_bench.txt).**
  - in_proj qkv|z: ks 2 132.5 vs ks 1 152.5 us (-13 %; err 8e-4), ~-0.5..-0.7 ms per check. nt 2 / 4 do not help.
  - attention q: ks 1 is best.
  - Goes into DN-KS v2 (REF2_DN_KS=2, the extended table) so tonight's REF2_DN_KS=1 stays as BigMoE measured it.
- **NEW-BASE KERNEL TABLE (Fri 08:57, quiet lock, C 224, arm batt = fri + DN-KS + D7BNAX_32 + att_single, 3c-N experts;
  data/stepb/stepb_batt_224_new_report.txt).**
  - Warm 2-row check alone 35.64 ms (fri this morning: 43.88). Drafter propose 6.4-6.8 ms.
  - 16-row dense kernels, chained (ms per check):

    | kernel | ms | GB/s |
    |---|---|---|
    | in_proj | 4.11 | 212 |
    | HC k3c | 2.63 | 143 |
    | HC k1c | 2.34 | 193-196 |
    | gated_delta_step | 2.08 | 118 |
    | out_proj / o_proj (ks 6) | 1.84 | 238 |
    | HC k2b | 0.94 | |
    | attention q | 0.91 | 239 |
    | router (8-bit MLX) | 0.90 | 79 |
    | shared gate\|up (ks 8) | 0.61 | 159 |
    | SDPA single call | 0.52 | |
    | shared down | 0.36 | |
    | a\|b | 0.29 | |
    | 8-bit 640 x 2,560 | 0.27 | |
    | HC write | 0.26 | |
    | prework | 0.25 | |
    | k\|v | 0.22 | |
    | the rest (glue, norms, rope, qsa) | ~1.2 | |

    Sum ~19.6 ms of chained kernel time.
  - Fit over the 4-bit kinds: t_sync 6.9 us, B 241 GB/s (the narrow-kernel intercept of 33 us is gone).
  - Experts (3c-N here): 10.6 ms at 2 rows, 56.4 ms at 16 rows (153 GB/s on the upper-bound bytes, 1.81x the read floor).
    FAST-NAX gives ~25 at 16.
  - Next levers by size:
    - HC reads (5.9 ms, k3c at 143 GB/s): -1..-2;
    - in_proj ks 2 (DN-KS v2): -0.5;
    - the 16-row GDN step: -1, needs a kernel, since the chunk form is dead;
    - the 8-bit router on a K-split: -0.5, needs a route-flip check;
    - the expert id glue: -1;
    - the drafter's small 17-row calls + concats: -0.5..-0.8.
- **QMM32 (Fri 09:01, share lock, GPU free; scripts/qmm32.py, qmm32_bench.py -> data/qmm32_bench.txt).**
  - qmm16c with two 16-row blocks per simdgroup sharing each decoded weight fragment (one weight read).
  - qmm32c == the two halves bitwise at every shape and ks.
  - 17-row calls (us), MLX / halves / qmm32c:

    | shape | MLX | halves | qmm32c |
    |---|---|---|---|
    | gate\|up | 216 | 92 | 74 |
    | down | 218 | 98 | 71 |
    | q | 116 | 59 | 47 |
    | o | 109 | 55 | 41 |
    | k\|v | 30 | 18.5 | 17.9 |
    | 2560 -> 640 | 20 | 15 | 16 |
  - Per propose: -0.44 ms vs the halves, -3.0 vs MLX. With the GPU free, even the small shapes beat MLX as halves; the
    08:19 bench that set the 4 MB threshold was contended.
  - Staged as D7BNAX_32=2 (candidate data/d17nax2/d7b_nax_cand.py: qmm32c for every eligible 17-32-row call, ks <= 5).
    =1 stays the halves BigMoE pre-flights.
- **GDN-STEP3 (Fable B's kernels/gdn_step3.py, ablation arms on the reference grid; Fri 09:03, share lock;
  scripts/gdn_step3_check.py): KILL by Fable B's own gate.**
  - var 0 == mlx_lm's kernel bitwise.
  - S 16 chained: mlx_lm 67.1, var 0 73.9, no simd_sum 79.8, k / q constants 78.6, both 69.4 us. So neither the
    reductions nor the k / q loads are the per-row cost.
  - Next suspect: each row step's uncached global loads of v / g / beta (16 dependent rounds). Ablation to try: all
    S rows' inputs preloaded into threadgroup memory before the time loop.
- **GDN-STEP3 prefetch ablation (Fable B's pf bit mask; Fri 09:12, share lock beside EvalSuite): STOP.**
  - Every pf arm is bitwise == mlx_lm (an exact candidate).
  - S 16 chained: mlx_lm 63.7, pf 0 74.9, pf 1 93.4, pf 4 74.1, pf 3 83.3, pf 7 69.6 us. The best arm is -7 % off pf 0
    (line >= 25 %).
  - mlx_lm's own kernel stays the fastest form, so the 16-row GDN step stays as it is.
- **INSTALLED Fri 09:12:23 (BigMoE's pre-flight done): ref2.py 438d4774 (DN-KS v2 behind REF2_DN_KS=2), d7b_nax.py d9e32317
  (qmm32c behind D7BNAX_32=2).**
  - Live self_test passes; both off by default; =1 values unchanged.
  - Backups: data/dnks2/ref2_before.py (1851aa5c), data/d17nax2/d7b_nax_before.py (dd9066d3).
  - Engine A/B next (quiet): batt / bv2 / batt2.
- **v2 engine A-B-A (Fri 09:13-09:19, quiet lock, C 224; arms batt / bv2 / batt2 on att_single; data/stepb/out/*_v2_*).**
  - Drafter: loop draft per cycle 6.90-6.94 / **6.18-6.24** / 6.98-7.03 ms; propose walls 6.52-6.78 / **5.68-5.90** /
    6.58-6.78. So **D7BNAX_32=2 is -0.75 ms per cycle (-0.85 per propose): PASS** on the drafter line (-0.5 for the
    v2 step).
  - Check side: the A arms drifted 2.1 ms apart (FLAT 41.99 vs 44.08; check + commit ~41.6 vs ~43.4: heat between the
    first and the last arm). bv2's FLAT (42.81) and check + commit (42.3-42.8) sit between them.
  - **DN-KS v2's in_proj ks 2 (expected -0.5) is NOT resolved here: VOID by drift.** Engaged: dn_ks 105,600 vs 89,760.
  - Recommendation for tonight's stack: D7BNAX_32=2 yes. Drafts are verified; it is bitwise == =1's halves on the big
    shapes, and k|v / 2560 -> 640 leave MLX. REF2_DN_KS stays 1 unless BigMoE's own A-B-A-B shows v2.
  - Lesson: back-to-back 45 GB arms drift ~2 ms in heat. A-B-A brackets it only when the drift is linear; this time
    batt ran cool. Small check-side effects (< 1 ms) need interleaved passes in one process.

## NEVER-WAIT in ref2's one-call (Leap, Fri 2026-11-06 09:2x-09:3x; the orchestrator's assignment, BigMoE's interface)
- Spec ~/FableGod/far/NEVERWAIT_TEST.md section 6; interface ~/BigMoE/docs/NEVERWAIT_IFACE.md; design data/nw/NW_DESIGN.md.
- Switches: REF2_NEVERWAIT=1 arms it; ref2.NW_ON (runtime, read per one-call; r57 sets it around verify passes);
  ref2.NW_RESET() per check; ref2.NWCNT = [] for per-call float32 counters [late groups, late uses, late mass, layer-47
  drops, layer-47 rank-0 stalls, forced uses]; REF2_NW_FORCE='L:e1,e2;L2:e3' (test only).
- How (no kernel change):
  - A cold DEMAND pair (post slot -1, not QOM's bit-21 optional use) runs one layer late.
  - Layer k's own kernel takes it at slot 0 (any valid slot) at weight 0, with its LSB at -2 (U3's no-LSB path). So
    nothing spins for it.
  - At layer k + 1 the same one-call kernel runs on layer k's pool and saved input with only the late pairs. Their -1
    slots resolve through layer k's release region, so it waits only if a read has not landed after a whole layer.
  - ref2's _hc_write adds the late term into layer k + 1's hyper before its own MLP-side write, through layer k's
    inject. A where-guard keeps zero terms exact, and the D5 carry still describes one write.
  - Layer 47: a cold rank 0 waits, the other late pairs drop.
  - While NW_ON, a one-call decline raises.
  - Kernels group resident pairs by slot (ooo1cq's key), so the weight-0 pairs cost one group.
- **Bit checks (scripts/nw_check.py -> data/nw_check.txt; 3c-N 1C QOM and NAX6-N 1C-Q, layers 20 -> 44, M 16): ALL PASS.**
  - (w) warm, NW on == off bitwise (3 reps x 2 forms; NWCNT zeros).
  - (f) FORCE == the ordinary-call reference (y_A without the forced pairs, y_B, and the B write == hc_write(h_B + delta x
    inj_A, ...)); counters match the forced pairs.
  - (c) cold demand experts at A's post == the same reference; no timeouts.
  - (l) layer 47: forced rank 0 kept, others dropped, nothing pending; counters right.
  - Bit check 1 (NW off == the live one-call): qomsplit_check with REF2_PATH = the candidate, (o) PASS and (s) PASS.
  - self_test PASS.
- **INSTALLED Fri 09:29:39: ref2.py 84d0ab40** (backup data/nw/ref2_before.py = 438d4774). Off by default; NW_ON False.
- Next: BigMoE's r57 side + its 2-prompt pre-flight (bit check 4), then the speed A/B Saturday night.
  - Stop rule: tonight's map arm KILL on code, or the snw ceiling < 0.3 S.
  - Costs to watch: one extra one-call + one write per layer even when nothing is late. If the ceiling passes, the
    first refinement folds the late pairs into k + 1's own call.
- **K3C diagnostic (hc8r kernel 8; Fri 09:31, share lock; scripts/k3c_diag.py -> data/k3c_diag.txt).**
  - Chained 48 calls: cold 23.5 us (163 GB/s on 3.83 MB), warm 12.9 us (warm / cold 0.55).
  - Mostly bytes: a perfect cold kernel at 250 GB/s is ~15.5 us, so the ceiling is ~-0.7 ms per check over 96 calls.
    Parked behind higher ms-per-hour items.
- **NAX1CQ-TUNE (FAST-NAX's expert one-call at ks / nsg_b / ks_b; Fri 09:32-09:38, share lock beside EvalSuite;
  scripts/nax1cq_tune.py -> data/nax1cq_tune.txt): no consistent winner.**
  - The best config changes with the routing regime (M 16 skewed (4, 4, 1) -4.3 %; M 16 random (2, 2, 2) -10.5 %; M 2
    (5, 4, 2) / (4, 4, 1) -10 %), and (5, 4, 2) is +17 % at M 16. All within contention noise.
  - At ~100-140 groups per call the kernel moves ~180 GB/s cold, so the expert lever is bytes (groups), not the kernel's
    split. ks 8 / 10 do not load (limits).
- **TOGGLE A/B tool (stepb flat2_driver, STEPB_TOGGLE='CFG_KEY=A|B'):** one process, measured passes alternate A B A B; per
  pass the zero-miss check + commit median; the paired (B - A) median. It cancels the ~2 ms drift between processes, for
  runtime-togglable switches (ref2 CFG read per call).
  - Pre-registered for DN-KS v2 (dn_ks 1 vs 2, 10 passes): PASS paired median <= -0.3 ms, KILL > -0.1.
- **DN-KS v2 by TOGGLE (Fri 09:34-09:36, quiet lock, C 224, one process, dn_ks 1 / 2 alternating over 10 passes, base
  D7BNAX_32=2 + att_single; data/stepb/out/stepb_tog_224_togks_*): BETWEEN, about -0.25 ms per 2-row check.**
  - Paired (2 - 1) zero-miss check + commit per pass pair: +0.51, -0.51, -0.32, -0.19 ms; the first pair had no zero-miss
    checks. Median -0.26.
  - All-checks check + commit, passes 3-10: 44.05 vs 43.80 (-0.25). Zero-miss cycles: 50.21 vs 50.03 (-0.18).
  - Line: PASS <= -0.3, KILL > -0.1. As with DN-KS v1, the engine sees about half the chained gain (overlap).
  - Keep REF2_DN_KS=1 in the stacks. =2 stays available.
  - The tool resolves ~0.2-0.3 ms with 4-5 pairs (pair noise ~+-0.5 ms). The read-out now skips pairs without zero-miss
    checks.
- **NO-WAIT CEILING on the engine's own blocks (Fri 09:38-09:40, quiet lock, C 224; FLAT-2 real-block passes,
  HumanEval/68, 160 tokens, 12 checks of 13-16 rows, kept 14.42 per check; data/stepb/out/stepb_{nax0,stack}_224_ceil_*).**
  - nax0 (FAST-NAX as this morning): draft ~10.0 ms, check + commit 67.2 ms (pass 3; 92 residual misses), cycle ~77 ms ->
    ~187 tok/s.
  - **stack (FAST-NAX + DN-KS + D7BNAX_32=2 + att_single + LAZY): draft ~7.2, check + commit 61.2 (85 misses), cycle ~68
    ms -> ~211 tok/s: -11 % cycle.**
  - The 300-at-K14 budget needs a ~47 ms cycle, so ~21 ms remain. Residual misses remain in both arms (all-resident is
    slightly faster). This prompt's K 14.4 is high; the 12-HumanEval average is ~9.4-10.
- **NEVER-WAIT v2 staged (Fri 09:41; data/nw2/ref2_cand.py md5 287b1245; NOT installed: lands only if tonight's ceiling
  passes, the orchestrator 09:3x).**
  - On FAST-NAX's 1C-Q a skipped pair is slot -2 (zero rows, no weight read) instead of slot 0 (a weight-0 read). This
    applies to the own call's late pairs and to the late call's other pairs. So a late call with nothing late reads
    nothing. 3c-N keeps slot 0 (its -2 reads rank 0's slot).
  - nw_check with REF2_PATH = v2: (w) / (f) / (c) / (l) all PASS on both forms.
  - Measuring NW's own overhead (the extra call + write per layer when nothing is late) needs BigMoE's r57 side
    (NW_ON around verify passes only). Then a TOGGLE A/B in one process.

## ENGINE GAP HUNT (Leap, Fri 2026-11-06 21:0x-21:1x; the orchestrator 21:0x: "dense ~33-36 ms per check in situ vs your
19.6 ms of chained kernels: where are the other ~13-16 ms?")
- Constraint: the evening's free memory (~15-31 GB, Kamen's apps open) does not fit the 45 GB engine. So the runs used
  small uniform pools with no victim tier (C 48 / 128), on BigMoE's sat STACK env (FAST-NAX + DN-KS + LAZY + D7BNAX_32=1
  + R57_FASTTAX).
- Tool: gpu_util_sampler.sh (ioreg 'Device Utilization %', ~10-20 Hz, no sudo) joined with stamped probe windows
  (stepb STEPB_DENSE / STEPB_FULL; gap_read.py).
- **The dense pass in situ at 16 rows, experts stubbed (no post, no reads), unsynced: 25.0 / 24.9 ms per forward (C 48 /
  C 128, 80 / 60 forwards). 20 ms inside the forward call (Python build; the GPU starts every 4 layers) + ~5 ms final
  eval wait. GPU busy 87-89 % over the window (incl. the rollbacks between forwards), so ~22 ms of GPU work per forward,
  GPU-bound.**
- The same forward's synced per-piece split sums 32-39 ms: one eval per piece, each paying a fresh command-buffer start.
  **So the "dense 33-36 ms" of the check's split is that accounting, not GPU time. The real gap over the 19.6 ms of
  chained kernels is ~2-3 ms (glue kernels + dependent dispatches) + ~1-3 ms of GPU idle (the start lag before the first
  4-layer submit).**
- Hence the full all-resident 16-row check ~51 ms (FAST-NAX, C 224, this morning's probe) = dense ~25 + expert path ~26.
  The real-block check + commit (61.2 at C 224 with residual misses; 78.5 in BigMoE's A/B) adds misses / spins and the
  commit. The expert path and the waits hold most of the check, not dense glue.
- FULL 16-row check at C 128 (real experts, warm repeats): 119.6 ms per forward, GPU 'busy' 78 %. NOT representative:
  the small uniform pools churn (2,545 misses per 2-row pass), and the GPU's spin-waits count as busy. The full check's
  GPU-busy vs wall needs C 224 (next free 45 GB slot).
- The expert path blocks the forward: the full check spends ~44.5 ms inside the forward call at C 224 against ~10-12 ms of
  leaf Python (gdn 2.3-2.8, att 1.6-2.6, HC 2-2.3, experts 3.2-6.1 ms). So the forward waits on the GPU inside the
  expert path (~25 ms). Whether that leaves the GPU idle between layers is the next measurement at C 224.
- Data: data/stepb/out/stepb_bstackd_48_gap*, stepb_bstackf_128_gap3_*, gpu_util_gap*.txt; scripts/stepb/gap_read.py,
  gpu_util_sampler.sh.
- **NEVER-WAIT v3 staged (Fri 21:1x; data/nw3/ref2_cand.py md5 dcd5f4f8; NOT live).** Diagnosis of BigMoE's NW v1 KILL
  (stalled calls 17 vs 17, +6.5 ms per check):
  - v1 left cold QOM optional uses and cold LSB sidecars (resident base) at -1. So each layer's own call still waited on
    the release, which lands with the layer's demand reads.
  - v3 with NW on: cold demand -> late (as before); cold optional -> dropped now (weight 0; the loader's -2 anyway); a
    cold sidecar with a resident base -> 3-bit now (U3's -2, lossy: the spec's 8.1 'cheaper answer'). FAST-NAX skipped
    pairs read nothing (v2).
  - NWCNT has 8 fields (+ optional dropped, + sidecars 3-bit).
  - nw_check PASS (warm / force / cold / layer 47; the new paths are not exercised by its scenarios).
  - Waiting for BigMoE's fday data on the waited pairs' kinds and the orchestrator's go.
- The C 224 gap run (full 16-row check GPU-busy vs wall) is slotted ~01:35 after BigMoE's ticket. Arm bstackF (the
  stack env, normal pools, STEPB_FULL / STEPB_DENSE 60 each), on NewIdeas' queue word.
- **NEVER-WAIT v3: install + engine A/B, pre-registered Fri 21:3x (the orchestrator's go 21:3x).**
  - Install: Saturday after BigMoE's night ticket and the HOLD.
    - Bit check 1 (qomsplit_check with REF2_PATH = v3: off == the live one-call) + nw_check, then the atomic install, OFF
      by default (REF2_NEVERWAIT arms, ref2.NW_ON per verify pass).
    - The v3 vs live diff sits entirely inside the NW-active block, so off == today by construction.
  - A/B (BigMoE's harness, 12 HumanEval, C 224, A-B-B-A): the sat STACK vs STACK + R57_NEVERWAIT=1 (v3).
    S = the stack's median gate service per check (~14 ms in fday).
    - PASS: median check >= 0.5 x S shorter AND kept per check within 2 %.
    - KILL: < 0.2 x S shorter, or kept down > 2 %.
    - BETWEEN otherwise.
    - Report the NWCNT fields (late / dropped-optional / 3-bit sidecars per check).
  - Quality: v3 adds lossy parts beyond nw1's map PASS (cold optional uses dropped now, cold sidecars 3-bit). NewIdeas
    decides whether they need their own map arm or a short same-scores check on code, planned in the same window.
- **NEVER-WAIT v3 INSTALLED, OFF by default (Fri 23:4x; Kamen opened the window at 23:30, so this landed before BigMoE's
  night ticket instead of after it; no engine process was running).**
  - Bit check 1 (share lock, 23:4x): qomsplit_check.py with REF2_PATH = data/nw3/ref2_cand.py PASS.
    - (o): v3 with NW off == live, bitwise, for QOM and non-QOM calls in both one-call forms (3c-N, NAX6-N 1C-Q),
      layers 20 and 44, with QCNT_1C equal.
    - (s): the split rule's reference PASS, 416 split drops exercised.
    - Data: data/qomsplit_check.txt.
  - nw_check on v3 (Fri 21:16, before the HOLD): (w) warm == off, (f) FORCE == reference, (c) cold == reference and
    (l) layer 47 all PASS (data/nw_check.txt).
  - Install: atomic (temp + mv). scripts/ref2.py 84d0ab40 -> dcd5f4f8, backup in data/nw3/ref2_before.py. ref2.apply()
    + self_test all True. NW_ON False, CFG neverwait False. REF2_NEVERWAIT=1 arms it.
  - Every changed line is inside `if nw:` (nw = NW_ON and CFG["neverwait"]), so the off path is today's code.
- **NEVER-WAIT v3 quality / speed gates (NewIdeas 21:4x, QUEUE dc9aff8 S1).**
  - Its own map arm first: Breakthrough preps nw3 tonight on the CPU. Cold sidecars at 3 bits (LSB-on-miss) are a new
    lossy path; the worst-case bound (every rank 0-1 term of unsure rows at 3 bits) stands in if residency can't be
    modelled.
  - The engine A/B runs speed-only beside it. NewIdeas' line replaces mine above: PASS >= +8 % tok/s, KILL < +3 %. It
    reports NWCNT (3-bit sidecars, optional uses dropped, late groups per check).
  - Note: BigMoE's 'stalled calls' counts calls that issued a read, not waits.
  - Adoption: map PASS + S4's short same-scores gate.

## GPU IDLE AT 16 ROWS + SB CADENCE: S0's input, S2's glue fix (Leap, Fri 2026-11-06 23:43-23:49, quiet locks)
Question (the orchestrator 21:3x; NewIdeas' S0 bar): on the stack env, is the GPU idle >= ~8 ms per 16-row check? If so,
the fully async post (S0) is Saturday's top build. Also S2's glue fix: SB cadence, submit every 2 layers vs 4.
- **Env:** BASE + R57_NAX=1 REF2_NAX_1CQ=1 REF2_DN_KS=1 REF2_GDN_LAZY=1 D7BNAX_32=1 R57_FASTTAX=1 (run_stepb arm bstackF).
  The ioreg "Device Utilization %" sampler (scripts/stepb/gpu_util_sampler.sh) ticked every 46 ms.
- **Method note:** a pass's average GPU busy includes its prefill. The 12 checks of a 160-token real pass take ~0.85 s of
  a ~7.1 s pass, so the per-pass averages (35-40 %) say nothing about checks. The numbers below average the samples over
  each pass's last 0.73-0.80 s, which holds the 11 steady checks. There the raw samples read 58-100 %, against the 0 %
  gap seen before the decode starts.
- "Busy" counts any resident kernel, including expert kernels spinning on the gate's release. "Idle" means no work queued.

**Real 16-row loop, C 224 (FLAT2_REAL: the engine's own blocks, rows 13-16, kept ~14.3; 3 measured passes per arm):**

| arm | pass | draft ms | check+commit ms | cycle ms | GPU busy (steady) | GPU idle per cycle |
|---|---|---|---|---|---|---|
| SB 4 (today) | real 2 | 8.03 | 63.03 (build 55.42, wait 6.50) | 71.06 | 88.6 % (17 samples) | 8.1 ms |
| SB 4 | real 3 | 8.00 | 62.48 (build 54.81, wait 6.36) | 70.48 | 88.1 % (17) | 8.4 ms |
| SB 4 | commit pass | 7.57 | 64.63 (build 54.22, wait 6.39) | 72.20 | 84.6 % (19) | 11.1 ms |
| SB 2 (STEPB_SB_N=2) | real 2 | 7.83 | 60.66 (build 54.54, wait 5.63) | 68.49 | 96.3 % (16) | 2.5 ms |
| SB 2 | real 3 | 7.96 | 59.98 (build 53.05, wait 5.47) | 67.94 | 96.1 % (16) | 2.7 ms |
| SB 2 | commit pass | 7.49 | 61.13 (build 52.85, wait 5.43) | 68.62 | 94.8 % (16) | 3.6 ms |

- **SB 4 to SB 2:** the cycle is -2.6 ms (-2.57 / -2.54 / -3.58), and idle drops from 8.1-11.1 to 2.5-3.6 ms per cycle.
  - Tokens look unchanged: kept per check is identical between the arms in all 4 real passes (FAST varies across
    passes with cache state, not between the arms).
  - On this prompt that is ~199-205 tok/s (SB 4) vs ~207-212 tok/s (SB 2), from kept / cycle. One prompt; not a benchmark.
- **2-row checks, in-process TOGGLE (C 208, 10 passes, ns:STEPB_SBN = 4 | 2; zero-miss checks):**
  - Paired check+commit B - A: -5.64 / -7.57 / -5.79 / -5.60 ms, **median -5.71 ms**. Cycles ~50.1 (SB 4) vs ~44.3 (SB 2).
  - Separate processes at C 224 agree: check+commit 46.45 / 46.24 (SB 4) vs 40.16 / 39.69 (SB 2).
  - At 2 rows the steady GPU busy is 80.8 % (SB 4) vs 67.9 % (SB 2). SB 2 is shorter but less busy, so at SB 4 ~10 ms
    per 2-row cycle is spin, not useful work.
- **Warm 16-row forward, back to back (FULL16 probe, 60 unsynced forwards):**

  | where | wall ms | GPU ms (busy %) | Python build ms |
  |---|---|---|---|
  | C 224 SB 4 | 45.80 | ~40.2 (88 %) | 40.21 |
  | C 224 SB 2 | 44.59 | ~36.8 (82 %) | 39.98 |
  | C 208 | 46.06 | ~41.3 (90 %) | 40.33 |

  - DENSE-ONLY (experts stubbed): wall 22.68, GPU ~20.5 (90 %), build 15.78 ms.
  - ~~So the expert call adds ~24 ms of Python per 16-row forward (~0.5 ms per layer).~~ **CORRECTED Sat 00:2x by XPT
    (next section):** the expert call is ~0.08 ms of Python per layer. The FULL vs DENSE build gap is Metal encoding plus
    waiting on the busy GPU inside mx.async_eval.
- **Verdict.**
  - S0 (fully async post): 8.1-11.1 ms of idle exists today, but SB 2 alone recovers it down to ~3 ms per cycle, below
    the 8 ms bar. Not worth building after SB 2.
  - S2: R57_SBN=2 is the glue fix (scheduling only; S2's A/B confirms the speed and token equality).
  - With SB 2 the 16-row cycle is ~96 % GPU-busy. What remains is GPU work plus spin on the gate's release (gate
    service 4-18 ms per check in these passes), which is S1's (NW v3) target.
- **Wiring for S2:** ~/Leap/data/s2wire/r57_s2.diff, against BigMoE's r57_screen.py fa8bee48 (applies clean), all off by
  default. BigMoE applies it.
  - R57_SBN=N replaces the literal 4 in ref_on's midloop lambda.
  - R57_SVH=1 / R57_DSUB=1 / R57_SV_CHECK=1: the SESSION-VOCAB hooks from data/r57_session_vocab.diff (Wed), re-based.
  - Smoke copy with outputs in ~/Leap: data/s2wire/r57_s2_smoke.py. Compile check and smoke after BigMoE's lock.
- **Data:** data/stepb/gap_night.txt; data/stepb/gpu_util_night.txt (C 208 run + TOGGLE) and gpu_util_real.txt (C 224);
  out/stepb_bstackF_208_night*, stepb_togs_208_sbn*, stepb_bstackF_224_real* / _real2*.
  - The 23:43 C 208 run measured 2-row FLAT passes and probes only: the waiter did not set STEPB_REAL. Corrected by
    the 23:46-23:49 C 224 real runs.

## XPT: the Python thread per real 16-row check, the cycle floor, and the S2 smoke (Leap, Sat 2026-11-07 00:23-00:27, quiet lock)
The orchestrator asked (Fri 23:5x) for the expert path's Python ms per layer at SB 2 in the steady window, so NewIdeas can
order S3 (the CPU-side lever) against S1, and for NW v3's ceiling priced against the Python floor. NewIdeas gave the slot.
- **Tool:** STEPB_XPT=1 (scripts/stepb/xpt.py; unit test scripts/stepb/test_xpt.py PASS). On real-block passes only, it
  wraps three call sites, recording wall (perf_counter) and thread CPU (time.thread_time) for each:
  - the trunk's DecoderLayer.__call__;
  - the expert call, type(l.mlp.switch_mlp).__call__;
  - mx.async_eval.
  - It also stamps thread CPU at each of run()'s F2C cycle stamps.
  - Calls are binned into cycles by those stamps, in entry order. Overhead ~0.4 ms per forward.
- **Env:** C 224, the stack (run_stepb arm bstackF), FLAT2_REAL passes (rows 12-16, kept 14.4).
  - The 3 steady passes per arm are real pass 2, real pass 3 and the commit pass.
  - Values are per-check medians over checks 2-12.

| arm | pass | cycle wall | Python CPU per cycle | layer calls wall / CPU | expert calls (48) CPU | async_eval wall / CPU | engine build |
|---|---|---|---|---|---|---|---|
| SB 2 | real 2 | 68.72 | 40.94 | 16.20 / 16.16 | 3.97 | 37.50 / 18.89 | 54.30 |
| SB 2 | real 3 | 68.00 | 38.91 | 15.24 / 15.24 | 3.75 | 37.36 / 18.02 | 52.93 |
| SB 2 | commit | 66.52 | 40.15 | 16.10 / 16.10 | 3.91 | 34.90 / 19.29 | 51.91 |
| SB 4 | real 2 | 70.51 | 39.86 | 15.66 / 15.64 | 3.80 | 38.42 / 17.20 | 54.81 |
| SB 4 | real 3 | 69.99 | 39.72 | 15.50 / 15.50 | 3.82 | 37.09 / 17.25 | 53.94 |
| SB 4 | commit | 67.79 | 39.37 | 15.53 / 15.53 | 3.77 | 36.35 / 17.23 | 51.94 |

- **The Python thread's CPU per cycle is 38.9-40.9 ms (SB 2)**, against a 66.5-68.7 ms cycle. Where it goes:
  - The graph build (48 layer calls, wall ≈ CPU: no waits there): 15.2-16.2 ms. Of that, the expert calls are 3.75-3.97 ms
    (**0.08 ms per layer**), and the dense + glue part is ~12.
  - Metal encoding inside mx.async_eval: 18-19.3 ms CPU.
  - Draft, commit and the loop: ~5.5 ms.
- **The waiting is inside mx.async_eval:** wall 35-37.5 ms vs CPU 18-19, so the thread waits ~16-19 ms per check there.
  This is back-pressure from the GPU, which is ~96 % busy at SB 2 (previous section). A further ~5.5 ms is the final
  wait after the build.
- **The cycle floor is ~40 ms of CPU**, so today's GPU-bound cycle (~65.5 ms busy) sits ~25-27 ms above it.
  - NW v3 (S1) priced against the floor: its target, the spin on gate releases (median gate service ~7 ms per check), fits
    inside that margin, so it is not floor-limited. Price it as GPU time.
  - S3 (CPU-side): the expert glue's Python is only ~3.9 ms per forward, so compiling it is worth at most ~3-4 ms of CPU,
    and only once GPU time per cycle nears ~40-45 ms. The larger CPU items are encoding (~19) and the dense graph build
    (~12); fewer dispatches cut those and GPU time together. **Order: S1 before S3; S3 parked.**
- **SB 2 vs SB 4 in this slot:** cycle -1.8 / -2.0 / -1.3 ms; with the 23:4x runs, -1.3..-3.6 ms per 16-row cycle. At SB 2
  async_eval's CPU is ~+1.5 ms (twice the submits) and its wait is lower.
- **cProfile (STEPB_XPT_PROF=1, out/stepb_bstackF_224_xptp_prof.txt):** the pass profile is dominated by each pass's
  prefill. prefill_call takes 5.5 of 6.8 s, and BigMoE's tier _read_rebuilt alone 3.1 s (48 calls, 65 ms each: the
  prompt's expert reads). So it does not separate the decode; XPT does.
- **S2 smoke PASS** (data/s2wire/r57_s2_smoke.py = BigMoE's r57 fa8bee48 + data/s2wire/r57_s2.diff, outputs in
  data/s2wire/r57/).
  - Env: the stack + R57_SVH=1 R57_DSUB=1 R57_SV_CHECK=1 R57_SBN=2 D7BNAX_32=2, C 224, 2 prompts. 517 tokens, rc 0,
    footprint peak 38.6 GB.
  - SESSION-VOCAB {prompts 2, committed 517, head2s 69, chk_rows 1104, chk_out 0, svh_extra 3, svh_calls 69, dsub_extra 3}:
    svh_calls == head2s (checks == proposes), chk_out 0, dsub_extra == svh_extra, all as pre-registered (SESSION-VOCAB, Wed).
  - BigMoE applied the diff in its r57_screen_fast.py (97a23bf). S2's speed + token-equality A/B is BigMoE's.
- **Data:** data/stepb/gap_xpt.txt, gpu_util_xpt.txt; out/stepb_bstackF_224_xpt2* / xpt4* / xptp* (+ _prof.txt);
  data/s2wire/s2smoke.log, data/s2wire/r57/s2smoke_*.

## S2b STEP 1 (in-situ price per dense piece) + S2c STEP 1 (own block vs alternating) (Leap, Sat 2026-11-07 00:42-00:44, quiet)
NewIdeas' QUEUE f270dee lines. C 224, SB 2 (S2's PASS), stack env (bstackF), ioreg sampler. Tools: STEPB_ABL / STEPB_S2C
(flat2_probe._abl_probe / _s2c_probe; the memo stub is unit-tested in test_abl.py).
- **S2b step 1.** The 16-row dense-only forward (every expert call returns zeros) at the 2-row probe check 8. 4 rounds x
  12 timed forwards per variant, interleaved. Each variant stubs one piece: its first call runs for real and is kept;
  later calls return the kept outputs through mx.depends on every input. Base: wall 21.85 ms, build 15.32, GPU busy 94 %.

| stubbed piece | wall saved (ms) | GPU saved (ms, busy % x wall) | note |
|---|---|---|---|
| HC reads, both (96 calls) | 6.82 | 11.3 | variant runs at 61 % busy, near its CPU build: wall is a lower bound |
| HC read, attention side (48) | 5.05 | 6.3 | ~105 us per layer |
| HC read, MLP side (48), with the attention write folded in | 2.37 | 2.9 | ~49 us per layer |
| HC write, MLP side (_hyper_residual_write, 48) | 3.58 | 3.3 | ~75 us per layer |
| GDN in_proj (36) | 4.22 | 5.9 | ~117 us per call vs ~113 chained: no excess |
| GDN whole (36, incl. in_proj) | 7.45 | 10.1 | |
| attention whole (12) | 3.53 | 4.3 | ~294 us per layer |
| MoE block, dense part (router + shared, 48) | 1.36 | 1.6 | |
| PLE (1) | 0.62 | 0.75 | |

  - **CORRECTED Sat 01:1x: these deltas are UPPER BOUNDS, inflated by the stub itself.**
    - mx.depends keeps the graph order. But MLX's concurrent encoder puts a barrier only where a kernel reads a buffer
      that an earlier kernel wrote. The stub's kept outputs are old buffers, so the consumers stop waiting for the
      stubbed piece's inputs and overlap with the work before it: the serial chain breaks.
    - Small pieces on the layer-to-layer chain are hit hardest. The HC write's "3.58 ms" is a ~3 us compiled kernel
      plus the broken chain.
    - Fix: a barrier-preserving stub (the kept output plus a zero term read from the inputs' buffers), calibrated by
      adding the same term to the real piece.
  - ~~The HC machinery costs ~10.4 ms; lead a': fold the MLP write into the next attention read (8.6 vs 2.4 ms).~~
    **Withdrawn:** a' is D5 (REF2_HC_PEND=1, section "D5", Wed 11-04): bitwise, KILLed as a speed lever. Isolated
    write + next read went 85.4 -> 83.0 us, or +0.3 us with R-73's compiled write on.
  - (c) in_proj nt 2: in situ equals chained, so there is no excess to win: KILL.
  - The FULL / DENSE probes of the same block: wall 44.53 (GPU ~35.8) / 21.87 (GPU ~18.8) ms.
- **S2c step 1 (real probe pass, check 6, 16 rows): BETWEEN.**
  - The check's own block back to back: wall 48.57 / 49.14 ms (same / same2, 0 misses).
  - Alternating with check 5's block: own 48.84. Cycling checks 4 / 5 / 6: own 49.22. So "a new expert set per forward"
    adds +0.3..+0.6 ms only: not the mechanism. The previous blocks themselves ran 43.9-45.5 ms (103 / 121 misses on their
    first touches).
  - The loop's check+commit at SB 2 is 59.1-63.0 ms (real passes 2 / 3 / commit), or ~57.9 at zero gate service (the
    SVC fit). So the excess reproduces at ~9 ms (>= 8), but no mechanism >= 6 ms is named yet.
  - The 16-row commit (stage pass, synced) is bimodal: ~0.47 ms on most checks, 4.6-5.2 ms on ~1/3 (a GDN replay); ~2 ms
    on average.
  - r57 host-syncs the drafts before the check (`dr = np.array(pth[0]).tolist()`), so the drafter cannot spill into it.
  - Step 1b: per-piece synced timing inside the live check vs the same block replayed synced right after. The sync cost
    is the same in both, so the difference names the pieces that carry the loop's excess.
- **Data:** data/stepb/gap_s2.txt, gpu_util_s2.txt, out/stepb_bstackF_224_s2b*.

## S2b STEP 1 REDONE (barrier kept) + S2c STEP 1b (live check vs its block replayed, per piece) (Leap, Sat 2026-11-07 00:54-00:56, quiet)
C 224, SB 2, stack env. The ablation now uses the barrier-preserving stub (`bar`: the kept output plus a zero term read
from every input's buffer) and its calibration (`cal`: the real piece plus the same term). Price = wall(cal) - wall(bar).
Both modes are unit-tested (test_abl.py).
- **S2b prices, in place, barrier kept (16-row dense-only, per check):**

| piece | price, wall ms (GPU ms) | the old depends stub said |
|---|---|---|
| HC read, attention side (48) | 2.12 (2.33) | 5.05 |
| HC read, MLP side, incl. the folded attention write (48) | 2.20 | 2.37 |
| HC write, MLP side (48) | 0.14 (-0.03) | 3.44 |
| attention (12) | 3.38 (2.42) | 3.53 |
| GDN (36), incl. in_proj | 7.65 (10.5) | 7.45 |
| GDN in_proj (36) | 4.63 (5.98) | 4.22 |
| MoE dense part: router + shared (48) | 1.17 (2.02) | 1.36 |
| D5 fold on (REF2_HC_PEND) | +0.04 saved | - |

  - Verdicts:
    - (a) HC in-loop excess: KILL. In place the reads cost 4.3 ms vs 5.95 chained; the "8.6 in situ" was synced
      accounting.
    - a' = D5: no gain (as on Wed 11-04).
    - (c) in_proj: KILL. S2b rests on (e) eprep and (f) the folds.
  - The zero term costs 0.0-0.6 ms per variant (cal vs base). GPU-ms estimates from busy % are noisy for the small
    pieces; wall is the price.
- **S2c step 1b (real probe pass, check 6, 13 rows): the live check synced per piece vs the same block synced right
  after (and again after the alternating forwards):** totals 68.42 vs 58.40 (59.18) ms, i.e. **+10.0 ms live.**

| piece | live | replay | replay 2 | live - replay |
|---|---|---|---|---|
| moe_pre (router + shared expert, evaluated before the expert call) | 10.02 | 4.17 | 4.43 | **+5.85** |
| gdn | 11.91 | 10.39 | 10.32 | +1.52 |
| experts | 27.42 | 26.05 | 26.39 | +1.37 |
| att | 6.43 | 5.51 | 5.50 | +0.93 |
| hc_w / hc_a / hc_m / hc_f / moe / ple | | | | ~0 each |

  - moe_pre is BigMoE's tier._call: shared expert + softmax(gate) + argpartition + sort, the same ops live and in the
    replay. Live they take ~209 us per layer vs ~87: the same ops run slower in the loop.
  - Candidates:
    - (1) the loader's background reads during the live check (look-ahead prefetch d = 1-3, miss reads). The replay
      has nothing to read.
    - (2) CPU contention on the encode: loader_main runs at user-interactive priority on a P-core, and synced timing
      includes the encode.
  - GDN's +1.5 fits the lazy commit's GDN replay; experts' +1.4 is gate service.
  - Unsynced, the own block back to back is 47.95 / 48.24 ms; alternating blocks 48.10 / 48.24 (own); the loop's
    check+commit 59.0-62.7.
  - **Verdict vs the line: BETWEEN.** The excess reproduces at ~10 ms (>= 8). The location holds 5.85 ms (just under
    6), and its mechanism is not proven.
  - Next: step 1b with the look-ahead prefetch off (BigMoE's switch), plus XPT's CPU vs wall around moe_pre.
- **Data:** data/stepb/gap_s2b.txt, gpu_util_s2b.txt, out/stepb_bstackF_224_s2b2*.

## S2c STEP 1b ARMS: keepalive / VTIER, and the thread's CPU per piece (Leap, Sat 2026-11-07 01:03-01:06, quiet)
C 224, SB 2, the stack env. Real check 6: the live check synced per piece vs its block replayed synced right after, now
with the thread's CPU per piece (time.thread_time at every wrapper stamp). One process per arm (BigMoE's suggestions).
NewIdeas' line: an arm names the cause if its moe_pre live - replay falls by >= 4 ms vs base.

| arm | total live / replay (ms) | moe_pre live - replay | moe_pre per layer: wall / CPU (live - replay) | experts live - replay |
|---|---|---|---|---|
| base | 65.77 / 55.21 | +5.65 | +0.117 / +0.031 | +1.71 |
| BIGMOE_SSD_KEEPALIVE_US=0 | 64.36 / 52.21 | +5.90 | +0.122 / +0.030 | +3.31 |
| R57_VTIER= (no tier copies) | 65.13 / 55.73 | +5.56 | +0.117 / +0.030 | +1.51 |

- **Neither switch is the cause** (no fall >= 4 ms). Look-ahead prefetch is off under gate+t, and the gated path has no
  host readback (post_ids / spin_wait are lazy GPU ops).
- **New: the Python thread spends ~+20 % more CPU on identical code in the live check**, on every piece.

  | piece | thread CPU live (ms) | thread CPU replay (ms) |
  |---|---|---|
  | GDN | 5.11 | 4.00 |
  | attention | 5.49 | 4.57 |
  | MoE block | 15.2 | 12.4 |
  | HC pieces | +0.1..+0.2 over the replay | |
  | whole forward ("layer") | 33.3 | 27.8 |

  - So ~half of the ~10.5 ms excess is the same Python and encode running slower, and the other half is waiting.
  - In moe_pre itself ~75 % of the excess is non-CPU waiting (+0.09 ms per layer) and ~25 % CPU.
- The same code ~20 % slower on the CPU points at core placement or clock: the decode thread runs at default QOS while
  BigMoE's loader_main spins at user-interactive on a P-core.
- Next (step 1c): arms STEPB_QOS=1 (the decode thread at QOS user-interactive; pthread_set_qos_class_self_np, tested rc
  0) and BIGMOE_IDLE_MS=0.5 (the loader naps when idle), and both.
- Data: data/stepb/gap_s2c.txt, out/stepb_bstackF_224_s2c_{base,ka0,vt0}*.

## S2c STEP 1c: core placement (Leap, Sat 2026-11-07 01:09-01:13, quiet; no PAUSE, die max 67-77 C, fans 6500)
Same probe as step 1b (real check 6, live synced per piece vs replay, thread CPU), C 224, SB 2. One process per arm.

| arm | live - replay total (ms) | moe_pre | thread CPU live - replay (forward) | experts live - replay | real pass 2 check+commit |
|---|---|---|---|---|---|
| base | +10.77 | +5.67 | +5.58 | +1.70 | 63.58 |
| STEPB_QOS=1 (decode thread QOS user-interactive) | +9.34 | +5.51 | +5.18 | +1.73 | 62.79 |
| BIGMOE_IDLE_MS=0.5 (loader naps when idle) | +9.43 | +5.28 | +4.25 | +1.48 | **77.62** (pickup latency) |
| both | +11.84 | +5.71 | +7.36 | +1.94 | 66.55 |

- **No arm names the cause** (none falls by >= 4 ms). Ruled out:
  - core placement via QOS;
  - the loader's spin (and IDLE_MS slows the loop: not a fix);
  - the keepalive and the tier copies (step 1b arms);
  - look-ahead prefetch (off under gate+t);
  - a host readback in the gated path (lazy post / spin).
- Next suspect: MLX's buffer cache, which BigMoE's stream.py:486 caps at 1 GB (BIGMOE_MLX_CACHE_MB, default 1024).
  - A live check after the drafter and the commit may allocate fresh Metal buffers (more CPU, plus waits in the
    encode); the replay right after reuses the buffers the live check freed.
  - Arm: BIGMOE_MLX_CACHE_MB=6144 vs base.
- Data: data/stepb/gap_s2d.txt, out/stepb_bstackF_224_s2d_{base,qos,idle,qi}*.

## S2c STEP 1d: MLX's buffer cache (Leap, Sat 2026-11-07 01:17-01:20, quiet; no PAUSE, die max <= 77 C)
Same probe, C 224, SB 2. Arms set BIGMOE_MLX_CACHE_MB (BigMoE's stream.py:486 caps MLX's buffer cache at 1 GB). Guards:
RSS > 45 GB, swap +2 GB, cycle > 2x (BigMoE's lines). None tripped.

| arm | live - replay total (ms) | moe_pre | lifetime peak footprint | MLX cache, live (GB) | MLX peak, live / replay (GB) |
|---|---|---|---|---|---|
| 1 GB (base) | +10.37 | +5.61 | 39.3 GB | 0.978 -> 0.831 | 36.71 / 36.59 |
| 3 GB | +11.24 | +5.74 | 39.7 GB | 0.959 -> 0.812 | 36.71 / 36.59 |
| 5 GB | +8.75 | +5.54 | 39.7 GB | 0.959 -> 0.811 | 36.71 / 36.59 |

- **Not the cause.** The cache stays under 1 GB in every arm, live and replay, so the cap never binds. The live check
  moves ~0.15 GB cache -> active; its MLX peak is only 0.12 GB above the replay's.
- **S2c, all measured tonight:**
  - Out: look-ahead prefetch, gate readback, keepalive, tier copies, QOS user-interactive, loader nap, MLX cache size.
  - What stands: the live check is ~+10 ms vs its own block replayed right after (synced; 6 runs). moe_pre is +5.6
    (uniform +0.12 ms per layer: ~0.03 CPU + ~0.09 wait), and the Python thread spends ~+20 % CPU on identical code
    in every piece.
  - Verdict: BETWEEN (location named, mechanism not).
  - Optional last test: replay the block right before the live check, to decide "first forward after the draft"
    against "live-only".
- Data: data/stepb/gap_s2e.txt, out/stepb_bstackF_224_s2e_{base,c3g,c5g}*.

## S2c STEP 1e: KEEP REF2's CACHES ACROSS THE LOOP's PER-CYCLE remove() -- PASS (Leap, Sat 2026-11-07 01:24-01:26, quiet)
**The in-loop excess is named and fixed.**
- **Mechanism.** r57's loop calls ref2.remove() at every cycle start (the drafter runs without REF2) and ref2.apply()
  before every check. remove() cleared REF2's per-weight caches: _SN (SMALL-N padded native copies), _DN, _HC8 (HC8
  parameter views). So every live check rebuilt them on first use. That means a pad plus _dn_repack (MLX ops on the
  GPU) for every SMALL-N weight, plus the Python rebuilds:
  - the shared expert's gate|up / gate in all 48 layers: the uniform +0.12 ms per layer in moe_pre;
  - GDN b|a (36 layers) and attention k / v (12 layers);
  - the HC8 views.
  The replay right after found them warm.
- **Arm.** STEPB_KEEP_REF2C=1, a harness patch: the loop's remove() only switches REF2 off (_ON) and drops the D5
  carry. The caches are identity-validated (e[0] is w), so the same arrays come back: the same bits by construction.
- **Result.** C 224, SB 2, stack env; one process per arm; no PAUSE, die max 75.5 C.

| | base | keep |
|---|---|---|
| live - replay total, synced (ms) | +9.39 (64.49 / 55.10) | **+0.48** (54.76 / 54.28) |
| moe_pre / GDN / attention / experts, live - replay | +5.41 / +1.64 / +0.77 / +1.73 | -0.05 / +0.01 / +0.26 / +0.42 |
| thread CPU, live - replay (forward) | +4.65 | +0.48 |
| **real loop check+commit, unsynced (passes 2 / 3 / commit)** | 62.11 / 59.08 / 58.38 | **54.90 / 51.66 / 50.37** |
| draft (ms) | 7.55 / 7.65 / 7.47 | 7.39 / 7.48 / 7.20 |
| lifetime peak footprint | 38.6 GB | 39.4 GB (within tonight's 38.6-39.7 spread) |

  - **-7.2 / -7.4 / -8.0 ms per 16-row check** in the unsynced loop. Cycle ~66-70 -> ~58-62 ms: ~+11-14 % tok/s on
    HumanEval/68 at C 224.
  - The pre-registered line (a fall of >= 7 ms in live - replay) is met: the fall is 8.9 ms. The replay's own time
    doesn't move.
- **Fix installed in live ref2 (01:3x), OFF by default:** REF2_KEEP_CACHES=1 / CFG["keep_caches"].
  - remove() then sets _ON off and drops the D5 carry, keeping _DN / _SN / _HC8.
  - ref2 dcd5f4f8 -> ddbc0621, atomic; backup data/keepc/ref2_before.py.
  - self_test PASS. With the switch off, remove() clears as before; with it on, it keeps the caches (checked).
- **Next:**
  - BigMoE's engine A/B at C 256 vs stackb+S2 (NewIdeas: PASS >= +4 %, KILL < +1.5 %, lifetime peak <= 45.0 GB).
  - The EXACT bitwise token check (NAX6-N, a few prompts, keep on vs off).
  - An equivalent r57-side form: toggle ref2._ON instead of remove() / apply() per cycle.
- **Data:** data/stepb/gap_s2f.txt, out/stepb_bstackF_224_s2f_{base,keep}*.

## S2b f: THE TWO FOLDS, chained (Leap, Sat 2026-11-07 01:28, quiet; scripts/fold_bench.py)
- Real shapes, 16 rows, 4-bit g64 native packs. DN-KS v1's ks per shape: qkv|z 16384 and q 12288 at ks 1, b|a 96 and
  k / v 512 at ks 10.
- Method: a dependent chain over distinct weight copies. Each layer's input depends on every output of the layer
  before, so the unfolded calls of one layer stay independent of each other, as in situ. Round-robin, 7 rounds.

| fold | unfolded per layer | folded per layer | saving per check | vs price | verdict |
|---|---|---|---|---|---|
| GDN: b\|a into qkv\|z (N 16384 + 96 + 32 pad = 16512, ks 1) | 124.0 us | 115.4 us | -0.31 ms (36 layers) | 1.6: 19 % | **KILL** |
| attention: k\|v into q (N 12288 + 512 + 512 = 13312, ks 1) | 105.1 us | 79.2 us | -0.31 ms (12 layers) | 1.1: 28 % | **BETWEEN** |

- Bits: each fold's rows == the unfolded calls at the same ks (bitwise, all segments). Against today's ks 10 for the
  small calls, the max rel diff is 0 (b|a) and 1.1e-7 / 3.0e-8 (k / v). So a fold moves those rows to ks 1, the stock
  K order.
- Why so small: in the chain the small call already overlaps the big one. The unfolded GDN pair costs 124 us vs ~113 + 45
  run alone.
- Note: the fold's 32 all-zero pad rows did NOT come out as exact zeros after MLX quantize + repack. They are sliced
  off, so this is harmless, but R-68's "a zero pack row gives exactly 0" doesn't hold for rows quantized from zeros by
  MLX.
- S2b now: a KILL, c KILL, a' (D5) no gain, f = GDN KILL / attention BETWEEN (-0.31 ms), e (eprep) priced ~0.4 ms
  [P]. S2b is small; tonight's large lever is S2c's REF2_KEEP_CACHES (-7.2..-8.0 ms per check).
- Data: data/fold_bench.txt.

## CYCLE AUDIT for KEEP's bug class + the commit replay (Leap, Sat 2026-11-07 01:3x, CPU only; NewIdeas' ask)
- **No other per-cycle rebuilds found.**
  - ref2's other caches (_WCAT, _DNR, _SH, _NATIVE_FLAGS) are never cleared by remove().
  - Leap's kernel modules (hc8r, qmm16c8, chain_fused, router_fused, glue_fused) keep lru_caches only.
  - draft_fused clears its forms / masks only on uninstall. Its drafter attention mask is keyed by the growing context,
    so it builds and evaluates one new mask per cycle (~0.1-0.3 ms).
  - leapk's ".clear()" is a tile clear inside kernel source.
- **One mergeable sync (r57).** Under FAST rules, mx.eval(dl_, rk_) (target log-prob gap and rank of each draft) is a
  second host round trip right after mx.eval(am, tp). Building them first and evaluating all four together saves
  ~0.13-0.2 ms per cycle, same bits.
  - The QOM-counter readbacks after the commit run only when counters exist. The measured cycle tail is ~0, so they
    don't wait on the commit.
- **The commit is bimodal exactly by partial acceptance.** The real commit pass (keep arm), commit ms per check against
  kept / verified rows:

  | check | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | kept / rows | 13/16 | 15/15 | 13/16 | 16/16 | 12/12 | 16/16 | 14/15 | 14/14 | 16/16 | 16/16 | 12/13 | 16/16 |
  | commit (ms) | 4.8 | 0.4 | 4.6 | 0.4 | 0.4 | 0.4 | 4.7 | 0.4 | 0.4 | 0.2 | 3.9 | 0.4 |

  So kept < rows costs 3.9-4.8 ms, and a full accept ~0.4 (MTPLX returns early).
  - On a partial accept, MTPLX's commit_verified_window replays 35 separate gated_delta_update calls over the kept rows
    from the pre-verify state (~92 us each at 16 rows, ~3.2 ms), plus the PLE-carrying layer's FULL forward over the
    kept rows, experts included. All lazy: r57 submits it with async_eval right after the check.
  - With the GPU ~96 % busy it adds nearly fully to the cycle: ~1.5 ms per cycle on HumanEval/68 (1/3 partial), more
    on lower-acceptance prompts. KEEP doesn't change it.
  - Exact lever (kernel work, after the freeze): the verify pass checkpoints each GDN state at rows 8 and 12 (~140 MB of
    extra writes per check), and the commit replays <= 4 rows from the nearest checkpoint. Same arithmetic, so
    bitwise-safe. Price ~1-3 ms per cycle by the partial-check share. Intermediate: one dispatch for the 35 replays.

## FCP: FIRSTCHK NAMED -- each prompt's check 0 re-wires the 32 GB expert pools (Leap, Sat 2026-11-07 08:28-08:30, quiet)
- Ask (the orchestrator 08:1x): sample the main thread and the GPU timing inside check 0 in-process, name the wait, say
  whether a fix is exact-safe. Setup (scripts/fcp/, pre-registered 831b987): BigMoE's FIRSTCHK control (fc_kill's fcctl)
  on a copy of r57_screen_fast.py (md5 70ed7810): v1' (final_env dd7ad0f, QOM sets humaneval), HumanEval 68 / 16 / 91,
  CAP 64, C 256, R57_A6=ctl R57_TCPU=1 R57_LOGHASH=1, plus fcprobe.py on every window of checks 0-3.

| fcpa, medians over the 3 prompts | check 0 | checks 1-3 | check 0 excess |
|---|---|---|---|
| wall ms (build + wait) | 345.2 | 86.9 / 99.7 / 91.9 | +253.3 |
| build ms (graph + the mid-loop async_eval) / wait ms | 311.1 / 44.1 | 73-86 / 13-14 | +232.3 / +30.4 |
| main-thread CPU ms / other threads' CPU ms | 41.4 / 713.4 | ~30.5 / 93-105 | +10.9 / **+611.1** |
| host wired memory (wire_count) | **+31.8 GB** | 0 | |
| host zero fills / decompressions (16 KB pages) | 8,857 / 1,501 | 0 / 0 | |
| MTLCompilerService CPU | 0 | 0 | no pipeline builds |
| mx.async_eval, 23 calls per check: wall / CPU ms | 294.2 / 17.5 | 66.1 / 11.3 | +228 / +6 |

- Sampler: +231 ms of the build excess sits at qwen4_exp.py:6491 (_forward's mid-loop mx.async_eval, the SB cadence).
- **Named: the GPU driver re-wires the 32.32 GB expert pools at every prompt's check 0** (wired +31.8 GB inside check 0,
  nothing after). The prefill computes experts from staging and scratch buffers, so the pools sit idle through it. The
  screen sets no wired limit (mx.set_wired_limit was 0; iogpu.disable_wired_collector 0, dynamic_lwm 1), so IOGPU's
  wired collector unwires them. Check 0's first command buffers then wait while the driver wires ~2 M pages (~0.6 s CPU
  over ~3 threads), and MLX throttles the main thread inside async_eval: blocked, little CPU. Not a pipeline build, not
  host (+11 ms), not SSD (misses as later checks).
- fcpb (+ FCP_NOCLEAR: tier.prefill_done's mx.clear_cache() skipped): excess +174 ms, -79 vs fcpa (BETWEEN on the
  100 / 30 ms lines); check 0 still wires +34.7 GB. The clear is a minor part; the re-wire stays.
- FAST v1' logits differ between any two processes (fcpa vs fcpb: 0 of 12 checks' hashes equal, misses differ by 1-11
  per check; tokens equal 3/3), so the bitwise check runs in the exact NAX6-N env (step 2).
- Prior art: K-80 (11-01) found "a one-time non-SSD cost inside the first check" with candidates (a) the empty buffer
  cache (NewIdeas' A6KEEP: KILL), (b) the gate's first call after the slot prefill, (c) first touch of decode-only
  buffers; R-80 ruled out kernel builds. FCP names it: the buffers are the pools and the cost is re-wiring them.
  mlx-lm wires model memory with mx.set_wired_limit (generate.py's wired_limit) for this kind of stall.

## FCP STEP 2: THE FIX -- mx.set_wired_limit for the run: check 0's excess GONE, bitwise in the exact env (PASS; 08:34-08:36, quiet)
- Arms (pre-registered 5565729): fcpw = v1' + FCP_WIRED=1 (mx.set_wired_limit(max_recommended_working_set_size = 55.66
  GB) once at start: mlx-lm's wired_limit, held for the whole run; the screen had 0), C 256, the same 3 HumanEval;
  fcpx0 / fcpx1 = the NAX6-N exact env (R57_NAX=1, no R57_FAST) without / with it, C 224, BigMoE's KEEP-proof prompts
  (HumanEval 68, 16, ifeval/1220), CAP 64, R57_LOGHASH=1.

| medians over 3 prompts | check 0 wall ms | checks 1-3 wall ms | check 0 excess | check 0 wires | other threads' CPU, check 0 | decode tok/s (CAP 64) / steady | lifetime peak |
|---|---|---|---|---|---|---|---|
| fcpa v1' | 345.2 | 86.9 / 99.7 / 91.9 | +253.3 | +31.8 GB | 713 ms | 96.0 / 156.9 | 43.0 GB |
| **fcpw v1' + wired** | **89.6** | 85.5 / 99.2 / 90.2 | **-0.7** | +0.2 GB | 93 ms | **159.5** / 158.4 | 43.0 GB |
| fcpx0 exact | 287.5 | 159.4 / 213.9 / 191.1 | +96.4 | +30.9 GB | 735 ms | 57.4 / 69.5 | 38.2 GB |
| **fcpx1 exact + wired** | **178.0** | 153.8 / 207.4 / 186.1 | **-8.2** | +0.2 GB | 350 ms | 58.4 / 72.8 | 38.2 GB |

- **fcpw: PASS** (excess -254 ms >= 100; check 0 wires 0.2 GB < 5; peak 43.0 <= 45.0). Tokens equal fcpa 3/3. Its
  logits hashes differ from fcpa's, as any two FAST v1' processes' do (fcpa vs fcpb, step 1).
- **EXACT: PASS -- fcpx1's verify-logits hash equals fcpx0's on all 43 checks of the 3 prompts, tokens equal 3/3.**
  Exact-safe by construction too: residency changes no kernel, input or order.
- The CAP 64 tok/s overstates the gain (check 0 is 1 of 4 checks). On v1' whole decode NewIdeas' FIRSTCHK-PRICE puts
  check 0 at 15.8 % of the wall: 1 / (1 - 0.158) = **+18.8 % [P]** (141.2 -> ~168 [P]); BigMoE's engine A/B prices it.
  Steady-state checks unchanged (158.4 vs 156.9), main-thread CPU unchanged (~30 ms per check).
- Prior art (published): mlx-lm's generate.py wires model memory with mx.set_wired_limit(max_recommended_working_set_size)
  for the length of a generation (macOS 15+, residency sets). The screens never called it, so IOGPU's wired collector
  unwound the idle pools during each prefill. Lever tag: published; the diagnosis is new.
- Not measured here: prefill time (whole runs 31 s vs 40 s, exact 27 vs 27 s: no visible cost), the volatile tier
  (R57_VTIER filled 0.00 GB in these runs; if its buffers are MLX allocations they would be wired too and stop being
  purgeable: BigMoE checks), the 12 GB engines (same stall expected, scaled to their pool size).
- Where it goes (owners): one call after the model load in BigMoE's screen / stream setup (and Fit12's / Breakthrough's
  engines). Kept below the 45 GB line: the wired limit caps what MLX wires; the footprint is unchanged.

## PLAN 220: LEAP'S BUILD ORDER for levers 1-3 (Leap, Sat 2026-11-07 14:1x; NewIdeas' PLAN 220, QUEUE 8dae435)
- **Measured input (CPU only, BigMoE's S6W files = v1' + wired, 12 HumanEval, C 256, two runs, 435 checks after check 0):**
  - partial accepts 38 / 39 % of checks;
  - main-thread time after the eval (accept logic + commit build + submit): partial 2.19 / 2.10 ms vs full 0.34 / 0.33;
  - the next draft: after a partial 6.10 / 6.07 ms vs after a full 5.14 / 5.15.
  - So a partial costs ~+1.8 ms of main-thread Python + ~+0.9 ms of GPU replay landing in the next draft = **~1.06 ms
    per cycle in all**. That is the ceiling for any lever on the commit.
  - Why: MTPLX's commit_verified_window builds, lazily, 35 gated_delta_update replays (compute_g, sigmoid, the kernel,
    the conv-state concat / slice / contiguous) plus the PLE layer's full forward over the kept rows. r57 async_evals it
    (r57_screen_fast.py:1379-1380). Its GPU work runs ahead of the next draft on the same queue.
- **#2 as specified (state checkpoints at verify rows 8 / 12): KILL on paper.** Checkpoints cut only replay rows (part of
  the ~0.9 ms GPU). They add 2 x 36 x 2 MB = 144 MB of state writes to EVERY check (~0.5-0.7 ms) and leave the 1.8 ms
  Python build untouched. My cost model (HANDOVER, Sat 01:4x) had them at break-even at ~1/3 partial already.
  - **#2' instead: the commit replay as ONE mx.compile'd call per kept count** (the 35 layers' g / beta / recurrence /
    conv updates; mlx_lm's kernel unchanged), behind a ref2 switch (default off). Bitwise test vs the stock loop on
    recorded commits, then the exact-env hash. [P] -1.2 ms main thread and ~-0.2 ms GPU per partial = **-0.5..-0.6 ms per
    cycle (x1.009)**.
- **#1 the dense program: mostly NOT exact-safe.** Today's numbers (S2b step 1 REDONE, stack env, C 224): the dense-only
  16-row forward is 21.96 ms wall, GPU ~20.9, Python graph build 15.2. The priced pieces sum to 16.7 ms; ~5.3 ms is
  unpriced (PLE, the final norm + head, glue).
  - Exactness: REF2's <= 16-row dense calls are bitwise == qmm16c per column. DN-KS (in v1') already took the
    narrow-GEMV K split (out_proj, a|b, k/v, gate|up, down). Its bits changed (<= 2e-3): FAST, and it went through the
    gate.
  - What stays bitwise-safe: (b) k3c's load tail (-0.9 [P], Fable B prices run ~2x high: ~-0.45); (d) attention's ~25
    other launches per layer after att_single (-0.5 [P]); HC k2b folded into its neighbour (-0.7 [P], step 0 B); and
    whatever the unpriced 5.3 ms holds (locate first).
  - Bit-changing (FAST: an S4-style gate by Mon 12:00, Kamen's / the orchestrator's call): a chunked GDN step at 16
    rows (2.07 ms per check today; the bitwise prefetch reached -7 % on Fri), an in_proj K split (-0.25 ms per 2-row
    check measured Fri: small), an HC8 restructure.
  - **So #1's exact-safe part is ~-1.5..-2.5 ms per cycle [P], not -8.**
  - The CPU floor too: in the live loop the main thread spends ~35 ms per check (build 15-16 + encode ~19, XPT) vs GPU
    ~47. GPU savings past ~12 ms per check would hit it. Fewer dispatches (eprep, attention launches) help both sides.
- **Build order (ms per day):**
  1. **Sat evening: #2'** (compiled commit replay): build, CPU/GPU bitwise test on recorded commits, an in-process toggle.
  2. **Sat night: one quiet slot to re-price on today's stack:** the ABL pieces + ple + head + the remainder; HC8
     k1c / k3c / k2b chains at 16 rows; eprep's in-place dispatch count.
  3. **Sun: #3 e** (Fable B's kernels/eprep.py: 4,000 cases, 0 differ) **+ f** (attention k|v fold, -0.31 [K]), then the
     top bitwise #1 items by step 2's prices (k3c tail, k2b, attention launches).
  4. **Sun evening:** the bundle's exact-env hash check (FCP's 43-check method), then BigMoE's B / X / X / B A/B
     (C 256, quiet) by Mon morning; NewIdeas' line (cycle >= 1 % lower, KILL if >= 0.5 % worse).
  5. The bit-changing list goes to NewIdeas for an S4 decision before Mon 12:00.
- **Honest arithmetic:** exact-safe bundle -2.5..-3.7 ms per cycle [P] = x1.04-1.06 -> ~189-192 tok/s [P]; with T16
  (x1.035, lossy) ~196-199. 220 needs the bit-changing dense kernels on top (another ~-4..-6 ms per cycle) or a new
  lever.
- **CORRECTION (14:1x) to the #2 lines above:** v1' runs REF2_GDN_LAZY=1 (ref2's lazy commit): the 35 pure-GDN layers
  do NOT replay at commit; the next check's GDN body replays the kept rows' packs in registers. **#2' (one compiled call
  for the 35 replays) is WITHDRAWN: that work no longer exists.** The measured partial cost (+1.8 ms main thread, +0.9 ms
  in the next draft; ceiling ~1.06 ms per cycle) is the PLE layer's full forward over the kept rows (the one non-lazy
  layer, MoE included) plus the lazy bookkeeping (_lz_commit_pre / post). Checkpoints still cannot pay (the GDN replay
  rows they would cut are already in registers). Next: the re-price slot splits the partial commit (lazy pre / MTPLX
  commit + the PLE forward / lazy post) before any #2 build; a lazy PLE layer is the candidate if that forward dominates.
  Build order now: (1) the re-price slot tonight (dense ABL + ple + head; the commit split; HC8 chains), (2) #3 e + f,
  (3) the bitwise #1 items by measured price, (4) #2 only if the split shows >= 0.5 ms per cycle reachable.

## PLAN 220 RE-PRICE SLOT, part 1 (Leap, Sat 2026-11-07 14:25-14:30, quiet) + a RE-CORRECTION of #2
- **RE-CORRECTION: the 14:1x "correction" was WRONG; v1' does NOT run the lazy GDN commit.** The v1' env (final_env
  dd7ad0f) has REF2_GDN_LAZY=1 but no R57_GDNB=1, and the lazy commit needs CFG gdn_body: it is inert. Evidence: the
  commit split's lazy wrappers never fired, and the 16-row chains show mlx_lm's gated_delta_step x36 with
  gdn_prework_exact / gdn_gated_norm_exact (the REF2-T chain, not gdn_body). So MTPLX's commit replays the 35 pure-GDN
  layers as written, and the 12:1x (first) analysis of #2 stands.
- **Commit split (fcpc / fcpd: v1' + wired, C 256, HumanEval 68 / 16 / 91 + the 3 warm-ups, CAP 128):**

| per commit, medians | partial (n 31 / 32) | full (n 33 / 34) |
|---|---|---|
| ref2 _commit, main thread wall = CPU | 1.08 ms | 0.16 ms |
| of it MTPLX's commit (35 GDN replays + the PLE layer + trims, built lazily) | 1.00 ms | 0.07 ms |
| of that the PLE layer's forward over the kept rows | 0.27 ms | - |
| GPU work it leaves (fcpd: an eval right after the commit) | **2.88 ms** | 0.18 ms |

  - With S6W's next-draft gap (+0.93 ms after a partial) and 38.5 % partials, the commit's cost per cycle is ~0.4 ms CPU
    + ~0.36 ms of GPU that the draft does not hide (~0.75 ms in all; the earlier 1.06 also counted the accept logic).
  - R-21 (10-29) already found the 35 replays state-bound (220 MB of fp32 state at ~185-200 GB/s): batching them saves
    <= 0.1 ms GPU. A compiled replay would save at most ~0.7 ms of CPU per partial (~0.27 ms per cycle): small.
- **New exact-safe candidate, ready now: R57_GDNB=1** (ref2 gdn_body, one-dispatch GDN body, bitwise == off; GDNB-2 PASS
  Fri: -0.71 ms per check at NSG 16 on the exact config with R-83 + DENSE-N-RAM). It is not in v1'. With it,
  REF2_GDN_LAZY (already in the env) engages too (Fri check: bits PASS, -0.56 ms per partial, +0.93 per full, ~-0.2 net
  per check). For BigMoE's A/B: v1' + wired vs + R57_GDNB=1 (lazy on, as the env has it) and, if that is worse than
  -0.7, + R57_GDNB=1 with REF2_GDN_LAZY unset.
- **The stepb arms ran without att_single** (R57_FASTTAX is handled only in r57_screen_fast.py; the stepb harness builds
  on r57_screen.py), so the dense pass used MTPLX's 12-head-chunk SDPA (144 calls per check). The S2b prices of Sat 00:54
  had the same gap. Attention's 3.51 ms and the 21.42 ms base are overstated, roughly by the chunked SDPA (~3 ms). The
  other pieces' prices: HC reads 1.90 + 1.99, HC write 0.24, GDN 7.42 (in_proj 4.65), MoE dense 1.11, **PLE 0.90**,
  **head (final norm + lm_head) 1.63** (~195 MB at ~120 GB/s). Part 2 re-runs the base, attention and the chains with
  STEPB_ATT_SINGLE=1.

## PLAN 220 RE-PRICE SLOT, part 2: v1''s dense program priced in place (Leap, Sat 2026-11-07 14:37-14:42, quiet)
- The stepb harness on r57_screen_fast.py itself (part 1's old screen had neither att_single nor SVH): arm v1f = v1''s
  dense side (FAST-TAX att_single on, SVH's screened head on, SB 2, KEEP, wired; QOM, NAX, DN-KS, HC8, DNR, HCB, SMALL-N,
  OOO-1C); v1fg = + R57_GDNB=1. C 224, check 8, dense-only 16-row forwards (4 rounds x 12), barrier-kept bar / cal stubs.
- **In-place prices (wall ms per 16-row check): base 19.79 (Python build 12.06); HC reads 1.82 + 1.95 = 3.77; HC write
  0.08; attention 2.64; GDN 8.90 (in_proj 4.62, the rest 4.28); MoE dense part 1.22; PLE 0.52; head (SVH) 0.78;
  unpriced ~1.9.** (Prices overlap a little: a stub can push the forward toward its 12 ms CPU build.)
- **16-row dependent chains (us per call x calls = ms per check; GB/s):**
  - in_proj qkv|z 113.6 x 36 = 4.09 (213; floor at 295: ~2.9); out / o_proj (DN-KS ks 6) 38.7 x 48 = 1.86 (236);
    attention q 76.1 x 12 = 0.91 (239).
  - **mlx_lm's gated_delta_step 58.6 x 36 = 2.11 (116 GB/s over 6.7 MB; 27.6 us at 2 rows).**
  - HC8 k1c 22.2-24.2 x 96 = 2.23 (~200); **k3c 26.6 x 96 = 2.55 (148)**; k2b 7.8 x 96 = 0.75 (60).
  - **router (MLX 8-bit quantized_matmul, 512 x 2560) 17.9 x 48 = 0.86 (84 GB/s)**; shared gate|up 13.1 x 48 = 0.63;
    down 7.2 x 48 = 0.35; router_fused 0.17; moe_tail 0.13.
  - SDPA (att_single, 24 heads x 16 rows x 397 keys) 42.6 x 12 = 0.51; attention's 8-bit 640 x 2560 24.0 x 12 = 0.29;
    k / v 0.20; rope 0.14; QSA pool / prepare 0.17; a|b 0.28; GDN prework 0.23; gated norm 0.17; hc_write 0.27.
- **GDNB in today's config: leap_gdn_body_gi 68.0 us per layer vs ~71.5 for the stock chain it replaces (step 58.6 +
  prework 6.5 + gated norm 4.7 + compute_g 1.7) = ~-0.13 ms per check; in place the base read 20.28 vs 19.79 (+0.49,
  noise).** Friday's GDNB-2 (-0.71) was on the exact config. Sent to BigMoE before its GDNB A/B (it was already running;
  its report data/gdnb/gdnb_report.txt prices body + lazy commit).
- **1b prices (bit-changing; S4-style gate) and build days:**

| 1b item | price per 16-row check | build | note |
|---|---|---|---|
| in_proj K split (REF2_DN_KS=2) | ~-0.3 ms (116.4 -> 108.2 us per call, part 1) | 0: built | ready for the Mon gate |
| chunked GDN step | ~-0.8..-1.0 ms (58.6 us -> a ~28 us state floor) | ~1.5 days | only if GDN-STEP-R (1a, below) misses |
| HC8 restructure | k3c 2.55 -> ~1.5 ms + k2b 0.75 | ~1 day | the k2b fold and the k3c tail can keep the arithmetic: 1a first |
| router on a native 8-bit kernel | ~-0.5 ms (84 GB/s today) | ~0.5-1 day | changes router bits (routing can move): 1b |

- **1a next (exact-safe), in order:** (1) **GDN-STEP-R**: mlx_lm's step with R dv rows per simdgroup (1,536 / R
  threadgroups, each dv row's arithmetic verbatim, bitwise by construction; tests the waves hypothesis Friday's
  ablations never touched): up to ~-1.0 ms; (2) the HC8 k2b fold + k3c tail with the same arithmetic: ~-1..-1.5 ms;
  (3) eprep (#3e) ~-0.3..-0.5.

## GDN-STEP-R: the 16-row GDN step's cost is arithmetic, not waves or load latency (Leap, Sat 2026-11-07 14:54-14:56, quiet)
- scripts/gdn_step_r.py: three bitwise variants of mlx_lm's gated_delta_step (each dv row's loop body verbatim), checked
  against mlx_lm's kernel on the engine's shapes (Hk 16, Hv 48, Dk = Dv 128; T 1 / 2 / 5 / 8 / 16 / 32; masked and not):
  R (R dv rows per simdgroup: 1,536 / R threadgroups) 48 / 48, PRE (every row's inputs in registers first) 24 / 24, TG
  (every row's inputs in threadgroup memory first) 18 / 18 cases bitwise == mlx_lm (y and the state).
- Dependent chains of 36 calls (us per call incl. a ~3 us link; mlx_lm's first arm runs on a cold clock, so mlx_lm2 at
  the end is the reference):

| T 16 | mlx_lm2 | R2 | R4 | R8 | PRE 1 | PRE 2 | TG 1 | **TG 2** | TG 4 |
|---|---|---|---|---|---|---|---|---|---|
| us per call | 68.0 | 73.2 | 72.5 | 75.8 | 247.1 (spills) | 159.8 | 68.4 | **61.6** | 69.0 |

  - At T 2 every form is 38.7-41.5 us.
- **Reading:** R8 does 8x the per-row work per thread in the same time, and neither the threadgroup copy nor the register
  copy removes the per-row cost (~2 us per row). So the 16-row step is bound by its arithmetic: ~5 M fp32 operations
  per row across the grid (decay, the kv FMA, the update, the out FMA and two 5-step simd_sums over 786 K state
  values), ~22 us at 16 rows on top of ~25 us of state traffic. Friday's GDN-STEP3 ablations fit this (prefetch -7 %).
- **Verdicts:** waves hypothesis KILL; GDN-STEP-R as a lever: **TG 2 = -9 % per call, ~-0.23 ms per check, bitwise
  (small 1a; fold into the bundle only if cheap).** The 16-row step can only get much cheaper with fewer operations:
  the chunked (WY / UT, matrix-unit) form, which changes bits. **1b chunked GDN step re-priced: ~-0.5..-0.9 ms per check
  [P], ~1.5 days, bit-changing.**

## HC8-K3C: no bitwise speedup (Leap, Sat 2026-11-07 15:00-15:03, quiet; scripts/hc8_k3c_abl.py)
- k3c (the HC read's up mix + tail, 96 calls per check, 3.81 MB per call) in a 96-call dependent chain over 96 cold
  up-mixers, arms interleaved after a warm-up (the first arm of an un-warmed run reads 30-45 % high: clock ramp).
- **Where the ~29 us go:** without the code loads 20.4-20.8, without the MMA 18.7-21.0 -> the code stream (~8.6 us) and
  the matrix work (~10.3) run back to back, on ~10 us of fixed work (launch, mix1 copy, tail, cross-stream sum, the
  link). The scattered tail loads cost nothing (constants: 30.1 vs 30.0): Fable B's suspect is cleared.
- **Bitwise variants (all == k3c on 12 weight sets), us per call vs k3c 28.7-29.0:** tail via threadgroup memory
  (TGT) 31.1-31.3; every group's codes loaded first (PF5) 34.4 (register pressure); K split across 5 simdgroups per
  stream with the group terms summed in k3c's order (KS5, 640-thread groups) 42.4.
- **Verdict: HC8 k3c tail KILL** (no bitwise form faster); **k2b fold not built** (a cross-threadgroup "last block"
  sync for <= ~0.2-0.3 ms per check: risk over gain before Monday). HC8's remaining levers change bits (1b).

## PLAN 220 1a BUNDLE (eprep + GDN TG 2): built, bitwise in the exact env, engaged (Leap, Sat 2026-11-07 15:04-15:09)
- **GDN TG 2** (scripts/gdn_step_r.py kernel_tg R 2: every row's inputs in threadgroup memory first, 2 dv rows per
  simdgroup, one kernel per T): every T 1-16 x 3 seeds 48 / 48 bitwise == mlx_lm's kernel (y and state); 61.9 vs 68.3 us
  per 16-row call (-9 %); a runtime-T form keeps only -2 % (66.7), so T is compiled in (T 1-16 precompiled at apply).
  ref2 d6f736fb (9948a7e): REF2_GDN_TG=1 patches mlx_lm's gated_delta_kernel (only then; bf16 q / k / v / beta, fp32 g
  and state, no mask, T <= 16; else mlx_lm's own).
- **eprep** (ref2 3b184972, 40025f0): REF2_EPREP=1; 400 / 400 cases bitwise at the kernel level.
- **Exact env (NAX6-N, C 224, HumanEval 68 / 16, ifeval/1220, CAP 64): fcpx0 vs fcpx2 (both switches on): the
  verify-logits hash equal on all 43 checks, tokens 3 / 3; TG engaged 3,348 calls (eprep 0: no QOM in the exact env).**
- FAST v1' + wired (fcpe, C 256, 3 HumanEval): TG 2,628 calls, **eprep 576 = every QOM one-call (ooo_1c_qom 576)**;
  tokens == fcpw's 3 / 3; decode 161.3 tok/s at CAP 64 (fcpw 159.5; noise at this size).
- Price [P]: TG ~-0.23 ms + eprep ~-0.3..-0.5 ms per 16-row check = ~-0.5..-0.7 ms (~-1 % per cycle). The judge is
  BigMoE's B / X / X / B engine A/B (NewIdeas' line: cycle >= 1 % lower; KILL if >= 0.5 % worse).
- **Correction (method only):** the FCP pair lines ("tokens equal 3 / 3") were computed from runs.json's absent 'out'
  field (None == None). Re-checked from the tokens files: fcpa vs fcpb, fcpa vs fcpw, fcpx0 vs fcpx1, fcpx0 vs fcpx2,
  fcpw vs fcpe are all 3 / 3 equal, so every claim stands. fcp_read.py now reads the tokens files.

## PLAN 220 (4): THE CYCLE'S GPU-IDLE SEAMS, located (Leap, Sat 2026-11-07 15:12-15:22, quiet; scripts/fcp/fcp_seams.py)
- FCP_SEAMS on v1' + wired (C 256, HumanEval 68 / 16 / 91, CAP 128): every mx.eval / async_eval timed, marks at c0 /
  c1 / cb / ce / c2 and around the draft's two readbacks (np.array(pth), np.array(cf): implicit evals), a 1 kHz sampler
  of the main thread. A blocking eval or readback drains the GPU queue, so the time from its end to the next submit is
  GPU idle. Medians over checks >= 4 (no probe snapshots there; checks 1-3 carry ~0.2-0.3 ms of the probe's own work).

| per cycle, ms | partial (n 6) | full (n 9) |
|---|---|---|
| draft: c0 -> its blocking eval (draft_fused.py:179) | 1.49 | 0.63 |
| draft: the readbacks (GPU busy: the drafter's tail + round trips) | 4.12 | 4.23 |
| **idle A': readback end -> the check's first submit** (post-draft Python 0.16-0.18 + c1 -> the first SB async_eval) | **2.47** | **1.27** |
| **idle B: the check's eval -> the commit's submit** (accept logic + the commit build) | **1.22** | **0.32** |
| C: commit sent -> the next draft's submit (idle only on a full accept; a partial's 2.9 ms replay runs here) | 0.23 | 0.19 |

- **GPU idle ~1.8 ms per full-accept cycle, ~3.7 per partial: ~2.5 ms per cycle at 38.5 % partial** (Friday's ioreg
  read: 2.5-3.6).
- Where the main thread is in those seams (sampler):
  - Gate.new_pass's os.path.exists(PAUSE) per check (~0.1-0.3, in the check's first SB group); the screen's
    disk_read_bytes / stats (~0.1-0.3, instrumentation at check start).
  - On partials only: a ThreadPool future wait (~0.8; MTPLX's PLE n-gram lookahead / stage), the PLE n-gram rows'
    memmap gathers (~0.4), and the commit build before its single submit (~0.9: 35 GDN replays + the PLE layer, then
    one async_eval).
- **Exact-safe levers (host scheduling only; BigMoE's loop, MTPLX via ref2's wrapper), ~-1..-1.4 ms per cycle [P]
  (x1.016-1.022):**
  - (a) the PAUSE stat off the pass path (a flag a watcher thread sets, or every Nth pass);
  - (b) the first SB async_eval after layer 0 instead of 2 (the GPU starts ~0.35 ms sooner);
  - (c) instrumentation (stats, disk_read_bytes) after the first submit;
  - (d) on partials, the PLE stage started at the accept decision (overlapping the commit build), not awaited at the
    next check;
  - (e) the commit's GPU work submitted in pieces (the GDN replays first, then the PLE layer).

## PLAN 220 seams (b) / (d) / (e) in ref2, default off: bitwise, engaged (Leap, Sat 2026-11-07 15:2x-15:31; BigMoE 15:3x ask)
- ref2 d6f736fb -> 1f85c231 (atomic swap, self_test True; backup data/seams/ref2_before.py). Installed only when switched
  on; scheduling only (no value changes):
  - **REF2_SB_FIRST=1 (b):** at the PLE-carrying layer (index 1; config ple_layer_ids [2]) of a 5-16-row pass outside the
    commit, mx.async_eval(layer 0's output) before layer 1 runs. MTPLX's own cadence (qwen4_exp.py:6490) sends after
    layer 1 with SB 2.
  - **REF2_PLE_LATE=1 (d):** MTPLX's _forward stages the PLE n-gram rows before layer 0's graph (memmap gathers + a
    page-touch pool, GPU idle). For 5-16-row passes outside the commit the stage is deferred to the PLE layer's call:
    the same inputs and history (layer 0 does not touch layer 1's cache). A deferral never consumed is dropped and
    counted (ple_late_stale). With (b) the gathers overlap layer 0's GPU work.
  - **REF2_COMMIT_SUB=N (e):** inside the commit, mlx_lm's gated_delta_update sends every N replays' new states
    (mx.async_eval) while Python builds the rest. The last group rides r57's own async_eval.
- **Exact env (NAX6-N, C 224, 3 prompts): fcpx0 vs fcpx3 (all three on, N 9): verify-logits hash equal on all 43 checks,
  tokens 3 / 3;** engaged sb_first 27, ple_late 27 (0 stale), commit_sub 104.
- FAST v1' + wired (fcpf): engaged sb_first 42, ple_late 42, commit_sub 92; tokens == fcpw 3 / 3.
- Time: BigMoE's xsafe A/B (X = v1' + wired + BIGMOE_PAUSE_EVERY_MS=50 + R57_S0CARRY=1 + these + R57_DQ1=1). The seams
  read prices the whole set at ~-1..-1.4 ms per cycle [P].
- **1a bundle (eprep + TG 2 + DQ1) A/B: BETWEEN** (BigMoE 15:21, 07dbca9: cycle +0.16 ms, bits PASS 43 / 43, both
  engaged). The priced -0.5..-0.7 ms per check did not reach the engine cycle.
- **Re-read of BigMoE's p1a files (p1ab1 / 2 vs p1ax1 / 2, per check, CPU only):** X kept fewer tokens per check (mean
  10.16 vs 10.84: FAST acceptance diverges), so per-prompt cycle medians mix different check populations. Pooled per
  check: check+commit -1.13 ms (median), cycle -0.46 (mean -0.74). Stratified, X-B medians:
  - full accept, misses <= 100 (~52 %): check+commit -0.94, cycle -0.71, draft +0.19;
  - full, > 100 misses: check -0.65, cycle -0.07;
  - partial, <= 100 misses (~39 %): check -0.13, cycle +0.10.
  - **Class-weighted cycle ~-0.34 ms (-0.55 %): a real gain under the 1 % line (BETWEEN stands), not ~0.** Read-outs of
    FAST A/Bs should stratify by accept type x miss band. Open: why the partial class gains ~0.

## F24 ROW-CHUNKING in ref2's tier path: REF2_TIER_CHUNK=1, bitwise, runs at C 112 (Leap, Sat 2026-11-07 18:52-18:58; BigMoE's ask, NewIdeas QUEUE 223bf55)
- The failure (BigMoE F24, data/f24): at C 112 a 16-row call has 160 ids > C_m 110, so it takes the sync path
  (_sync_slots -> _tier_slots); the first, ungated (exact) check needed 120 distinct experts > 110 base slots -> assert.
- ref2 1f85c231 -> c176a5ad (default off; atomic swap, self_test True; backup data/tchunk/ref2_before.py). With
  REF2_TIER_CHUNK=1, a tier call whose distinct experts exceed the base slots (or whose 4-bit ranks exceed the sidecar
  slots) runs as contiguous row groups that fit (_tier_row_groups, greedy, in order). Each group goes through the
  unchanged body (_tier_rows: ensure, the 3c / NAX kernel, mx.eval), so a later group may evict an earlier group's
  experts (the ext evicts only experts "not needed now"; no cross-call pins). Rows are concatenated back in order.
  Never engages when the union fits. REF2_TIER_CHUNK_CAP=N (test only) groups at N.
- CPU: the grouping on 5,000 random cases (rows 1-16, K 10, caps 40-440): every group within both caps, rows covered in
  order, a fitting call kept whole: 0 bad.
- **GPU proof: the exact env (NAX6-N) at C 112 with chunking (fcpxk; its 16-row checks take the chunked sync path) vs the
  exact env at C 224 (fcpx0; the gate path): verify-logits hash equal on all 43 checks, tokens 3 / 3.** Engaged: 16
  chunked calls (32 groups) of 737 tier calls; lifetime peak 24.7 GB; no abort.
- FAST v1' + wired at C 112 (fcpk, 3 HumanEval, CAP 64): runs through (21 chunked calls, 42 groups; no assert or
  free-slot abort), but 29.4 tok/s at CAP 64 (cycle ~540 ms: every 16-row check is on the sync path, 160 ids > C_m 110)
  and lifetime peak 25.4 GB (> 24.0: C must go lower). For comparison E24's exact s10b stack is 41.3 tok/s at 23.86 GB.

## PREFILL 500 (Kamen 23:4x MUST): MLX's kernels at prefill M, batch invariance, the full-row logits (Leap, Sat 23:44-23:49, quiet)
- **PF-KBENCH (scripts/pf_kbench.py; random 4-bit g64 weights at the 125B's shapes; ms per call -> TFLOPS):**

| kernel (K -> N) | M 64 | M 256 | M 1024 | M 2048 | M 4096 | M 8192 |
|---|---|---|---|---|---|---|
| GDN in_proj 2560 -> 16,384 | 13.3 | 22.4 | 27.0 | 28.3 | 28.9 | 29.1 |
| attention q 2560 -> 12,288 | 6.5 | 9.7 | 26.7 | 28.0 | 28.7 | 29.0 |
| out_proj 6144 -> 2560 | 2.2 | 16.8 | 23.9 | 26.0 | 27.3 | 28.3 |
| shared gate|up 2560 -> 1280 | 0.8 | 3.2 | 4.9 | 11.8 | 24.3 | 26.2 |
| shared down 640 -> 2560 | 1.0 | 4.4 | 11.4 | 16.9 | 20.2 | 24.1 |
| head 2560 -> 248,320 | 24.9 | 28.2 | 29.3 | 29.5 | (2 GB) | (4 GB) |
| **experts: gather_qmm sorted, 512 x top 10 (gate|up + down)** | 1.2 | 2.8 | **7.1** | 9.7 | 11.8 | **13.8** |

  - Dense 4-bit GEMMs reach ~27-29 TF from 1K rows; the experts reach half that (20-160 rows per expert).
  - Per prefill token ~13 GFLOP [estimate]: dense ~7, experts ~4.7, attention <= ~1.2 at 8K.
  - Compute floor ~0.97 ms per token with 1K-row expert batches (~1,000 tok/s), ~0.65 at 8K rows (~1,500). After
    BigMoE's expert-once streaming, **the expert GEMMs bound prefill (PF-B's target)**.
- **Batch invariance, kernel level (the same rows inside calls of different M, bitwise vs M 1024):**
  - 4-bit qmm: equal from M 16 (in_proj, att q), 128 (out_proj, shared down), 256 (shared gate|up); M 1024 = 2048 =
    4096 = 8192 everywhere tested.
  - **The 8-bit router (N 512) only from M 541 (differs at <= 512).**
  - gather_qmm (sorted): 1024 = 2048 = 4096 = 8192.
- **End to end (fcpp1 / fcpp8: exact env, C 192, agent/edit10 1,565 + agent/write13 4,730 tokens, CAP 16): R57_PFCHUNK 1024
  vs 8192 -> the verify-logits hash differs on 9 of 9 decode checks (tokens equal 16 + 16).** A one-call prefill is not
  bitwise == today's 1024-row chunks: a prefill piece (attention over the prefix, the GDN chunked recurrence, HC bf16
  GEMMs or the grouped MoE prefill) is not batch-invariant. Locating it is next.
  - **Design that is bitwise by construction:** layer-major with 1024-row sub-batches (each layer sees exactly today's
    inputs). Only the expert streaming changes. Sent to BigMoE.
- **The full-row logits:** the screen's prefill calls the model with MTPLX's defaults (emit_logits True, logits_keep 0):
  full-vocab logits for every row (508 MB + ~1.3 TFLOP, ~45 ms, per 1024-row chunk; 2.35 GB at one 4,730-row call).
  Lifetime peak 37.4 -> 41.6 GB at C 192 in fcpp8. Only lg[0, -1] is used. Fix: emit_logits=False on all but the last
  chunk, logits_keep >= 256 on the last (the head is row-invariant there). Whole runs: fcpp1 66 s vs fcpp8 42 s (the
  one-call prefill streams experts once: BigMoE's PFX measures it).

## CHAT 150 / PREFILL 500: CH2 anatomy + the priced F1a and PF-B plans (Leap, Sun 2026-11-08 00:0x; NewIdeas QUEUE 9c65385)
- **CH2 (fcpch: v1' + wired = L, C 256, IFEval 1220 / 3280 / 1561, CAP 128, quiet; small n; per check):**
  - 2 rows (n 13): wall 54.5 ms (build 48.0 + wait 6.4), main-thread CPU 29.3, the loader threads' CPU 74.
  - 3 rows: 61.7 ms; 4 rows: 67.9.
  - Not CPU-bound: the main thread waits ~25 ms inside the build, so the GPU binds at low rows.
  - Cycle 55.8 (full) / 64.2 (partial) ms: the MTP head's draft 6.6-9.2 ms (c0 -> its last eval), GPU-idle seams
    ~2.4 (check start) + 0.3-1.2 (accept / commit) + ~0.5.
  - BigMoE's N45 (n 664): the MTP head already verifies a median 3 rows on chat (adaptive by its cut). A 1-row check
    costs 52.7 (exact) / 42.1 (L) ms; +~3-4 ms per row and ~2.6 ms of draft per drafted token. **The fixed per-check
    cost, not the row count, sets the chat cycle.**
- **The 2-row map (16-row dependent chains at FLAT-2's 2 rows, v1''s dense program, gap_p220b; us x calls = ms):** every
  dense call is padded to 16 rows (REF2's <= 16-row form).
  - HC8 5.9: k1c 2.46, k3c 2.47, k2b 0.96. GDN in_proj 4.35 (200 GB/s). out / o_proj 1.93. GDN step 1.12. att q 1.02.
  - The 2 -> 16-row pads (Concatenate) 1.44. Router (MLX 8-bit) 0.89 (84 GB/s). Shared gate|up + down 1.13.
  - SDPA 0.53. Small glue ~1.5 (rope, norms, QSA prep, moe_tail, router_fused, gated norm, prework, hc_write, a|b, k|v).
  - ~20-22 ms of GPU vs a dense byte floor of 3.1 GB = ~11-12 ms.
- **F1a plan (low-row dense kernels, 1-6 rows; bit-changing allowed, IFEval-gated vs NAX6-N), priced per 2-row check:**
  1. Row-native low-row forms: no 2 -> 16 pad (Concatenate 1.44 ms) and no 16-row work in the GEMVs. One kernel family
     at M 1-8 with the K split chosen per shape (DN-KS's knowledge): in_proj toward ~280 GB/s (-1.2), out / o_proj
     (-0.4), att q (-0.3), router 8-bit native (-0.6), shared (-0.5). **-4..-4.5 ms, ~1.5 days.**
  2. HC8 read at low rows: fold k2b into k3c's prologue (at 2 rows k2b's partials are ~56 KB per threadgroup, ~2 us
     from L2; bitwise possible) -0.7..-0.9; row-native k1c / k3c -0.8..-1.2. **-1.5..-2 ms, ~1 day.**
  3. Glue fusion at low rows (rope + norms, moe_tail + router_fused, gated norm + prework): **-0.6..-1 ms, ~1 day.**
  - Total ~-6..-7.5 ms of GPU per 2-row check: ~20-22 -> ~14 ms dense, chat check ~45-55 -> ~38-48 ms, x1.13-1.18 on
    the chat cycle [P]. Gated set target ~Wed (NewIdeas). The experts at 2 rows (NAX ~14 ms at all-resident) and the
    MTP draft (~2.6 ms per drafted token) are the next walls for 150.
- **PF-B plan (prefill expert GEMMs; after BigMoE's layer-major port PF-A):**
  - Today MLX's gather_qmm (sorted) reaches 7.1 TF at 1K rows and 13.8 at 8K (dense qmm 27-29). Its tiles run
    part-filled at 20-160 rows per expert.
  - A grouped expert GEMM over a layer's whole prompt (rows bucketed per expert, full-tile MMA via MPP's matmul2d on
    the 4-bit / 3c records, weights read once) targets ~18-22 TF. Experts then take ~0.2-0.25 ms per token vs ~0.66
    at 1K-row sub-batches: prefill compute ~0.97 -> ~0.55 ms per token (~1,800 tok/s compute floor) [P].
  - Exactness: a new reduction order changes prefill bits vs gather_qmm, so it is a lossy prefill unless the decode
    still matches AR (R-56-style audit). The bitwise route keeps gather_qmm on all rows at once (pairs invariant across
    1K-8K: measured) at 13.8 TF: ~x1.4 on the experts vs 1K-row sub-batches, exact, ~0.5 day once PF-A lands.
  - Recommended order: the bitwise all-rows gather_qmm first (with PF-A), then the MPP grouped GEMM as the lossy
    option. ~1.5-2 days for the kernel and its audit.
- **LOWROW (Sun 00:0x, quiet; scripts/lowrow_bench.py; dependent chains over cold weights; us per call at M 1 / 2 / 4):**
  - in_proj 2560 -> 16,384: REF2's padded qmm16c 131-135 (175-180 GB/s); v1''s native qmm16n 120.9 (chain table);
    **MLX's own GEMV 96-101 at M 1-2 (234-245 GB/s)**, 115.5 at M 4.
  - att q: qmm16c 122; native 85.3; MLX 86-88 (equal to native).
  - out_proj: qmm16c 100-104 (ks 1); DN-KS ks 6 40.3; MLX 46.7-48 (DN-KS better).
  - Narrow ones, DN-KS better than MLX everywhere: shared gate|up 13.5 vs 18-20; shared down 10.1 vs 15.6-18; a|b
    8.7 vs 18-23; k / v 9.7 vs 22-44.
  - **The 8-bit router (MLX) 18.5-25.5 at 55-75 GB/s** (a native 8-bit DN-KS form ~5-8).
- **F1a re-priced (per 2-row check):**
  - **Exact-safe** (no gate):
    - (i) in-kernel row padding: REF2's 16-row kernels read rows >= M as zeros instead of a materialized
      Concatenate pad. The same arithmetic, so the same bits. -1.4 ms (the 2 -> 16-row pads, 1.44 ms), ~1 day.
    - (ii) the k2b fold into k3c at <= 4 rows: -0.6..-0.7 ms, ~2 h.
  - **Bit-changing** (IFEval gate):
    - (iii) in_proj on a row-native GEMV at <= 2 rows: -0.7 ms.
    - (iv) a native 8-bit router: -0.5 ms.
    - (v) glue fusion: -0.6..-1 ms.
  - **Total ~-3.8..-4.3 ms per 2-row check** (the narrow GEMVs were already DN-KS's), x1.07-1.08 on the chat cycle
    [P]. The earlier -6..-7.5 counted DN-KS's own gains twice.

> (a quote from the project owner, kept private)
- Arms on L (v1' + wired, C 256), IFEval 1220 / 3280 / 1561, CAP 128: fcpch (plain + the ioreg GPU sampler) and fcpcs
  (MoE blocks synced before and after in check windows). Field meanings in the per-check records: io = demand SSD busy
  ms, svc = gate service ms on calls with misses (the GPU spinning on the loader), **stl = the number of layer calls
  that stalled (out of 48), not ms.**
- **Plain, per check (medians):**
  - 2 rows (n 13): wall 53.8 ms, main-thread CPU 29.2, other threads' CPU 70.5, **GPU busy 91 %**.
  - 4 rows (n 10): wall 69.1, main 33.2, GPU busy 92 %.
  - Screen stats: misses 75.5, io 14.7 ms, svc 17.3 ms, stl 26 of 48, ck 50.6, draft 8.2.
- **Synced (n 33; syncs inflate the wall to 97.9 ms):**
  - Sum of the waits before each MoE block (the dense GPU work since the last one) 29.4 ms; MoE builds (Python) 5.2;
    MoE waits (router + shared + gate + experts) 47.8; main CPU 27.8.
  - Less ~0.15 ms per sync: **dense ~22 ms GPU** (the 2-row chain map agrees: 20-22), **MoE ~40 ms incl. ~17 of
    miss spin, so experts + router + shared ~16 ms**.
- **Decomposition of a 2-row chat check (~54 ms, GPU-bound): dense GPU ~22 + experts / router / shared ~16 + miss spin ~17
  (they sum to the wall); main-thread CPU ~29, overlapped.**
- Order for my lane:
  - With misses gone (M1: BigMoE / Breakthrough), GPU ~38 > CPU 29: G2 (experts) and G1 (dense) first.
  - Then the CPU build binds (B1).
  - G2's floor at 2 rows is ~2.2 GB (ranks 0-1 at 4 bits, 2-9 at 3 bits, ~23 MB per row per layer): ~8 ms, so
    ~-6 ms lossless. More needs lossy chat routing (top-8, lower-bit tail ranks), gated.
- **For a 22 ms step: misses -17 (M1), experts -8..-12 (G2 + lossy routing), dense -8..-12 (G1 + more), main CPU
  29 -> <= 15 (B1), draft 8 -> ~4 (D1).** Every piece has to land.
- **G1 exact piece REF2_PAD_IN (ref2 95585132, 1cb5d24; scripts/qmm26m.py):** qmm26's native kernel pads rows in the kernel
  (rows >= M load constant zeros, only M rows written). Bitwise == qmm16n on the padded x: 320 / 320 (M 1-16, five shapes,
  DN-KS's K splits).
- **Chat A/B on L (fcpcb vs fcpcp, 3 IFEval, CAP 128):** check+commit medians by rows: 1: 45.4 -> 47.7; **2: 44.1 ->
  43.9**; 3: 50.5 -> 49.9; 4: 56.6 -> 56.4 ms. **No measurable gain.**
- **Chat on L is not run-to-run deterministic.** Any two plain runs (fcpch / fcpcs / fcpcb) agree only on check 0's hash;
  tokens differ on 2 of 3 prompts. Engine-level bit checks for chat levers must use the exact env (or kernel-level
  proofs).
- **Reading: while misses stall ~26 of 48 layer calls per chat check (~17 ms of gate spin), dense-side GPU savings move
  into the spin and do not shorten the check.** The 1a bundle showed the same (gains only on low-miss checks). **M1
  (misses) gates every GPU lever on chat.** My levers need measuring with misses removed (replays with the experts
  resident) until M1 lands.
- **B1 re-scoped from fcpch's 1 kHz main-thread samples (checks with <= 3 rows, n 19):** of the ~48 ms build phase,
  **~44 ms is inside mx.async_eval / mx.eval (MLX's C++ graph walk + Metal encode, plus throttle waits on the busy GPU)
  and only ~4 ms in Python graph construction** (qwen4_exp / ref2 / bigmoe frames). The ~29 ms of main-thread CPU per
  chat check is mostly MLX's per-dispatch encode cost: ~3,000 kernels per check at ~7 us each.
  - So B1 is a dispatch diet, not a Python compile. Each removed kernel saves ~7 us of CPU and ~1.5-2 us of GPU.
  - Today at 2 rows: ~60-65 dispatches per layer (HC glue, GDN glue, attention glue, MoE id / mask glue, pads).
  - REF2_PAD_IN alone removes ~340 per check (~2.4 ms of CPU), invisible while the GPU binds.
  - Target ~30 per layer: ~-12 ms of CPU per check [P].
- **Dispatch diet, counted (v1f 2-row kernel records, per check):** ~1,600 wrapped kernel calls + ~480 glue flushes.
  - By piece: GDN 396 (+158 glue), MoE dense 432 (+123), HC reads 480 (+157; hc_m 288, hc_a 192), attention 168
    (+33), HC write 48, experts 48 (the one-call), PLE 12, rest ~16.
  - Plan [P]: HC pads in-kernel + the k2b fold ~-300 (hc8f, built; its check is pending); GDNB's one-dispatch body
    ~-180 (built: R57_GDNB); a fused MoE dense side (router + shared + tail) ~-240 (new); attention glue ~-60.
  - **~-800..-1,000 dispatches = ~-6..-10 ms of encode CPU per check, plus ~-1.5..-2 ms of GPU.** It counts once M1 and
    G1 / G2 bring the GPU under the CPU.
- BigMoE 00:2x: L's chat arm (gate+t) has the look-ahead OFF already. QOM never engaged on chat (the FAST gate ran on
  DFlash2 drafts only); fixed with R57_QOM_MTP=1|2 (M1-A / M1-B, chs2). C4: no single lever drop makes chat == AR.
- **HC8-F (scripts/hc8f.py; ref2 388ca3f1, REF2_HC8_FOLD=1, default off): K2 folded into K3 for HC8 reads with <= 4 real
  rows.** Every K3 threadgroup recomputes k2b's rinv / mix1 / xsum / inject in a prologue, in k2b's order. **Bitwise ==
  k2b + k3c on the real rows: 30 / 30 cases (M 1-4 and 16).** One dispatch less per HC read (96 per chat check, plus its
  dependency gap). Self_test True with the switch off and on. For the M1 winner's chat bundle with REF2_PAD_IN / GDNB /
  eprep + TG 2.
- **K1M + HC8-G + G2 priced (Sun 00:49-00:52, QUIET lock; data/hc8f_check.txt, data/hc8g.txt, data/g2_bench.txt,
  data/hc8f/k1m_gr.txt; scripts/hc8f.py, hc8g.py, g2_bench.py, k1m_gr_check.py; live ref2 bb055eb4, installed 00:54,
  self_test PASS, both new switches default off).**
  - **K1M (exact, REF2_HC8_K1M=1):** HC8's K1 reads the M-row h / pending, with the pad rows' loads guarded in-kernel.
    Its outputs equal K1's on the zero-padded inputs on all 16 rows. **96 / 96 kernel cases bitwise** (v4 and v5,
    pending on / off, M 1-4). read_k1m (K1M + k2b + k3c) == hc8r.read(padded) on all 16 rows. **24 / 24 through
    ref2._gr** (fold, fold + K1M, K1M vs the unfolded path).
  - HC8 read timing, us per read (dependent chains of 48 reads over 8 cold weight sets, pending write on):

    | M | v4 padded (today) | fold, padded | fold + K1M | K1M + k2b + k3c | HC8-G GEMV (best) |
    |---|---|---|---|---|---|
    | 1 | 54.1 | 54.1 | 46.8 | 46.9 | 34.9 |
    | 2 | 54.2 | 55.7 | 48.6 | 46.5 | 40.8 |
    | 3 | 53.9 | 57.1 | 49.9 | 46.8 | 56.2 |
    | 4 | 53.8 | 59.3 | 52.3 | 47.3 | 66.3 |

  - **Verdicts:**
    - HC8-F (the fold) alone: **KILL**, +0.0..+0.53 ms per check. K3F's per-threadgroup k2b prologue costs more than
      the k2b launch it removes.
    - **K1M: PASS**, -0.62..-0.74 ms per check at 1-4 rows, bitwise. It replaces REF2_HC8_FOLD in the bundle (BigMoE
      told 00:5x for CHS4).
    - **HC8-G (FAST, REF2_HC8_G=2):** -12.0 / -5.7 us per read vs K1M at 1 / 2 rows (-1.15 / -0.55 ms per check);
      slower at 3-4 rows, where the switch hands over to K1M. Error vs v4: max rel 9.5e-4 on mixed, 13 of 153,600
      values differ (bf16 ulps); inject equal; written bitwise.
  - HC8-G per kernel (sweep): K1G 14.3 / 18.5 / 39.2 us at M 1 / 2 / 4 (the 3.3 MB floor at 285 GB/s is 11.6 us);
    K3G 18.7 / 21.3 / 29.3 us (includes ~2 us of the chain's dependency op). Both grow with M; at M 4 likely register
    pressure (16 x M inputs per lane plus RPS x M sums).
  - **G2 priced: experts at 1-4 rows are already near the machine's rate.** Quiet; real pools of layers 20 / 44
    (native+p11+pad, all resident, C 512), top-10, skewed routing, no U3. Base record 2.06 MB, sidecar 0.72 MB.
    us per layer call:

    | M | base groups | MB per call | NAX6-N 1cq (L today) | 3c-N one-call | 3c-N / NAX |
    |---|---|---|---|---|---|
    | 1 | 10 | 22.1 | 120.5 (183 GB/s) | 105.8 (209 GB/s) | 0.877 |
    | 2 | 18 | 40.8 | 196.6 (207 GB/s) | 185.2 (220 GB/s) | 0.942 |
    | 3 | 26 | 58.3 | 268.7 (217 GB/s) | 271.5 (215 GB/s) | 1.011 |
    | 4 | 34 | 75.4 | 338.2 (223 GB/s) | 348.3 (216 GB/s) | 1.030 |
    | 8 | 61 | 135.9 | 585.6 (232 GB/s) | 664.4 (204 GB/s) | 1.135 |
    | 16 | 102 | 228.5 | 984.8 (232 GB/s) | 1262.4 (181 GB/s) | 1.282 |

  - **Reading:** a 2-row check's experts read ~2 GB, ~9.4 ms on L's kernel (not ~14 ms; the board's "~1 GB, ~3.5 ms"
    floor was off by 2x). L's kernel is already at ~73 % of 285 GB/s. **G2's ceiling is ~-2.5 ms per 2-row check at
    the bare floor.** 3c-N gives -0.55..-0.71 ms at 1-2 rows (FAST; it differs from NAX by up to 1.5-2e-2 relative).
    It is not wired: 3c-N needs U3's -2 sidecar fallback (BigMoE's LP1 kernel patch) without LP1's loader change.
  - **Sum of my GPU levers on a 1-2-row chat check:** K1M -0.7 (exact) + HC8-G -0.55..-1.15 (FAST) + 3c-N -0.55..-0.71
    (FAST, unwired) + row-native in_proj ~-0.5 (FAST) = ~-2.3..-3.1 ms of ~43. Kernels alone do not move the 1-2-row
    floor (BigMoE: ~39-42 ms even at < 30 misses). Next: measure where that floor goes (GPU busy, per-layer router-id
    waits, encode) on L + M1-B before building more.
- **NGEMV + G2 wired + one-call determinism (Sun 01:15-01:19, share lock, GPU ~free; data/ngemv.txt, data/d1c_check.txt;
  scripts/ngemv.py, g2k.py, g2_check.py, d1c_check.py; live ref2 8f5933c7, both new switches default off, self_test PASS).**
  - **NGEMV (FAST, REF2_NGEMV_ROWS=N):** one GEMV on ref2's native 4-bit layout (lane = 4 rows x 16 k of a group,
    threadgroup = a 32-row tile with KS simdgroups splitting K) for DENSE-N, SMALL-N and DENSE-N-RAM calls at <= N rows.
    The first build left half the M 4 outputs unwritten (64 threads for 128 outputs at KS 2); fixed. v2 extracts each
    code once per group. Check v2: max rel error vs qmm16n <= 3.8e-3, 0.1-0.2 % of outputs differ (bf16 ulps), all 7
    shapes. vs L's qmm16n (DN-KS 1) per check: **M 1 -1.41 ms, M 2 -0.80 ms, M 4 +3.46 ms** (in_proj 1 / 2 rows 90.5 /
    93.3 us vs 103; narrow calls 5-7 us vs 8-13). Use N = 2. Through ref2._dnr_qmm: 6 / 6 engaged, max rel diff 2.1e-3.
  - **G2 wired (FAST, REF2_G2_ROWS=N):** FAST-NAX one-calls at <= N rows run 3c-N's one-call with LP1's tier-3 line
    (g2k; U3's -2 sidecars, no loader LP1). G2-CHECK: 36 / 36 cases finite and within 5e-2 of the NAX one-call (worst
    2.4e-2, the two kernels' known difference), M 1-3, QOM on / off, late pairs. Price: -0.55..-0.71 ms per check.
  - **One-call determinism (BigMoE C4 found E + OOO_1C + NAX_1CQ != AR and different run to run):** on real pools with
    static slots, 1cq and plain 1c are bitwise identical over 8 repeats in 16 / 16 configs (1-6 rows, all resident,
    late pairs released before or 5 ms after the launch). So the kernels and the in-kernel poll are deterministic; the
    live difference needs the loader: (a) a victim slot reused while this layer's resident pairs still read it (the
    one-call runs them beside the misses), or (b) S7B-PROBE's same-command-buffer stale lines. Sent to BigMoE; a
    coherent(system) 1cq variant can split (b).
- **COH-1CQ (Sun 01:2x, share lock; scripts/coh1cq.py, data/coh/d1c_coh.txt; live ref2 11fa14ff, REF2_NAX_COH=1, default
  off): BigMoE ruled out slot reuse in the ext (make_room never evicts a call's need list; reserve() only for look-ahead /
  oracle, off in E / L), so S7B-PROBE's stale-line hazard is the live suspect for E + OOO_1C + NAX_1CQ's run-to-run diffs.**
  - FAST-NAX's one-call kernels A / B with every expert-pool read coherent(system): the body's code / LSB pointers in
    tierglu_coh's "cn" form, the p11 scale / bias words through a coherent cast inside renamed P11 helpers placed after
    the SYS block. Learned: `coherent(system)` is rejected in struct members, in function parameters that must bind, and
    in header code before the SYS block (MLX's preamble); inside helpers after SYS, `coherent(system) device` casts compile.
  - Bitwise == the plain kernels on static slots: 16 / 16 configs (odd repeats coherent, even plain; 1-6 rows, all
    resident and late pairs). BigMoE runs E + OOO_1C + NAX_1CQ +- REF2_NAX_COH=1, two runs each, after CERT: == AR and the
    same across runs means (b) is the cause.
- **CHS4 [K] for Leap's levers (BigMoE, Sun 07:13-07:42, quiet, 12 IFEval, C 256, one pass per arm, snapshot 342f914,
  ref2 11fa14ff; data/chs4/chs4_report.txt). COH CONFIRMED: stale slot lines caused the one-call's diffs.**
  - **COH:** ycoc / ycoc2 (E + OOO_1C + NAX_1CQ) 37.7 / 38.0 tok/s, == AR 3 / 12 both, different first diffs, and the
    two runs diverge from each other at 9 / 12 prompts. ycoh / ycoh2 (+ REF2_NAX_COH=1) 37.8 / 37.9, **== AR 12 / 12
    both and token-identical to each other**, ck 57.4 / 57.2 vs 57.9 / 57.9 ms (no cost). Hypothesis (b) holds: inside
    one command buffer the one-call's plain loads saw lines of a slot's previous expert (S7B-PROBE); coherent(system)
    pool reads see the loader's fresh bytes. In E the one-call is now exact but not faster (yes1 = E + SB1 38.3 in the
    same batch; x3e = E 37.0 in chs3), so E stays as it is.
  - **L has the same bug:** s5p_env (= @L) carries REF2_OOO_1C=1 + R57_NAX=1 + REF2_NAX_1CQ=1, so every L / M1-B number
    to date includes stale expert reads. ooo1cq uses one kernel pair A / B for every case (U3 sure rows and QOM drops are
    -2 slots inside it) and coh1cq swaps both, so REF2_NAX_COH=1 covers all of L's one-call reads. Asked BigMoE (07:47)
    to put it into L and chat_env.txt before EvalSuite's M1-B IFEval gate. The plain 3c-N one-call and G2 (g2k) read
    slots plain too: G2 stays off.
  - **Low-row levers on M1-B** (yb / ybb mean 49.9 tok/s, ck 40.15): bundle (ybl) 49.6, ck 40.8 = wash; bundle +
    HC8_G=2 (yblg) 53.1 (+6.4 %), ck 39.4 (check median -1.6 %, 2-row median 33.0 -> 31.3 ms; part of the gain is kept,
    2.58 vs 2.44); FAST low-row (yblf = yblg + NGEMV + G2 at <= 2 rows) 50.4, ck 40.1 = wash. On L: bundle (y0l) 44.8 vs
    y0 / y0b 44.2, ck 46.5 vs 48.3.
  - **Read:** a GPU win of 1-2 ms on 1-2-row checks is below one-pass whole-decode noise (same-config pairs y0 / y0b and
    yb / ybb differ by 1.6 / 2.4 %). The 1-row checks were scheduler-bound (46.7 ms; NewIdeas' SB1 -> 35.4 ms), which hid
    the 1-row kernels. CHS5 arm asked: the stack (with SB1) + REF2_NGEMV_ROWS=2, read on chs_strat's 1-2-row median.
- **MLX NAX SORTED-GATHER BUG + PF-B (Sun 08:13-08:25, share locks beside EvalSuite's gate; scripts/pfb_inv.py,
  pfb_diag*.py, pfb_order.py, pfb_fix.py; data/pfb_*.txt). BigMoE's PFX: PF-A plm1 195 / 368 / 441 tok/s; long prompts
  compute-bound.**
  - **Bug (MLX 0.32.2, affine_gather_qmm_rhs_nax, quantized_nax.h:1547):** `sgp_sm = align_M ? SM : min(SM,
    short(max(0, (M - (y_row + tm)))))`. When the call's row count M (tokens x top-k) is not a multiple of the tile
    height, the `short` overflows wherever more than 32,767 rows follow, and those simdgroups neither compute nor
    store: the rows stay UNWRITTEN (zeros in a fresh buffer, stale data in a reused one). Seen: 79,320 rows -> 32,776
    rows unwritten (13,784..46,559; the formula predicts 13,785..46,552); 47,300 -> 14,560; 38,360 -> 5,600; 40,960
    (aligned) -> 0. Deterministic, the same in any call order and with the input evaluated first (not a race);
    mx.argsort is fine. Only this kernel has the cast (dense NAX qmm does not).
  - **CORRECTION 08:5x: a rediscovery.** This is BigMoE's LEDGER BUG1 (found earlier by PrefillKernels,
    align_probe.py: "wrong rows at bm 64 when the routed row count is not a multiple of 64"; MLX 0.32.4 fixes it).
    bm 64 starts at rows / E >= 64, i.e. M >= 32,768 at E 512, exactly where the short can overflow. New here: the
    trigger line (quantized_nax.h:1547, > 32,767 rows following) and the measured counts. I broadcast it before
    grepping the owner's LEDGER; corrected to the orchestrator, the lounge and Fit12.
  - **Who is hit:** any SwitchGLU (mlx_lm) call with > 32,768 sorted rows (> 3,276 tokens at top-10) and M % 64 != 0
    that does not pad. BigMoE's tier prefill (BIGMOE_TIER_PF_GROUP, D1i's tier path) already pads to x64 (tier.py
    BUG1), so Fit12's 12 GB tier prefill is likely unaffected (Fit12's GPU proof queued). EvalSuite's audit (a67de24):
    every gate and ref is safe; exposed = 3 b12-family agent rows (edit02 / write11 / write12). BigMoE's pe4096 /
    pe8192 arms and fast24's prefill are unchecked. Safe: 1,024-token chunks (<= 10,240 rows: PF-A plm1, pe1024,
    exact E), decode, dense calls.
  - **Workaround, exact:** pad the sorted rows to a multiple of 64 (copies of the last row, same expert) so the
    kernel takes its aligned path: pfb_fix.switch_glu_padded is bitwise == the 1,024-token calls at 1.6K / 4.7K /
    7.9K, no unwritten rows.
  - **PF-B priced (exact):** one padded MoE call per layer vs PF-A's 1,024-token calls: x1.42 / x1.89 / x1.96 at
    1.6K / 4.7K / 7.9K (7.9K: 136 -> 69 ms per layer, 5.7 -> 11.2 TF; GPU shared, indicative) = ~1.8 s of 12.9 at
    4.7K and ~3.2 s of 18.0 at 7.9K. My first unpadded timing (x2.84, 16.8 TF) was inflated by the skipped rows.
    Using it needs PF-A's loop split at the MoE (each chunk's attention part in order, then ONE MoE call per layer,
    then each chunk's residual). The same split lets the layer's expert read overlap its attention parts: PFX fits
    TTFT = SSD bytes / 11.3 GB/s + 1.5 ms per token within 0.1 s at all three lengths, i.e. the reads (~6.0 s at
    7.9K) are serialized with compute (~11.9 s), because _read_rebuilt is synchronous in the first chunk's call.
  - **CHS5 (BigMoE 08:12): NGEMV_ROWS=2 KILL as a speed lever** (1-row check 31.9 vs 31.4 ms, 2-row 30.0 vs 30.7; its
    +3.5 % whole decode is different tokens). The check is now CPU-bound (bld ~31.5 ms of ~33, wt ~3): GPU-only kernel
    levers at 1-2 rows stop paying.
- **ENC2 + PF-B dense invariance (Sun 08:28-08:5x, share locks beside EvalSuite's gate; scripts/enc2*.py,
  pfb_dense_inv.py; data/enc2*.txt, pfb_dense_inv.txt).**
  - **ENC2 (check-side CPU lever: SB's mid-loop async_eval on a second thread while the main thread builds the next
    layers):** MLX 0.32.2 releases the GIL in eval (a side thread keeps 99-104 % of its solo rate). Streams are
    thread-local ("There is no Stream(gpu, 0) in current thread"): the encoder thread must create the stream and the
    main thread build the check's graph on it (`with mx.stream(enc.s)`), with every eval, the final one included, run
    on the encoder thread. That form is bitwise == serial and survived 200 checks. Speed on a 48-layer toy: x0.96 by the
    pre-registered line (KILL). The toy turned out GPU-bound per kernel (build 2.2 ms, async_eval blocked ~55 ms), and
    its timings swung 50 -> 35 ms with EvalSuite's load, so the gain is unpriced. A narrow variant read x1.34 post hoc,
    and a same-thread new stream read x0.98-1.01. In the real check, the gain needs the GPU busy time per check
    (unknown) to sit below the CPU's ~31.5 ms. Engine form: wrap the check's forward in the encoder stream and route
    every mx.eval (gate readbacks, final) through that thread (BigMoE's loop). Parked as a lead.
  - **PF-B batching risk (BigMoE R57_PFLM=3 batches the router + shared expert over 4-8K rows):** MLX's
    quantized_matmul (q4, q8 g64) is per-row bitwise invariant at 1,024 vs 4,096 / 7,932 / 8,192 rows for the router
    (2560->512), shared gate / up (2560->640), shared down (640->2560) and shared-expert gate shapes. Plain bf16
    matmul is NOT (K 2560: 210-538 rows differ). The model quantizes the router (8-bit) and shared expert (4-bit), and
    ref2's DENSE-N-RAM unpacks to MLX's quantized_matmul above 16 rows, so the batching should keep the logits hash.
    **CORRECTION (Sun 16:5x):** the quantized arms skipped N < 32, so the shared-expert gate (q4, N 1) was only tested in bf16.
    It is NOT invariant: MLX runs it through qmm_splitk with an M-dependent split (PrefillKernels FINDINGS N3, 10-18). PF-B's
    logits-hash break came from it: see "PF-B EXACTNESS" below.
- **GPU-UTIL (Sun 08:5x, read-only, ioreg "Device Utilization %", 40 x 0.5 s; data/gpu_util_m1b.txt): during EvalSuite's
  M1-B + COH chat gate decode the GPU is busy 81 % (median 81, p10 74, p90 86), the one-call's spin on expert loads
  included.** At a ~40 ms cycle that leaves ~8 ms of GPU idle per cycle: the ceiling for every CPU-side lever (ENC2,
  compiled glue, fewer ops) is ~-20 % cycle even at zero CPU cost; GPU levers face an 81 %-busy GPU. 150 tok/s (x2.7
  over 56) cannot come from the check side; it needs more kept tokens per check (BigMoE's read too).

- **PF-B EXACTNESS: the break is shared_expert_gate (N 1, MLX split-K), a REDISCOVERY of PrefillKernels N3; the fix is
  that gate per chunk (Sun 16:4x-16:5x, share lock; scripts/pfb_moe_inv.py, pfb_seg.py; data/pfb_moe_inv.txt,
  pfb_seg.txt, pfb_bigmoe_report.txt).**
  - Trigger: BigMoE's PFB A/B (d44ec7a). PF-B runs x1.15-1.22 over PF-A (pb38: 218 / 438 / 499 tok/s on 1.6K / 4.7K /
    7.9K, peak 46.2 GB vs 45.1). But the decode logits hash vs PF-A is 0/7, 0/24, 0/22 in every arm, pb3n included, and
    on edit10 too (one 1,565-row MoE call vs 1,024 + 541: below the 2,048-row DQ-GEMM lane and below BUG1).
  - Test [K]: the engine's whole prefill MoE block, i.e. mlx_lm 0.31.3 Qwen3NextSparseMoeBlock.__call__ (qwen4_exp's
    SparseMoeBlock calls it at prefill when the switch is unfused), plus qwen4_exp._hyper_residual_write (hc 4).
    - Real shapes and quantization: router q8 g64 N 512, SwitchGLU q4 g64, shared expert q4 g64, shared_expert_gate
      q4 g64 N 1. Synthetic weights; skewed routing (up to 6,216 rows on one expert).
    - PF-A's shape: 1,024-token chunk calls with the stock SwitchGLU. PF-B's shape: pflm._split_layers' grouping
      verbatim, switch_glu_padded, Y sliced per chunk.
  - Result: only shared_expert_gate differs: 304 / 1,565 rows (edit10's shape), 2,326 / 4,730 and 4,451 / 7,932 at TOK
    4,096. The difference reaches shared_y / out / write on 64 / 449 / 850 rows. Everything else is ==: router,
    softmax, argpartition, take, norm, padded SwitchGLU, weighted sum, shared expert, residual write. TOK 1,024 (one
    chunk per group) is ALL ==.
  - Fix [K]: shared_expert_gate per PF-A chunk inside the group, everything else batched: ALL == at all six shapes
    (1.6K / 4.7K / 7.9K x TOK 4,096 / 8,192). Probe: one call vs 1,024-row calls differ from M 1,100 up (at M 2,048,
    rows 0..2,047 differ). The N 640 control is ==.
  - Mechanism (PrefillKernels FINDINGS N3, 2026-10-18, real layer-0 weights: 4,748 of 8,192 rows differ): MLX 0.32.2
    runs narrow-N quantized matmuls through qmm_splitk with split_k = 512 / ceil(M / 32), so their bits depend on M.
    N >= 512 gets split 1 -> qmm_t_nax, which is per-row invariant. PrefillKernels' PK2 split (experts per group, the
    rest of the block per chunk) was already bit-identical.
  - CORRECTION: my 08:5x dense-invariance PASS skipped quantized N < 32, and my "bitwise" PF-B advice to BigMoE
    covered the switch alone. This is a REDISCOVERY, not new: the owners' grep found N3 before any broadcast.
  - Peak [P]: PF-B's +0.9-1.1 GB is the group's transient MoE activations (sorted xs, h, y, y x scores: ~1.6 GB at
    8,192 tokens vs ~0.2 GB at 1,024).
  - Next (BigMoE builds): pb38 + the gate per chunk vs pa1 (expect logits hash == everywhere), and a TOK 2,048 arm
    for the 45.0 line.

- **ENGINE PACKAGE v0 (Leap, Sun 2026-11-08 19:4x; orchestrator 19:2x, ~/Papers/MONDAY.md + RELEASE section 4 TBD-9):**
  builder `release/make_release.py <snap>` (8763fcb) -> a fresh git repo: BigMoE's frozen snapshot taken read-only (git
  archive of the commit holding it), the 43-module union of all 174 R57_MODMAN manifests + bigmoe/, ext/, evalsuite/,
  copygate / relax_core, renamed roots (engine/ kernels/ third_party/ drafters/ keepset/ harness/), paths -> @ROOT@,
  comment-safe scrub (public-snapshot rules; 3 quotes redacted, all in comments / docstrings), leak check, syntax check;
  `run.py` = thin CLI (text prompt -> a harness item via evalsuite.data.chat; c45_runner's base env + the config line;
  the decode loop run unchanged via runpy in a child process; prints whole decode / whole request / first token /
  lifetime peak from runs.json); 6 configs: exact / lossy (L + COH) / chat (zs) x 45 (C 256) / 32 (C 160), lines
  parse-checked == BigMoE's (chat_env + SB1 == pfd ZS == chat32; lcoh == s5p + COH).
  - [K] package from snapshot pfd (BigMoE ba7dc548): 218 files, 6.2 MB; leaks 0, syntax errors 0, live research paths 0.
  - [K] clean-clone PREP (data/release/pfd_prep_report.txt): clone -> install.sh --offline (venv from wheels rebuilt from the
    research venv's installed files, release/wheelhouse.py: no network; mtplx's 321 .py == the venv's) -> extension build
    -> store + weights linked: rc 0 in 5-7 s; dry runs of all 6 configs OK. Import smoke with HOME = an empty dir: 51 / 51
    modules import from the clone, 0 from outside it.
  - [K] CONVERSION TEST (data/release/pfd_convert_report.txt): the three store steps (repack_records -> make_records3
    (MoonShots b660ce9, verifier: 0 mismatches) -> convert_records_p11 --pad) on layers 0 and 47 from the safetensors:
    L00.bin and L47.bin == models/bare/records3 BYTE FOR BYTE (1,426,063,360 B each), index entries ==; per layer 1-2 s
    repack, 4-7 s split + verify, 12-31 s p11. Full store [P]: 48 x ~20-40 s = ~16-32 min, ~4.4 GB extra disk at a time.
  - Next: SMOKE run Mon 07:00 (NewIdeas' slot): all 6 configs x 2 IFEval keys x CAP 64 with HOME empty, tokens vs
    verify/ (exact: AR prefix), module + open-file closure; FINAL on BigMoE's rel1 at 12:05.
  - [K] REBUILT from BigMoE's provisional release snapshot rel0 (aa1ca3b, FROZEN 358; Mon 00:12): 220 files, 6.3 MB, leaks
    0, syntax errors 0, live paths 0; 43-module union, 0 missing; EvalSuite 754b7e98 (evalsuite/ == 54bc8398's). chat-45
    turns PF-D on with BigMoE's default (R57_PFD=1,R57_PFD_V=4,R57_PFD_AHEAD=1: tokens bitwise equal to zs without it,
    BigMoE WED_TESTS Mon 00:09); exact-45, lossy-45 and the three 32 GB configs: off (BigMoE: not measured with it). Smoke
    clone re-prepped (data/release/rel0_prep_report.txt): install rc 0 in 7 s, 6/6 dry runs; chat-45's env has the three
    PF-D keys, the other five none. The 07:00 SMOKE runs this clone.
  - [K] DFLASH2 PIN + LICENCE (BigMoE's packaging items, Mon 00:2x-00:3x): every config loads the block drafter's base
    incoai/Qwen3.8-Flash-Next-DFlash2 (Inco AI, Apache-2.0 per its model card) via snapshot_download = the cache's default
    branch. run.py's engine process now pins revision a22a1744 (local files only; e532d27); install.sh checks the cache;
    run.py stops with the fetch command if missing. CPU test PASS (data/release/drafter_pin_test.txt): a snapshots-only
    cache resolves through run.py, the unpinned control fails, the real cache == the measured snapshot; v1 (revision
    only) FAILED offline (huggingface_hub 1.33 asks the Hub for a commit's file listing). third_party/LICENSE-dflash2 +
    NOTICE-l120_ft (our fine-tune's notice of changes); README / THIRD_PARTY credit Inco AI. Package 63f41fa (222 files,
    leaks 0); smoke clone re-prepped 00:32 (rc 0, 6/6 dry runs).
  - [K] SMOKE Mon 07:01 on 63f41fa (HOME empty; 2 IFEval prompts, cap 64): **FAIL (7)**. exact-45 / exact-32 tokens 2/2
    prefix-equal vs verify/; lossy-45/32 + chat-45/32 + the text prompt crashed: `No module named 'tierglu'`. Cause: the
    QOM arm's 3-bit sure rows make the loop call bigmoe.tier._tier_glu(), which imports tierglu from
    ~/MoonShots/patches/planes_engine; BigMoE's module manifests record only Leap / TrainLab / Breakthrough / snapshot
    files, so the build (which uses them) never saw it. The package's kernels/ms_gdn/tierglu.py is byte-identical (md5
    dc81729b in MoonShots @2aeb5ad, clean; same for gdn_exact / gdn_exact_patch). Also the closure check flagged the
    weight links (it resolves links but did not allow the weights: a check bug). FIX: make_release rewrites that path to
    @ROOT@/kernels/ms_gdn (tier.py 102, loader.py 45) + lists the other 43 ~/ paths in MANIFEST; the closure check allows
    the files clone/weights/* resolve to. Package 0a20f26 (4 files differ from 63f41fa); CPU checks PASS (_tier_glu with
    HOME empty loads the clone's copy; the fixed check passes the 07:01 exact records 0/0, the no-links control fails).
    Re-smoke armed 07:11. Data: data/release/smoke_0700.txt + smoke_0700_runner.log.
  - [K] RE-SMOKE Mon 07:13 on 0a20f26 (cap 64): **FAIL (2)**: the crash and the closure flags are gone (closure 0 / 0 on all
    six configs; the text prompt runs), exact-45 / exact-32 2/2, lossy information only (no reference yet), chat-45 / chat-32
    prompt 1 == but prompt 2 differs (token 4 / token 60). The chat references ran prompt 1 to 192 / 184 tokens (12 prompts,
    cap 256) before prompt 2; the smoke cut it at 64, and the chat configs' relaxed acceptance over the drafter's proposals
    can depend on what the process decoded before (hypothesis; re-smoke 2 tests it). TEST FIX (dbe5ea6): the test's default
    cap is 256 (the references' cap) and check_tokens.py --history checks a lossy / chat prompt only after the reference's
    earlier prompts matched in full (CPU tests: 6/6 cases as designed). make_release also refuses a live research path now.
    Re-smoke 2 armed 07:21. Data: data/release/smoke_0700.txt (RE-RUN 1) + smoke2_0713/.
  - [K] RE-SMOKE 2 Mon 07:28 on dbe5ea6 (cap 256): **CLEAN-CLONE PASS** (221 s). exact-45 / exact-32 2/2; chat-45 / chat-32
    2/2 with the reference's history, both prompts to the reference's full length (192 + 235 / 184 + 256 tokens), which
    confirms the history dependence; lossy information only (references from BigMoE's REL ROWS on rel1, ~10:00); closure
    0 / 0 on all six; the text prompt gives a complete answer; chat-45's runs show PF-D active (48 calls, 0 fallbacks).
    Data: data/release/smoke_0700.txt (RE-RUN 2) + smoke3_0728/.
  - [K] PACKAGE TOOLS Mon 07:3x (CPU; tested, in src for the rel1 build): (1) build gate: make_release refuses to commit with
    a live research path (negative copy: 2 paths listed, NOT COMMITTED; data/release/build_gate_test.txt); (2) the clean-clone
    test reports *.info.json references and never fails on them (for L + COH if BigMoE's REPEAT arm rlr_256 vs rli_256 is
    below 12/12; stub test); (3) convert/convert_store.sh builds the store layer by layer: L0 + L47 byte-identical to the
    existing store, index merged, resume skips, a failing step stops it (rc 1); v1 failed (its report inside the store folder
    made the p11 step refuse) -> report in <out>.work (data/release/convert_store_test.txt). Full store ~15-31 min [P].
  - [K] LOSSY LABEL + REL1 BUILD Mon 09:5x-10:0x (CPU): BigMoE's REL ROWS on rel1 (09:47-09:57, quiet, BM 53dfff5): REPEAT
    rli_256 == rlr_256 12/12 token-identical, IFEval C 256 == C 160 12/12 (rli_256 / rli_160 / rlr_256 tokens md5-equal
    f6076f09). By the rule set before the run, both lossy references ship as HARD checks (refs_rel1.json:
    lossy-45_rli256, lossy-32_rli160). Pre-check: the 07:28 smoke's lossy tokens (rel0, same code) == the references 2/2
    at full length (154 + 139) at 45 and 32 GB, --history (data/release/lossy_precheck.txt). rel1 build 242553d: 225
    files, 6.3 MB, leaks 0, unsafe 0, live 0, syntax 0; vs the smoke-tested rel0 package: snapshot name, convert_store.sh,
    README, the .info branch of the test, the two lossy refs (data/release/rel1_build.txt). Clone cleanclone/final
    prepped 09:58 (share lock; install rc 0 in 10 s; six dry runs). FINAL runner armed 09:59 (task, attached): quiet
    ticket at 12:05, last start 14:30.
  - [K] KERNELS v1.2 number check Mon 10:0x (NewIdeas NUMBER_CHECK section 5 + editor note 17; each re-read from raw):
    (1) Strata vs ours first token 2.7-4.6 s (MAIN Table 7.1 2.73-4.56); (2) PF-B peaks 46.23 / 45.09 / 45.81 GB from
    BM data/r57/pd{b38s,pa1,b32s}_256_runs.json mem (the report rounds to 0.1); (3) DN-KS -7.74 ms = five kinds incl.
    shared down 16.7 -> 9.1 us (four = 7.38); (4) 6.4 table: stepb's KB are KiB of all bytes (att q 17,744 KiB = 18.17
    MB incl. in/out), so its GB/s (243-247) were right and the MB column was KiB/1000: now 24.2 / 18.2 / 9.1 / 2.0 / 0.84
    / 0.22 MB; dnks_bench's 236 GB/s counts weights only (17.69 MB); (5) S 16: replica 295-296 GB/s, NAX6-N 74 %, FAST
    79-81 %, NAX6 58-59 % (k90_arm012 QUIET); R13: both probe runs (barrier 16/20 and 20/20; B5 0/20) + rough cost
    (+0.7 / +1.5 %, 3c-N +26.3 % one run; coh_check.txt); 180.3 = 142.77 whole decode pooled over s1n1-2 (geo 142.1);
    dispatch 1,000-4,000 x 1.24-1.56 us = 1-6 ms.
  - [K] LOSSY = CARD-GATED CONFIG Mon 10:3x (CPU, read-only git archive; after EvalSuite's card lcoh PASS 10:07): card
    snapshot (BM e057abf) vs rel1 (0b4df8c), snapshot names normalised: bigmoe/ + ext/ identical; the decode loop differs
    only by the PF-D block (R57_PFD, off in lossy) + its record field; pflm.py (PF-A / PF-D only) + converter + 4 extra
    scripts are imported by no lossy run (smoke3 modules.json 848 entries, rli_256 33). Env: the package's lossy line ==
    card_lcoh_env.txt except the card's lcb6 in R57_QOM_SETS; run.py names --kind code 'humaneval' (QOM on) and chat
    'multi' (off), as the card. -> the card PASS applies to the shipped lossy-45 (C 256; the card did not run C 160).
    RELEASE section 4 says so (data/release/lossy_card_equiv.txt).
  - [K] FINAL CLEAN-CLONE PASS Mon 12:10-12:14 (quiet, granted 12:10:13 at EvalSuite m1b's process boundary; 216 s; swap
    6.56 -> 5.44 GB): rel1 package 242553d, clone prepped 09:58, HOME empty, offline, 2 IFEval prompts x cap 256 per config.
    exact-45 / exact-32 2/2 prefix-equal vs ar_ref (153 + 256); lossy-45 / lossy-32 2/2 vs rli_256 / rli_160 (154 + 139, full
    length; hard checks); chat-45 / chat-32 2/2 vs pdz4a / zs32 (192 + 235, 184 + 256, full length); closure 0 / 0 outside
    on all six (834-849 modules, 3,268-3,353 files); text prompt 87 tokens, the same answer as smoke3. CPU cross-checks: 12/12
    tokens == the refs read directly and == smoke3 (rel0); lossy / chat prompts took the engine runs' checks and kept per
    check exactly, at x0.996-1.020 whole decode, x0.996-1.012 whole request, median first token within 0.11 s (CORRECTED
    12:5x from "0.1 s", NewIdeas; 2 prompts each,
    information). RELEASE TBD-9 filled + a one-Mac limit bullet (data/release/final_1210.txt; archive final_1210/).
  - [K] TRIAL README-ONLY REBUILD Mon 12:2x (CPU, scratch out dir): with the drafted README (de0b1c9), vs the tested 242553d
    only README.md + MANIFEST.json differ (MANIFEST: README's sha256 + the leap / evalsuite commit ids; evalsuite/ unchanged
    since 98c6072d) -> the post-16:00 README rebuild keeps every tested file byte-identical (data/release/readme_rebuild_try.txt).
  - [K] FULL STORE CONVERSION Mon 16:07-16:20 (share lock, after m1b): the tested package's convert_store.sh rebuilt all 48
    layers from the safetensors in 571 s (11-14 s per layer; repack 1-2, split + verify 4-6, p11 6-7 s) into a temp folder;
    cmp: 48/48 layer files + index.json byte-identical to records3; swap flat; temp deleted (data/release/convert_full.txt).
  - [K] FINAL PACKAGE PUSH Mon 16:2x: github kamenlevi/exactstream (PRIVATE) main 061f4f1 = rel1 package with the final
    README, harness pinned to the tested EvalSuite 98c6072d (make_release.py --es-commit; HEAD's cli.py had changed); vs the
    tested 242553d only docs + notices; draft release v1.0.0 with the drafter files (data/release/github_dryrun.txt).
  - [K] FINAL PUSH Mon 17:46: github main 16f3885 = the package (no 'laptop'; MAIN's new title; complete report listed)
    + docs/ = Breakthrough's final site (13 files, SHA-256 == its list); repo PRIVATE, release DRAFT, Pages OFF.

- **KERNELS v1.0 numbers derived from raw records (Leap, Sun 2026-11-08 19:5x; no GPU):**
  - [K] NATIVE-FRI RUN 2 pooled (scripts/natfri_pooled.py -> data/natfri/natfri_pooled.txt): over the 13 prompts kept in
    all four arms, whole decode NAX6-N 44.74 vs 3c-N 42.12 tok/s = x1.0622 (the report's steady tpsw ratio x1.0665 is
    information only); lifetime peak 38.11 vs 38.10 GB; that screen has no pfs, so whole request / first token cannot be
    derived.
  - [K] CHS4 coherent one-call pooled from BigMoE data/r57/y{coh,coh2,coc,coc2}_256_runs.json: coherent 37.8 / 37.9 tok/s
    whole decode, 21.1 / 21.3 whole request, first token 2.83 (2.23-3.79) / 2.80 (2.17-3.73) s; plain 37.7 / 38.0 and
    20.0 / 20.1 tok/s, 2.77 / 2.79 s; peak 41.69-41.70 GB.
  - PF-B EXACTNESS CONFIRMED IN THE ENGINE (BigMoE pfd part B, report at commit a6d5347, snapshot ba7dc54): PF-B TOK 8192
    + seg per chunk: tokens == PF-A and logits hash 8/8, 28/28, 22/22; 215 / 432 / 500 vs 189 / 358 / 433 tok/s
    (x1.14 / 1.21 / 1.16); peak 46.20 vs 45.10 GB; TOK 2048: 212 / 418 / 486 tok/s, 45.80 GB (still over 45.0).
  - MLX issue #4632 (opened 5 Oct 2026 by another user, closed 6 Oct; not reproducible on 0.32.3; fix attributed to PR
    #3922) predates our BUG1 (PrefillKernels N3, 10-18): ours is an independent finding. Root cause line verified in the
    installed 0.32.2 header: quantized_nax.h:1547, `min(SM, short(max(0, (M - (y_row + tm)))))`.

## TONIGHT AT 32 GB (Leap, Tue 2026-11-10 00:0x; orchestrator 23:2x on Kamen's yes): drafter pair, then prefill
- DRAFTER PAIR (data/drafter_1109.md), one 2,048-token code request through chat.py, chat-32, share lock, nothing else
  running: DFlash2 102.4 tok/s whole decode (6.14 kept/check, cycle 59.5 ms, draft 6.3 ms, peak 31.84 GB) vs MTP head
  85.3 (6.02 kept/check, cycle 67.9 ms, draft 20.4 ms, peak 31.60 GB) = x1.20 for DFlash2 on code; the gap is the draft
  cost, not acceptance; copy drafter 0 of 2,612 positions. Single runs, back to back (dfl first).
- PREFILL TEST (data/prefill_1109.md), chat-32 through run.py, 64 tokens, first token = prefill / lifetime peak:
  10K: default 78.41 s (127 tok/s) 34.01 GB | PF-A 22.71 s (440, x3.45) 34.83 GB | PF-B 20.04 s (498, x3.91) 35.77 GB;
  30K: default 252.24 s (119 tok/s) 35.48 GB | PF-A 68.55 s (437, x3.68) 38.49 GB | PF-B 58.02 s (517, x4.35) 40.83 GB.
  Tokens AND decode statistics: PF-A == default at 10K and 30K, PF-B == default at 10K; PF-B at 30K differs from token
  43 (first 42 match; 100 vs 107 rows). Decode stats are identical across the other paths, so PF-B's prefill at 30K is
  probably not bitwise (BUG until shown otherwise; the engine's PF-B check ran at <= 8K). Next: R57_LOGHASH PF-A vs PF-B
  at 30K + the exact-32 decode (~3 min GPU).
- PREFILL EXACTNESS AT 30K WITH EXACT DECODING (exact-32, R57_LOGHASH=1, 29,988 tokens, 64 tokens; Tue 00:57-01:07,
  orchestrator's go; data/prefill_1109.md + prefill_1109/exact30k/): default vs PF-A: tokens 64/64 and verify-logits
  hashes 26/26 EQUAL (first token 238.60 vs 62.64 s = x3.81; peak 34.93 vs 38.16 GB) -> PF-A bitwise to 30K. PF-B vs
  PF-A: hashes 0/25 at TOK 8192 (192 calls) AND at TOK 4096 (384 calls), first difference at check 1, tokens from the
  40th -> PF-B NOT bitwise at 30K. Published PF-B evidence (TOK 8192 and TOK 2048 = 3-4 calls/layer) is hash-equal to
  7.9K, so the call count alone doesn't explain it; length / position open. Papers: Editor's erratum (fail version).

## 24 GB PACKAGE CHECK (Leap, Tue 2026-11-10 09:59-10:02; orchestrator 08:4x; data/w24_pkg.md)
- `--mem 24` in the local clean clone = w13b as gated (card_w24d_env.txt blob e06f495e, C 100, w24 decode loop, PFCHUNK
  1024 kept by W24e). Replay of 6 gate rows (3 IFBench + 3 GPQA, CAP 2048): tokens 6/6 EQUAL, prompt tokens equal; peak
  23.79 GB. 12 HumanEval (W24d keys, CAP 256): tokens 12/12 identical to BigMoE's w13eh / w13fh; whole decode 135.9 tok/s
  (vs 130.33 pooled), peak 23.78 GB, first token 3.63 s median. 36 GB M5 Max: unverified (wired limit = the machine's
  recommended working set; GPU arch suffix differs -> MLX tiles may differ).
- PUBLIC CANDIDATE with the 24 GB line (release/out/pkg_w24 0f076a5; docs-only rebuild pkg_w24b b02d0ba; not pushed):
  CLEAN-CLONE PASS 10:23 (all 8 configs 2/2 token-equal to their references, closure 0/0, 271 s); serve.py --mem 24
  first real-engine test PASS (OpenAI + Anthropic streams well-formed, 31.9 / 32.1 tok/s, peak 23.68 GB, 0 errors;
  SIGINT ignored when started from a script, SIGTERM has no clean shutdown path: reported, not reworked).

## TUE AFTERNOON (Leap, 13:46-14:02; data/tue_afternoon_1110.md)
- 36 GB MAC SIMULATION of the 24 GB package (mlock ballast 22 / 28 GB, GPU working set capped at 27 GB in-process): HE x12
  138.1 / 137.3 tok/s (free memory 135.9), tokens 12/12 identical, peak 23.77 GB; 3K prompt 25.03 GB; 3-turn chat OK;
  swap flat / +0.4 GB. Unverifiable here: another GPU type, SSD, thermals.
- PF-A AT 24 GB (chat.py): identical answers; 6,980-token first token 42.06 -> 17.70 s (x2.38), peak 25.67 -> 25.98 GB.
  FINDING: the 24 GB line peaks at 25.67 GB on a 7K prompt (default prefill), above the label's 25.1 GB (gate prompts <= 2.8K).
- PF-B BISECT (exact-32, TOK 8192, hashes vs PF-A): 10K 25/25, 15K 28/28, 20K 25/25, 25K 24/24 (4 groups) -> exact to 25K;
  30K 0/25 -> breaks in (25K, 30K]; group count ruled out; hypothesis: an M-dependent matmul at the 30K group shapes.
- 24 GB PEAK BY LENGTH: 25.82 GB at 10K, 28.17 GB at 30K (default prefill); PF-A 26.54 / 30.60 GB (x2.55 / x2.72 first
  token). KERNEL CHECK: PF-B's grouped router + shared expert differ from the per-chunk calls only on a short final chunk
  (292 rows at 30K) -> exactness depends on the last chunk's size; fix = router + shared expert per chunk slice (not built).
- PF-B FIX (research copy, R57_PFLM_SEGALL=1): bitwise = PF-A on the short-tail prompt (5,412 tokens: 20/20, unfixed 0/20)
  and at 30K (26/26, unfixed 0/25); speed kept: x1.17 / x1.15 over PF-A, 30K prefill 555 tok/s (unfixed 567); peak
  unchanged (40.42 GB). Exact PF-B clears the 500 tok/s prefill line at 30K on the 32 GB line.
- 1.1.0 CANDIDATE pkg_110c ON THE REAL ENGINE (clean clone; lossy-24 + chat-24 keys, a text prompt, chat.py and serve.py
  --mem 24; Tue 20:57-20:59): PASS. 2/2 keys prefix-equal with the w13eh references on both configs; closure 0 / 0;
  peaks 23.49-23.71 GB (all under 24); chat.py rc 0 (25.3 tok/s); serve.py streams both formats and stops on SIGTERM.
- TOOL CALLS ON THE REAL ENGINE (serve.py working copy --mem 24, PF-A; Claude Code + Codex, contained; Tue 21:03-21:16):
  probe 27/30 (the 3 misses = one prompt answered in prose on all 3 endpoints; every call made was right); Claude Code
  3/3 tasks; Codex 2/3 + 1 partial (appended on the same line); side lane flagged; prefix reuse 8/8 follow-ups whole;
  per turn without a cache 30.5-32.2 s (Claude Code, 12.3-12.6K tokens) / 21.7-23.2 s (Codex, 8.4K); peak 27.32 GB.
- 1.1.0 ANSWER LENGTH (Kamen 21:3x): chat.py / serve.py --max-tokens default 16,384 -> 65,536 (CAP is only a stop bound:
  no memory by itself); candidate rebuilt as pkg_110d (328a19f; differs from the GPU-tested pkg_110c in 2 defaults + help
  text); clean-clone prep + stand-in start check PASS (banner "up to 65536 tokens per answer").
- 1.1.0 chat.py FIXES (Kamen's chat, 21:4x-22:2x): paste = one message at any length (non-canonical reader, bracketed paste;
  the tty's 1,024-byte line cut a 1,800-character paste), live numbers in the window title (the frozen display came from
  the UI sizing itself from the log fd: 30-row fallback), no research-PAUSE wait, /file PATH message; pty test 21/21;
  in Kamen's clone 22:22 (real-engine start rc 0); candidate pkg_110e (7943ddb).
