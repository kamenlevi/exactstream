# LUNACY: round 1, think-first version (Lunatic, Tue 3 Nov 2026, 00:0x-00:5x EET)

## How this version was made
- **v1 (00:03-00:38)** started from the wall files, as the original brief said. Kamen's change (00:4x): generate from `PHYSICS.md` and first principles only, and open files only to kill-test.
- **This version** regenerates the assumptions and the ideas from PHYSICS.md alone, using:
  - the physics law
  - "what would 10× need"
  - inverting an assumption
  - removing a constraint
  - combining dead ideas
  - questioning the measurement
- **Only after the list was complete** did I open files, for the numbers and for TRIED repeats.
- **Origin tags** on every idea:
  - **FP**: I would have it from the physics page without having seen the engine.
  - **FP + data**: an FP idea whose decisive number came from a kill test on recorded data.
  - **engine**: only someone who had read the engine would have it.
- **Marks:** [K] measured (source named), [P] arithmetic on measured numbers, [G] judgement.
- **Rules kept:**
  - CPU only; 4 Python runs in all (1 failed at import, then `lu_union.py`, `lu_batch_ssd.py`, `lu_alpha.py`, < 1 min of CPU; outputs `lu_*.out` here); the think-first ideas needed arithmetic and greps only.
  - No GPU, downloads or agents; writes only in `~/FableGod/lunatic/`.
  - Bytes and matrix-unit groups are priced on the union per block.

## 0. The answer
1. **The law, from the physics page:**
   - tok/s ≤ RAM bandwidth × tokens kept per pass ÷ bytes moved per pass (the dense 3.4 GB plus the union of the experts the pass touches).
   - The SSD obeys the same law with misses in place of the union.
   - 350 tok/s means ≤ 0.77 GB of weights moved per kept token at 270 GB/s. One token alone touches 4.4 GB, so every byte moved has to serve ~6 kept tokens.
2. **What 10× (1000 tok/s) needs:**
   - At most 0.27 GB of RAM traffic and 12 MB of SSD traffic per kept token, so each byte has to serve ~16 tokens and ~99 % of expert uses have to hit RAM.
   - One stream gets there only on 128-token runs at ~0.995 agreement, i.e. copies.
   - Many streams get there: S1 below, with the sidecar dropped at night, reaches **~550-850 tok/s aggregate on FAST** at the measured kernel cost.
3. **Single-stream EXACT 350 is not reachable on this Mac.** With a perfect drafter on the rows where the 125B is sure, agreement stops at 0.9485 (the 125B's own close calls). That is ×1.36 on bytes per kept token: ~400 tok/s at 100 % of RAM bandwidth, ~245-280 at the measured kernel efficiency [K lu_alpha, AUDIT_350]. EXACT reaches 350 only as aggregate (S1) or on copy-heavy work.
4. **S1, parallel same-domain streams (FP):**
   - Expert routing is shared across the domain: 4 independent code answers × 16 rows touch 179 experts per layer, against 166 for 64 consecutive rows of one answer [K lu_union].
   - So 4 streams per pass cut RAM per token to ×0.47 while SSD bytes per token stay flat at C 256-320 [K lu_batch_ssd].
   - Result: ~2× aggregate EXACT at night, bitwise per stream. With the sidecar dropped (M-26: 8.1 MB per kept at C 320), FAST at 8 streams is no longer SSD-bound.
   - Never tested here: TRIED has no multi-stream line.
5. **S2, deep relaxed blocks (FP + data):**
   - The drafter's misses on 125B-sure rows are a horizon effect (99.6 % right at block position 0, 88.4 % at 12-15) and as frequent as its mid-row misses.
   - Fixing the horizon, relaxing tie and mid rows (FAST), and using 64-row checks halves bytes per kept token. This is the only single-stream ×2 found.
6. **Dead:**
   - two-speed EXACT behind FAST
   - Jacobi tails and the MTP head as the drafter (both already tried)
   - certified early exit and Freivalds checks
   - the idle video engines and texture units as decompressors
   - cross-expert deltas
   - the whole model in RAM (55.5 GB needed against ~52; also unnecessary, see I4)
7. **The system:** price every idea on the three walls; let FAST own single-stream 350 and EXACT own ~250 single-stream plus 350 aggregate; and measure an agentic turn. At the target, prefill would be ~70-80 % of a turn's time and no current number sees it.

## 1. The ten assumptions (generated from the physics page)

| # | assumption | method that questions it | verdict |
|---|---|---|---|
| 1 | **Speed is one answer's tokens per second** | inverting | **break (S1)**: aggregate across parallel streams |
| 2 | **A pass is filled with one answer's speculative rows**, so its depth is capped by the drafter | combining dead ideas (32-row checks + batching) | **break (S1)**: fill the wide pass with other answers' rows, which are accepted at their own rate (0.66-0.79) instead of the 10-20 % of rows 17-32 |
| 3 | **The drafter's job is knowing what the 125B knows** (the close calls) | extreme case: what would 10× kept need? | **break (S2)**: for wide blocks the binding misses are horizon misses at rows the 125B is sure of |
| 4 | **Wider passes lose** (the union grows faster than kept) | physics law | true at agreement 0.909, false at 0.974 (S2); S1 is the wide pass that works |
| 5 | **EXACT 350 single-stream is reachable by stacking levers** | physics law + measured agreement | **break the target**: EXACT single-stream ≈ 250; 350 is aggregate or copies |
| 6 | **RAM is a cache in front of the SSD** | removing a constraint ("as if it all fit") | the SSD stops binding before it all fits: night + sidecar drop = 8.1 MB per kept (M-26) |
| 7 | **The EXACT reference is fixed** (3 bits + the 0.77 MB top-2 extra-precision part) | changing what is optimized | **question it**: the reference is a design choice; re-basing EXACT on a format FAST proves equal (no sidecar) halves EXACT's SSD bytes (Kamen's call; I4) |
| 8 | **The GPU does the work; CPU, Neural Engine and video engines idle** | first principles | **holds**: RAM bandwidth is shared, so other units only fill latency gaps (≤ 30 %) |
| 9 | **The measurement is decode tok/s on from-scratch code and short chat** | questioning the measurement | **break**: an agentic turn is prefill + decode + copies + parallel sub-tasks; at 350, prefill dominates |
| 10 | **Progress is levers adopted in %, from 10-minute screens, routed to component owners** | questioning the system | **break**: price on the three walls; give structural bets one owner and a multi-day build. S1 and S2 are both invisible to a 10-minute single-stream screen |

## 2. The ideas (generated first; kill-tested after)

| # | idea (origin, method) | mechanism | math [cited] | kill test → verdict |
|---|---|---|---|---|
| I1 | **Parallel same-domain streams** (FP; inverting #1) | 2-8 independent requests (agent sub-tasks, files, tests and docs) decoded as one merged pass per step: one expert pass per layer over all streams' rows; per-stream drafts and states | union ×0.81 / ×0.58 / ×0.39 of B × single at B 2 / 4 / 8 [K lu_union]; RAM per row 0.66 → 0.47 → 0.31 → 0.20 GB; SSD per row at C 320 43.8 → 40.8 → 39.1 → 48.8 MB, at C 256 79.1 → 81.0 → 77.5 → 103.0 [K lu_batch_ssd] | replay PASSED (flat SSD to B 4) → **ALIVE: S1** |
| I2 | **Deep relaxed blocks** (FP + data; extreme case) | fix the drafter's horizon at 125B-sure rows; FAST accepts tie and mid drafts the 125B ranks top-3 with p ≥ 0.1; 64-row checks | sure rows 99.6 → 88.4 % from block position 0 to 12-15; sure-row misses 3.93 % of code tokens = mid-row misses 4.1 %; agreement 0.909 → 0.974; bytes per kept 1.18 → 0.59 GB (first-order) [K lu_alpha] | replay PASSED → **ALIVE: S2** |
| I3 | **Two-speed: FAST runs ahead, EXACT certifies behind in prefill-size batches** (FP; inverting #10 of v1) | EXACT verifies FAST's text in 64-128-row passes | certifying costs ~2.4 ms per token at 64 rows [P lu_union, AUDIT_350 fit]; FAST and EXACT diverge at 1.34 % of code tokens [K DAY9_q1], so ~30 % of FAST's run-ahead is discarded; EXACT ≈ 140 tok/s, the same as direct | **DEAD** (UX value only) |
| I4 | **The whole model in RAM** (FP; removing "RAM is a cache") | drop the sidecar, pack the scales, hold all 512 experts per layer | 512 × 48 × 2.15 MB = 52.8 GB (47.0 packed) plus today's 8.5 GB non-pool lifetime peak (BigMoE HANDOVER:607) = 55.5 GB, against ~52 GB available at night | **DEAD as a build**, and unnecessary: the sidecar drop alone gives 8.1 MB per kept at C 320 (M-26, TRIED:35); S1 uses that fact |
| I5 | **Re-base EXACT on the lean format** (FP; changing what is optimized) | once FAST proves "all ranks at 3 bits" scores the same, make that the EXACT reference | code SSD per kept at C 256: 46.3 → 24.5 MB (−47 %) [K M-26]; removes EXACT's day SSD exposure and the extra-precision groups: ~125 → ~160 tok/s at 45 GB [P on AUDIT_350]. The strict KLD gate failed it (M-02, ×1.24-1.83); task-score results are pending (t3_all) | **DECISION for Kamen** after EvalSuite; ×1.3 → HANDOFF |
| I6 | **The 125B drafts for itself** (Jacobi tails; the built-in MTP head) (FP) | reuse the verifier's rejected-tail predictions, or chain the native MTP head | Jacobi / lookahead KILL ≤ 5 % (TRIED:598); MTP chain at depth 15 ×1.3 on chat and agent, ×0.75 on code (D-17, TRIED:520); MTP depth 1 gets the 125B's close calls 86 % right vs DFlash2's 66 % (KWALL b, TRIED:538) | **DEAD as new** (MTP Head owns it; the depth-1 result belongs in S2's horizon fix) |
| I7 | **Certified early decision** (interval bounds on low-precision passes; Freivalds checks with precomputed rᵀW) (FP) | prove the argmax without the full-precision pass | a residual stack's error grows like Π(1 + Lip) over 48 layers (1.3^48 ≈ 3·10^5); 2-bit passes flip tokens (B7 / B9 KILL); Freivalds checks a claimed product but cannot produce one | **DEAD** |
| I8 | **Idle units as decompressors** (video engines, texture formats, the GPU's lossless framebuffer compression) (FP; the page lists idle engines) | store weights compressed, let fixed-function hardware decode | lossless gain ≤ 9 % (codes at their 2.72-bit entropy floor, SSD_WALL §2); lossy texture formats at 3.56 bpp equal today's 3.5, at 2 bpp they are a 2-bit tier (KILL); hardware decoders write into RAM, so every byte is written and read again | **DEAD**; the CPU-decoded compressed warm tier (+9-13 % slots) → HANDOFF |
| I9 | **Idle units as co-processors** (CPU attention at long context; dense layers on the Neural Engine) (FP) | overlap them with the GPU's expert streaming | RAM bandwidth is shared; the GPU's dense pass already runs at ~65 % of bandwidth and the experts at 61 % of streaming, so only those gaps can be filled; within one stream the layers are serial | ≤ 30 % → HANDOFF (it fits best inside S1's multi-stream pipeline) |
| I10 | **Cross-expert delta coding** (FP) | store an expert as a delta from a resident sibling | 2× needs pairwise correlation ≥ ~0.87; fine-grained experts are near-orthogonal by prior; the shared-basis form (MoBE) is lossy and PARKED (TRIED:246) | **DEAD by prior** (a one-minute CPU check would settle it) |
| I11 | **Measure an agentic turn** (FP; questioning the measurement) | count a turn as prefill of the tool output + decode + copied spans + parallel sub-tasks | prefill streams ~half the experts from the SSD per chunk at 45 GB (~26 GB → ~2.2 s at 12 GB/s), so a 3K-token tool output costs ~2-3 s, against 0.86 s for 300 tokens decoded at 350 [P]; measured TTFT at 24 GB is 12.7 s for 2.9K tokens (TRIED:763); copy spans run at ~0.995 agreement → ×2.6 bytes per kept at 128 rows [P lu_union] | **ALIVE as a system change**: at the target, prefill is ~70-80 % of a turn |
| I12 | **Order-invariant exact arithmetic** (engine) | pre-rounded sums are exact in any order, so any split or path gives the same bits | covered by Fable B's F1 + REF3-T and cloud task 17's integer-exact option; per-group scales limit it | **DEAD as new** → HANDOFF |

## 3. Survivor briefs

### S1. Parallel same-domain streams: the Mac's real capacity, and the 1000 line as aggregate

- **Assumptions it breaks:** #1 (speed is one stream), #2 (a pass holds one answer's rows), and the folk rule that batching an SSD-streamed MoE multiplies expert loads.
  - It is also a combination of two dead ideas: the 32-row check died because rows 17-32 of one answer are mostly rejected; batching was never tried because the SSD was assumed to explode.
  - Filling the wide pass with rows from *other answers*, each accepted at its own rate, revives both.
- **Mechanism:**
  - 2-8 independent requests from one session decode as one merged pass per step. Candidates: an agent's sub-tasks, several files, code + tests + docs, or best-of-n.
  - Per-stream drafts are batched (the drafter is read once). Per-stream states: 112 MB of GDN state each (36 × 3.1 MB) plus attention KV.
  - One expert pass per layer covers the union of all streams' rows.
  - The reference is batch-invariant, so every stream stays bitwise identical to its solo EXACT run.
- **Why it sounds crazy:** batching on a one-user laptop, against the SSD folk rule. And it does not move the single-stream number at all.
- **The math [K lu_union, lu_batch_ssd, M-26; P]:**
  - **Domain-wide sharing:** code's union grows like a vocabulary, ∝ N^0.47 (76.6 → 283 experts per layer from 16 to 256 rows). Four independent code answers × 16 rows ≈ 64 rows of one answer (179 vs 166).
  - **RAM per token:** ×0.71 / ×0.47 / ×0.30 at B = 2 / 4 / 8. Expert groups per token ×0.58 at B = 4.
  - **SSD per token:** ×0.98 / ×0.89 at B = 4 (C 256 / 320); +11-30 % at B = 8; ×1.6 at C 160 (dead at 24 GB).
  - **EXACT at night (C 320):** single-stream ~145-155 is GPU-bound (AUDIT_350 §2). At B = 4 it reaches ~250-310, capped by the SSD at ~34-38 MB per kept ≈ 310-345. **About ×2.**
  - **By day, 45 GB:** ~125 → ~190 (SSD-capped), **×1.5**.
  - **FAST at night with the sidecar dropped:** M-26 gives 8.1 MB per kept at C 320 single-stream, ~9 at B = 8, so the SSD needs only ~6 GB/s even at 700 tok/s. Per 8-stream pass (128 rows, ~84-101 kept):
    - experts: 237 groups per layer × 48 × 10.15 µs = 115 ms (×0.7 with Q1 ≈ 80)
    - dense: ~31 ms (compute-bound at 128 rows)
    - drafts: ~10 ms
    - **→ ~550-850 tok/s aggregate**, ~9-13× today's single stream; ~1,000 at the RAM roofline.
  - Each stream runs at ~0.5× (B = 4) to ~0.3× (B = 8) of its solo speed.
- **The kill test it passed:** lu_batch_ssd (per-layer LRU, prompt-warmed, aligned 16-row steps). SSD per token is flat to B = 4 at C 256 and 320.
  - Caveats: no sidecar, no M-17 per-layer sizes, no rejected rows. Single-stream numbers calibrate at ×0.85-0.89 to SSD_WALL's plain LRU.
- **First real experiments:**
  1. **MoonShots, CPU, 10 min:** the band45 / cachesize45 replay with B = 2 / 4 / 8 interleaved code captures, M-17 sizes, the sidecar (and the 3333333333 plan), and real check rows including rejected ones. **PASS:** SSD MB per kept at B = 4 ≤ 1.15× B = 1 at C 278 and 320. **KILL:** > 1.3×.
  2. **BigMoE, GPU, ~1 h at night:** two, then four HumanEval prompts as one merged pass per step. Check every stream bitwise against its solo run; measure aggregate tok/s and SSD MB per kept. **PASS:** aggregate ≥ 1.6× solo at C 320, all streams bitwise. **KILL:** < 1.25×.
> (a quote from the project owner, kept private)
- **Prior art:** batched offloaded-MoE inference (MoE-Lightning, Fiddler), continuous batching (vLLM), chunked prefill alongside decode (Sarathi-Serve).
  - New here: the measured domain-wide Heaps' law of expert routing, its consequence that same-domain batching costs no SSD bytes per token on an SSD-streamed laptop MoE, and its combination with the sidecar drop into an aggregate 1000 line.
  - Tag: adapted.

### S2. Deep relaxed blocks: the only single-stream ×2 on FAST, and the drafter target it implies

- **Assumptions it breaks:** #3 (the drafter's enemy is knowledge) and #4 (wider passes lose).
- **Why the physics points here:**
  - Single-stream bytes per kept fall only if kept per pass grows faster than the union. From lu_union's curve that needs agreement ≥ 0.98 (×1.4 at 64 rows) or ≥ 0.99 (×2 at 128 rows).
  - So the question was which misses keep agreement below 0.98.
- **The data [K lu_alpha: 3,283 tested code positions; DFlash2's HPD-0 walk joined with the 125B's top-k]:**
  - 86.5 % of positions are 125B-sure (p1 ≥ 0.99), 12.0 % mid, 1.5 % tie. DFlash2 is right 95.5 / 65.9 / 30.0 % on each.
  - At sure rows, by block position: 99.6 % (0), 98.3 (1-3), 96.1 (4-7), 93.1 (8-11), 88.4 (12-15). Each is conditional on every earlier block token being right, so this is a horizon, not a cascade.
  - Sure-row misses are 3.93 % of all code tokens, as many as the mid-row misses (4.1 %). 67 % of them sit in the 125B's top-8, and the drafter flags only 26 % with cf < 0.5.
  - The native MTP head at depth 1 gets mid rows 86 % right (KWALL b), so the close calls are reachable near the anchor.
- **Mechanism:**
  1. **Horizon fix (MTP Head):** a second refinement step over block positions 8-15, conditioned on the first pass's tokens (re-anchor on what has been decided instead of extrapolating 12 tokens from one anchor), or a deeper-conditioned tail head.
  2. **FAST relaxed tie / mid rows (EvalSuite):** accept the draft when the 125B ranks it top-3 with p ≥ 0.1. Mid rows go 66 → 82 %, ties 30 → 74 %. Sure-row misses are never relaxable (the 125B gives the draft < 0.01).
  3. **64-row checks (BigMoE):** the union grows 76.6 → 165.6 per layer (×2.16 for ×4 rows).
- **Why it sounds crazy:** it says the drafter effort went to the close calls, which the knowledge wall proved hopeless, while the binding term for wide blocks is the drafter's horizon at rows the 125B is certain of. It also revives wide checks.
- **The math (first-order: independent misses, a relaxed token's effect on later text ignored):**

  | rule | agreement | best bytes per kept |
  |---|---|---|
  | strict, today | 0.909 | R16 1.18 GB |
  | R2 relaxed | 0.935 | ~1.0 |
  | strict + perfect sure rows | 0.9485 | R32 0.87 (**×1.36 on EXACT**, its biggest lever) |
  | R2 + perfect sure rows | 0.974 | R64 0.59 (**×2.0 on FAST**) |

  - Q1 adds its ×0.6-0.8 on expert bytes.
  - FAST's RAM-bound ceiling moves from ~300 toward ~600; ~360-420 at the measured ~61-70 % efficiency. That is the first single-stream FAST route past 350 that does not need the expert kernel to beat its measured ceiling.
- **First real experiments:**
  1. **MTP Head, CPU replay of the HPD-0 walk** with a two-step block (re-draft positions 8-15 given the first pass's tokens). **PASS:** sure-row accuracy at j 8-15 ≥ 97 % (from 88-93 %). **KILL:** < 93 %.
  2. **EvalSuite:** same scores on code / math / IFEval for the R2 rule, in the RELAX harness.
  3. **BigMoE:** 64-row FAST checks priced with the width replay once (1) passes.
- **Owner:** MTP Head (it owns the drafter and the 98 % track). The target becomes "125B-sure rows at block positions 8-15", not the close calls.
- **Prior art:** block-diffusion drafters (DFlash), iterative refinement in diffusion LMs, relaxed verification (MARS, typical acceptance), position-decay curves (EAGLE).
  - New here: the decomposition (horizon misses equal close-call misses and bind wide blocks) and the ×2 target that follows.
  - Tag: adapted.

## 4. HANDOFF
One line each, for NewIdeas' queue: `~/FableGod/lunatic/HANDOFF.md`.

## 5. Reproducibility and method note
- `lu_union.py`: union by block length, cross-stream union, per-kept pricing on the 36 routing captures.
- `lu_batch_ssd.py`: per-layer LRU SSD replay for B parallel streams.
- `lu_alpha.py`: the HPD-0 code walk joined with the map's top-k (the knowledge probe's join).
- All three run on `~/BigMoE/.venv/bin/python3 -I` with nice 15, recorded data only. The think-first ideas (I4-I11) were killed with arithmetic and greps of TRIED.md, BigMoE HANDOVER:607 and MoonShots' M-26 line.
- Nothing committed; no messages sent.
- Honest note on method: the replays in S1 and S2 were run in v1, before the think-first change. S1 passes the "would I have it without their engine?" test (batching is the first idea of the physics law; the replay only killed or kept it). S2's idea (wide blocks need ≥ 0.98 agreement) is FP; its horizon finding came from the kill test.
