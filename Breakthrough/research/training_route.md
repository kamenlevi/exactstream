# The training route: making small-RAM routing natural instead of approximating it (2026-10-20 night)

## Why inference-only tricks hit a wall on this model
- Flat routing: rank 0 holds 21 % of the top-10 mass, ranks 0-1 36 %, 90 % needs 8.8 ranks (gate_mass.py).
- Cold tail: skipped experts are not reused soon (Belady oracle recovers 0.295 -> 0.266 missed mass at 10 %), and
  adjacent tokens share few experts (our pass replay: 4-row passes 106 vs 111 loads/token; Edge0: adjacent tokens agree
  on ~1/4 of a layer's experts). So neither caches nor speculation amortise the tail.
- Quality needs ~85-95 % of the mass computed (wait-to-85 % = ar4's KLD), which costs ~300 MB/token at 10 % residency:
  ~40 tok/s SSD-bound on the M5 Pro.

## What the literature that reaches small RAM did
- **Edge0** ("The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing Prediction",
  arXiv 2609.18063): Qwen3.6-35B-A3B (40 x 256, top-8) served at **K = 4** with a **trained next-layer pre-router whose
  prediction is the routing** (so reads overlap compute and nothing is dropped), recovered by distillation: heads
  imitate the next router, then ~2M rows SFT with student routing, then ~200k rows on-policy reverse-KL distillation;
  a LoRA recovery adapter kept unmerged (merging + re-quantising erases most of it). Mac mini M4 Pro 24 GB: 20.4 tok/s
  decode, 2.9 GiB active; 8B tier 28.0 tok/s at 1.5 GiB. Quality vs fp16 base: average 79.2 vs 83.2 (-3.9, includes
  int4); IFBench 57.9 vs 61.7. SSD per step at K=4 ~51-59 MiB. Pre-router on a 16 GB M2: +80 % decode.
- **ZEDA** (2605.18643): two-stage self-distillation lets post-trained MoEs (Qwen3-30B-A3B, GLM-4.7-Flash) skip >50 %
  of expert FLOPs at marginal loss.
- **Cache-Aware Joint Router Adaptation** (2609.04895): trains auxiliary cache routers (temporal reuse) while keeping
  the native top-K; "Temporal Router" predicts same-layer reuse.
- **SPICE** (2608.21240): shared expert + trained low-rank surrogate per routed expert for low-confidence misses;
  GSM8K -3.0..-3.5 points vs exact (DeepSeek-V2-Lite, Qwen2-57B): surrogates alone are not quality-neutral.
- Inference-only, no training: ACE (2609.05228) 20 % of slots skipped by score ~free on Qwen3.6-35B; SERE
  (2602.07616) similar-expert re-routing; both far below our ~29 % mass at 10 % residency, chosen by the cache.

## What a training route would look like for the 125B (to be priced, not started)
1. Reduce K 10 -> 5-6 with distillation (Edge0 / ZEDA): halves expert bytes per token and per pass (RAM roofline
   8.3 -> ~6.4 GB per 4-row pass) and roughly halves frugal's cold misses; inference-only top-6 costs KLD ~0.02
   (our m70 probe, 5.9 ranks), so distillation has to recover ~0.01.
2. Concentrate the router (entropy / cache-locality regularised distillation of the gate only, 48 x 2560 x 512 =
   63M parameters): if ranks 0-1 held ~70 % of the mass, wait-top-2 frugal would sit near today's wait-to-70 % KLD
   (0.020) at today's bytes (56 MB/token at 10 %).
3. Recovery LoRA on dense parts (attention / GDN / shared expert), unmerged, distilled on-policy against the exact model.
Cost on this Mac: the teacher is the exact streamed model (~15-40 tok/s); millions of rows are out of reach locally
(Edge0 used ~2.2M rows); a gate-only distillation with cached teacher top-k logits is the only small first step.
Decision for Kamen (not ours): whether a trained variant of the 125B is acceptable for the breakthrough line.

## Router concentration costs accuracy (2609.04895, read 23:50)
Joint full-backbone + auxiliary-router training of Qwen3-30B-A3B (48 x 128, top-8; 16 % residency) with a
differentiable cache-coverage loss: hit rate +10 to +33 points, simulated load 1353 -> 974 MB/token (GSM8K) at
Temporal-Router weight 0.1-0.3 with GSM8K 85.44 -> 85.44 / MATH -0.40 / CSQA -1.23; raising the weight to 2.0
concentrates routing (top-8 share of all mass 36.8 -> 48.6 %, entropy 3.96 -> 3.51, effective experts 53 -> 35) but
GSM8K falls 10.84 points. No wall-clock numbers. Native top-K is kept at inference. Lesson: mild concentration is
nearly free with task SFT, strong concentration is not; a 125B route would need distillation, not task SFT.

## Feasibility after the orthogonality result (2026-10-21 02:15)
- A layer-local router re-fit (gate-only, frozen experts) cannot reduce K: the routed experts' outputs are mutually
  orthogonal (best |cos| median 0.024), so no reweighting of a top-5 subset reproduces the dropped five contributions.
  Only end-to-end adaptation (the rest of the network learning to work without them, Edge0's LoRA + SFT + on-policy RKL)
  can, which is a training run on the full model.
- On this Mac: forward + backward through the 125B with streamed experts is ~5-10 tokens/s at best; Edge0's smaller
  on-policy phase alone (~200k rows, ~40M tokens) would take months; the SFT phase (~2M rows) years. A K-reduction or
  locality distillation of the 125B therefore needs rented GPUs (order of 8x H100 for days), which is Kamen's decision
  (cost, and whether a trained variant counts as "the same model").
- Inference-only, the remaining quality lever is confidence-gated coverage (IDEAS D1c), which keeps the weights.
