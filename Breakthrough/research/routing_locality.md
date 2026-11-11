# Routing locality at tiny expert caches

2026-10-19. SR = self-reported, not replicated; "ours" = measured here.

## 1. Published trade-offs

| Method | What it changes | Fetch/miss cut (residency) | Quality delta | Training | 256-512 experts? | Src |
|---|---|---|---|---|---|---|
| Wait-top-2 (ours) | wait ranks 1-2; resident rank 10-19 substitute; else drop | 152→20 loads/token (10 %) | GSM8K-150 identical at 19-25 % (35B) | none | yes | ours |
| "Which misses…" (known) | fetch rank-1 or gate ≥0.15, substitute rest | n/a | PPL +0.35 % vs +5.7 % substitute-all (~25 %); GSM8K +0.4 to −8.4, tracking lost gate mass (12.5 %) SR | none | 512 | 9 |
| Cache-conditional (known) | rerank toward resident experts | >50 % | GSM8K/MMLU <0.1 % SR | none | no | 3 |
| Cache-aware joint router adaptation | backbone + retention-router post-train, soft top-B coverage loss | −31 % at 15.6 %; −70 % at loss weight 2.0 (quality unreported) | GSM8K 0.0, MATH −0.4, CSQA −1.2 (−31 % run) SR | full model, eval-task train splits | 128 | 1 |
| ReMoE (known) | router-only locality losses | −4 % (6.3 %), −7 % (9.4 %) | GSM8K −0.76 SR | router | no | 2 |
| Cacheable by Design? | trained locality + training-free rerouting | −60 % trained, −80 % stacked | PPL ≤+3.4 %, failed its ≤1 % gate SR | pretraining ≤340M | no | 4 |
| PreMoE | per-task static subset | keep 50 % / 12.5 % | MATH500 97.2 / 72.0 % (DeepSeek-R1) SR | none | 256 | 7 |
| ZEDA | zero-output experts + self-distillation | >50 % of expert FLOPs | "marginal", 11 math/code/IF sets SR | post-train | 128 | 10 |
| ACE (known) | calibration-free skipping | 50 % skipped | +4.15 pt avg vs best rival SR | none | 256 | 11 |
| MoNE | pruned expert → its mean output | 25 % pruned | −0.14 zero-shot avg SR | calibration | no | 12 |
| Speculating Experts (known) | compute predicted, not routed, experts | n/a | GSM8K 95.0→57.6 % (router predictor), 91.8 % (better one) SR | predictor | 128 | 14 |

## 2. Best published change at ≤10 % residency; what is open

- No paper shows reasoning-equal quality from a routing change at ≤10 % residency on 256-512 experts. The only ≤10 % data: ReMoE (−4 to −7 % misses, 64 experts) and SpecMD's exact Least-Stale prefetch (88 % hits at 5 % SR, but ~4× LRU's bytes in [1]'s replay).
- Best quality-safe trained result: [1] (−31 % misses at 15.6 %, GSM8K unchanged), via full-backbone tuning on the eval tasks' train splits. Best training-free: cache-conditional reranking (~2×, ≤64 experts).
- Our wait-top-2 cut (7.6× at 10 %) exceeds every published one (different models); unknown is its quality with 186 of 480 rank-uses per token substituted or dropped.
- Working sets (Denning): 10 % = 51 slots/layer, about a 10-token working set (26 at W=4, 66 at W=16). 2.5 % = 13 slots, under even 4 tokens' 26: only "ranks 1-2 exact (≤8 distinct per 4 tokens), rest approximated" fits.
- Perplexity hides reasoning loss ([4]; our 14 points at ~4 % PPL). Substituting co-activated "buddies" cost 4 ARC points even at 75 % residency (BuddyMoE [13] SR). Dropping the bottom quarter is free (DyMoE); dropping half costs points unless trained (MoBiLE: naive halving −8.5 % GSM8K SR [15]; ZEDA trains).
- Open: per-layer drop/substitute/correct choice for those 186 ranks; training under the real miss policy; admission control; any 512-expert evidence at ≤10 %.

## 3. Router-only fine-tune or small learned substitute (64 GB Mac)

- **Params:** routers 48 × 2560 × 512 = 63M (~1 GB with fp32 Adam). Smaller: per-expert selection bias (24.6k) or per-layer temperature.
- **Substitute:** per-expert mean output (closed form, 126 MB for all 24,576 experts) plus a per-layer rank-32 correction from input and expert embedding (<10M params), fitted on traced (input, dropped set, true output). Prior art: MoNE [12], LightMoE-Hao [16].
- **Data:** 5-20M tokens of the 4-bit model's own generations on mixed math/code/knowledge/long prompts; no eval overlap.
- **Objective:** KL(full-routing teacher ‖ student under the simulated cache + wait-top-2) + locality (ReMoE's −log reuse mass, or [1]'s soft top-B coverage, B = slots) + KL trust to the original router.
- **Hours (estimate):** experts (~68 GB) exceed RAM, so end-to-end backprop re-streams them each step (~9 s, hidden under compute at ≥16k-token steps). At ~24 GFLOP/token (~4B active, with recompute) and 15-25 effective TFLOPS: 3-5 h per 10M tokens. Layer-local fitting: under 1 h.
- **Closest recipe:** ReMoE (router-only, frozen experts, CE + KL trust + locality, 100k OpenHermes samples, 2k steps). Its small gains at 6-9 % suggest the miss policy must be in the loop.

## 4. Ideas

1. **Budgeted waiting** by gate mass × layer sensitivity, unused budget carried across tokens (leaky bucket, as in video rate control). With equal-size experts, greedy by weight is optimal for 0/1 imprecise computation [19].
   Prior art: "Which misses…" (perplexity only), AdapMoE, LExI [17]. Left: layer-weighted budgets at ≤10 %, scored by answer agreement.
2. **Drop with correction:** add gate × mean output for low-variance experts (MoNE's criterion; cf. loop perforation), else drop and renormalize.
   Prior art: MoNE (static, zero-shot only); DyMoE, ACE. Left: dynamic misses, reasoning tasks.
3. **Near-tie reranking:** a resident expert replaces a missing one only within δ logits, δ smallest in layers 10-39.
   Prior art: [3]; ReMoE reports rerouting lifting hits 23.7→63.9 % at 6.3 % (quality unseen). Left: 512 experts, our layer map.
4. **Train under the miss policy** (QAT analogue; codec encoders track the decoder's references).
   Prior art: [1], ReMoE, ZEDA, Sticky Routing [5] (−59 % switches, tiny models), [4] (negative); Read-ME [18] knows routing ahead but rebuilds the MoE. Left: router-only with the policy in the loop.
5. **Prompt working set and phases (Denning):** pin the prefill's most-used experts, stream ranks 1-2, re-pin at reasoning-stage changes.
   Prior art: PreMoE; EASY-EP [8] (50 % kept, "comparable" SR); SwapMoE [6] (routes within a resident set); LightMoE-Luo [20]; SAEM [21]; [22] (shared experts lower locality). Left: pinned + streamed hybrid.
6. **Admission control (CDN):** admit ranks 1-2 at once, ranks 3-10 on second touch (TinyLFU [23] / S3-FIFO one-hit-wonder filtering); pin a small long-term set (codec long-term references). Exact, no quality cost.
   Prior art: none found for experts; SpecMD [24], fMoE change eviction only.
7. **Coarse copy + background fetch** (texture streaming samples a coarser mip but still loads the tile [25]): fetch skipped misses for later tokens.
   Prior art: HOBBIT, SliceMoE, D2MoE. Left: at 6-12 GB only mean vectors or rank-8 stand-ins for layers 10-39 (~1.3 GB) fit.

## 5. Cheapest experiments

- **E1, trace replay (minutes, no GPU):** second-touch admission, a protected long-term set and Least-Stale at 2.5/5/10 %; log lost gate mass per token and layer. Build if exact loads at 10 % fall from 152 to ≤116 (60 % of the LRU→Belady gap), or substituted + dropped ranks fall ≥25 % at ≤20 loads. Widen to ≥20 prompts first.
- **E2, 35B proxy (one GSM8K-150 run per variant):** wait-top-2 at 10 % and 5 %, missing low ranks handled as substitute (today), drop + renormalize, drop + mean, or layer split (0-9 substitute, 10-39 drop + mean). Build if one keeps ≥95 % identical answers and GSM8K within 3 points (n=150 noise) at 10 %.
- **E3, 125B (offline, then one run):** near-tie reranking + budgeted waiting on traces, then GSM8K-20 at 10 %. Build if loads stay ≤20/token while lost gate mass halves, with ≥19/20 and ≥90 % identical answers.

Fine-tune only if E2-E3 leave a gap.

## 6. Sources

1. https://arxiv.org/abs/2609.04895
2. https://arxiv.org/abs/2605.27081
3. https://arxiv.org/abs/2412.00099
4. https://arxiv.org/abs/2608.18261
5. https://arxiv.org/abs/2607.08780
6. https://arxiv.org/abs/2308.15030
7. https://arxiv.org/abs/2505.17639
8. https://arxiv.org/abs/2504.06792 (NeurIPS'25)
9. https://www.researchsquare.com/article/rs-11268552
10. https://arxiv.org/abs/2605.18643
11. https://arxiv.org/abs/2609.05228
12. https://arxiv.org/abs/2507.00390 (ICLR'26)
13. https://arxiv.org/abs/2511.10054
14. https://arxiv.org/abs/2603.19289
15. https://arxiv.org/abs/2510.12357
16. https://arxiv.org/abs/2603.12645
17. https://arxiv.org/abs/2509.02753
18. https://arxiv.org/abs/2410.19123 (not re-checked)
19. https://researchwith.njit.edu/en/publications/scheduling-imprecise-computation-tasks-with-01-constraint/
20. https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1062/
21. https://arxiv.org/abs/2608.21614
22. https://en.papernotes.org/ICLR2026/llm_efficiency/not_all_models_suit_expert_offloading_on_local_routing_consistency_of_mixture-of/
23. https://arxiv.org/abs/1512.00727 (not re-checked)
24. https://arxiv.org/abs/2602.03921 (ICML'26)
25. https://docs.unity3d.com/Manual/svt-how-it-works.html
