# Deeper and retrained draft models for speculative decoding (2023-2026): drafter depth, deep-position acceptance, and fixes for depth-wise decay

**Key numbers first** (details and caveats in the sections below):
- **Depth:**
  - DFlash with 3 / 5 / 8 layers: Math500 τ 5.64 / 5.99 / 6.33. Speedup peaks at 5 layers. — [DFlash](https://arxiv.org/abs/2602.06036)
  - DFlash 2: 15 layers vs 5 gives +5.9 points of conditional top-1 at draft position 6 (78.73 vs 72.86). — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
  - AngelSpec: 5 → 7 layers gives +0 (4.25 → 4.25). — [AngelSpec](https://arxiv.org/abs/2607.25852)
  - MoA: a deeper EAGLE at a fixed trainable budget fell from τ 1.45 to 1.01. — [MoA](https://arxiv.org/abs/2410.03804)
  - Scylla fit: acceptance = 0.74·log10(layers) + 4.61. — [arXiv 2505.07858](https://arxiv.org/abs/2505.07858)
  - FlexDraft uses 10 of the target's own layers by default. — [FlexDraft](https://arxiv.org/abs/2605.20022)
- **Deepest per-position numbers found:** about 86% conditional acceptance held flat at positions 8–14 (the 15th drafted token). Qwen3.5-4B, MATH-500, T=1, DFlash 2. At position 14, DFlash reaches 77.5% and native MTP 77.9%. Nothing was found for position 16 or beyond. — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **MTP recursion collapses without training for it:**
  - FastMTP: vanilla about 70 / 10 / 0% at steps 1–3, vs 80 / 56 / 36% after self-distilled fine-tuning. — [FastMTP](https://arxiv.org/abs/2509.18362)
  - MTP-D: an untrained loop falls to 0.6% cumulative acceptance by head 3. — [MTP-D](https://arxiv.org/abs/2603.23911)
  - AngelSpec: training-time test raises marginal p3 from 0.266 to 0.524. — [AngelSpec](https://arxiv.org/abs/2607.25852)
- **Drift fixes, measured gains:**
  - EAGLE-2 → EAGLE-3: τ 4.05 → 6.13. — [EAGLE-3](https://arxiv.org/abs/2503.01840)
  - HASS over EAGLE-2: +8–20% τ. — [HASS](https://arxiv.org/abs/2408.15766)
  - Over DFlash: DSpark +26–30% at γ=15 ([DSpark](https://arxiv.org/abs/2607.05147)), DSpine +27.8% ([arXiv 2609.36173](https://arxiv.org/abs/2609.36173)), DFlash 2 +21% ([DFlash 2 blog](https://inco.ai/blog/dflash2/))
- **Highest code τ:** HumanEval 9.1 on Qwen3.5-27B with DFlash; 8.09 HumanEval / 7.23 MBPP on Qwen3-Coder-30B-A3B. All include the bonus token. — [DFlash](https://arxiv.org/abs/2602.06036)
- **What it would take to go 30 tokens deep** (my calculation): accepting all 30 drafts half the time needs a conditional rate of at least 0.977 at every position (τ ≈ 22). The flat 0.86 above gives τ ≈ 7.1 at 30 drafts.

**Not reached in this pass** (cut for time; no deep dives started):
- EAGLE-1 and EAGLE-2 layer-count ablations
- Per-position values plotted in figures: EAGLE-3 Fig. 7, PosS Fig. 4, DSpark Fig. 2–3, Nemotron Fig. 4, FlexDraft's depth figure
- Medusa per-head accuracies
- The DeepSeek-V3 MTP depth sentence
- Qwen3-Next's MTP layer count
- Retrieval and suffix drafting on code
- Trained drafters with a block of 32 or more
- Remaining appendix text of AngelSpec, MiMo-V2-Flash and Nemotron

Conventions used in these notes: τ (also called MAL or acceptance length) = mean tokens committed per verification cycle. "Incl. bonus" means the count includes the target's own free/correction token. A conditional rate at position i is P(draft i accepted | drafts 1..i-1 accepted). A marginal or cumulative rate is P(drafts 1..i all accepted). Cross-paper τ values are not comparable unless the target, block or tree size, temperature, thinking mode and baseline training budget match. Several papers re-train baselines in-house and get much lower numbers than the original authors. For example, DFlash on Qwen3-8B at T=0 averages τ 6.49 in its own paper but 3.77 as a baseline in DSpine.

## 1. Does adding layers/depth to a drafter improve acceptance (especially at deep positions), and under what training regime? Does depth trained in from the start differ from depth bolted on later?

### Takeaway
Depth raises acceptance, but by small and quickly shrinking amounts. It pays off mainly in block-parallel drafters, where the cost of depth is paid once per block:
- DFlash with 3, 5 and 8 layers gives Math500 τ 5.64, 5.99 and 6.33, but speedup peaks at 5 layers.
- In DFlash 2 tests, 15 layers instead of 5 raises conditional top-1 at draft position 6 by 5.9 points.
- AngelSpec saw no gain going from 5 to 7 layers.

Autoregressive drafters pay for depth on every drafted token, so EAGLE-3 still uses a single layer. At a fixed trainable budget, a deeper EAGLE did worse (τ 1.45 fell to 1.01 in the MoA ablation). Block size, multi-step or on-policy training, injecting target features into every layer, and dependencies within the block all move acceptance more than layer count does. Native MTP modules are trained shallow during pretraining (1 layer in DeepSeek-V3, MiMo and Qwen3.5; 2 shared-weight layers in Nemotron 3 Super). Their extra steps are nearly always added after pretraining, and those steps must be trained for recursion. Reusing a one-step module without that training collapses after the first extra token.

### Cited Findings

**Block-parallel / diffusion drafters (depth paid once per block)**
- **DFlash defaults:** 5 draft layers (8 for Qwen3-Coder), block size 16 (10 for LLaMA-3.1). Five target hidden layers are fused and injected as K/V into *every* draft layer, and the embedding and LM head are shared and frozen. τ explicitly includes the bonus token. — [DFlash, arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
- **DFlash depth ablation** (block 16, 5 target features; speedup / τ):

  | Layers | Math500 | HumanEval | MT-Bench |
  |---|---|---|---|
  | 3 | 4.69× / 5.64 | 3.90× / 4.61 | 2.38× / 3.18 |
  | 5 | 4.71× / 5.99 | 3.96× / 4.94 | 2.35× / 3.37 |
  | 8 | 4.64× / 6.33 | 3.96× / 5.29 | 2.23× / 3.50 |

  τ rises with depth on every task. The authors pick 5 layers for the best average speedup. — [DFlash](https://arxiv.org/abs/2602.06036)
- **DFlash, same depth drafted in parallel vs autoregressively** (Qwen3-4B, 5 layers, block 8; τ / speedup, GSM8K / HumanEval / MT-Bench):
  - EAGLE-3-style 5-layer AR drafter with input fusion: 4.2 / 2.1×, 4.3 / 2.2×, 3.1 / 1.4×
  - DFlash-AR with KV injection: 4.8 / 2.4×, 4.6 / 2.3×, 3.4 / 1.5×
  - Block DFlash with input fusion: 3.5 / 2.9×, 3.5 / 2.9×, 2.6 / 2.0×
  - Block DFlash with KV injection: 4.2 / 3.3×, 4.0 / 3.2×, 3.0 / 2.2×

  A 5-layer AR drafter matches or beats the parallel drafter on τ but is much slower end to end. — [DFlash](https://arxiv.org/abs/2602.06036); the same table appears in the [LMSYS DFlash/Spec V2 blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- **DFlash train vs test block size** (8-layer drafter, Math500 τ): train 16 / test 16 gives 6.33, 16/8 gives 5.09, 8/16 gives 5.02, and 8/8 gives 5.21. Training on the larger block generalizes down, but not up. — [DFlash](https://arxiv.org/abs/2602.06036)
- **DFlash 2 depth test** (Inco AI blog, 18 Aug 2026; vendor-reported, not peer reviewed). Qwen3-4B, GSM8K, T=0, Recall@1 conditioned on earlier positions being right:
  - 5-layer DFlash: 85.39 at position 0, falling to 72.86 at position 6
  - 5 layers plus a two-tap local convolution: 77.61 at position 6
  - **15 layers: 78.73 at position 6**

  — [Inco AI, DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **FlexDraft (19 May 2026) builds a deep drafter from the target itself.** It reuses the target's final N layers, tuning only their attention projectors while the FFNs stay frozen. Mask tokens enter at layer L−N+1 and attend bidirectionally within the block and to the clean prefix. The paper says "We therefore use 10 layers by default" and that 13 layers "yields slightly higher speedup in the parallel setting" at the cost of more parameters and draft compute. Trainable parameters are about 6% of the target. Depth-curve values appear only in a figure. — [FlexDraft, arXiv 2605.20022](https://arxiv.org/abs/2605.20022)
- **FlexDraft: full-parameter vs attention-only tuning** (τ / speedup): GSM8K 3.78 / 3.04× vs 5.86 / 4.23×; HumanEval 2.57 / 2.10× vs 3.99 / 2.94×; MT-Bench 2.14 / 1.60× vs 3.06 / 1.90×. Keeping the target's FFN knowledge frozen wins. — [FlexDraft](https://arxiv.org/abs/2605.20022)
- **FlexDraft results** (Qwen3-8B, batch 1; the main tables do not state temperature):
  - 300K-sample training: GSM8K τ 6.12 / 4.57×, MATH 5.98, HumanEval 4.15 / 3.25×, MBPP 3.93, MT-Bench 3.22
  - Scaled training: GSM8K 7.98 / 5.88×, MATH 8.03 / 5.79×, HumanEval 5.55 / 4.26×, MBPP 6.16 / 4.46×, MT-Bench 4.16 / 2.58×
  - GSM8K τ by temperature: 5.61 (T=0.3), 5.11 (T=0.6), 4.61 (T=1.0)

  The paper defines L_acc = r_acc + 1, which includes the bonus token, but the tables only say "average acceptance length". — [FlexDraft](https://arxiv.org/abs/2605.20022)
- **AngelSpec DFly, depth vs block size** (T=1, no thinking, average τ): 3 layers / block 5 gives 3.29; 5 / 5 gives 3.39; 5 / 8 gives 4.25; 7 / 8 gives 4.25. Going from 5 to 7 layers added nothing, while going from block 5 to block 8 added 0.86. — [AngelSpec, arXiv 2607.25852](https://arxiv.org/abs/2607.25852)
- **DSpark depth ablation** (1–5 layers, block 7): gains rise monotonically with depth, with the largest jump from 1 to 2 layers. A 2-layer DSpark beats the 5-layer DFlash in every domain. Values are only in a figure. — [DSpark, arXiv 2607.05147](https://arxiv.org/abs/2607.05147)
- **DSpark in DeepSeek-V4 production:** max draft length γ=5, a three-layer MoE backbone, and sliding-window attention of 128. Per-user speed is 60–85% faster (Flash) and 57–78% faster (Pro) at matched throughput. — [DSpark](https://arxiv.org/abs/2607.05147)
- **DSpine (adjacent causal injection):** 5 layers, block 16. No depth ablation is reported. — [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)
- **Speculative Pipeline Decoding drafter depth L_s** (Qwen3.5-4B, step acceptance ratio α = N/K):
  - Pipeline width n=8, L_s = 1/2/3/4: α = 0.2977 / 0.3589 / 0.3721 / 0.3799 at T=0, and 0.2708 / 0.3276 / 0.3396 / 0.3435 at T=1
  - n=4: L_s=1 gives 0.5352 and L_s=4 gives 0.6119 (T=0)
  - The paper's operating rule is L_s ≈ L/n − 1.

  — [SPD, arXiv 2605.30852](https://arxiv.org/abs/2605.30852)

**Autoregressive / EAGLE-family drafters (depth paid per drafted token)**
- **EAGLE-3** uses a single decoder layer ("inputted into a single layer decoder") and reports no layer-count ablation. Its ablation on LLaMA-3.1-8B MT-bench: EAGLE-2 3.16× / τ 4.05; removing the feature constraint 3.82× / 5.37; adding fused features (EAGLE-3) 4.40× / 6.13. Appendix A says EAGLE-3 raises draft tree depth from 6 to 8 at the same node count. — [EAGLE-3, arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- **MoA: "EAGLE (more layers)"** (HumanEval, chain drafting, max 4 tokens):

  | Variant | Params (total / trainable) | tok/s | τ |
  |---|---|---|---|
  | EAGLE | 1.3B / 250M | 30 | 1.45 |
  | EAGLE (more layers) | 1.3B / 250M | 27 | **1.01** |
  | EAGLE (more params) | 1.45B / 400M | 29 | 1.28 |
  | MoA (TLI=0) | — | 40 | 2.14 |

  The paper added this baseline "to ensure that the benefit of our architecture does not come from simply adding decoder layers or parameters". It does not explain why the deeper EAGLE did worse. — [MoA, arXiv 2410.03804](https://arxiv.org/abs/2410.03804)
- **MoA on SpecBench** (LLaMA3-8B-Instruct, mean of 6 tasks): MoA (TLI=1) 1.95× / τ 4.50 vs EAGLE-2 1.78× / τ 3.60. On-policy, T-step-bounded training gives about +33% τ over EAGLE. Layer self-attention, which addresses partial observability, adds about 6% tok/s. — [MoA](https://arxiv.org/abs/2410.03804)
- **Scaling Laws for Speculative Decoding** fits acceptance (accepted tokens per cycle) = 0.74·log10(D) + 4.61, where D is the number of draft decoder layers. The tested range is unclear: one section says 1, 2 and 5 layers, another says gains hold "up to 10 decoders".
  - The drafter was pretrained on 10B tokens and then fine-tuned on ShareGPT.
  - Pretraining-token law: 5.10 with no pretraining, 5.43 at 100B tokens, saturating past 50B.
  - Scylla vs EAGLE-2 (T=0 means): Vicuna-7B 4.68 → 6.32; LLaMA3-8B 4.18 → 5.36; LLaMA3-70B 4.68 → 5.53.
  - Scylla vs EAGLE-3 (Vicuna-13B): EAGLE-3 6.62, Scylla 6.31, Scylla+8SFT 6.81. The abstract's "+0.3 over EAGLE-3" does not match this table.

  — [arXiv 2505.07858](https://arxiv.org/abs/2505.07858)
- **Gumiho:** a 2-layer serial Transformer drafts the first 2 tokens, and 5 parallel MLP heads draft the next 5. Going from 2 to 3 serial layers raises τ but lowers speedup (shown only in a figure). Vicuna-7B MT-Bench: 3.15× / τ 5.29 vs EAGLE-2 2.88× / 5.00. The paper reports gains of 4.5–15.8% over EAGLE-2. — [Gumiho, arXiv 2503.10135](https://arxiv.org/abs/2503.10135)
- **FastEagle:** N cascaded light layers, where layer i predicts position t+i, all in one forward pass.
  - LLaMA-3.1-8B, T=0 mean: 4.74× / τ 5.79 vs EAGLE-3 3.94× / 5.85.
  - Vicuna-13B HumanEval: 6.64× / 7.10 vs EAGLE-3 5.97× / 7.14.
  - Ablation: replacing the cascade with parallel heads drops MT-Bench τ from 6.13 to 4.81.

  — [FastEagle, arXiv 2509.20416](https://arxiv.org/abs/2509.20416)

**Head-style drafters (Medusa, Hydra, Clover, Falcon)**
- **Medusa:** each head is a single feed-forward layer with a residual connection, and 5 heads are used.
  - Medusa-1 (frozen backbone): 2.18× on Vicuna-7B, 2.33× on 13B.
  - Medusa-2 (joint training with the backbone): 2.83× on both.
  - Acceleration rate 3.47 (7B) and 3.51 (13B), counting the base token. No ablation on head depth.

  — [Medusa, arXiv 2401.10774](https://arxiv.org/abs/2401.10774)
- **Hydra / Hydra++:** heads are sequentially dependent.
  - Hydra++ deepens each head MLP to 4 layers; "scaling to 5 layers and beyond provides no additional benefit".
  - It adds teacher distillation and one extra self-attention "prefix" decoder layer (PrefixMLP), worth 1.12× average acceptance length and 1.08× throughput.
  - Hydra++ speedup over autoregressive decoding: 2.70× (Vicuna 7B), 2.50× (13B), 2.53× (33B), or 1.27–1.31× over Medusa.

  — [Hydra, arXiv 2402.05109](https://arxiv.org/abs/2402.05109)
- **Clover-2:** the augmenting block grows to 2 decoder layers, which the paper says gives about a 9% overall benefit. Tokens per step vs EAGLE improve by 2.9% on average (Vicuna-7B) and 3.6% (LLaMA3-8B). — [Clover-2, arXiv 2408.00264](https://arxiv.org/abs/2408.00264)
- **Falcon:** a semi-autoregressive drafter "equivalent to merely two Transformer layers". Emitting more tokens per pass lowered acceptance (Vicuna-7B MT-Bench):
  - k=2: 3.10×, τ 3.34, α 77.64%
  - k=3: 2.96×, τ 2.52, α 72.21%
  - k=4: 2.86×, τ 2.35, α 70.78%

  — [Falcon, arXiv 2412.12639](https://arxiv.org/abs/2412.12639)

**MTP modules: trained in pretraining vs bolted on later**
- **DeepSeek-V3:** acceptance of the second-token MTP prediction is 85–90% across generation topics, giving 1.8× TPS. — [DeepSeek-V3 report](https://arxiv.org/pdf/2412.19437)
- **MiMo-7B:** pretraining uses only one MTP layer, because "multiple MTP layers yield no further improvement" in preliminary studies. After pretraining, the layer is replicated, the main model and first MTP layer are frozen, and two new MTP layers are fine-tuned. Acceptance on AIME24 is about 90% for the first MTP layer and above 75% for the third. — [MiMo-7B, arXiv 2505.07608](https://arxiv.org/abs/2505.07608)
- **MiMo-V2-Flash:**
  - One MTP head is trained in pretraining (loss weight 0.3, then 0.1) and replicated K times in post-training. Three MTP layers are released.
  - Each block has 0.33B parameters, a dense FFN and sliding-window attention with a 128-token window.
  - Acceptance length reaches about 3.6 on WebDev. It follows y = 4(1 − 0.58·x^0.58) against next-token cross-entropy x (R² 0.995).
  - Speedups are 2.20–2.42× at acceptance 3.6 (16K input / 1K output, batch 32–128). The abstract's 2.6× is not matched by the table; the closest is 2.70× at batch 96 with acceptance 3.8.

  — [MiMo-V2-Flash, arXiv 2601.02780](https://arxiv.org/abs/2601.02780)
- **Nemotron 3 Super:**
  - Two MTP layers with shared parameters, trained in pretraining (loss scale 0.3) and continued through SFT.
  - A final "MTP Healing" stage freezes the rest of the model and trains only the MTP heads on generated responses.
  - SPEED-Bench, draft length 7, average acceptance length: Nemotron 3 Super 3.45, Qwen3-Next 3.33, DeepSeek-R1 2.70. On coding: 3.78, 4.32 and 2.99.
  - Acceptance falls steadily with draft index.

  — [Nemotron 3 Super, arXiv 2604.12374](https://arxiv.org/abs/2604.12374)
- **Qwen MTP layer counts:** the Qwen3.5-4B config has `mtp_num_hidden_layers: 1` — [Qwen3.5-4B config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json). The Qwen3-Next-80B-A3B-Instruct config.json has no MTP keys at all (48 layers) — [Qwen3-Next config.json](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)
- **Native Qwen3.5-4B MTP, run recursively, vs a post-hoc 0.6B DFlash drafter** (SGLang, B200, greedy, thinking on, concurrency 1):

  | Workload | MTP s=3 | DFlash b=4 | MTP s=7 | DFlash b=8 | MTP s=15 | DFlash b=16 |
  |---|---|---|---|---|---|---|
  | GSM8K | 3.42 | 3.43 | 5.13 | 5.30 | 6.18 | 6.75 |
  | MATH500 | 3.50 | 3.53 | 5.35 | 5.65 | 6.47 | 7.48 |
  | HumanEval | 3.45 | 3.55 | 5.19 | 5.68 | 6.15 | 7.72 |
  | MBPP | 3.27 | 3.42 | 4.61 | 5.24 | 5.33 | 6.53 |
  | MT-Bench | 3.27 | 3.23 | 4.63 | 4.70 | 5.61 | 5.93 |

  Throughput: MTP s=15 reaches only 1.48–1.79×, while DFlash b=16 reaches 3.40–4.60×. — [z-lab/Qwen3.5-4B-DFlash model card](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)
- **Qwen 3.5 397B-A17B** (HumanEval, concurrency 1, greedy, thinking on): DFlash at block 16 gives more than 4.3× baseline and 1.5× the throughput of native MTP at 7 steps. — [LMSYS blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- **FastMTP** (MiMo-7B-RL, single shared MTP head reused recursively, greedy):
  - Vanilla MTP acceptance at steps 1/2/3: about 70% / 10% / 0%.
  - After fine-tuning on self-distilled data: 80% / 56% / 36%, with math above 50% at step 3.
  - Mean τ rises from 1.83 to 2.73 (2.66 with vocabulary compression), and speedup from 1.21× to 2.03×.
  - The head was trained for up to 7 drafts, but speed peaks at K=3.

  — [FastMTP, arXiv 2509.18362](https://arxiv.org/abs/2509.18362)
- **MTP-D (self-distillation during pretraining; 1 dense layer per head; K=4):** looping the DeepSeek-style MTP without training drops cumulative acceptance to 0.6% by head 3, vs 26.70% for MTP-D. A looped extension from 4 to 16 heads keeps the 16th head at 5–10% cumulative acceptance. — [MTP-D, arXiv 2603.23911](https://arxiv.org/abs/2603.23911)
- **AngelSpec Hy3 MTP** (one Transformer layer reused recurrently, D=3): training-time test plus rollout raises mean MAL from 2.58 to 2.99 at T=0 and from 2.54 to 2.90 at T=0.9. — [AngelSpec](https://arxiv.org/abs/2607.25852)
- **Apple, "Your LLM Knows the Future":** the Tulu3-8B SFT model stays frozen. Training adds gated LoRA (rank 128) and a 2-layer MLP sampler head, with k=8 mask tokens. Tokens per step: HumanEval 5.35, GSM8K 5.22, AlpacaEval 2.52, average 3.17. The guaranteed token is counted. — [arXiv 2507.11851](https://arxiv.org/abs/2507.11851)
- **LayerSkip** (layer dropout plus early-exit loss; trained four ways: from scratch, continual pretraining, domain fine-tuning and task fine-tuning): self-speculative speedups of up to 2.16× on CNN/DM, 1.82× on coding and 2.0× on TOPv2. — [LayerSkip, arXiv 2404.16710](https://arxiv.org/abs/2404.16710)

### Inferences
- **Depth pays only when drafting is parallel.** In DFlash's 3→5→8 test, τ rises about 0.3–0.35 per step but speed peaks at 5 layers. In autoregressive drafting, each extra layer is paid on every drafted token. That explains why EAGLE-3 stays at 1 layer and why a 5-layer AR drafter reaches 2.1× where the parallel version reaches 3.3× at the same τ.
- **Depth matters more at deeper positions.** Both DFlash 2's position-6 result (+5.9 points for 15 vs 5 layers) and DSpark's widening lead over DFlash as γ grows point this way. A cheap local convolution recovered about 80% of the 15-layer gain (+4.75 of +5.87 points), so better mixing within the block substitutes well for raw depth.
- **Depth follows a log-linear law with sharp diminishing returns.** Scylla's slope of 0.74 per decade of layers implies roughly +0.22 τ from 1 to 2 layers and +0.74 from 1 to 10. That matches DSpark (biggest gain from 1 to 2 layers), AngelSpec (5 to 7 gave nothing) and Hydra (5+ MLP layers gave nothing).
- **The MoA result is not an argument against depth in general.** It shows that a deeper but thinner AR drafter is worse at a fixed budget.
- **Trained in vs bolted on: the observed pattern.** Pretraining-time MTP is deliberately kept to 1–2 layers. MiMo found more MTP layers did not help pretraining, and Nemotron shares weights across its two. Extra depth or steps are then added after pretraining by replicating and fine-tuning (MiMo), self-distilled fine-tuning (FastMTP), training-time test plus rollout (AngelSpec), or "MTP healing" (Nemotron). Without that step, recursion collapses (FastMTP about 70→10→0%; MTP-D 0.6% at head 3).
- **The best bolted-on drafters beat native MTP once drafts are long.** DFlash at block 16 beats Qwen3.5's native MTP at 15 steps on τ (MATH500 7.48 vs 6.47; HumanEval 7.72 vs 6.15) and by about 2.5× on throughput. At 3–4 drafts the two are tied.
- **"Reuse the target's depth" is a distinct, promising design point.** FlexDraft tunes only the attention of the target's own last 10 layers, which beat full-parameter tuning by about 50% on τ. LayerSkip and Kangaroo use the target's early layers.

### Gaps
- No layer-count ablation was found for EAGLE-1 or EAGLE-2. EAGLE-3 reports none.
- MoA's "EAGLE (more layers)" result lacks per-position data, the τ convention is unclear, and the absolute τ values (1.01–2.14 on HumanEval) are unusually low, which may mean the bonus token is excluded.
- No controlled study compares the same depth trained in pretraining against the same depth added post hoc. The evidence above is circumstantial and comes from different models.
- I did not re-read the DeepSeek-V3 text giving the MTP depth used. The 85–90% figure is for the second-token prediction.
- Qwen3-Next's MTP layer count is not in its config.json. The Qwen3.5-4B config says 1 layer.
- FlexDraft's and DSpark's depth curves, and Gumiho's serial-depth ablation, are shown only in figures.
- Medusa per-head accuracies do not appear in the text.

## 2. What are the best published per-position (conditional) acceptance numbers at deep positions (>= 8, ideally >= 16), and on which tasks?

### Takeaway
The deepest per-position numbers I found come from the DFlash 2 vendor blog: conditional acceptance holds flat at about 86% across draft positions 8–14 (0-indexed; position 14 is the 15th drafted token). The setting is Qwen3.5-4B on MATH-500 at T=1 with thinking on. At position 14, DFlash reaches 77.5%, DSpark 79.9% and native MTP 77.9%. No source reports positions of 16 or more. arXiv papers mostly report only positions up to 7, often only in figures, and several "per-position" numbers in MTP papers are actually cumulative survival rates. Code and math consistently score higher than chat.

### Cited Findings

**Deep positions (>= 8)**
- **DFlash 2** (Qwen3.5-4B, MATH-500, thinking on, T=1.0, top-p 0.95, top-k 20, presence penalty 1.5, lossless rejection sampling). Conditional acceptance by position:

  | Drafter | Pos 8 | Pos 9 | Pos 10 | Pos 11 | Pos 12 | Pos 13 | Pos 14 |
  |---|---|---|---|---|---|---|---|
  | DFlash 2 | 85.95% | 85.99% | 86.41% | 86.46% | 86.43% | 86.02% | 86.48% |
  | DFlash | 81.29% | — | — | — | — | — | 77.48% |
  | DSpark | — | — | — | — | — | — | 79.86% |
  | MTP | — | — | — | — | — | — | 77.85% |

  — [Inco AI, DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **DFlash 2 mean τ on the same model:** MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash 2 5.97. By benchmark (MTP / DFlash / DSpark / DFlash 2):
  - HumanEval: 4.84 / 5.43 / 5.80 / 6.28
  - MBPP: 4.16 / 4.49 / 4.96 / 5.41
  - MATH-500: 5.04 / 5.42 / 6.20 / 6.76
  - MT-Bench: 3.90 / 4.26 / 4.77 / 5.20

  — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **DFlash:** a block-8 drafter accepts the *entire* block 35.7% of the time on Math500. — [DFlash](https://arxiv.org/abs/2602.06036)
- **MTP-D (2B dense base model, greedy, max 100 new tokens, K=4).** Head-4 cumulative acceptance (CAR) / per-head acceptance (AR):
  - GSM8K: 72.76 / 92.89 (plain MTP 68.52 / 92.09)
  - MATH: 46.42 / 81.81 (40.27 / 79.23)
  - TriviaQA: 48.52 / 82.85 (43.75 / 81.96)

  A looped 16-head version keeps the 16th head at 5–10% cumulative acceptance. — [MTP-D](https://arxiv.org/abs/2603.23911)

**Positions 1–7 (conditional unless marked)**
- **DSpark** (offline, T=1.0, block 7, chain drafting; target model for the figure not captured):
  - Position 1, math: DFlash 0.88, EAGLE-3 0.81, DSpark 0.93
  - Position 1, chat: DFlash 0.72, EAGLE-3 0.53
  - Across the block: EAGLE-3 on chat *rises* from 0.53 to 0.74; DFlash on code *falls* from 0.87 to 0.78; DFlash on chat falls from 0.72 to 0.63
  - DSpark claims a "high and stable" conditional rate across the block, without numbers.

  — [DSpark](https://arxiv.org/abs/2607.05147)
- **FastEagle** (Fig. 3, MT-Bench, T=0): FastEagle falls from 0.81 to 0.74; EAGLE-3 stays about 0.78–0.81 and is "the most stable overall"; EAGLE-2 "degrades substantially" from 0.69 to 0.51. — [FastEagle](https://arxiv.org/abs/2509.20416)
- **PosS** (MT-Bench, Llama-3-8B-Instruct, depth 8): EAGLE-2's per-position accuracy is below 65% from position 5 onward. HASS holds up only for the first four positions. PosS stays "substantially higher until the last position" (figure only). — [PosS, arXiv 2506.03566](https://arxiv.org/abs/2506.03566)
- **AngelSpec Hy3 MTP** (D=3). MAL = 1 + Σp_i, so the p_i values are **marginal** (cumulative) rates:
  - T=0: p1/p2/p3 go from 0.799 / 0.518 / 0.266 to 0.814 / 0.653 / 0.524 with training-time test plus rollout.
  - T=0.9: from 0.789 / 0.499 / 0.253 to 0.792 / 0.619 / 0.486.
  - p3 on HumanEval rises from 0.387 to 0.757; on GSM8K from 0.290 to 0.706.
  - HumanEval at T=0: 68.1% / MAL 3.04 → 85.1% / 3.55.

  — [AngelSpec](https://arxiv.org/abs/2607.25852)
- **FastMTP:** after fine-tuning, steps 1/2/3 accept 80% / 56% / 36% (math above 50% at step 3). Because τ 2.73 ≈ 1 + 0.80 + 0.56 + 0.36, these are marginal rates. — [FastMTP](https://arxiv.org/abs/2509.18362)
- **MiMo-7B:** about 90% acceptance at the first MTP layer and above 75% at the third (AIME24). — [MiMo-7B](https://arxiv.org/abs/2505.07608)
- **DeepSeek-V3:** 85–90% acceptance for the second-token prediction. — [DeepSeek-V3](https://arxiv.org/pdf/2412.19437)
- **Qwen3-Next-80B-A3B-Instruct-FP8** with one speculative token: mean acceptance length 1.91 and per-position rate 0.906. This comes from a search-engine summary of a PR log that I did not open. — [ROCm/ATOM PR #772](https://github.com/ROCm/ATOM/pull/772)
- **One MTP layer in practice** (search-engine summaries; pages not opened):
  - TensorRT-LLM on B200 reports an average accepted length of 1.92 with one MTP layer on DeepSeek-R1, a 1.78× per-user TPS speedup. — [TensorRT-LLM blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html)
  - SGLang reports an average acceptance length of 2.18 with a 3-token window on DeepSeek-V3, and up to 60% higher output throughput. — [LMSYS MTP blog](https://www.lmsys.org/blog/2025-07-17-mtp)
- **Falcon average acceptance rate α** (Vicuna-7B; MT-Bench / HumanEval / GSM8K): Medusa 60.53 / 63.26 / 61.09%; EAGLE 74.59 / 76.18 / 75.47%; Falcon 77.64 / 79.92 / 78.03%. — [Falcon](https://arxiv.org/abs/2412.12639)
- **GLIDE:** acceptance rate "generally falls between 55% and 70%". — [GLIDE, arXiv 2402.02082](https://arxiv.org/abs/2402.02082)

**Code vs chat**
- **Nemotron SPEED-Bench** (draft length 7, average acceptance length):
  - Coding: Qwen3-Next 4.32, Nemotron 3 Super 3.78, DeepSeek-R1 2.99
  - Roleplay: 2.17 / 2.82 / 2.19
  - Writing: 2.69 / 2.99 / 2.41

  — [Nemotron 3 Super](https://arxiv.org/abs/2604.12374)
- **DFlash** (Qwen3-8B, T=0, block 16): τ is 7.27 on LiveCodeBench, 6.50 on HumanEval and 5.95 on MBPP, against 4.24 on MT-Bench. — [DFlash](https://arxiv.org/abs/2602.06036)
- **Qwen3.6-27B MTP:** acceptance is reportedly 79–89% on code generation but 62–70% on factual tasks. This is a secondary news summary of a Reddit post, and I did not verify it. — [aiweekly](https://aiweekly.co/alerts/qwen3-mtp-loses-throughput-edge-in-tool-heavy-agents)

### Inferences
- **The DFlash 2 numbers are self-consistent.** With a flat conditional rate p, τ ≈ 1 + Σ_{i=1..N} p^i. At p=0.86 over 15 drafts, that gives τ≈6.50, close to DFlash 2's measured 6.76 on MATH-500. The DFlash block-8 full-acceptance rate of 35.7% implies a geometric-mean conditional rate of about 0.86 over 7 drafts, assuming block 8 means one anchor plus 7 drafted tokens, which is DSpine's convention. MTP-D's 5–10% cumulative at head 16 implies a mean conditional rate of about 0.83–0.87.
- **About 86% conditional appears to be the current published ceiling at deep positions.** At that rate:
  - Going from 15 to 30 drafts only raises τ from 6.5 to 7.1.
  - The chance of accepting all 15 drafts is 0.10; all 30, 0.011.
- **What a 30-deep target would require:** a conditional rate of at least 0.977 for a 50% chance of accepting all 30 drafts. That gives τ(30) ≈ 22.3; p=0.97 gives τ ≈ 20.4 and p=0.98 gives 23.3. That is 11–12 points above the best published deep-position rate, on math, with a 4B target. These figures come from my own calculation.
- **Conditional curves are not comparable across methods because of survivorship.** EAGLE-3's chat curve rises from 0.53 to 0.74 because only easy prefixes survive to deep positions. Also check whether a paper reports marginal or conditional rates: AngelSpec's and FastMTP's per-step rates are marginal. AngelSpec's implied conditional rates after training-time test are p2/p1 ≈ 0.80 and p3/p2 ≈ 0.80, up from 0.65 and 0.51.
- **Code and math beat chat at every depth where data exists,** including DSpark position 1, DFlash per-task τ and the Nemotron categories. The deep-position DFlash 2 data is on math, not code.

### Gaps
- No primary source reports per-position acceptance at position 16 or beyond.
- Several per-position curves exist only as plots: EAGLE-3 n-α (Fig. 7), PosS (Fig. 4), DSpark (Fig. 2, positions 2–7), Nemotron 3 Super (Fig. 4) and DSpine's layer-wise readout.
- The DFlash 2 numbers come from a vendor blog, not peer review. Attribution is also unclear: the blog is Inco AI's ("Our team released DFlash in January"), the DFlash arXiv authors are UC San Diego / z-lab, and some DFlash 2 checkpoints carry z-lab names.
- The deep-position data covers math only. No deep-position code numbers were found.
- The DSpark appendix (the part I read) has no per-position tables.

## 3. What fixes for compounding feature drift / depth-wise acceptance decay exist, and what gains do they report at deep positions?

### Takeaway
The fixes fall into four families:
1. **Train the drafter on its own predictions:** EAGLE-3 training-time test, HASS alignment, AngelSpec training-time test with rollout, MoA on-policy training, Falcon CSGD, FastMTP self-distilled data.
2. **Keep draft layers grounded in target state** and away from drafted features: DFlash KV injection, GLIDE target-KV cross-attention, Speculative Pipeline Decoding, FlexDraft target layers, LayerSkip and Kangaroo early exits.
3. **Draft in parallel but add cheap dependencies within the block:** DSpark Markov head, DFlash 2 path selector plus convolution, DSpine adjacent causal injection, AngelSpec hidden-correction head.
4. **Specialize by position:** PosS, Gumiho, FastEagle.

Family 1 gives the biggest jumps for autoregressive and MTP recursion: EAGLE-2 to EAGLE-3 τ 4.05 → 6.13, and AngelSpec marginal p3 0.266 → 0.524. Family 3 gives +16–30% τ over DFlash, and the only quantified flattening of deep-position decay (DFlash 2 holds 86% at position 14 where DFlash drops to 77.5%). Most papers report average τ rather than deep-position gains.

### Cited Findings

**Train on own predictions (exposure-bias fixes)**
- **EAGLE-3 training-time test:** the drafter feeds its own outputs back during training, one native step plus two simulated steps. Together with dropping the feature constraint and fusing low, middle and high features, τ goes from 4.05 to 6.13 on LLaMA-3.1-8B MT-bench. — [EAGLE-3](https://arxiv.org/abs/2503.01840)
- **EAGLE-3 τ by target:**

  | Target | Temp | HumanEval | MT-bench | GSM8K |
  |---|---|---|---|---|
  | Vicuna 13B | 0 | 7.54 | 6.65 | 6.29 |
  | LLaMA-3.1 8B | 0 | 6.74 | 6.13 | 6.23 |
  | LLaMA-3.3 70B | 0 | 6.52 | 5.63 | 6.15 |
  | DeepSeek-R1-Distill-LLaMA 8B | 0 | 6.38 | 5.58 | 6.93 |
  | LLaMA-3.1 8B | 1 | 5.82 | 4.24 | 4.59 |

  The paper is ambiguous on the bonus token. It says "tokens generated per drafting-verification cycle" but also that this "corresponds to the number of tokens accepted from the draft". — [EAGLE-3](https://arxiv.org/abs/2503.01840)
- **HASS harmonized context alignment** (training is split into n steps so the drafter learns from its own earlier-step features). Ablation on LLaMA2-Chat 7B, mean τ over MT-bench, HumanEval and GSM8K:

  | Variant | T=0 | T=1 |
  |---|---|---|
  | EAGLE-2 + Top-K | 4.78 | 4.57 |
  | Align-2 | 5.11 | 4.86 |
  | Align-3 | 5.15 | 4.92 |
  | Align-4 | 5.16 | 4.91 |
  | Align-5 | 5.11 | 4.86 |

  Overall HASS gains over EAGLE-2 are 8–20% on τ. For example, LLaMA3-70B HumanEval goes from 5.02 to 5.78 and LLaMA2-13B HumanEval from 5.57 to 6.05 (T=0). HASS's first-step rate is slightly *lower* than EAGLE-2's on some targets. — [HASS, arXiv 2408.15766](https://arxiv.org/abs/2408.15766)
- **AngelSpec training-time test plus rollout on a pretrained MTP block:** marginal p3 rises from 0.266 to 0.524 (T=0) and mean acceptance from 52.8% to 66.4%. The paper frames this as fixing "exposure bias". An end-to-end total-variation loss with LK warm start gave the best average accepted length in a small-MoE study (3.96, vs 3.95 for LK alone and 3.03 for TV alone). — [AngelSpec](https://arxiv.org/abs/2607.25852)
- **FastMTP self-distilled fine-tuning** of a recursive MTP head: step 2 rises from about 10% to 56%, step 3 from about 0% to 36%, and speedup from 1.21× to 2.03×. The abstract's "82% over vanilla MTP" does not match the table, which gives 2.03/1.21 ≈ +68%. — [FastMTP](https://arxiv.org/abs/2509.18362)
- **MoA on-policy, T-step-bounded training** (K sampled from 5 to 15 during training): about +33% τ and +26% tok/s over EAGLE. — [MoA](https://arxiv.org/abs/2410.03804)
- **Falcon Coupled Sequential Glancing Distillation** (Vicuna-7B MT-Bench; speedup / τ / α):

  | Tree attention | CSGD | Speedup | τ | α |
  |---|---|---|---|---|
  | No | No | 2.06× | 1.64 | 62.86% |
  | No | Yes | 2.27× | 2.13 | 68.68% |
  | Yes | No | 2.65× | 2.78 | 74.37% |
  | Yes | Yes | 3.10× | 3.34 | 77.64% |

  The deltas claimed in Falcon's text do not match its own table. — [Falcon](https://arxiv.org/abs/2412.12639)
- **MTP-D** adds gradient-detached distillation from the main head into the MTP heads during pretraining. Head-4 cumulative acceptance gains about 7.5% and speedup about 22.9% at K=4. Looped extension from 4 to 16 heads adds another 35.1%. — [MTP-D](https://arxiv.org/abs/2603.23911)

**Ground every draft layer in target state / avoid consuming drafted features**
- **DFlash KV injection** (target features projected into K/V at every draft layer; Qwen3-4B, 5 layers, block 8; GSM8K / HumanEval / MT-Bench):
  - Block drafter: τ 3.5 / 3.5 / 2.6 with input fusion vs 4.2 / 4.0 / 3.0 with KV injection
  - AR drafter: 4.2 / 4.3 / 3.1 (EAGLE-3-style, input fusion) vs 4.8 / 4.6 / 3.4

  Five target features beat three (Math500 τ 5.64 vs 5.38). Removing target features entirely leaves 5-layer τ at 3.38–4.61 at T=0. — [DFlash](https://arxiv.org/abs/2602.06036)
- **GLIDE (with CaPE):**
  - The drafter cross-attends to the target's KV cache; draft layer m attends to target layer N_T − N_D + m, i.e. the upper layers.
  - Training uses a block-wise mask (L=5) to simulate the cache delay.
  - One layer for 7B/13B targets and two for 33B (47M / 71M parameters).
  - Acceptance is 55–70%, or 5.4–23.5 points above a vanilla drafter. Vicuna-13B on code reaches 68.6% and an expected 2.24× speedup.
  - CaPE widens the proposal set at low-confidence positions (S(p) = 7/5/3/1).
  - Reported speedups: up to 2.17× for GLIDE alone and 2.61× with CaPE on Vicuna. The paper's own figures are inconsistent (2.5× vs 2.61×).
  - No per-position numbers.

  — [GLIDE, arXiv 2402.02082](https://arxiv.org/abs/2402.02082)
- **Speculative Pipeline Decoding:** the target is split into n pipeline stages, and the drafter predicts one token per step from multi-depth target anchor features. It "never conditions on draft-generated features". In the paper's words: "Difficulty is bounded by the fixed width n rather than an open-ended draft horizon." Wall-clock speedups (Qwen3.5-4B unless noted):

  | Setting | SPD (n=8, L_s=3) | EAGLE-3 (k=7) |
  |---|---|---|
  | T=0, overall | 2.53× | 1.97× |
  | T=1, overall | 2.05× | 1.46× |
  | T=0, HumanEval | 3.23× | 2.10× |
  | T=0, MT-Bench | 1.93× | 1.65× |
  | Qwen3.5-9B, T=0 | 2.67× | 2.54× |

  At n=16 the theoretical speedup is 3.70× but wall-clock is only 1.43×. No per-position data; the paper reports the step ratio α only. — [SPD](https://arxiv.org/abs/2605.30852)
- **Kangaroo:** a shallow sub-network of the target plus an adapter drafts, and a second early exit stops drafting below a confidence threshold. Up to 1.68× on Spec-Bench, with 67M added parameters vs 591M for Medusa-1. — [Kangaroo, arXiv 2404.18911](https://arxiv.org/abs/2404.18911)
- **LayerSkip:** the model drafts by exiting at early layers and verifies with the remaining ones. Up to 2.16× on CNN/DM and 1.82× on coding. — [LayerSkip](https://arxiv.org/abs/2404.16710)
- **FlexDraft:** the drafter is the target's own upper layers with tuned attention and frozen FFNs (results above). — [FlexDraft](https://arxiv.org/abs/2605.20022)
- **Decay along a separate axis: context length.** On Qwen3.5-27B, base DFlash τ on hotpotqa falls from 4.91 at 1K to 3.61 at 16K. A long-context fine-tune gives 4.99 → 6.05 instead. On gov_report at 32K, τ goes from 2.09 to 3.56. — [DFlash](https://arxiv.org/abs/2602.06036)

**Parallel drafting plus cheap dependencies within the block**
- **DSpark:** a DFlash backbone plus a rank-256 Markov transition head, with a calibrated confidence head and a scheduler that drops low-confidence suffixes. Its lead over DFlash widens with proposal length:
  - γ=7: +16% math, +15% code, +18% chat
  - γ=15: +30% math, +26% code, +22% chat

  Round latency rises only 0.2–1.3%. τ at T=1, block 7, Qwen3-8B, DSpark vs DFlash vs EAGLE-3:
  - HumanEval: 5.52 vs 4.64 vs 4.33
  - MBPP: 5.16 vs 4.36 vs 3.96
  - MT-Bench: 3.72 vs 3.11 vs 2.66

  τ includes the bonus token. — [DSpark](https://arxiv.org/abs/2607.05147)
- **DFlash 2:**
  - Keeps the top-16 candidates at each position.
  - A low-rank bilinear path selector costs +2.0M parameters and +0.6% latency.
  - A two-tap dynamic depthwise convolution costs +16.5M parameters (3%) and +0.7% latency.
  - Qwen3-4B 5-layer ablation (GSM8K τ, T=0 / T=1): DFlash 4.27 / 3.78 → adding the DSpark correction 4.49 / 4.08 → adding path selection 4.61 / 4.25. The oracle over the top-16 candidates reaches 6.79 at T=0.
  - Deep-position flattening: about 86% at positions 8–14, vs DFlash 81.3% → 77.5%.

  — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **DSpine (adjacent causal injection):**
  - After every layer, each position gets a gated message from its predecessor's predicted feature, so dependency propagates through depth while all positions update in parallel.
  - Uses block-causal rather than bidirectional attention, cosine supervision at every layer in an output-embedding space, and a transition cache over K²=16² candidate pairs.
  - Mean τ on Qwen3-8B (T=0 / T=1): DFlash 3.77 / 3.39, DSpark 4.32 / 3.92, DFlash2 4.39 / 3.90, **DSpine 4.82 / 4.42** (+27.8% over DFlash at T=0).
  - Component ablation on MATH-500: 4.93 without refinement or injection, 5.11 without refinement, 5.49 full.
  - Injection adds at most 0.15 ms per round and about 33M parameters. τ includes the bonus or correction token.

  — [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)
- **AngelSpec DFly ablation** (Hy3, T=0, average τ): DFlash backbone 3.77 → DFly backbone 4.40 → adding a Markov head 4.56 → adding a hidden-correction head 4.60 → adding code/math data 4.75. On Hy3-295B-A21B, DFly-8 runs 1.98–2.40× faster than autoregressive decoding, and 10.5–11.8% faster than DFlash. — [AngelSpec](https://arxiv.org/abs/2607.25852)

**Position specialization**
- **PosS** (separate draft layers specialized by position). Average τ:
  - Llama-3-8B, T=0: EAGLE-2 4.06, HASS 4.39, PosS-2 4.55
  - Llama-2-13B, T=0: 4.79, 5.32 and 5.37 (PosS-3)

  The best single-dataset gain over HASS is 4.62 → 4.83 τ (Alpaca). PosS-1 uses about six times as many draft layers as EAGLE-2. — [PosS](https://arxiv.org/abs/2506.03566)
- **Gumiho** (serial layers for early tokens, parallel heads later) and **FastEagle** (cascaded per-position layers): results are in Section 1. — [Gumiho](https://arxiv.org/abs/2503.10135); [FastEagle](https://arxiv.org/abs/2509.20416)

### Inferences
- **Existing MTP heads:** the cheapest proven fix is training-time test or self-distillation on the drafter's own outputs and the target's rollouts. The step-3 gains are large (FastMTP from about 0 to 36%; AngelSpec from 0.27 to 0.52 marginal). These are post-hoc fixes that need no change to the target.
- **Long parallel drafts:** dependencies within the block are where recent gains are concentrated. DSpark, DFlash 2 and DSpine each beat DFlash by roughly 16–30%, at under 1–2% added latency. Only DFlash 2 shows the deep-position curve itself flattening.
- **The fixes partly stack.** DFlash 2 builds on the DSpark correction, and DSpine beats both DSpark and DFlash2 as baselines on Qwen3-8B. But each paper re-trains its baselines, and absolute τ differs widely between papers (DFlash on Qwen3-8B: 6.49 in its own paper vs 3.77 in DSpine).
- **Structural fixes** (Speculative Pipeline Decoding, GLIDE, self-speculative early exit) remove drift by construction but cap draft length or acceptance in other ways: pipeline width, 55–70% acceptance, or 1.7–2.2× speedups. None reports deep-position data.

### Gaps
- No per-position acceptance at depth 8 or beyond is reported for Speculative Pipeline Decoding, DSpine, HASS, PosS (figure only), GLIDE, Kangaroo, LayerSkip or EAGLE-3 (figure only).
- How Kangaroo and LayerSkip behave at deep positions is not reported; both work with short drafts.
- No paper reports a head-to-head of all the fixes on one target and one training budget.

## 4. What is the longest draft with reported high acceptance, and the highest acceptance length (tokens per verification) on code benchmarks like HumanEval / MBPP?

### Takeaway
The longest draft with documented high per-position acceptance is a 16-token block (15 drafted tokens) whose conditional rate stays at about 86% through the last position (DFlash 2, math). Proposal lengths of 15–16 also appear in DSpark (γ=15), Speculative Pipeline Decoding (n=16), native Qwen3.5 MTP run 15 steps (τ 6.15–6.47) and MTP-D (16 heads, but only 5–10% cumulative at head 16). The highest code acceptance length found is **τ 9.1 on HumanEval** for Qwen3.5-27B with DFlash (SGLang, concurrency 8). Next come Qwen3-Coder-30B-A3B with DFlash (HumanEval 8.09, MBPP 7.23), Qwen3.5-9B / 35B-A3B with DFlash (HumanEval 7.9), the z-lab Qwen3.5-4B DFlash at block 16 (HumanEval 7.72) and EAGLE-3 on Vicuna-13B (HumanEval 7.54, tree). All the DFlash figures include the bonus token.

### Cited Findings

**Longest drafts**
- **DFlash 2:** block 16 with 15 drafted positions; conditional acceptance is 86.48% at position 14 on MATH-500 with Qwen3.5-4B. — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **DFlash:** block 16 by default.
  - Qwen3-4B / Qwen3-8B at T=0: average τ 6.54 / 6.49 and MATH-500 τ 7.84 / 7.87.
  - In SGLang on B200: Math500 τ 8.01 for both Qwen3-4B and Qwen3-8B.

  — [DFlash](https://arxiv.org/abs/2602.06036)
- **DSpark:** proposal lengths up to γ=15–16 tested; its lead over DFlash is largest at γ=15. — [DSpark](https://arxiv.org/abs/2607.05147)
- **Speculative Pipeline Decoding at n=16** (Qwen3.5-4B, T=0): 3.70× theoretical speedup, but only 1.43× wall-clock. — [SPD](https://arxiv.org/abs/2605.30852)
- **Native Qwen3.5-4B MTP at 15 steps:** MATH500 τ 6.47, HumanEval 6.15. — [z-lab card](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)
- **MTP-D at 16 heads:** 16th-head cumulative acceptance 5–10%. — [MTP-D](https://arxiv.org/abs/2603.23911)
- **Apple MTP with 8 masks** (9 tokens per step at most): 5.35 tokens per step on HumanEval. — [arXiv 2507.11851](https://arxiv.org/abs/2507.11851)
- **FastMTP:** the head was trained for up to 7 drafts, but speed peaks at 3. — [FastMTP](https://arxiv.org/abs/2509.18362)
- **MiMo v2.5-Pro-UltraSpeed** uses DFlash to exceed 1k output tokens per second. No drafter details, block size or τ are given. — [LMSYS blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)

**Highest code τ by source** (target, setting, τ)
- **DFlash, SGLang, concurrency 8** (appendix A.4; HumanEval τ / speedup):
  - Qwen3.5-27B: **9.1 / 3.9×**
  - Qwen3.5-9B: 7.9 / 3.4×
  - Qwen3.5-35B-A3B: 7.9 / 2.3×
  - Qwen3.5-4B: 7.3 / 2.9×, vs native MTP 6.3 / 1.6× (MTP step count not captured)

  — [DFlash](https://arxiv.org/abs/2602.06036)
- **DFlash, Qwen3-Coder-30B-A3B** (8-layer drafter; SGLang, B200): HumanEval τ **8.09** (3.5× at concurrency 1), MBPP **7.23** (3.2×), LiveCodeBench 6.42 (2.6×). — [DFlash](https://arxiv.org/abs/2602.06036)
- **DFlash at T=0, block 16:**
  - Qwen3-4B: HumanEval 6.64, MBPP 6.09, LiveCodeBench 7.09
  - Qwen3-8B: HumanEval 6.50, MBPP 5.95, LiveCodeBench 7.27
  - At T=1, Qwen3-8B HumanEval is 5.52.

  — [DFlash](https://arxiv.org/abs/2602.06036)
- **z-lab Qwen3.5-4B DFlash, block 16** (greedy, thinking on): HumanEval 7.72 (4.60× at concurrency 1), MBPP 6.53. — [z-lab card](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)
- **EAGLE-3** (tree depth 8): Vicuna-13B HumanEval 7.54 (T=0) and 6.22 (T=1); LLaMA-3.1-8B 6.74 (T=0). — [EAGLE-3](https://arxiv.org/abs/2503.01840). FastEagle's re-run gets EAGLE-3 7.14 and FastEagle 7.10 on Vicuna-13B HumanEval. — [FastEagle](https://arxiv.org/abs/2509.20416)
- **FlexDraft with scaled training** (Qwen3-8B): HumanEval 5.55, MBPP 6.16. — [FlexDraft](https://arxiv.org/abs/2605.20022)
- **HASS** (T=0): LLaMA2-Chat 13B HumanEval 6.05; LLaMA3-70B 5.78. — [HASS](https://arxiv.org/abs/2408.15766)
- **DFlash 2 on HumanEval:** Qwen3.5-4B 6.28 (T=1, thinking on); Muse Glimmer 30B 5.66 (block 16, vs DFlash 4.11); Qwen3.8-27B 4.39 (block 8). — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
- **DSpark** (T=1, block 7): Qwen3-8B HumanEval 5.52, MBPP 5.16; Gemma4-12B HumanEval 5.64. — [DSpark](https://arxiv.org/abs/2607.05147)
- **DSpine** (Qwen3-8B, T=0 / T=1): HumanEval 4.94 / 4.63, MBPP 4.98 / 4.61. — [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)
- **AngelSpec Hy3 MTP** (D=3): HumanEval MAL 3.55. — [AngelSpec](https://arxiv.org/abs/2607.25852)

**Bonus-token conventions**
- **Included explicitly:**
  - DFlash: "including the bonus token produced by the target model" — [DFlash](https://arxiv.org/abs/2602.06036)
  - DSpark: all reported accepted-length metrics include the target's bonus token — [DSpark](https://arxiv.org/abs/2607.05147)
  - DSpine: τ = E[N_acc] + 1 — [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)
  - AngelSpec MTP: MAL = 1 + Σp_i — [AngelSpec](https://arxiv.org/abs/2607.25852)
  - Medusa: the acceleration rate counts the base token — [Medusa](https://arxiv.org/abs/2401.10774)
  - Apple MTP: the minimum value is 1 — [arXiv 2507.11851](https://arxiv.org/abs/2507.11851)
  - DFlash 2: its Table 1 counts the verifier's next token — [DFlash 2 blog](https://inco.ai/blog/dflash2/)
  - FlexDraft: L_acc = r_acc + 1 in the method section, though the tables are not explicit — [FlexDraft](https://arxiv.org/abs/2605.20022)
- **Excluded:** Clover-2's Figure 4 metric is "extra generated tokens (excluding the first one) per step". — [Clover-2](https://arxiv.org/abs/2408.00264)
- **Ambiguous or not stated:** EAGLE-3, HASS, PosS, FastEagle, FastMTP, Falcon, Scylla, Hydra, MoA, GLIDE, Nemotron 3 Super, MiMo-V2-Flash, and Speculative Pipeline Decoding (which reports only α = N/K). The z-lab card defines accept length as "completion_tokens / spec_verify_ct". — [z-lab card](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)

### Inferences
- **The z-lab metric** divides all completion tokens by the number of verify steps, so it should include the bonus token by construction. This is my inference; the card does not say so.
- **On standard code benchmarks, the published frontier for trained drafters is τ ≈ 8–9** (DFlash at block 16 on 27B–30B Qwen targets). That is roughly equivalent to a flat conditional rate of 0.88–0.92 over 15 drafts. Code τ appears to rise with target size and coder specialization (Qwen3.5-4B 7.3 → 9B 7.9 → 27B 9.1 HumanEval at concurrency 8).
- **High τ does not mean high speedup when drafting is sequential.** Native MTP at s=15 reaches τ 6.1–6.5 but only 1.5–1.8×. Speculative Pipeline Decoding at n=16 drops to 1.43× wall-clock. Long drafts pay off only when draft cost stays roughly flat in draft length (parallel or block drafters).

### Gaps
- Not covered here: retrieval and suffix-based drafting such as SuffixDecoding and prompt lookup. These can report much longer accepted runs on code-editing and agentic workloads.
- No trained drafter with a block of 32 or more and reported acceptance was found in this pass.
- The DFlash appendix row with HumanEval τ 9.1 does not state block size or thinking mode in what I extracted.
- MiMo v2.5-Pro-UltraSpeed (over 1k tok/s with DFlash) gives no acceptance numbers.
- Some cross-paper EAGLE-3 re-runs differ from the original. On Vicuna-13B HumanEval, the EAGLE-3 paper reports 7.54 and FastEagle's re-run 7.14. On LLaMA-3.1-8B, 6.74 vs 6.34.
