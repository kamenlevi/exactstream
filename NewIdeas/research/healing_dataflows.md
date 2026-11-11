# Healing altered dataflows in pretrained LLMs and MoEs

*Literature sweep, 2026-10-21, from arXiv abstract and HTML pages. Every number is the authors' own report; none were reproduced. "Zero-shot" means the dataflow was changed with no training. "Avg" is each paper's own benchmark mean.*

## (a) Module output added one layer late

- **FarSkip-Collective**, arXiv 2511.11505 (Nov 2025; v3 May 2026, MLSys 2026). This is the closest match.
  - **Models and change:** DeepSeek-V2-Lite 16B, Qwen3-30B-A3B and Llama-4 Scout 109B, with every layer converted. Layer k's attention misses layer k−1's routed-expert output, which is added once its all-to-all finishes. Layer k's MoE misses its own attention output. Inputs are at most one layer stale.
  - **Healing:** full-parameter KL self-distillation from the original model, on up to 10B instruction tokens.
  - **Results:** zero-shot, Qwen3 scores random on MMLU and 0% on HumanEval+. Healed avg goes 64.5→62.0, 75.9→73.7 and 76.0→75.1. Cross-entropy SFT on the same data reaches only 55.0 / 59.5 / 65.6.
  - **Token ablation (300M tokens, DS-V2-Lite):** converting the last 50% of layers gives 63.9, 75% gives 56.3, 100% gives 48.6. At 1.2B tokens, 100% gives 50.3.
  - **Speed:** time to first token up to 1.34× faster, decode up to 1.25×.
- **Ladder Residual**, 2501.06589 (Jan 2025, ICML 2025).
  - **Model and change:** Llama-3.1-8B-Instruct, upper 16 of 32 layers. Each module reads a residual that lacks the previous module's output.
  - **Healing:** 3B tokens of Infinity-Instruct, 2 epochs, lr 5e-6. LoRA vs full is not stated.
  - **Results:** avg 56.11 → 41.65 zero-shot (GSM8K 85.0→10.5, HumanEval+ 60.4→30.5) → 57.61 healed. Converting 20 layers gives 53.86.
  - **Caveat:** there is no same-data control. IFEval rose from 52.6 to 59.8, so part of the gain is the data.
  - **Other:** trained from scratch at 3.5B, Ladder costs 1.2 avg points. Speed +21% at TP8, batch 1.
- **SPD (Sync-Point Drop)**, 2502.20727 (Feb 2025, ICML 2025).
  - **Models and change:** LLaMA2-7B/13B/70B and OPT. Drops the tensor-parallel all-reduce after attention.
  - **Healing by sensitivity:** at least 44% of blocks (about 75% on 70B) are dropped zero-shot. Sensitive blocks get block-local MSE distillation on 128×2048 WikiText-2 tokens for 10 epochs. Extremely sensitive blocks also need head-grouping initialisation.
  - **Results:** 70B loses 0.94% accuracy for about 20% lower latency. With every sync dropped, distillation recovers 20–28 points on 7B/13B.
- **ScMoE**, 2404.05019 (Apr 2024; v3 May 2025). Top-1 experts read the previous layer's post-attention state, with a shared expert on the current one. It matches or beats top-2 (GPT3-MoE-XL PPL 16.46 vs 17.52), but only in models trained from scratch (up to 6.7B). There are no retrofit experiments.
- **Pipeline-Native Transformers**, 2608.23841 (Aug 2026), trained from scratch.
  - **Results:** expert_delay=2 costs nothing on TinyStories (d=512, L=6): PPL 6.50 vs 6.52 synchronous. A 30.9B pipeline-native MoE decodes 5.94 tok/s on 32 vCPUs, and overlapping disk reads adds up to 1.68×.
  - **Retrofit claim:** the author calls the design "not retrofittable to existing checkpoints". This was not tested, and FarSkip and Ladder show retrofits work.

## (b) Consecutive layers in parallel, and layer skipping

- **Layer Parallelism (LP)**, "Leveraging the true depth of LLMs", 2502.02790 (Feb 2025; TMLR 2026).
  - **Models and change:** Llama 2 7B, Llama 3.2 3B, Qwen3 4B/14B. Each pair of consecutive layers reads the same input, and their outputs are summed.
  - **Zero-shot:** 1.19× throughput at −1.5 avg on Llama 2 7B. Cutting Qwen3-4B from 36 to 27 effective layers drops MMLU 70.2→44.1 and GSM8K 85.0→3.6.
  - **Healing:** only the LP layers are trained, on RedPajama for 4,096 steps × batch 32 (tokens not stated). Qwen3-4B recovers to MMLU 62.0 and GSM8K 48.3. Qwen3-14B at 40→35 layers: GSM8K 61.6→81.7 (base 82.3).
  - **Insight:** the authors recover about 50% of the loss and were "unable to fully recover" the rest. Avoid early layers and keep a sequential tail; triplets do worse.
- **FFN Fusion**, 2503.18908 (Mar 2025; NeurIPS 2025). Fuses runs of consecutive FFNs into one wide parallel FFN, in blocks whose attention Puzzle had already removed.
  - **Puzzle-49B:** going from 26 sequential FFNs to 2 gives MMLU 80.73→79.98 and MT-Bench 8.87→8.25 zero-shot. After 25B tokens of KD: 80.56 and 9.00.
  - **Llama-3.1-405B→253B:** 49 FFNs fused into 4 groups give 82.76/8.35 zero-shot and 85.17/9.10 after about 65B tokens of KD (84.23/8.83 before fusion).
  - **Whole-block parallelism:** running whole attention+FFN blocks in parallel, two 4-block groups cost 0.7 MMLU, and four groups drop it to 56.1. Healing was not tried.
- **LayerSkip**, 2404.16710 (Apr 2024; ACL 2024). Continual pretraining with layer dropout and an early-exit loss: 52B tokens for Llama 2, 419B for Llama 3 8B. Last-layer quality held on Llama 2 but fell on Llama 3 8B (MMLU 66.5→60.5, HumanEval 37.8→28.7).
- **FlexiDepth**, 2503.23798 (Mar 2025; COLM 2025). Llama-3-8B-Instruct stays frozen; only a router and adapter are trained (Tulu-v2, about 7 h on 8×A100). It skips 6–8 layers per token at about 100% retention, but shows no wall-clock speedup.
- **"Tandem" / parallel attention+MLP conversion:** I found no paper that converts a pretrained sequential LLM to parallel blocks and measures healing. Trained from scratch, parallel blocks cost about 1 point (Ladder paper, 3.5B: 63.03 vs 64.11).

## (c) Pre-gated / predicted routing

- **Pre-gated MoE**, 2308.12066 (Aug 2023; ISCA 2024). Switch-Base-8/128 and Switch-Large-128: block N's gate picks block N+1's experts.
  - **Training:** pre-gates are trained during task fine-tuning from the pretrained weights, using the baseline's recipe (134M tokens per task).
  - **Results:** quality ≈ the original (Large-128 SQuAD EM 81.9 vs 82.4). Looking ahead 2–3 layers does worse.
- **Edge0**, 2609.18063 (Sep 2026).
  - **Models:** Qwen3.6-35B-A3B (40 layers × 256 experts, served at top-4 instead of top-8) and an 8B Ling model.
  - **Change:** a per-layer MLP prerouter, warm-started from the next layer's router, reads the post-attention state plus the expert sets used at tokens t and t−1. It predicts the next layer's experts for token t+1. **The prediction is the routing** (layers 7–38).
  - **Healing:** a recovery LoRA (r=16, on attention and shared-expert projections; routed experts frozen). It is trained first by SFT on about 2M rows of teacher text with prerouting on, then by on-policy reverse-KL on about 200k rows.
  - **Results:** with int4, prerouting and top-4 together, avg 79.2 vs 83.2 for the fp16 teacher (AIME −6.1). The 8B gets 69.9 vs 72.7.
  - **Missing:** no ablation without the LoRA, and no PPL, KLD or hit rate.
  - **Speed:** 20.4 tok/s at 2.9 GiB on a 24 GB M4 Pro.
  - **Warning:** distillation alone "produce[s] repetitive, collapsed text".
- **Speculating Experts**, 2603.19289 (Mar 2026). Qwen3-30B-A3B and GPT-OSS-20B/120B.
  - **Change:** layer l+1's router is applied to an estimate built from layer l's state, and those experts run. The model is not fine-tuned.
  - **Results:** recall@k is about 90% on Qwen past layer 2, yet avg falls 0.817→0.641 (GSM8K 0.950→0.576). GPT-OSS barely moves (0.799→0.779 and 0.849→0.857).
  - **Fix:** small estimators (4–45M params, KL to the true router, 4–5M tokens) lift Qwen to 0.750. Using them only in the high-drift early layers gives 0.766.
  - **Speed:** time per token falls 5–14%.
- **Read-ME**, 2410.19123 (Oct 2024; NeurIPS 2024). Llama-2-7B-chat is rebuilt as a 4.7B-active MoE with one pre-gating router that does not depend on the layer. About 1B tokens; MMLU 45.3→38.9.
- **Prefetch-only** (outputs unchanged, so nothing to heal): ProMoE 2410.22134, AdapMoE 2408.10284, Fate 2502.12224, PROBE 2602.00509, pre-attention predictors 2511.10676 (93–98% accuracy), SeqMoE 2609.12978, MoE-CORE 2610.01950. Separately, 2609.17940 shows that using several layers of routing history improves prediction (OLMoE R² 0.60→0.67).

## (d) Reducing top-k

- **ZEDA**, 2605.18643 (May 2026).
  - **Models and change:** Qwen3-30B-A3B (top-8 of 128, plus 64 zero experts) and GLM-4.7-Flash (top-4 of 64, plus 32). Zero experts take 51–53% of the top-k slots.
  - **Training:** SFT on teacher samples (60k prompts), then on-policy reverse-KL (320 steps). 30 h (Qwen) or 61 h (GLM) on 8×H200.
  - **Results:** avg 74.9→74.2 and 72.5→71.8. Static halving with the same training gives 73.0 / 70.9. Stage ablation on Qwen: SFT 73.6, OPD 72.9, SFT then OPD 74.2. Decode is about 1.19× faster.
- **Training-free halving**, 2609.04575 (Sep 2026). On Qwen3.6-35B-A3B, going from 8 to 4 experts costs 4.65 MMLU with standard renormalisation but only 0.35 when weights are normalised over the top 16. Qwen3.5-397B at 10→5 loses 0.55. Which experts run matters more than their weights, and perplexity and accuracy favour different settings.
- **OEA**, 2511.02237 (Nov 2025). On Qwen3-30B, top-5 is statistically indistinguishable from top-8 on most benchmarks (GPQA is the exception); top-4 and top-3 degrade.
- **Others:** AdaMoE 2406.13233 (Mixtral, null experts plus fine-tuning: −14.5% FLOPs, +1.69 ARC-C). LDA 2609.09241 is a training-free fix for the output-scale inflation that fewer experts cause.

## (e) Healing after structural changes

| Paper | Change | Healing | Result |
|---|---|---|---|
| 2403.17887 (Mar 2024) | Drop deep layers, Llama-2 | QLoRA r 2–64, C4, 164–328M tokens | Loss near unpruned; MMLU barely recovers; collapse at 45–55% remains |
| 2402.02834 (Feb 2024) | Depth-prune Vicuna-7B (PPL 17.1) | LoRA (50k Alpaca) vs CPT (37–271B tokens) | PPL at 20%: 18.8 vs 14.3; at 45%: 37.0 vs 16.0; at 60%: 68.9 vs 17.1 |
| 2411.15558 (Nov 2024) | −25% layers, Llama-3.1-8B-It (0.630) | 27.7M tokens, 2 epochs | LoRA 0.527; lm_head plus last 3 layers 0.581 |
| 2410.09982 (Oct 2024) | −6 blocks, Llama-3.1-8B-It (69.4) | LoRA, 50k samples, 5 epochs | None 51.8; SFT 56.7; teacher-regenerated data 63.3 |
| 2407.14679 (Jul 2024) | Width/depth pruning | KD vs CE | 4B MMLU 37.8–42.5 (KD) vs 24.6 (CE); 94B tokens final |
| 2410.10254 (Oct 2024) | Linear attention | Attention transfer + LoRA, 40M tokens | Llama-3-8B MMLU 66.6→52.8 |
| 2409.13710 (Sep 2024) | Remove LayerNorm, GPT-2 small | About 500M tokens | Loss gap about 0.05; HellaSwag −0.5 |

## Principles

1. **Damage grows with how many layers see a perturbed input, and how early.**
   - Late-half conversions heal with 10^8–10^9 tokens: FarSkip at 50% loses 0.6 after 300M tokens, and Ladder over 16 layers shows no net loss after 3B.
   - All-layer conversions need about 10^10 tokens and still lose 1–2.5 points.
   - Early layers are fragile. LP avoids them, and Edge0 and the Speculating Experts hybrid keep exact routing there.
2. **Perturbations heal cheaply; deletions do not.**
   - Changes that keep module inputs statistically close heal with 10^7 tokens or fewer, or none. Examples: dropping a sync on an insensitive block, renormalised top-k halving, prerouting in later layers.
   - Changes that remove computation (deep pruning, linearisation, deep LP, whole-block parallelism) recover the loss but only part of the reasoning and knowledge. Healing repairs "the pruning interface" (2403.17887).
3. **Token budgets:**
   - 10^5–10^7: block-local distillation and router heads (SPD, estimators).
   - 10^7–10^8: LoRA for mild pruning or linearisation.
   - 10^8–10^9: partial conversions and task-level pre-gating.
   - 10^9–10^10: full-depth conversions.
   - 10^10–10^11: severe pruning or new early-exit behaviour (FFN Fusion, Minitron, CPT, LayerSkip).
4. **Distill from the original model.**
   - KL to the teacher beats cross-entropy on the same data by 7–14 points (FarSkip). Minitron and 2410.09982 agree.
   - For routing changes, warm up with SFT on teacher outputs, then use on-policy reverse-KL (ZEDA, Edge0).
   - Intermediate-state losses are mixed: they helped Minitron and hurt FarSkip (46.0 vs 48.6).
5. **LoRA is enough for small, low-rank errors**, such as quantisation, routing changes with experts frozen, or linearisation. Large shifts need full or partial-layer training (+5.4 over LoRA in 2411.15558). The full-depth conversions train all parameters.
6. **Measure generation, not perplexity.** GSM8K and HumanEval collapse while MMLU and loss look fine: Ladder zero-shot loses 5 points of MMLU but 74 of GSM8K. Track teacher KLD plus GSM8K, HumanEval and AIME, against a same-data control.
7. **Warnings.**
   - **Merging a LoRA into int4 weights erases most of it.** Its deltas (RMS about 1e-3) are smaller than one 4-bit step. Only 34% of the change survives on an attention projection, 2% on a dense one, and 18% at the logits (Edge0). Serve the adapter unmerged, or train quantisation-aware (QA-LoRA 2309.14717).
   - **Strongly trained models break more easily:** Llama 3 more than Llama 2 under LayerSkip, and Qwen3 more than GPT-OSS under speculative routing.
   - **Some headline numbers disagree with their own tables.** Speculating Experts says Hybrid recovers "~37%" of the GSM8K gap, but its table implies about 98%. LP's and FarSkip's speedups vary within each paper.
