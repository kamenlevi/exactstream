# Distilling Qwen3.8-Flash-Next on rented GPUs: plan and cost

*NewIdeas, 2026-10-21. Prices come from provider pages read today. Throughput is estimated from one published measurement. Nothing was run.*

## Three corrections to the brief

- **The model is 125B-A6B.** The model card says "125B with 6B activated, plus 51B n-gram embedding and 4B MTP".
  - From the config and shard headers, active parameters per token are:
    - routed experts: 2.36B (of 120.8B)
    - dense parts: 3.65B, of which GDN is 2.09B, QSA 0.62B, gated-residual mixers 0.63B, shared experts 0.24B and routers 0.06B
    - LM head: 0.64B
  - So a forward pass is **~14 GFLOP/token**. At 12B active, every compute line below would double.
- **The bf16 repo is 360 GB** (180B params): the ~250 GB core, the n-gram table (~102 GB) and MTP (~8 GB). There is also an official FP8 copy (186 GB).
- **QSA is dense causal attention up to 2,048 tokens.**
  - MTPLX's `qwen4_exp.py` says "dense == sparse in this regime".
  - So training on sequences of 2k tokens or fewer needs no sparse backward.
  - No open implementation trains the indexer.

## 1. Hardware and prices ($/GPU-h, 8-GPU nodes)

| GPU | BF16 dense | On-demand | Spot / preemptible |
|---|---|---|---|
| H100 80 GB | 989 TF | Lambda 3.99, Together 3.99, RunPod 3.99 (community 2.69), Nebius 4.50, CoreWeave 6.16 | Together 1.99, CoreWeave 2.46, Nebius from 0.79 (volatile) |
| H200 141 GB | 989 TF | DigitalOcean 4.47, RunPod 5.29 (community 3.59), Nebius 5.40, Together 5.99, CoreWeave 6.31 | CoreWeave 2.62, Together 2.99 |
| B200 180 GB | 2,250 TF | Lambda 6.69, RunPod 7.99, Together 8.19, Nebius 8.50, CoreWeave 8.60 | Nebius 3.10–4.10 (live), Together 4.09, CoreWeave 4.26 |
| MI300X 192 GB | 1,307 TF | DigitalOcean 2.59 (1.91 with a 12-month commitment), Hot Aisle 3.39, Crusoe 3.45 | none found |

Notes:
- **Vast.ai** had no 8× H100 or 8× B200 listings. 8× H200 was 4.81–5.05.
- **Lambda's** 8× H100 and 8× B200 nodes showed out of stock. Its multi-node clusters need 16+ GPUs for at least 2 weeks.
- **Ingress is free** at Lambda, RunPod, CoreWeave, Nebius and Crusoe.

## 2. Memory and frameworks

Per-GPU memory on 8×80 GB, assuming:
- EP8 for the experts and ZeRO-3 for the dense parts
- the n-gram table in host RAM and MTP not loaded
- full recompute, 8k tokens/GPU, and a chunked LM-head + KL loss

| Item | GB/GPU |
|---|---|
| Routed experts: bf16, frozen, shared by teacher and student | 30.2 |
| Embeddings + LM head (frozen) | 2.5 |
| Trainable dense + frozen teacher copy (2 × 7.3 GB, sharded) | 2.3 |
| AdamW for 3.65B params (LoRA: ~0.2) | 6.4 |
| Activations: 4-stream residual at 20 KB/token/layer, plus one layer's working set | 9.5 |
| KL chunks, all-to-all buffers, CUDA context, fragmentation | ~10 |
| **Total** | **~61 (~69 at 16k tokens)** |

**Verdict: it fits only with every one of those measures.**
- vLLM requires the n-gram table to be off-GPU on H100.
- Unchunked fp32 logits take 8 GB per 8k tokens.
- ms-swift measured 72.5–77.7 GiB/GPU for LoRA-SFT of this model at 8k tokens, with no teacher loaded.
- Stock online-teacher KD loads a second model (+250 GB). Sharing the frozen experts needs a "teacher mode" flag in the model code. The alternatives are a teacher server, or top-k logprobs saved during generation.
- **Default to 8×H200 or 8×B200.** H100 is the budget option.
- If only the last N layers change, compute the shared prefix once. That saves ~35% of FLOPs for a half-depth conversion.

**Frameworks.** Merged training support for this model exists in four:

| Framework | Trains Flash-Next? | KD / on-policy |
|---|---|---|
| ms-swift + mcore-bridge (Megatron) | **Yes**, all parts incl. MTP. TP/PP/CP/EP. LoRA can wrap experts and router; 8-GPU recipe | GKD: `beta` 0 = forward KL, 1 = reverse; `lmbda` = on-policy share (vLLM); teacher server with top-k |
| NeMo AutoModel | **Yes**, FSDP2+EP; no TP/PP/MTP | Online-teacher KD; no on-policy |
| Axolotl | **Yes**: QLoRA / NVFP4 MoE-LoRA; full FT untested; no EP+TP | Offline KD from stored logprobs |
| transformers + TRL | **Yes** since 26 Aug, at reference speed (eager/SDPA) | TRL DistillationTrainer: on-policy GKD |
| Megatron-LM / -Bridge | Draft PRs only | Offline top-k KD; ModelOpt online KD |
| verl / NeMo-RL / slime | Open PR / feature branch / not found | All do on-policy distillation (ZEDA used slime) |
| torchtitan, DeepSpeed, LLaMA-Factory, Unsloth | No: Qwen3-Next or Qwen3.5/3.6 only; Unsloth is inference-only | — |
| Tinker | No: fixed model list, no custom dataflow | OPD cookbook |

GDN hybrids (Qwen3-Next, Qwen3.5/3.6) train almost everywhere. FLA's chunked GDN kernel and causal-conv1d have backward passes.

**Pick:**
- **ms-swift** for stages 1–2. It is the only stack that has this model, EP, LoRA, forward/reverse KL and on-policy rollouts.
- **NeMo AutoModel** is the plain-PyTorch fallback.
- Either way, the dataflow patch must also go into vLLM.

## 3. Throughput and cost

**FLOPs per training token** (routed experts frozen, dense parts trained, teacher online):
- Teacher forward 14 + student forward 14 + student backward 22 ≈ **50 GFLOP**. The experts get activation gradients only. With LoRA, ~43.

**MFU** (model FLOPs ÷ dense peak; recompute counts as overhead):

| GPU | Base | Range |
|---|---|---|
| H100/H200 | 10% | 6–25% |
| B200 | 8% | 4–20% |
| MI300X | 6% | 3–15% |

Anchors:
- **This model:** NeMo AutoModel full fine-tuning on 64×H100 is the only measurement. It ran at 86.85 TFLOP/s/GPU, or 8.8% MFU and 2.1k tok/s/GPU, on mock data with balanced routing.
- **Similar models:**
  - An open Megatron-Bridge PR reaches 28% on Qwen3.5-35B-A3B (H100, forced balance).
  - Qwen3.5-35B on B300 gets ~8%.
  - ModelOpt distilled Qwen3.6-35B-A3B at ~3.2k tok/s per GB200.

At base MFU, one node does ≈16k tok/s on H100/H200, 29k on B200 and 12k on MI300X.

**Training compute only:**

| Tokens | Node-h: H100/H200 · B200 · MI300X | H100 3.99 / spot 1.99 | H200 4.47 / spot 2.62 | B200 6.69 / spot 4.09 | MI300X 2.59 |
|---|---|---|---|---|---|
| 3e8 | 5.3 · 2.9 · 6.7 | $169 / $84 | $190 / $111 | $155 / $95 | $139 |
| 1e9 | 18 · 9.7 · 22 | $565 / $282 | $633 / $371 | $519 / $317 | $462 |
| 3e9 | 53 · 29 · 67 | $1,690 / $845 | $1,900 / $1,110 | $1,560 / $952 | $1,390 |
| 1e10 | 177 · 97 · 223 | $5,650 / $2,820 | $6,330 / $3,710 | $5,200 / $3,180 | $4,620 |

At the top of the MFU range, which an optimized Megatron path might reach, divide these by ~2.5.

Other costs:
- **Teacher generation:** ~3 node-h per 1e8 tokens. This assumes vLLM FP8 at 10k output tok/s per node, which is probably conservative; InferenceX lists 13–30k tok/s per H200.
- **On-policy reverse KL:** 20–40 node-h. ZEDA's took 20 h on 8×H200 for a 3B-active model.
- **Evals:** ~1 node-h per model variant.
- **Debugging:** 5–15 node-h per software stack.

| Stage | Node-h | H100 on-demand | H200 on-demand | Spot |
|---|---|---|---|---|
| 0: proxy dry run | 6 GPU-h | – | ~$27 | – |
| 1: go/no-go, 3e8 | ~20 | $640 | $715–960 | $320–420 |
| 2: to 3e9, plus on-policy | ~100 | $3.2k | $3.6–4.8k | $1.6–2.1k |
| 3: full, 1e10 | 190–240 | $6.1–7.7k | $6.8–11.5k | $3.0–5.0k |

**Whole program:** $10–17k on-demand, or $5–7.5k on spot. B200 cuts training hours by ~45% at a similar price per token.

## 4. Data

What the papers used:
- **FarSkip:** up to 10B tokens of instruction data, KL to the original model, 2^16–2^18 tokens/step, lr 2e-5.
- **ZEDA:** the teacher answered 60k prompts (17k math and 15k code from AceReason-1.1-SFT, plus 28k chat from Llama-Nemotron). Then 320 on-policy steps × 16 prompts × 2 samples.
- **Edge0:** ~2M rows of teacher text, then ~200k on-policy reverse-KL rows, with LoRA r=16.
- **Thinking Machines:** ~150 on-policy steps, about 77k prompts × 4 samples.

**Recommendation:**
- Forward KL supervises the teacher's full distribution at every position, so open text heals almost as well as teacher text. FarSkip used ordinary instruction data.
- Teacher generations still matter for the thinking format, tool calls and long agentic-coding contexts.

**Stage-1 mix** (3e8 tokens ≈ 150k rows of 2k tokens):
- **40% teacher generations**, thinking on and off.
  - Prompts come from OpenThoughts3 and AceReason (math, code), WildChat and Tulu-3 (chat, instruction following), Nemotron-v2 (multilingual), and tool-use and agentic-coding transcripts.
  - Save the top-20 logprobs while generating (~120 B/token). All three dataflow variants can reuse them.
- **40% open responses**, scored by the teacher.
- **20% raw code, math and web text.**
- Hold out 1% for KLD measurement.

**Larger budgets:**
- 1e9 tokens ≈ 0.5M rows; 1e10 ≈ 5M.
- Later stages add 8–32k-token sequences and about 300 on-policy steps × 64 rollouts.

## 5. Risks

1. **The software is two months old.**
   - The patch must agree across training, vLLM and MLX.
   - Test teacher-mode parity (KLD ≈ 0) first. One model card reports the n-gram table silently loading as noise.
2. **MFU** swings the cost 2–3×. Measure it in the first hour.
3. **Frozen experts may not be enough.**
   - FarSkip trained everything.
   - Full training needs ~2.3 TB: at least 4×8 H100 or 2×8 B200 with InfiniBand.
4. **4-bit may erase the healing.**
   - In Edge0, merging LoRA into int4 kept only 2–34% of it.
   - Options: keep the trained dense tensors at 8-bit (+1.8 GB), serve LoRA unmerged, or train against fake-quantized weights.
5. **MTP drift.** Re-fit the draft head (~1e8 tokens), or MTPLX's 2–3× speedup drops.
6. **Hidden damage.**
   - Distillation alone gave repetitive text in Edge0.
   - GSM8K can collapse while perplexity looks fine.
   - Training on 2k-token sequences may not heal behaviour at 100k-token contexts.
7. **Spot preemption.** Trainable state is ~51 GB (LoRA: under 1 GB), so checkpoint every 20 min.
8. **License.** Fine-tunes are allowed. A separate license is needed only to run a model-as-a-service or AI-coding-assistant business.

## 6. Cheapest staged plan

**Stage 0 (~$30):**
- On the Mac, measure each candidate's zero-shot damage and its real 4-bit tok/s gain.
- Dry-run on Qwen3.6-35B-A3B (same GDN+MoE family) on 1×H200 for ~6 h: a parity test, and a KLD-recovery curve to 3e7 tokens.

**Stage 1, go/no-go (3e8 tokens, $320–960):**
- **Setup:**
  - One node.
  - Partial conversion: the last 24 layers, Edge0-style layers ≥7, or a ZEDA top-k cut.
  - Routed experts frozen.
  - LoRA r16–64 on GDN, QSA and the shared expert, with routers and mixers trained in full.
  - Forward KL to the teacher's top-k, sequences ≤2k tokens, ~1,150 steps × 2^18 tokens.
- **Evals:**
  1. Teacher KLD on 2M held-out tokens: mean, p99 and top-1 agreement.
  2. Generation benchmarks: GSM8K, MATH-500, AIME25 avg@8, HumanEval+/MBPP+ and IFEval; subsets of LiveCodeBench, MMLU-Pro and MGSM; repetition rate. Compare the teacher, the zero-shot model, the healed model, and a same-data control (the unmodified model given the same LoRA run).
  3. The 4-bit Mac build through EvalSuite's gate, plus measured tok/s.
- **Go** if all of these hold:
  - ≥75% of the zero-shot KLD gap closes.
  - Benchmarks are within ~1 point of the teacher. FarSkip's half conversion lost 0.6 points at 3e8 tokens.
  - There is no collapse.
  - The 4-bit gain survives.
- **Kill** if <50% of the gap closes. FarSkip's full conversion gained only 1.7 points from 4× more tokens.

**Stage 2 (to 3e9 tokens):**
- More layers and longer sequences.
- On-policy reverse KL: ms-swift GKD with `beta=1`, or slime+SGLang as in ZEDA.
- An MTP re-fit.

**Stage 3 (1e10 tokens):** only if the stage-2 slope justifies it. FarSkip still lost 0.9–2.5 points after 10B tokens.

## Sources

- **Model:**
  - HF API: [bf16](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next-FP8)
  - Local: `~/BigMoE/models/bare/config.json`, LICENSE, MTPLX `qwen4_exp.py`
- **Prices and specs:**
  - Prices: [Lambda](https://lambda.ai/pricing), [Together](https://www.together.ai/pricing), [RunPod](https://www.runpod.io/pricing), [CoreWeave](https://www.coreweave.com/pricing), [Nebius](https://nebius.com/prices), [DigitalOcean](https://www.digitalocean.com/pricing/gpu-droplets), [Hot Aisle](https://hotaisle.xyz/pricing), [Crusoe](https://www.crusoe.ai/cloud/pricing), [Vast.ai API](https://console.vast.ai/api/v0/bundles)
  - Throughput: [InferenceX](https://inferencex.semianalysis.com/compare/qwen-3-8-flash-next-h100-vs-h200)
  - Peak FLOPS: [MI300X](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf), [B200/H100](https://bizon-tech.com/blog/nvidia-b200-b100-h200-h100-a100-comparison)
- **Serving recipes:** [vLLM](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), [SGLang](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- **ms-swift:** [recipe](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html), [GKD](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Megatron-SWIFT/GKD.md), [mcore-bridge](https://github.com/modelscope/mcore-bridge/blob/main/src/mcore_bridge/model/gpts/qwen4_exp.py)
- **NeMo AutoModel:** [model page](https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/large-language-models/qwen/Qwen3.8-Flash-Next.md), [performance](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/performance-summary.mdx), [KD](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/llm/knowledge-distillation.mdx)
- **Other trainers:** [Axolotl](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/qwen3.8-flash-next), [transformers #48337](https://github.com/huggingface/transformers/pull/48337), [TRL](https://huggingface.co/docs/trl/main/en/distillation_trainer), [verl #7749](https://github.com/verl-project/verl/pull/7749), [NeMo-RL OPD](https://docs.nvidia.com/nemo/rl/nightly/about/algorithms/on-policy-distillation.html), [slime](https://github.com/THUDM/slime), [Tinker](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)
- **Megatron:** Megatron-LM [#7393](https://github.com/NVIDIA/Megatron-LM/pull/7393); Megatron-Bridge [#6060](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/6060), [#5098](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/5098), [#5318](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/5318); [ModelOpt tutorial](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md)
- **Papers:** [FarSkip](https://arxiv.org/html/2511.11505), [ZEDA](https://arxiv.org/html/2605.18643), [Edge0](https://arxiv.org/html/2609.18063), [Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/), `research/healing_dataflows.md`
- **Data:** [OpenThoughts3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M), [AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT), [Tulu-3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture), [Nemotron v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)
