# Acceptance limits for deep speculative drafts (16-64 tokens), and the 1,000 tok/s single-user question (2023-2026)

**Key numbers first (details and sources are in the sections below).**
- **Formula.** Tokens per verify = (1−α^{γ+1})/(1−α), capped at 1/(1−α) ([Leviathan 2023](https://arxiv.org/abs/2211.17192)). At α=0.98 and γ=30 that is 23.3 tokens per verify, and all 30 tokens are accepted with probability 0.545. At α=0.95 it is 15.9; at α=0.90 it is 9.6. These figures are my calculation from the formula.
- **Best published learned drafters (chain-equivalent α are my calculations).**
  - DFlash, Qwen3-8B at T=0: τ 6.49 on average and 7.87 on MATH-500 with 16-token blocks, about α 0.86–0.89 ([DFlash](https://arxiv.org/html/2602.06036v1)).
  - EAGLE-3 at T=0: τ 5.9–7.5 at depth 8 ([EAGLE-3](https://arxiv.org/html/2503.01840)).
  - MTP heads: 0.92→0.68→0.38 by position on GLM-4.5-Air ([Liu et al. MLSys 2026](https://arxiv.org/html/2601.11580)); 85–90% for the second token on DeepSeek-V3 ([DeepSeek-V3](https://arxiv.org/abs/2412.19437)).
- **Closest evidence of about 98% per position at depth about 20.** It holds only on the best code-editing requests, which mostly copy their input. With up to 20 tokens proposed per step, the best per-request means were 18.3 for a draft model and 15.0 for n-gram ([Liu et al. 2026](https://arxiv.org/html/2601.11580)), which works out to about α 0.986 and 0.965. SuffixDecoding accepts 6.3–7.8 tokens per step on agentic workloads ([SuffixDecoding](https://arxiv.org/html/2411.04975v2)). No learned drafter is near 98% held flat to depth 16–30 on open-ended text.
- **Near ties and self-agreement.** On chat, Qwen3-4B's 5th-percentile top-1 probability is 0.391 ([Sharma 2026](https://arxiv.org/html/2606.30265)). Qwen3-235B at T=0 produced 80 unique completions out of 1,000 because of batch-size numerics, with the first split at token 103 ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). My rough reading is that the target agrees with itself on about 99.5–99.7% of tokens.
- **Fastest single-user speeds on 100B+ models.**
  - Cerebras: 969 tok/s on Llama 3.1 405B, dense 16-bit ([Cerebras](https://www.cerebras.ai/blog/llama-405b-inference)); 2,522 tok/s on Maverick ([secondary](https://getcoai.com/news/ai-chip-startup-cerebras-outperforms-nvidias-blackwell-in-llama-4-test)); about 3,000 tok/s on gpt-oss-120B (vendor claim).
  - NVIDIA DGX B200: 1,038 tok/s on Maverick, using EAGLE-3 with draft length 3 and FP8 ([NVIDIA](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)).
  - DeepSeek-R1 671B on GPUs: about 250 tok/s.
  - M3 Ultra: about 16–20 tok/s on 671B at 4-bit.
- **Not covered (time limit).** Groq figures; 2026 records beyond GPT-5.3-Codex-Spark (over 1,000 tok/s, model size undisclosed); full texts of Yin et al. and Pankratov–Alistarh, which I read only as abstracts; quantized self-draft "ceiling" measurements; acceptance numbers for prompt-lookup and predicted-outputs; per-position curves to depth 30; and an official NeurIPS 2026 tutorial site, which I did not find.

Terms used below. α = per-position acceptance probability. τ or AL = tokens produced per draft-verify cycle, counting the bonus token (most EAGLE-family and vLLM papers use this). "Chain-equivalent α" = the constant α that would give the observed τ at draft depth γ under Leviathan's formula. Chain-equivalent α values are my own calculations and appear only under Inferences. Vendor claims are marked "(vendor claim)" and secondary sources "(secondary)".

## 1. What does theory say about acceptance limits?

### Takeaway
The classic papers set per-position acceptance at α = E[min(p,q)] = 1 − TV(p,q). They give expected tokens per verify as (1−α^{γ+1})/(1−α), which can never exceed 1/(1−α), so extra draft depth only pays when α is very close to 1. Theory from 2024-2026 adds three results:
- The verifier is near-optimal already. Standard rejection sampling is optimal among unbiased rejection-based schemes, and optimal block verification adds only about 5-8%. The binding limit is how far the draft distribution is from the target's.
- Under greedy acceptance, rejections are governed by the target's top-1 vs top-2 margin, i.e. by near-tie steps.
- An entropy "speed-of-light" bound says accepted tokens per iteration grow only like log(verifier capacity)/entropy.

### Cited Findings
**Leviathan, Kalman, Matias (ICML 2023), primary**
- Acceptance definition: α = E(β) = 1 − E(D_LK(p,q)) = E(min(p,q)) (Corollary 3.6). D_LK is a symmetric divergence in [0,1] (Corollary 3.4). — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Tokens per run: if the β are i.i.d., tokens per run follow a geometric distribution capped at γ+1, with E[#tokens] = (1−α^{γ+1})/(1−α) (Eq. 1). — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Walltime improvement is (1−α^{γ+1})/((1−α)(γc+1)), where c is the draft/target cost ratio.
  - c was always below 0.05 for their small drafts.
  - Some γ improves walltime exactly when α > c (Corollary 3.9).
  - For negligible-cost drafts (c≈0, e.g. n-gram lookups), the improvement is bounded above by 1/(1−α).
  - Source: [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Table 1 (c = 0): α=0.9, γ=10 gives 6.86x speed for 1.60x arithmetic. α=0.8, γ=5 gives 3.69x for 1.63x. α=0.6, γ=2 gives 1.96x. — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Table 3, empirical α measured on 10K target tokens:
  - LaMDA 137B with LaMDA 100M / 2B / 8B drafts: 0.61 / 0.71 / 0.75 at T=0 and 0.57 / 0.71 / 0.74 at T=1.
  - T5-XXL (EnDe) with T5-small / base / large: 0.75 / 0.80 / 0.82 at T=0 and 0.62 / 0.68 / 0.71 at T=1.
  - T5-XXL (CNNDM): 0.65 / 0.73 / 0.74 at T=0 and 0.53 / 0.55 / 0.56 at T=1.
  - Unigram and bigram drafts: 0.03–0.23.
  - GPT-like 97M target with a 6M draft: 0.88 at T=0 and 0.89 at T=1.
  - The paper's summary: drafts "a couple of orders of magnitude smaller" than the target give α between 0.5 and 0.9.
  - Source: [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Paper's remark: "the sharper the adjusted distribution, the higher the α values". α and walltime are higher with argmax decoding (T=0). — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Measured T5-XXL speedups (TPU-v4, batch 1): EnDe 3.4x at T=0 (γ=7, α=0.75) vs 2.6x at T=1 (α=0.62). CNNDM 3.1x vs 2.3x. — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)

**Chen et al. (DeepMind 2023), primary**
- Setup: Chinchilla 70B target, 4B draft (1.8 ms/token vs 14.1 ms/token for the target), K=4, batch 1, 16 TPU v4.
  - XSum, nucleus p=0.8: 1.92x (7.52 ms/token).
  - XSum, greedy: 2.01x (7.00 ms/token).
  - HumanEval, p=0.95, T=0.8: 2.46x (5.73 ms/token).
  - Source: [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
- Figure 1: accepted/(K+1) falls as K grows. Speedup "plateaus or even regresses" with K; XSum is optimal at K=3. Variance grows with K. — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
- On HumanEval and greedy XSum, the speedup beat the hardware's theoretical memory-bandwidth limit for autoregressive sampling. — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
- Acceptance depends on domain and decoding method. The authors' explanation for HumanEval's higher acceptance: common subsequences in code, shorter tokens, and temperature sharpening of both draft and target logits. They also found acceptance "robust to the exact parameters" of nucleus, top-k and temperature. — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)

**Optimality and lower bounds for the verifier**
- Yin, Chen, Huang, Wang (NeurIPS 2024) model decoding as a Markov chain and tie the components of speculative decoding together "via total variation distances". The claims that expected rejections equal summed TV distances, with a matching instance-dependent lower bound that makes standard SD optimal among unbiased rejection-based methods, come from a search-result summary of the PDF, not from my own reading. — [Yin et al. 2024](https://arxiv.org/abs/2411.00841)
- Block verification (Sun et al., ICLR 2025) verifies the whole draft jointly. It is optimal in expected accepted tokens per iteration among valid verifiers, never worse than token-level verification, and gains only about 5–8% in wall-clock time (seen via a search summary of the proceedings abstract). — [Sun et al. ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3e710b42b1a9ed898f607ec0f4fcc971-Abstract-Conference.html)
- Pankratov & Alistarh (Dec 2025), "Speculative Decoding Speed-of-Light", claim "the first 'tight' lower bounds" on the runtime of any deterministic speculative generation algorithm. Their bound is E[X] ≤ (μ + μ_(2))·log(P)/μ² + O(1), where:
  - X = tokens accepted per iteration
  - P = verifier capacity
  - μ = expected entropy of the verifier's output distribution
  - μ_(2) = expected second log-moment
  - The proof uses a branching-random-walk analogy. The abstract says Llama experiments "validate" the bound but gives no numbers. — [Pankratov & Alistarh 2025](https://arxiv.org/abs/2512.11718)
- SpecTr-GBV combines multi-draft generation with greedy block verification and claims optimal expected acceptance length within i.i.d. draft generation (search summary only). — [SpecTr-GBV](https://arxiv.org/html/2604.25925v1)

**Sharma, "When Is a Draft Accepted? A Theory of Acceptance in Speculative Decoding" (arXiv 2606.30265, 29 Jun 2026), primary**
- General result: greedy, additive- and multiplicative-relaxed, top-m, entropy-thresholded (Medusa-style) and greedy-tree criteria all have rejection regions that are lower level sets of the target distribution. — [Sharma 2026](https://arxiv.org/abs/2606.30265)
- Exact single-token certificate: rejection needs KL ≥ R_p = s·KL(p|_A/s ‖ Uniform(A)) (Thm 4.4). R_p ≤ (2/k)·log 2 ≤ log 2 (Thm 4.6), so no single-token criterion of this form guarantees acceptance once KL(p‖q) > log 2. — [Sharma 2026 (HTML)](https://arxiv.org/html/2606.30265)
- Strict greedy:
  - The margin is γ_p = p(x0) − p(x1). If KL ≤ ε and γ_p > √(2ε), acceptance is guaranteed (Prop. 4.8).
  - The bound is tight: g⁻¹(ε) = √(2ε)(1 − ε/6 + O(ε²)) (Thm 4.9).
  - The paper's conclusion: greedy SD "depends extensively on the target distribution's margin", so low-margin (near-tie) steps flip with little KL.
  - Source: [Sharma 2026 (HTML)](https://arxiv.org/html/2606.30265)
- Relaxed and tree acceptance:
  - Additive relaxation (accept if p(y) > p(x0) − t) makes t, rather than the target margin, the controllable quantity.
  - Greedy tree with the drafter's top-m candidates: acceptance is guaranteed if γ_m > √(4ε/(m+1))·(1+O(ε)), and the tree certificate is at most log(m+1).
  - The paper says branching cuts the needed margin by about √((m+1)/2), but elsewhere calls the certificate "(m+1)/2 times larger". The two statements are inconsistent.
  - Source: [Sharma 2026 (HTML)](https://arxiv.org/html/2606.30265)
- Qwen3 measurements (UltraChat-200k, 500 prompts, 128 generation steps each):
  - Target top-1 probability, mean / median / p5 / p25: Qwen3-1.7B 0.845 / 0.969 / 0.343 / 0.846; Qwen3-4B 0.871 / 0.988 / 0.391 / 0.886.
  - Top-10 mass at p5: 0.901 (1.7B) and 0.927 (4B).
  - Over 40% of steps have top-1 probability close to 1.
  - A tree with m=2 roughly doubles the median certificate over greedy; larger m gives diminishing returns.
  - These are analytic certificates, not measured end-to-end acceptance.
  - Source: [Sharma 2026 (HTML)](https://arxiv.org/html/2606.30265)

**Xia et al., "Acceptance-Aware Draft Model Training" (arXiv 2609.24150, v1 21 Sep 2026, v2 10 Nov 2026), primary**
- Objective, chain greedy case: E[L_t] = Σ_{k=1..γ} Π_{j=0..k−1} a_{t+j}, where a = q(x*) is the draft's probability on the target's greedy token. In the sampling case, a = 1 − TV(p̃,q̃) on temperature-scaled distributions. — [Xia et al. 2026](https://arxiv.org/html/2609.24150)
- Why standard training falls short: KL/CE and acceptance share the same global optimum only if the drafter can reproduce the target exactly. With limited capacity, KL spends capacity on the overall mismatch rather than on the tokens the target accepts. — [Xia et al. 2026](https://arxiv.org/html/2609.24150)
- Greedy-tailored losses barely move τ.
  - Tree setup: depth 7, branching (4,3,2,1,1,1,1), 128 nodes.
  - Qwen3-8B/0.6B: τ 4.494 (KL) → 4.514 (EAL+GRPO).
  - Qwen3-32B/4B: 5.057 → 5.088.
  - Chain γ=7, Qwen3-8B/1.7B: 3.649 → 3.698.
  - Source: [Xia et al. 2026](https://arxiv.org/html/2609.24150)
- Sampling-tailored losses move τ much more.
  - Qwen3-8B/0.6B tree at T=1: 4.998 (KL) → 5.750 (WTV).
  - Qwen3-32B/4B: 5.960 → 6.671.
  - Gemma4-31B/E2B chain at T=1: 5.699 → 6.102.
  - A gpt-oss-120b/Qwen3-4B pair was evaluated, but its values did not come through.
  - Source: [Xia et al. 2026](https://arxiv.org/html/2609.24150)

**Abdellatif, "Acceptance Dynamics Across Cognitive Domains in Speculative Decoding" (arXiv 2604.14682, v2 27 Sep 2026), primary abstract**
- Setup: tree SD with a TinyLlama-1.1B draft and a Llama-2-7B-Chat-GPTQ target; 99,768 speculative nodes from 200 prompts across code, math, logic and chat. — [Abdellatif 2026](https://arxiv.org/abs/2604.14682)
- Only chat consistently gives an expected accepted length above 1.0 token per step. Chat has both the highest entropy and the highest acceptance. — [Abdellatif 2026](https://arxiv.org/abs/2604.14682)
- The entropy–acceptance correlation is negative but weak in every domain (ρ between −0.20 and −0.15). Task type predicts acceptance better than tree depth. — [Abdellatif 2026](https://arxiv.org/abs/2604.14682)

**Liu, Yu, Park, Stoica, Cheung, "Speculative Decoding: Performance or Illusion?" (MLSys 2026 oral; arXiv 2601.11580), primary**
- Setup: vLLM 0.10.1/0.11.1 on H100, covering n-gram, EAGLE/EAGLE-3, draft-model and MTP methods. All experiments use temperature 0. — [Liu et al. 2026](https://arxiv.org/html/2601.11580); [MLSys oral page](https://mlsys.org/virtual/2026/oral/3782)
- The speedup model (1−α^{k+1})/((1−α)(kc+1)) assumes verifying k+1 tokens costs about as much as one, which holds only when decoding is memory-bound. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- Verification takes 42–95% of execution time (72% for Llama3-70B at BS=1). n-gram drafting takes under 2%; EAGLE drafting 12–20% at BS=1. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- Oracle comparison: an oracle whose proposal length always equals the accepted length reaches about 2.75x (n-gram, InstructCoder, Llama3.1-8B, BS=1) against about 2.1x for the best fixed k=5. Combining EAGLE and n-gram per position adds up to 2.2x more, and the introduction cites 4.9x over no-SD for adaptive combinations. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- MTP decay (GLM-4.5-Air, GPQA-Main): position-wise acceptance falls 0.92 → 0.68 → 0.38. The authors blame the single MTP module being "trained primarily to predict the first token accurately but reused autoregressively". — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- Tree vs chain, as AL / acceptance rate:
  - Qwen3-8B, GSM8K: chain k=3 2.25/0.415; tree k=6 2.51/0.300; tree k=21 2.92/0.095.
  - Llama3-70B, ShareGPT: 2.29/0.429 → 2.55/0.310 → 2.93/0.097.
  - At BS=64, k=21 trees drop below 1x speedup on every workload.
  - Source: [Liu et al. 2026](https://arxiv.org/html/2601.11580)

### Inferences
- Combining the results above: once the verifier is optimal (Yin; block verification adds only 5–8%), per-position acceptance is bounded by (a) how close the drafter is to the target (TV/KL, which is a capacity question) and (b) the target's own margin/entropy at each step (Sharma; Pankratov–Alistarh). Depth helps only multiplicatively through α^k.
- The Pankratov–Alistarh bound says accepted tokens per iteration scale like log(P)/μ: doubling verifier width adds a constant, while halving entropy roughly doubles the bound. This is consistent with Lookahead decoding's "log(FLOPs)" trade-off and with SuffixDecoding's finding that low-entropy tasks give long runs (both in section 2). I did not read the bound's assumptions or the definition of "deterministic" in the full text.
- Acceptance-aware training shows that, under greedy verification, changing the loss is not the lever (τ gains of +0.01 to +0.04). Drafter capacity or input information (e.g., target hidden states, as in EAGLE-3/DFlash) is.

### Gaps
- I read only the abstracts of Yin et al. and Pankratov–Alistarh. Their exact theorem conditions and empirical numbers were not verified.
- Abdellatif's per-depth acceptance profile values are not in the abstract.
- Sharma's Table 2 (certificate quantiles) was truncated in the fetch.
- No official "NeurIPS 2026 speculative decoding tutorial site" was found. The only lead is a Digg aggregator post saying Baseten engineers built an interactive speculative-decoding tutorial for the NeurIPS Education Track and released a "LosslessBench" harness; it is unverified ([Digg, secondary](https://digg.com/ai/rqh584h2)). An earlier COLING 2025 tutorial proposal exists ([arXiv 2503.00491](https://arxiv.org/abs/2503.00491)).

## 2. Is there a published draft method with ~95–98% per-position acceptance that stays flat to depth 16–30? Which tasks allow it?

### Takeaway
No published learned drafter shows about 95–98% per-position acceptance holding flat to depth 16–30 on open-ended text. As of 2026 the best learned drafters reach τ of about 6–8 per cycle on math and code at T=0 (DFlash with a 16-token block; EAGLE-3 trees of depth 8), and MTP heads decay steeply after the first position. Accepted runs of 15–20+ tokens appear only on copy-dominated or low-entropy workloads:
- code editing (InstructCoder)
- agentic and structured outputs (AgenticSQL, SWE-Bench agents)
- "speculative edits"

There, n-gram, suffix-tree or draft-model methods sometimes accept most of a 20-token proposal, but only on the best requests or tasks.

### Cited Findings
- **Performance or Illusion (MLSys 2026, T=0)**:
  - In the long-draft measurement, each method proposes up to 20 tokens per step. The authors record how many are generated (accepted + bonus) and then advance one token.
  - Per-request generation lengths: EAGLE 2.7–7.4; n-gram 1.1–15.0; draft model 5.6–18.3. The earlier fetch tied these to InstructCoder with Llama3-70B; the follow-up found no stated pairing or temperature, and the experiments elsewhere use T=0.
  - Source: [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- **Same paper, Table 4 dataset means** (proposal length not stated):
  - InstructCoder: n-gram 7.27 (std 4.19), EAGLE 4.24, EAGLE-3 4.18.
  - CNN/DM: EAGLE-3 3.08, n-gram 2.33.
  - ShareGPT: EAGLE-3 3.06, n-gram 2.02.
  - GSM8K: EAGLE-3 3.02, n-gram 1.41.
  - GPQA-Main: n-gram 4.20 vs EAGLE-3 2.78.
  - AIME22-24: n-gram 3.53 vs EAGLE-3 2.62.
  - Source: [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- **Same paper, when copying wins**: n-gram beats EAGLE/EAGLE-3 whenever prompt–output overlap is above about BLEU-4 0.6 (Llama3.1-8B), with up to 53% (k=3) and 100% (k=5) higher speedup. On long Qwen3-8B-Thinking outputs, n-gram tokens per step rise from about 1.6–3.7 (outputs under 4K tokens) to about 2.7–5 (over 13K), then fall when the final answer begins. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- **SuffixDecoding (Oliaro et al., NeurIPS 2025)**:
  - It speculates adaptively: MAX_SPEC = α·p, where p is the pattern-match length, and α∈[1,4] "works well for agentic applications". — [SuffixDecoding arXiv v2](https://arxiv.org/html/2411.04975v2)
  - Setup for the numbers below: Llama-3.1-8B-Instruct, one H100, batch 1, simulated greedy verifier.
  - AgenticSQL: 6.35 mean accepted tokens per step (linear) / 6.24 (tree), acceptance rate 0.415 / 0.375, against EAGLE-3 at 3.16 (0.036).
  - SWE-Bench: 7.82 (linear) / 7.55 (tree), against PLD at 3.17.
  - Spec-Bench (open-ended): suffix-only 1.75 vs EAGLE-3 4.65.
  - Speedups: 5.3x on AgenticSQL; about 2.5x on SWE-Bench (PLD 1.50x); 1.8–4.5x end-to-end in OpenHands (32B model, 4-way tensor parallelism, vLLM).
  - Some speculation branches advance output "by dozens of tokens or more in one step". — [SuffixDecoding arXiv v2](https://arxiv.org/html/2411.04975v2)
- **SuffixDecoding entropy vs performance**: empirical output entropy is 0.086 for AgenticSQL Extract, 0.171 Enrich, 0.738 Classify, about 1.5 for the SQL steps, 2.50 Spider, 2.95 Magicoder, 3.43 WildChat. Lower entropy predicts better performance. — [SuffixDecoding arXiv v2](https://arxiv.org/html/2411.04975v2)
- **SuffixDecoding headline numbers differ by version**: the NeurIPS abstract says "up to 3.9x", 2.2x over EAGLE-2/3 and 1.6x over Token Recycling, against the arXiv text's 5.3x and 2.8x. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115453); [Snowflake Arctic Inference blog](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic)
- **Lookahead decoding (Fu, Bailis, Stoica, Zhang, ICML 2024)**: up to 1.8x on MT-bench and 4x on code completion with strong scaling across multiple GPUs. It trades per-step log(FLOPs) for fewer decoding steps. — [Fu et al. 2024](https://arxiv.org/abs/2402.02057)
- **REST, retrieval-based (He et al., NAACL 2024)**: 1.62x–2.36x on 7B/13B models, single batch, code or text generation. — [He et al. 2024](https://arxiv.org/abs/2311.08252)
- **Cursor "Editing Files at 1000 Tokens per Second" (14 May 2024)**:
  - A fine-tuned Llama-3-70B fast-apply model with "speculative edits": deterministic speculation with no draft model, relying on the model's strong prior for the draft tokens.
  - About 1,000 tok/s (about 3,500 chars/s), roughly 13x over vanilla Llama-3-70B and 9x over their earlier GPT-4 speculative edits. Served on Fireworks.
  - No acceptance statistics were published.
  - Source: [Cursor blog](https://cursor.com/blog/instant-apply)
- **DFlash (block-diffusion drafter, ICML 2026)**:
  - Block size 16 for Qwen3. On Qwen3-8B at T=0, τ is: GSM8K 6.54, MATH-500 7.87, AIME25 7.08, HumanEval 6.50, MBPP 5.95, LiveCodeBench 7.27, MT-Bench 4.24; average 6.49 (4.86x).
  - EAGLE-3 averages 2.96 (tree 16) and 3.40 (tree 60) on the same setup.
  - At T=1, DFlash averages τ 5.48 (4.03x).
  - With block-8 inference on Math500, the whole block is accepted in 35.7% of cases.
  - Training and inferring at block 16 gives Math500 τ 6.33, against 5.21 at 8/8.
  - The paper notes that "Errors at early positions within a draft block invalidate all subsequent tokens."
  - Source: [DFlash arXiv](https://arxiv.org/html/2602.06036v1); [ICML 2026](https://proceedings.mlr.press/v306/chen26bv.html)
- **EAGLE-3 (Li et al., 2025)**:
  - Tree depth 8, 48–60 draft nodes, T=0. τ reaches 7.54 (Vicuna-13B, HumanEval), and averages 6.23 for LLaMA-3.1-8B and 5.88 for LLaMA-3.3-70B.
  - EAGLE-3's acceptance stays "almost unchanged" as more of its inputs are its own predictions, while EAGLE's drops (Fig. 7, values not tabulated).
  - Speedup rises roughly in proportion to training data.
  - Source: [EAGLE-3](https://arxiv.org/html/2503.01840)
- **MTP heads**:
  - DeepSeek-V3: second-token acceptance of 85–90% across generation topics, giving 1.8x TPS ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437), via search snippet). A later DeepSeek paper gives 80–90% in real-world use ([Insights into DeepSeek-V3](https://arxiv.org/pdf/2505.09343)).
  - SGLang reports average acceptance lengths of 2.18 and 2.44 for DeepSeek-V3 MTP ([LMSYS blog](https://www.lmsys.org/blog/2025-07-17-mtp), via search summary).
- **NVIDIA EAGLE-3 on Llama 4 Maverick**: AL rises to 3.44 at draft length 5, but speedup peaks at 2.06x at draft length 3. — [NVIDIA blog](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)

### Inferences
- Chain-equivalent α (my calculation from Leviathan Eq. 1):
  - Performance or Illusion, InstructCoder per-request bests (k=20, at most 21 tokens): draft model 18.3 → α ≈ 0.986; n-gram 15.0 → α ≈ 0.965.
  - Performance or Illusion, InstructCoder n-gram dataset mean 7.27 at k=20 → α ≈ 0.87. The paper does not state k for this table.
  - DFlash, Qwen3-8B at T=0: MATH-500 7.87 (γ=15) → α ≈ 0.89; average 6.49 → α ≈ 0.86; MT-Bench 4.24 → α ≈ 0.77.
  - EAGLE-3, Vicuna-13B HumanEval, 7.54 at depth 8 → about 0.955. This is a tree, so it measures per-level coverage by several candidates, not single-chain α.
  - NVIDIA Maverick, 3.44 at draft length 5 → α ≈ 0.77.
  - SuffixDecoding SWE-Bench 7.82 → α ≈ 0.87 if γ≈32; γ is not reported.
- The only published evidence close to "98% per position at depth about 20" is therefore the top per-request results on code editing, where outputs largely copy the input. A 1B draft model on InstructCoder reached about 0.986 chain-equivalent on its best requests. For learned drafters on math, code and chat, published chain-equivalents sit around 0.77–0.92.
- Long accepted runs are mainly a property of the workload's entropy (copying, JSON schemas, repeated agent steps), not of the drafter. Copy-style drafters (n-gram, suffix tree, speculative edits) have variance 2–5x higher than EAGLE-style drafters (Performance or Illusion), so they reach very long runs on some requests and nearly nothing on others.

### Gaps
- No paper gave tabulated per-position acceptance for positions 1..20 (Performance or Illusion shows only heatmaps), or the distribution of accepted-run lengths (e.g., the share of steps accepting more than 16 or 30 tokens) for SuffixDecoding or n-gram.
- The original prompt-lookup-decoding README, Token Recycling, SAM-decoding, LLMA (inference with reference) and OpenAI/Fireworks "predicted outputs" were not checked directly. No acceptance statistics for speculative edits were found.

## 3. How does greedy (T=0) verification compare with sampling in achievable acceptance?

### Takeaway
For drafters trained with KL/CE, greedy acceptance is consistently higher than T=1 sampling:
- Leviathan, T5: α 0.75 vs 0.62.
- EAGLE-3, LLaMA-3.1-8B MT-bench: τ 6.13 vs 4.24.
- DFlash, Qwen3-8B average: 6.49 vs 5.48.

The gap shrinks as drafts get bigger or better (LaMDA 8B draft: 0.75 vs 0.74; EAGLE-3 70B: 5.88 vs 5.66). It can even reverse with sampling-aware tree verification or training (Qwen3-8B/0.6B tree: τ 4.49 greedy vs 5.75 at T=1 with the WTV loss). Theory explains why: greedy acceptance is all-or-nothing on matching the argmax, so it hinges on near-tie margins, while lossless sampling gives partial credit, 1 − TV(p,q).

### Cited Findings
- Leviathan (Table 3), T=0 vs T=1:
  - T5-small on EnDe: 0.75 vs 0.62.
  - T5-large on EnDe: 0.82 vs 0.71.
  - LaMDA 137B with a 2B draft: 0.71 vs 0.71; with an 8B draft: 0.75 vs 0.74.
  - The paper notes α and walltime are higher with argmax decoding.
  - Source: [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
- Chen et al.: Chinchilla greedy XSum 2.01x vs nucleus 1.92x. Acceptance was "robust" to sampling parameters, and temperature sharpening helps on HumanEval. — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
- EAGLE-3 mean τ at T=0 vs T=1:
  - LLaMA-3.1-8B: 6.23 vs 4.92, with MT-bench 6.13 vs 4.24.
  - LLaMA-3.3-70B: 5.88 vs 5.66.
  - Vicuna-13B: 6.62 vs 5.67.
  - DeepSeek-R1-Distill-LLaMA-8B: 5.84 vs 4.89.
  - Source: [EAGLE-3](https://arxiv.org/html/2503.01840)
- DFlash, Qwen3-8B: average τ 6.49 at T=0 vs 5.48 at T=1, and AIME25 7.08 vs 4.73. — [DFlash](https://arxiv.org/html/2602.06036v1)
- Acceptance-aware training (tree, depth 7, 128 nodes): with a KL-trained drafter, T=1 τ is already above greedy (Qwen3-8B/0.6B 4.998 vs 4.494; Qwen3-32B/4B 5.960 vs 5.057). With the WTV loss it goes higher (5.750 and 6.671). In chain mode greedy wins for KL drafters (Qwen3-8B/0.6B 3.051 vs 2.823), but WTV at T=1 (3.169) slightly beats greedy. — [Xia et al. 2026](https://arxiv.org/html/2609.24150)
- Sampling acceptance per token is E[min(p,q)] = 1 − TV (Leviathan Cor. 3.6). Greedy acceptance requires the argmax to match and depends on the margin p(x0) − p(x1) (Sharma, Prop. 4.8 / Thm 4.9). Relaxed (lossy) criteria such as additive t, multiplicative α and top-m enlarge the acceptance region mainly on low-margin steps. — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192); [Sharma 2026](https://arxiv.org/html/2606.30265)
- Performance or Illusion runs everything at T=0, so the largest production study has no sampling data. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)

### Inferences
- Chain-equivalent α (my calculation):
  - EAGLE-3 LLaMA-3.1-8B MT-bench: about 0.90 at T=0 vs 0.79 at T=1.
  - EAGLE-3 LLaMA-3.3-70B mean: about 0.889 vs 0.878.
  - DFlash Qwen3-8B AIME25: about 0.876 vs 0.794.
  - Larger targets and drafters seem to close the T=0/T=1 gap.
- For very deep drafts, greedy is the more favourable regime when the target is confident, since sharpening helps. Its failures concentrate on near-ties, which is where a lossy relaxed rule (accept any token within t of the top-1) would buy the most per-position acceptance. That trades exactness for depth.
- The reversal in Xia et al.'s trees suggests that at T=1 a multi-candidate tree with probability-matched sampling can cover more of the target's acceptable mass than top-m greedy covers the single argmax. This is specific to their setup and contradicts the EAGLE-3 and DFlash direction, so treat it as unsettled.

### Gaps
- No study found that holds the drafter fixed and reports per-position acceptance curves out to depth 16–30 at both T=0 and T=1.

## 4. Why might 98% at depth 30 be fundamentally hard? Is there evidence of a "target self-agreement" ceiling?

### Takeaway
Six things work against 98% flat at depth 30:
1. Compounding: α^k.
2. Near-tie steps in the target, where any imperfect drafter flips the argmax.
3. Drafter capacity: KL-optimal is not acceptance-optimal unless the drafter can reproduce the target.
4. Train/inference mismatch: heads trained on ground-truth prefixes, then reused on their own outputs.
5. An entropy-based cap on tokens per iteration.
6. Verification and drafting costs that grow with depth or width.

No paper was found that directly measures acceptance against a "target self-agreement" ceiling. The closest evidence is that the same 235B model at T=0 gave 80 distinct completions out of 1,000, diverging at near-tie tokens, purely from batch-size-dependent numerics. Even a numerically perturbed copy of the target would therefore fall slightly below 100% under strict greedy matching.

### Cited Findings
- **Compounding and depth decay**:
  - Expected tokens is (1−α^{γ+1})/(1−α), capped at 1/(1−α). — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
  - accepted/(K+1) falls as K grows. — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
  - MTP decays 0.92 → 0.68 → 0.38 (GLM-4.5-Air). — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
  - DeepSeek-V3 second-token acceptance is 85–90%. — [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
  - In DFlash, early errors invalidate the rest of the block, and EAGLE-3's acceptance length "quickly saturates due to limited model capacity" as target information becomes "diluted" with depth. — [DFlash](https://arxiv.org/html/2602.06036v1)
- **Near ties**:
  - Greedy rejection is controlled by the target margin.
  - For Qwen3-4B on chat, the p5 of top-1 probability is 0.391 and p25 is 0.886, even though the median is 0.988.
  - Source: [Sharma 2026](https://arxiv.org/html/2606.30265)
- **Thinking Machines (Horace He et al., Sep 2025), "Defeating Nondeterminism in LLM Inference"**:
  - Qwen3-235B-A22B-Instruct-2507, prompt "Tell me about Richard Feynman", T=0, 1,000 completions of up to 1,000 tokens.
  - 80 unique completions; the most common appeared 78 times.
  - All 1,000 agreed on the first 102 tokens. At token 103, 992 continued "Queens, New York" and 8 "New York City".
  - Cause: kernels are not batch-invariant, and server load changes the batch size.
  - With batch-invariant kernels, all 1,000 were identical. The cost on Qwen3-8B: 26 s (default vLLM) → 55 s (unoptimised deterministic) → 42 s (with an improved attention kernel).
  - Source: [Thinking Machines blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- Chen et al. made the same point earlier: "Even with greedy sampling, a single token deviating due to numerics could result in two sequences diverging wildly." — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)
- **Capacity**:
  - α rises with draft size: LaMDA 100M → 8B drafts give 0.61 → 0.75; T5-small → large give 0.75 → 0.82. — [Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)
  - KL and acceptance optima coincide only with exact reproduction, and greedy-tailored losses add +0.01–0.04 τ. — [Xia et al. 2026](https://arxiv.org/html/2609.24150)
  - EAGLE-3's speedup scales with training data. — [EAGLE-3](https://arxiv.org/html/2503.01840)
  - Small draft models beat EAGLE heads in Performance or Illusion (75–81% on GSM8K). The authors attribute EAGLE's weaker acceptance to fine-tuning on a "relatively limited dataset". — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- **Train/inference mismatch**:
  - EAGLE's acceptance drops as more of its inputs are self-predicted; EAGLE-3's "training-time test" keeps it almost flat. — [EAGLE-3](https://arxiv.org/html/2503.01840)
  - DFlash trained at block 16 generalises to block 8, but not the other way round. — [DFlash](https://arxiv.org/html/2602.06036v1)
  - An MTP head trained for the first token and reused autoregressively decays steeply. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
- **Entropy**:
  - E[X] ≤ (μ + μ_(2))·log(P)/μ² + O(1). — [Pankratov & Alistarh 2025](https://arxiv.org/abs/2512.11718)
  - Lower output entropy predicts longer suffix-decoding runs (0.086 for AgenticSQL Extract vs 3.43 for WildChat). — [SuffixDecoding](https://arxiv.org/html/2411.04975v2)
  - The entropy–acceptance correlation is weak within domains (ρ −0.15 to −0.20). — [Abdellatif 2026](https://arxiv.org/abs/2604.14682)
- **Costs**:
  - Verification is 42–95% of runtime, and the acceptance rate collapses as trees widen (0.415 → 0.095 from k=3 to k=21). k=21 trees fall below 1x at batch 64. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
  - NVIDIA found AL keeps rising with draft length, but net speedup peaks at length 3 because of draft cost. — [NVIDIA blog](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)

### Inferences
- **Arithmetic** (my calculation from Leviathan Eq. 1):

  | α | tokens per verify, γ=16 | γ=30 | γ=64 | cap 1/(1−α) | P(all 30 accepted) |
  |---|---|---|---|---|---|
  | 0.98 | 14.5 | 23.3 | 36.6 | 50 | 0.545 |
  | 0.95 | 11.6 | 15.9 | 19.3 | 20 | 0.215 |
  | 0.90 | 8.3 | 9.6 | 10.0 | 10 | 0.042 |

  With draft cost c per token, α=0.98 and γ=30 give 23.3x at c=0, 17.9x at c=0.01, 14.5x at c=0.02 and 9.3x at c=0.05. A 30-deep draft therefore needs a drafter costing about 1–2% of a target step, or a parallel one-pass drafter like DFlash.
- **Near-tie budget**: 98% leaves a 2% error budget. In Qwen3-4B chat, about 5% of steps have top-1 ≤ 0.39. If a strong drafter matches the target's argmax on about 60% of those steps, they alone cost about 2% and use the whole budget, before any errors on confident steps. Flat 98% is therefore plausible on low-entropy workloads (code edits, structured or agentic text, copying) or with lossy relaxed acceptance at ties, and implausible on open-ended chat for an exact greedy verifier. This is illustrative and depends on the assumed 60%.
- **A rough "self-agreement" number from Thinking Machines**: only 78 of 1,000 runs matched the modal completion over up to 1,000 tokens. That implies per-token agreement with the mode of roughly 0.078^(1/L): about 0.9975 at L=1,000 and 0.995 at L=500. A drafter that is the target itself under different numerics (batch size or kernels, and plausibly a quantized copy) would sit around 99.5–99.7% per token on that prompt. A 98% learned drafter would be within about 1.5–2 points of this numerics ceiling. This is very rough: one prompt, unknown completion lengths, divergences clustered at a few near-tie positions.
- Since the verifier is already near-optimal (Yin; block verification), the remaining levers are drafter–target agreement (capacity, target-feature inputs, on-policy training), workload entropy, and lossy acceptance rules.

### Gaps
- No direct measurement of a "target self-agreement ceiling" was found, e.g. the per-token greedy acceptance of a quantized or numerically perturbed copy of the target used as its own drafter. Papers on quantized self-drafts (e.g., QSpec-style work) were not checked.
- The share of near-tie steps for 100B+ targets, or for code and agentic text, was not found. Sharma's statistics cover only Qwen3-1.7B/4B on UltraChat.

## 5. Highest single-user decode speed reported for 100B+ models, what made it possible, and why ~1,000 tok/s is rare

### Takeaway
Single-user speeds of about 1,000+ tok/s on 100B+ models have been reported, but almost only on wafer-scale SRAM hardware (Cerebras):
- Llama 3.1 405B, dense 16-bit: 969 tok/s (Nov 2024).
- Llama 4 Maverick: 2,522 tok/s, measured by Artificial Analysis (May 2025, secondary report).
- gpt-oss-120B: about 3,000 tok/s; Qwen3-Coder-480B: "up to 2,000" tok/s (Aug 2025, vendor claims).

GPUs have done it once: NVIDIA's 8-GPU DGX B200 reached 1,038 tok/s on Maverick (May 2025), which required EAGLE-3 speculation, FP8 and heavy kernel work. Other reference points:
- GPU per-user records for DeepSeek-R1 671B are about 250 tok/s.
- Apple Silicon (M3 Ultra) runs DeepSeek-R1 671B at 4-bit at about 16–20 tok/s.

The main obstacles are memory bandwidth per token, speculative speedups of only about 2–3x in practice (verification-dominated), and serving economics that favour batching.

### Cited Findings
- **Cerebras, Llama 3.1 405B (18 Nov 2024)**: 969 tok/s with a 1,000-token prompt, 16-bit weights, 240 ms TTFT, 539 tok/s at a 100K-token prompt. The blog compares this to 12x the best GPU result, 8x SambaNova and 75x AWS. It does not say how many systems were used or whether speculative decoding was involved. — [Cerebras blog (vendor claim)](https://www.cerebras.ai/blog/llama-405b-inference)
- **Cerebras, Llama 4 Maverick**: 2,522 tok/s against NVIDIA Blackwell's 1,038 tok/s in Artificial Analysis testing. — [CO/AI (secondary)](https://getcoai.com/news/ai-chip-startup-cerebras-outperforms-nvidias-blackwell-in-llama-4-test); press release (not fetched): [Cerebras](https://www.cerebras.ai/press-release/llama4PR)
- **Cerebras, Llama 4 Scout**: over 2,600 tok/s, "19x faster than the fastest GPU solutions" per Artificial Analysis (Apr 2025). — [AIwire (secondary)](https://dev.aiwire.net/2025/04/10/cerebras-claims-worlds-fastest-inference-for-meta-llama-4/)
- **Cerebras, gpt-oss-120B**: about 3,000 tok/s (Aug 2025). — [insideHPC (secondary, vendor claim)](https://insidehpc.com/2025/08/cerebras-reports-3000-tokens-per-second-inference-on-openai-gpt-oss-120b-model/)
- **Cerebras, Qwen3-Coder-480B**: up to 2,000 tok/s (1 Aug 2025, vendor) — [Cerebras blog](https://www.cerebras.ai/blog/qwen3-coder-480b-is-live-on-cerebras). An InfoWorld reviewer "not once" reached 2,000 tok/s and quotes another user seeing under 100 TPS on small tasks — [InfoWorld (secondary)](https://www.infoworld.com/article/4055909/).
- **NVIDIA, DGX B200 with 8 Blackwell GPUs (May 2025)**:
  - Over 1,000 TPS/user on Llama 4 Maverick (400B), measured independently by Artificial Analysis; Tom's Hardware gives 1,038, up from SambaNova's previous 792. 72,000 TPS per server at the highest-throughput setting. — [NVIDIA blog](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/); [Tom's Hardware (secondary)](https://www.tomshardware.com/tech-industry/artificial-intelligence/dgx-b200-blackwell-node-sets-world-record-breaking-over-1-000-tps-user)
  - What was needed:
    - FP8 for GEMMs, MoE and attention.
    - Fused kernels: FC13+SwiGLU, and AllReduce+RMSNorm+Quantize.
    - Programmatic Dependent Launch, CUDA Graphs with on-device verification, and an overlap scheduler.
    - An EAGLE-3 draft at length 3 (2.06x).
    - Overall about 4x over the prior Blackwell baseline, on 64 TB/s of aggregate HBM3e.
    - Source: [NVIDIA blog](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)
- **DeepSeek-R1 671B**:
  - NVIDIA DGX B200: over 250 tok/s per user (FP4, concurrency 1). — [NVIDIA blog](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance)
  - SambaNova: 198 tok/s on 16 SN40L chips (Feb 2025). — [SambaNova press](https://sambanova.ai/press/fastest-deepseek-r1-671b-with-highest-efficiency)
  - SambaNova later cited Artificial Analysis averages of about 250 tok/s per user, against 19 tok/s per user for GPU providers. — [SambaNova blog (vendor)](https://sambanova.ai/blog/only-inference-provider-with-high-speed-support-for-the-largest-models)
- **OpenAI GPT-5.3-Codex-Spark (12 Feb 2026)**: "more than 1000 tokens per second" on Cerebras WSE-3, research preview. Model size undisclosed. — [OpenAI](https://openai.com/index/introducing-gpt-5-3-codex-spark/); [Tom's Hardware (secondary)](https://www.tomshardware.com/tech-industry/artificial-intelligence/openai-lauches-gpt-53-codes-spark-on-cerebras-chips)
- **Apple Silicon**: DeepSeek-R1 671B, MLX 4-bit, M3 Ultra 512 GB: about 18.4 tok/s on a short prompt and 6.4 tok/s generation after a 13,140-token prompt. — [r/LocalLLaMA mirror (community, secondary)](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1j9jfbt/m3_ultra_runs_deepseek_r1_with_671_billion); [hardware-corner (secondary)](https://hardware-corner.net/mac-studio-m3-ultra-deepseek-llamacpp)
- **Cursor, 70B model (below 100B)**: about 1,000 tok/s through speculative edits on copy-heavy rewrites. — [Cursor blog](https://cursor.com/blog/instant-apply)
- **Arguments for why ~1,000 tok/s is hard**:
  - Cerebras' bandwidth argument: a 70B model at 16-bit needs about 140 GB moved per token, so 1,000 tok/s needs about 140 TB/s. An H100 has 3.3 TB/s; a WSE-3 has 21 PB/s of on-chip memory bandwidth. — [Cerebras press release (vendor)](https://cerebras.ai/press-release/cerebras-launches-the-worlds-fastest-ai-inference); [Hot Chips 2024 slides](https://www.hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf)
  - Speculative decoding gains are "highly workload-dependent". Verification dominates runtime, measured results fall well short of oracle bounds, and speedup shrinks as batch size grows. — [Liu et al. 2026](https://arxiv.org/html/2601.11580)
  - Speculative sampling can beat the bandwidth roofline for autoregressive decoding: Chinchilla 70B on HumanEval and greedy XSum. — [Chen et al. 2023](https://arxiv.org/abs/2302.01318)

### Inferences
- **Records by category**:
  - Highest reported for a model with 100B+ total parameters: about 3,000 tok/s (gpt-oss-120B, Cerebras, vendor claim).
  - Highest for a dense 100B+ model: 969 tok/s (Llama 3.1 405B).
  - Highest on GPUs: 1,038 tok/s (Maverick, with speculation).
  - From memory, not checked here: gpt-oss-120B has about 5.1B active parameters, Maverick and Scout about 17B, Qwen3-Coder-480B about 35B. The ~1,000+ tok/s results are therefore mostly on MoE models with small active parameter counts, or on SRAM hardware.
- **Roofline extension of Cerebras' arithmetic**:
  - 405B at 16-bit is about 810 GB per token, so 1,000 tok/s needs about 810 TB/s.
  - DGX B200's 64 TB/s caps dense 16-bit 405B decode at about 79 tok/s before speculation, ignoring KV cache and communication.
  - On bandwidth-bound hardware, the multiplier must come from tokens per verify. Going from a 50 tok/s baseline to 1,000 tok/s needs about 20+ accepted tokens per verify, i.e. α ≈ 0.97–0.98 at depth about 30 (table in section 4).
  - This is why "deep drafting at 97–98%" and "1,000 tok/s on bandwidth-limited hardware" are essentially the same problem.
- **How Cerebras does it**: Llama 3.1 405B at 16-bit (about 810 GB) cannot fit on one wafer's SRAM (about 44 GB per WSE-3 by Cerebras' public spec, not verified here). It must be pipelined across many wafers, and the speed comes from SRAM bandwidth rather than speculation, which the blog does not mention.
- **Economics**: providers optimise server throughput (72,000 TPS per server vs 1,000 TPS per user on the same box), and speculative gains fade at high batch sizes. Single-user 1,000 tok/s is therefore a niche product (Cerebras, Codex-Spark) rather than the GPU-serving default.

### Gaps
- Groq numbers for 100B+ models (e.g., Kimi K2, gpt-oss-120B) were not collected.
- Late-2025/2026 Cerebras or other records (e.g., GLM-4.x, Qwen3-235B) and any 2026 GPU records (GB300/Rubin) were not verified.
- The number of Cerebras systems used for 405B, and whether any Cerebras record uses speculative decoding, is not disclosed in the sources read.
- The Artificial Analysis primary pages were not fetched; the 2,522 and 792 figures come from secondary reports.
