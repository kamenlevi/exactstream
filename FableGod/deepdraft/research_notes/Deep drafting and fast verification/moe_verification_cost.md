# How cheap can the verify pass be in speculative decoding on MoE target models (2023-2026)?

*Scope note (2026-11-14): figures come from arXiv HTML and abstract pages, Cohere's blog and the NeurIPS 2026 poster listing. A summarising fetch tool read them, so check any number that carries a decision against the PDF. Lines marked "my calculation" are my own arithmetic and appear in no source. Name map: **EVICT** is the method of arXiv 2605.00342 ("Making Every Verified Token Count"), and **EcoSpec** is the method of arXiv 2607.12696 ("Less Experts, Faster Decoding"). They are two papers, not four. Uniform-routing reference used throughout (the MoESD formula): N(t) = E·(1 - (1 - K/E)^t).*

**Key numbers (details and conditions in the sections below):**
- **Cheapest verify pass measured (Qwen3-30B-A3B, A100 SXM4 80GB, SGLang, batch 1):** 32 tokens cost 1.62x a one-token pass, 16 tokens 1.47-1.52x, 8 tokens 1.34-1.39x. — [EVICT](https://arxiv.org/html/2605.00342)
- **Where "20-31%" comes from:** Cohere's blog, on an unnamed 128-expert top-8 MoE at batch 1. A 4-token pass costs 1.25x a one-token pass. Its union is 20.36 experts per layer, against 25.4 if tokens routed independently and 29.1 under uniform routing. — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- **Union growth with depth (Qwen3-Coder-30B-A3B, A100):**

  | Tokens per verify | 2 | 4 | 9 | 11 |
  |---|---|---|---|---|
  | Measured unique experts per layer | 12.1 | 18.2 | 27.5 | 30.2 |
  | Uniform-routing prediction (my calculation) | 15.5 | 29.1 | 56.4 | 65.1 |

  - Cost per added expert rises about 3.2x (0.08 to 0.27 ms) once a pass has 5 or more tokens.
  - The offline-oracle speedup ceiling is about 2.34x. — [2609.22156](https://arxiv.org/html/2609.22156)
- **Union-only (dedup) kernels:** they run at 0.32-0.75 of the bandwidth of kernels that read each expert once per token, so they only win at 5-8 tokens per pass, or never. On llama.cpp Metal (M4 Max), real speculation on Qwen3-30B-A3B was slower than plain decoding: 0.877x at k=1 and 0.517x at k=8. — [2610.34440](https://arxiv.org/html/2610.34440)
- **No correlation savings on DeepSeek-V3.1:** a 4-token pass touches 31.4 experts per layer, about the uniform prediction of 30.5 (my calculation). — [EcoSpec](https://arxiv.org/html/2607.12696)
- **Is the union a proven floor?** Only by definition: Lean-checked positivity in 2610.34440, plus Nsight profiles showing a dedup kernel's DRAM reads equal the union within 0.2-2.5%. I found no formal I/O lower bound and no exact method that goes below the union. — [2610.34440](https://arxiv.org/html/2610.34440)
- **Methods:**

  | Method | Exact? | Reported result |
  |---|---|---|
  | EVICT | lossless | 1.21-1.25x over EAGLE-3 |
  | EcoSpec | lossless | up to 1.62x over plain decoding |
  | DraftExpert | lossless | with experts offloaded, a K=4 verify costs 2.35-3.30x a one-token decode step; truncation plus prefetch brings it to 1.25-1.40x |
  | AcceptMoE | lossy | 1.29x with all weights on the GPU, 2.06x with experts offloaded; -0.27 accuracy points |
  | MoE-Spec | lossy | +10-30% throughput; -4 to +7 points per task |

  — [EVICT](https://arxiv.org/html/2605.00342), [EcoSpec](https://arxiv.org/html/2607.12696), [DraftExpert](https://arxiv.org/html/2607.24434v1), [AcceptMoE](https://arxiv.org/html/2608.02989), [MoE-Spec](https://arxiv.org/html/2602.16052v1)

**Not covered (stopped at the time limit):**
- Full text of the Cohere NeurIPS 2026 paper (OpenReview blocked the fetch).
- Union-size tables for 16-32-token passes on 256-512-expert models.
- Full texts of MoE-SpeQ and SpecMoEOff.
- An official NeurIPS 2026 speculative-decoding tutorial site.
- MLX verify-cost data beyond 9 tokens.
- Verify-cost ratios for DeepSeek MTP in production serving.
- A PDF spot-check of the extracted numbers.

## Q1. What do the cost models and bounds say: MoESD, "The Limits of Speculation" (2609.22156) and "Expert Bytes, Kernel Bandwidth and the Best Draft Length" (2610.34440)?

### Takeaway
All three model verify time as a fixed part plus a part that grows with the distinct experts (bytes) the block touches. MoESD gives the independent, uniform-routing union formula and a saturation threshold, with no closed-form best draft length. 2610.34440 proves (partly in Lean) an exact condition for speculation to win, and shows that the best draft length is a monotone staircase in the acceptance rate. It also shows that union (dedup) kernels beat per-slot kernels only at 5-8 tokens per pass, or never, on today's llama.cpp CUDA/Metal and MLX-LM kernels. 2609.22156 gives an offline-oracle ceiling of about 2.34x for Qwen3-Coder-30B-A3B with EAGLE-3 on one A100, and measures a roughly 3.2x jump in per-expert verify cost once the verify microbatch reaches 5 tokens.

### Cited Findings
**MoESD (arXiv 2505.19645, NeurIPS 2025)**
- Expected number of activated experts for t tokens: N(t) = E·(1 - ((E-K)/E)^t). The paper treats each expert's activation as a Bernoulli indicator, assumes them i.i.d., and assumes uniform activation ("reasonable for well-trained MoE models"). It does not model load imbalance. — [MoESD v4](https://arxiv.org/html/2505.19645v4)
- Saturation threshold: T_thres = ceil(log_(1-rho)(1-tau)), with rho = K/E and tau "usually a large ratio such as 0.95". Past this point, extra verify tokens add little expert traffic. The paper gives no numeric T_thres for any model. — [MoESD](https://arxiv.org/html/2505.19645v4)
- Speedup model: Speedup = sigma(gamma+1) / [gamma·T_D(B,1)/T_T(B,1) + T_T(B,gamma)/T_T(B,1) + T_reject/T_T(B,1)], where sigma = [(1-alpha^(gamma+1))/(1-alpha)]/(gamma+1). It defines "target efficiency" = T_T(B,1)/T_T(B,gamma). There is no closed-form best gamma, and only gamma = 2, 3, 4 were tested. — [MoESD](https://arxiv.org/html/2505.19645v4)
- Measured results:
  - Qwen2-57B-A14B-Instruct with a Qwen2-0.5B draft reaches 2.29x (2xGPU-B, HumanEval, T=0, gamma=4). It reaches 2.18x on 2xGPU-A, 2.25x on 4xGPU-C and 2.08x on 4xGPU-A.
  - Mixtral-8x7B with an EAGLE head reaches 1.79x (2xGPU-A, HumanEval, T=0, gamma=4).
  - The GPU models are not named. Batch size was swept from 1 to 100.
  - MoE speculative speedup becomes "more pronounced" above batch 16, while dense (OPT-30B) speedups keep falling. — [MoESD](https://arxiv.org/html/2505.19645v4)
- The formula is checked only graphically (Fig. 1) on DeepSeek-V2-Lite-Chat and Qwen1.5-MoE-Chat. The paper gives no numeric error and no routing-correlation measurement. — [MoESD](https://arxiv.org/html/2505.19645v4)
- Later papers reuse the formula. XShare: DeepSeek-R1 (N=256, k=8) gives about 57 experts at batch 8 and about 163 at batch 32, which "closely match our empirical observations" (plain batches, no speculation). — [XShare 2602.07265](https://arxiv.org/html/2602.07265). EVICT restates it as N_uniq(t) = M[1-(1-K/M)^t]. — [EVICT 2605.00342](https://arxiv.org/html/2605.00342)

**NeurIPS 2026 follow-up to MoESD (Cohere authors)**
- "Why Speculative Decoding Works Better Than Predicted on Sparse MoE Models" (Ranjan, Teru, Venkitesh, Locatelli):
  - Covers 10 sparse MoEs, 30B to 1T parameters, 8 to 512 experts.
  - Prior work predicted a speedup peak at intermediate batch sizes. Here, recent fine-grained models show higher speculative speedup at low batch size.
  - The authors give three reasons: small per-expert weights leave bandwidth underused and raise L2 hit rates; routing overlaps across draft tokens; and GPU execution dynamics.
  - A maximum-entropy Iterative Proportional Fitting (IPF) model predicts unique-expert counts with "less than 4% error". — [NeurIPS poster page](https://neurips.cc/virtual/2026/poster/150811); full text at [OpenReview fC3FzfodMG](https://openreview.net/forum?id=fC3FzfodMG) (blocked by a bot check, not read)

**Lan, arXiv 2610.34440 (25 Oct 2026, sole author)**
- Cost of a pass of n tokens with kernel p: tau_p(n) = t0 + D(n)/beta_D + E_p(n)/beta_p(n) + phi_p(n). Here D(n) = D + kv(n-1) is the bytes read once per pass, and phi_p is fixed work such as kernel launches.
  - Per-slot expert bytes: E_slot(n) = m·n·W.
  - Dedup expert bytes: E_dedup(n) = sum_l w_l·U_l(n), the expected union.
  - Union under reuse: E_dedup(k+1) = mW + mWk(1 - rho_bar_k). — [2610.34440](https://arxiv.org/html/2610.34440)
- With geometric acceptance a, draft length k beats plain decoding if and only if V(k) - V(0) < (a + a^2 + ... + a^k)·V(0).
  - Speedup = G_a(k)·V(0)/V(k), with G_a(k) = 1 + a + ... + a^k.
  - Threshold for one draft token: a0 = V(1)/V(0) - 1. — [2610.34440](https://arxiv.org/html/2610.34440)
- The best draft length k*(a) never decreases as a rises: a monotone staircase. Jumps of more than one length are allowed, and no single-peak assumption is made.
  - Step points: a_j = max_{i<j} min_{k>=j} alpha_{k,i}.
  - No single rule of the form "rho_bar_k > c" decides the answer for every routing. — [2610.34440](https://arxiv.org/html/2610.34440)
- Dedup is faster than per-slot exactly when u(n) < r(n). Here u = E_dedup/E_slot is the unique share and r = beta_dedup/beta_slot is the bandwidth ratio. The saving is S(n) = (E_slot/beta_slot)·(1 - u/r). — [2610.34440](https://arxiv.org/html/2610.34440)
- Error bounds: a relative bandwidth error of at most epsilon gives a step-time error delta <= epsilon/(1-epsilon). "A 10% bandwidth error moves the predicted speedup by at most 25%." The best draft length is the least sensitive prediction. — [2610.34440](https://arxiv.org/html/2610.34440)
- Lean-checked modules include SpeedupCond, Staircase, StepPoints, UnionJump, MoreReuse, Positivity and ErrBounds. The time-form threshold and the epsilon-to-delta conversion were done by hand only. — [2610.34440 appendix](https://arxiv.org/html/2610.34440)
- Hardware and models, all at batch 1:
  - Apple M4 Max (546 GB/s) running llama.cpp Metal and MLX-LM.
  - RTX 4070 Ti SUPER (672 GB/s, 48 MiB L2) running a modified llama.cpp CUDA build.
  - Models: OLMoE-1B-7B-0125 (64 experts, top-8), Qwen3-30B-A3B (128, top-8), gpt-oss-20b (32, top-4), DeepSeek-V2-Lite (64 routed, top-6, plus 2 shared). — [2610.34440](https://arxiv.org/html/2610.34440)
- Validation:
  - Measured savings fall within the error bound at every CUDA and MLX-LM point, and at 36 of 48 Metal points.
  - The crossover is predicted within one pass size in 10 of 12 configurations.
  - On DeepSeek-V2-Lite (CUDA), timings taken before the speculative runs give the best draft length within one at all 22 acceptance values.
  - Measured CUDA speculative throughput over prediction has a median of 0.93-1.04 per configuration (range 0.74-1.18).
  - Fitting bandwidths on some models and predicting a held-out one misses by 9-206% (median 23% CUDA, 47% Metal). — [2610.34440](https://arxiv.org/html/2610.34440)

**"The Limits of Speculation", arXiv 2609.22156 (Amankulov, Mamatin)**
- Setup:
  - Speculation-budget choice is posed as an offline Stochastic Shortest Path over reference sequences, solved by an Oracle that simulates MoE verify cost counterfactually.
  - Qwen3-Coder-30B-A3B-Instruct (128 experts, top-8) with an EAGLE-3 draft (SpecForge).
  - One A100 80GB, bf16, B=1, full CUDA graphs, SpecBench. The Oracle analysis uses 80 math prompts. — [2609.22156](https://arxiv.org/html/2609.22156)
- Verify-latency model: T = beta0(m) + alpha(m)·U + c·L_ctx + eps, with m = K+1 tokens and U the unique experts per MoE layer.
  - alpha is about 0.078-0.084 ms per expert for m <= 4, and about 0.264-0.268 ms per expert for m >= 5 (about 3.2x higher).
  - c is about 1.1e-3 ms per context token.
  - In-sample R^2 is 0.9669-0.9991; out-of-sample MAPE is 0.26-0.59%.
  - Measured median latency drops from m=4 to m=5 (12.63 to 10.94 ms). This coincides with a change in beta0 and was not re-tested with a randomized sweep. — [2609.22156](https://arxiv.org/html/2609.22156)
- Draft cost: T_D(K) = 0.459K + 0.073 ms. — [2609.22156](https://arxiv.org/html/2609.22156)
- The Oracle "limit":
  - Speedup plateaus at about 2.34x beyond Lmax = 6.
  - Mean accepted length saturates at about 2.1.
  - The mean chosen draft length stays at or below about 2.8 even with Lmax = 12.
  - The authors call this an upper bound for reproducing the reference text, not a policy achievable online. — [2609.22156](https://arxiv.org/html/2609.22156)
- In "Delta Space", rejected candidates form a "strict linear boundary" through the origin. The authors read it as a local necessary condition on marginal cost against expected progress, Delta E[Cost]/Delta E[a]. No numeric slope is reported. — [2609.22156 abs](https://arxiv.org/abs/2609.22156)
- Per a search snippet of the PDF, the paper treats the expert union as an observed cost signal, not a causally identified traffic volume. — [2609.22156 PDF](https://arxiv.org/pdf/2609.22156)
- Metadata oddity: the abs page shows v1 dated 26 Aug 2026, but the ID prefix is 2609. — [2609.22156 abs](https://arxiv.org/abs/2609.22156)

**Other cost models**
- Cascade (arXiv 2506.20675):
  - Utility = gain in effective token rate (ETR) / (t_iter,spec / t_iter,base). Theorem 4.2: t_spec = t_base/U_spec.
  - It predicts speedup with R^2 = 99.4% over 120 data points. Speculation is turned off when U < 1. — [Cascade](https://arxiv.org/html/2506.20675v1)
- EVICT: verify cost C(k) is profiled offline into a lookup table, and the method picks k* = argmax E[A(T_k)]/C(k). The authors say this is equivalent to minimising time per output token. — [EVICT](https://arxiv.org/html/2605.00342)
- Cohere blog, Amdahl-style estimate: T(4)/T(1) of about 1.46x, from a routed-expert time fraction f = 0.30 and unique(4)/k = 2.55. The measured value is 1.25x, so the estimate is about 17% high. — [Cohere blog, 21 Apr 2026](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- Cohere dynamic speculative decoding (DSD) post:
  - For the MoE Command A+, the best K is non-monotonic in batch size: low at low batch, higher in the middle (DSD picks K=5 at BS 16-32, K=3 elsewhere), lower again at high batch.
  - For the dense Command A, the best K falls monotonically with batch size. — [Cohere DSD blog, 10 Jul 2026](https://cohere.com/blog/hardware-aware-dynamic-speculative-decoding)

### Inferences
- MoESD saturation points at tau = 0.95 (my calculation):

  | E/K | Tokens to saturate |
  |---|---|
  | 8/2 | 11 |
  | 64/8 | 23 |
  | 128/8 | 47 |
  | 256/8 | 95 |
  | 512/10 | 152 |

  So for fine-grained MoEs, a 16-32-token verify block is far from saturation. Under uniform routing the union is still growing fast at that size:

  | E/K | Union at 16 tokens | Union at 32 tokens |
  |---|---|---|
  | 128/8 | 82.4 (10.3x top-k) | 111.8 (14x top-k) |
  | 256/8 | 102.0 | 163.3 |
  | 512/10 | about 139 | about 240 |

  Measured routing correlation shrinks these a lot (see Q3).
- Cost model plus measured cost curve, not acceptance alone, sets the best block size. 2610.34440's inequality and staircase mean the measured V(k) curve decides k*. Where V(k) grows nearly linearly in k, as with per-slot kernels on Metal, speculation can lose even at good acceptance. This happened in practice (Q4).
- Reading of the alpha jump in 2609.22156 (my reading; the paper only reports the jump): it suggests a change of kernel configuration or regime in the SGLang MoE path once the batch reaches m >= 5. The marginal-expert cost measured at small m may therefore not carry over to 16-32-token blocks on the same stack.
- No paper derives a universal closed-form ceiling on speedup for MoE. The only numeric "limit" is the 2.34x oracle for one model, one GPU and one prompt subset.

### Gaps
- MoESD's numeric saturation thresholds, per-batch target-efficiency values and the identities of GPU-A/B/C were not available in the extracted text.
- The full text of the NeurIPS 2026 Cohere paper was not reachable (OpenReview bot check). Per-model unions, T(K+1)/T(1) per model and hardware remain unknown.
- No official NeurIPS 2026 speculative-decoding tutorial site was found. The nearest are a COLING 2025 tutorial proposal ([arXiv 2503.00491](https://arxiv.org/abs/2503.00491)) and an unconfirmed aggregator report of a NeurIPS Education Track tutorial with a "LosslessBench" harness ([Digg](https://digg.com/ai/rqh584h2)). Neither covers the MoE verify cost as far as the snippets show.

## Q2. Which methods cut verify cost, by how much, and are they exact or lossy?

### Takeaway
There are two families.
- **Lossless:** methods that change which tokens get verified, or how and when expert bytes are fetched. These are EVICT, EcoSpec, Cascade, DraftExpert, SP-MoE and dedup kernels. On a GPU with all weights resident they gain about 1.1-1.25x over a tuned EAGLE-3/MTP baseline. Under offloading they gain more (1.3-3.5x).
- **Lossy:** methods that limit the verifier to fewer experts than the natural union. These are MoE-Spec, AcceptMoE and XShare. They gain 10-30% with weights resident and about 2x under offloading. Reported accuracy cost ranges from -0.27 points (AcceptMoE mean) to several points per task (MoE-Spec: -4 to +7).

### Cited Findings
**Lossless (the target's verification rule is unchanged)**
- EVICT (arXiv 2605.00342; Pan, Tao, Wang, Zhao, Wang, Zhang; v1 1 May 2026, v2 5 Sep 2026):
  - Method: before verification, cut the draft tree down to its cost-effective prefix. No training, no hyperparameters.
  - Models: Qwen3-30B-A3B; Qwen3-235B-A22B (TP=8); Ling-flash-2.0 (103B total, 6.1B active, 256 experts, top-8; TP=4).
  - Setup: A100 SXM4 80GB, SGLang, BS=1. EAGLE-3 tree: steps=4, topk=8, draft_tokens=32. — [EVICT](https://arxiv.org/html/2605.00342)
  - Speed: 1.08-2.35x over AR. Over EAGLE-3: 1.21x mean on Qwen3-30B-A3B, 1.22x on Ling-flash-2.0, 1.25x on Qwen3-235B-A22B. Qwen3-30B-A3B at T=0: 176.34 to 202.82 tok/s. — [EVICT](https://arxiv.org/html/2605.00342)
  - Against EAGLE-3 on Qwen3-30B-A3B: verified tokens fall 74.7%, activated experts fall only 32.5%, verify latency falls 26.6%. — [EVICT](https://arxiv.org/html/2605.00342)
  - Mean accepted tokens: 3.65 to 3.09 (Qwen3-30B-A3B), 3.57 to 3.04 (Ling-flash-2.0, T=0). — [EVICT](https://arxiv.org/html/2605.00342)
  - "Remains fully lossless without modifying the target model's verification semantics." — [EVICT](https://arxiv.org/html/2605.00342)
- EcoSpec (arXiv 2607.12696; Xie et al.; 14 Jul 2026):
  - Method: draft score S(t) = P(t)/(DeltaCost(t|B) + eps), where DeltaCost is the number of the token's experts not already in the verify buffer. A light expert predictor supplies the routes; its top-K overlap is 80% (DeepSeek-V3.1), 82% (Qwen3-235B-A22B) and 93% (GPT-OSS-120B).
  - Setup: 8x H200, a HuggingFace Transformers prototype, BS=1, draft 3 steps x top-2, gamma = 4 including the bonus token. — [EcoSpec](https://arxiv.org/html/2607.12696)
  - Results, greedy (EAGLE-3 or MTP baseline to EcoSpec):

    | Model | Unique experts per layer per step | Verification HBM reads | Verify time per step |
    |---|---|---|---|
    | Qwen3-235B-A22B | 23.7 to 20.5 (-13.5%) | 99.3 to 88.1 GB (-11.3%) | 0.832 to 0.730 s |
    | GPT-OSS-120B | 11.6 to 10.6 (-8.6%) | 6.0 to 5.5 GB (-8.0%) | 0.113 to 0.090 s |
    | DeepSeek-V3.1 (MTP baseline) | 31.4 to 31.2 (-0.6%) | 97.3 to 96.8 GB (-1.0%) | 0.980 to 0.930 s |

    The aggregation unit for the HBM-read figures is not stated in the extracted text. — [EcoSpec](https://arxiv.org/html/2607.12696)
  - Speed over AR at T=0: Qwen3-235B 1.36x (EAGLE-3 1.22x), peak 1.62x on MT-Bench; GPT-OSS-120B 1.31x (1.14x); DeepSeek-V3.1 1.15x (MTP 1.10x). At T=1: 1.38x, 1.30x and 1.33x, against 1.28x, 1.18x and 1.28x. — [EcoSpec](https://arxiv.org/html/2607.12696)
  - Lossless by construction ("does not modify the target-model verification rule"). The extracted text reports no empirical check that outputs match. — [EcoSpec](https://arxiv.org/html/2607.12696)
- Cascade (arXiv 2506.20675):
  - Setup: 5 MoEs (Mixtral, Phi-3.5-MoE, OLMoE, DeepSeek-V1 16B, Qwen1.5-MoE) on one RTX 6000 Ada 48 GB, vLLM, single batch. n-gram drafting on all five; EAGLE on Mixtral only. Static K = 1-3.
  - Static speculation slowed 18 of 35 model-task pairs, by up to 1.5x. Worst static slowdowns: 26% (K=1), 38% (K=2), 54% (K=3).
  - Cascade limits the worst slowdown to 5%, beats static K by 7-14% in throughput, and reaches up to 1.6x over no speculation. — [Cascade](https://arxiv.org/html/2506.20675v1)
- DraftExpert (arXiv 2607.24434; 27 Jul 2026):
  - Method: the model drafts for itself with the shared expert, its top-1 routed expert and one distilled draft expert per layer. Verification adds expansion-aware truncation and prefetch.
  - Platforms (experts offloaded): RTX 4090 with routed experts in host memory; Snapdragon 8 Elite phone with routed experts in flash.
  - Lossless: "all emitted draft tokens remain verified by the frozen full MoE".
  - Decode speed 1.29-1.61x over AR with offloading (mean 1.45x).
  - Verify-side cost, as a multiple of one-token AR decode: fixed K=4 costs 2.35-3.30x; adding confidence/expansion truncation brings it to 1.48-1.76x; adding prefetch brings it to 1.25-1.40x (86-88% hit, 12-14% waste). — [DraftExpert](https://arxiv.org/html/2607.24434v1)
- SP-MoE (arXiv 2510.10302):
  - Method: SD-aware expert prefetch for offloaded MoEs.
  - Setup: Mixtral-8x7B, Phi-3.5-MoE and DeepSeek-Lite on RTX 3090, RTX 4090 and A100-40GB over PCIe 4.0, BS=1.
  - Expert loading is 69.4% of iteration latency on average. Top-1 expert prediction is about 88-89%.
  - TPOT 1.07-3.5x better than offloading+SD baselines.
  - Uses standard greedy accept/reject; the paper does not use the word lossless.
  - Per-expert sizes: Mixtral 336 MB, Phi-MoE 150 MB, DeepSeek-Lite 16.5 MB. — [SP-MoE](https://arxiv.org/html/2510.10302v2)
- MoE-SpeQ (arXiv 2511.14102): a quantized draft model predicts experts so they can be prefetched from host memory. Reported up to 2.34x over the best offloading framework for Phi-MoE. This comes from a search snippet only; a secondary review says 3.3x. — [MoE-SpeQ abs](https://arxiv.org/abs/2511.14102)
- Dedup (union) kernels (2610.34440): exact, and they request fewer bytes, but their effective bandwidth is lower (r = 0.32-0.75).
  - Measured crossover, where dedup becomes faster: CUDA OLMoE 5 tokens; CUDA gpt-oss 5-6; Metal OLMoE 7; Metal gpt-oss 6-8.
  - Never on Metal Qwen3 or MLX-LM OLMoE. — [2610.34440](https://arxiv.org/html/2610.34440)

**Lossy (the target's expert computation is changed)**
- MoE-Spec (arXiv 2602.16052; McDanel, Li, Surineni, Khaitan; 17 Feb 2026):
  - Method: per layer, sum router probabilities over all tree tokens and keep the top-B experts.
    - "Substitution" (default) re-routes each token to k experts inside that shortlist.
    - "Truncation" drops experts outside it.
  - Setup: A100 80GB (1 GPU for OLMoE and Qwen3-30B-A3B, 2 for Mixtral), FP16, single request, default tree of 63 tokens (swept 3-511). — [MoE-Spec](https://arxiv.org/html/2602.16052v1)
  - Results: "10-30% higher throughput" than EAGLE-3 at similar quality. Mean relative gain: Mixtral 27%, Qwen3 16%, OLMoE 6%.
    - Examples at T=0: MBPP on Mixtral 2.3x vs 1.7x; GSM8K on Qwen3 2.4x vs 1.9x.
    - Acceptance length falls 1.4% on average.
  - Accuracy change vs EAGLE ranges from -4 to +7 points per task (e.g. Qwen3 HumanEval T=0: -4; Mixtral HumanEval T=0: +7).
    - Aggressive budgets: Qwen3 keeps about 95% of AR quality at 2.0x; Mixtral keeps about 78% at maximum speedup. — [MoE-Spec](https://arxiv.org/html/2602.16052v1)
- AcceptMoE (arXiv 2608.02989; Liang, Chen, Mo, Wang, Li, Ma, Luk; 4 Aug 2026):
  - Method: the verifier ranks experts by router score times an offline "commitment" probability, and sizes its own expert set per block (per-pair mean 23-34 experts).
  - Models: Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B and GPT-OSS-120B. EAGLE-3 with 5 steps, topk 10, 64 draft tokens. SGLang 0.5.12.post1, BS=1.
  - Weights resident (RTX PRO 6000 Blackwell): 1.290x mean over standard SD (range 1.217-1.339x), up to 257 tok/s.
  - Offloaded (RTX 5090 32 GB, 48 expert slots per layer): 2.06x. — [AcceptMoE](https://arxiv.org/html/2608.02989)
  - Host-to-device traffic, MiB per output token:

    | Model | Plain AR | Standard SD | AcceptMoE | Cut vs SD |
    |---|---|---|---|---|
    | Qwen3-Instruct (BF16) | 888 | 2394 | 633 | -73.6% |
    | Qwen3-Coder (BF16) | 721 | 2025 | 464 | -77.1% |
    | GPT-OSS-120B (MXFP4) | 439 | 1243 | 298 | -76.0% |

    Standard SD spends 65-78% of wall time waiting for expert fetches; AcceptMoE spends 27-35%. — [AcceptMoE](https://arxiv.org/html/2608.02989)
  - Accuracy, mean over 12 model-task pairs:

    | Setting | Mean accuracy |
    |---|---|
    | Plain AR | 91.01 |
    | Standard SD | 90.90 |
    | AcceptMoE | 90.63 (-0.27 vs SD) |
    | MoE-Spec at the same budget | 88.15 |
    | Count-based selection | 82.72 |

    Per-pair change ranges from -1.22 to +0.54. Mean accepted length is about 4.39 for both AcceptMoE and standard SD. — [AcceptMoE](https://arxiv.org/html/2608.02989)
- XShare (arXiv 2602.07265; Vankov et al.; 6 Feb 2026):
  - Method: batch-aware expert selection; in speculative mode, selection is hierarchical and correlation-aware.
  - Setup: GPT-OSS-120B, EAGLE-3 length 3, BS=4, H100 (p5en.48xlarge, TP=1), vLLM.
  - Up to 14% throughput gain. One table entry shows +16.9% on IFBench.
  - Lossy. In one configuration accuracy changes by +1.67 / +2.80 / -0.36 points (AIME2025 / GPQA / AALCR). Without warm-up it falls by 20.83 / 3.89 / 19.61 points.
  - The text and tables disagree in places. — [XShare](https://arxiv.org/html/2602.07265)
- Speculating Experts (arXiv 2603.19289; 9 Mar 2026): runs predicted experts instead of re-fetching the router's choice. Up to 14% lower TPOT than on-demand CPU loading. Accuracy is "generally" kept, so the method is lossy. — [abs](https://arxiv.org/abs/2603.19289)
- TEAM (arXiv 2602.08404; ICML 2026), for MoE diffusion LMs: up to 2.2x with "negligible" degradation (lossy). — [TEAM](https://arxiv.org/abs/2602.08404)

### Inferences
- With weights resident on a strong EAGLE-3 or MTP baseline, lossless methods buy roughly 1.1-1.25x. The 2-3.5x figures appear only under offloading, where verify cost is mostly slow-link expert bytes.
- EVICT shows verify cost is far from linear in tokens. Cutting verified tokens by 74.7% removed only 32.5% of activated experts and 26.6% of latency. The union and the fixed per-pass work dominate, not the token count.
- Union-aware drafting barely helps on DeepSeek-V3.1 (-0.6% to -1.6% union) compared with Qwen3-235B (-13%). This matches the lack of measured overlap on DeepSeek-V3.1 at 4 tokens (Q3).
- When capping is lossy, how the cap is chosen matters more than how tight it is. At the same budget, acceptance-weighted selection (AcceptMoE) lost about 0.3 points against SD, while router-mass selection (MoE-Spec) lost about 2.8 points, per AcceptMoE's own comparison.

### Gaps
- No study compares all of these methods on the same hardware. AcceptMoE is the only one that benchmarks against MoE-Spec.
- EcoSpec ran in a HuggingFace prototype (0.8 s per verify step on 8x H200), so its gains may differ in optimised engines.
- MoE-SpeQ was seen only through its abstract snippet. I found no result for "SpecMoEOff".

## Q3. How much does routing correlation between adjacent tokens shrink the expert union? Is there data for 16-32-token verify blocks?

### Takeaway
The "20-31%" figure comes from Cohere's April 2026 blog, a secondary write-up of the authors' NeurIPS 2026 paper. On an unnamed 128-expert, top-8 MoE verifying 4 tokens, the union was 20.36 experts per layer. That compares with 25.4 under an independence baseline and 29.1 under uniform routing, about 20% and 30% lower. Primary papers show the shrink growing with block length: about 50% below uniform at 8-11 tokens on Qwen3-30B-A3B-class models, and 38% below uniform for a 31-token OLMoE tree. DeepSeek-V3.1 showed no shrink at 4 tokens. Union counts for 16-32-token blocks are rare.

### Cited Findings
- Cohere blog (Ranjan, 21 Apr 2026; vLLM, MT-Bench, unnamed 128-expert top-8 MoE, GPU not named):
  - Adjacent-token expert overlap at step 1 is 0.381 (about 3.0 of 8 experts). The independence baseline gives 0.118 and uniform gives 0.0625.
  - At "step 4" overlap is still 0.299, 2-4x the independence prediction.
  - Unique experts for K=3 (4 tokens): 20.36 measured, 25.4 independence, 29.1 uniform. The post says temporal correlation reduces unique experts by 20-31%.
  - A 4-token verify touches about 2.5x top-k, against 3.2-3.6x predicted.
  - Overlap is stable: 0.377-0.385 across 13 Spec-Bench categories and 0.378-0.386 across 7 languages. — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- NeurIPS 2026 poster: routing overlap across draft tokens shrinks the unique-expert count below the independence prediction, and the IPF model predicts the count with under 4% error. — [poster](https://neurips.cc/virtual/2026/poster/150811)
- 2610.34440 (teacher-forced WikiText, plus checks on verified text):
  - Share of a token's experts already used by the previous token, rho(2): 0.363 (OLMoE), 0.480 (Qwen3-30B-A3B), 0.492 (gpt-oss-20b). Uniform routing gives m/N = 0.125, 0.0625 and 0.125.
  - The same share at n=9: 0.762, 0.780 and 0.856.
  - Unique share u: 0.72-0.83 at n=2. At n=8: OLMoE 0.50, Qwen3 0.43, gpt-oss 0.36, DeepSeek-V2-Lite 0.62.
  - Verified text matches WikiText windows within 0.06. The exception is gpt-oss at depth 4096, which is 0.08-0.16 lower because its outputs repeat.
  - The RTX 4070 Ti SUPER and M4 Max route the same expert set on 0.947 of 11,845 (row, layer) pairs. — [2610.34440](https://arxiv.org/html/2610.34440)
- 2609.22156 (Qwen3-Coder-30B-A3B, A100, SpecBench, EAGLE-3), median unique experts per layer by verify size m:

  | m | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Unique experts | 12.1 | 15.5 | 18.2 | 20.6 | 22.6 | 24.5 | 26.1 | 27.5 | 29.0 | 30.2 |

  — [2609.22156](https://arxiv.org/html/2609.22156)
- EcoSpec (gamma = 4, BS=1), unique experts per layer:
  - Qwen3-235B-A22B (top-8 of 128): 23.7 greedy, 24.0 sampling. GPT-OSS-120B (top-4 of 128): 11.6 and 11.9. DeepSeek-V3.1 (top-8 of 256) with MTP: 31.4 and 31.4.
  - Several requests together (EAGLE-3 to EcoSpec):
    - Qwen3-235B: 23.7 to 20.5 (BS1), 44.6 to 36.1 (BS2), 54.9 to 45.7 (BS4; 16 tokens per pass).
    - GPT-OSS-120B: 11.6 to 10.6 (BS1), 22.3 to 19.5 (BS2), 32.6 to 28.1 (BS4). — [EcoSpec](https://arxiv.org/html/2607.12696)
- XShare (GPT-OSS-120B): expert overlap between consecutive speculative tokens is "often 2-3x larger" than between independent tokens. — [XShare](https://arxiv.org/html/2602.07265)
- MoE-Spec (OLMoE-1B-7B, 64 experts, top-8):
  - A 31-token EAGLE-3 tree loads 39 experts per layer.
  - A 127-token tree loads 54 of 64. The text also gives 54 at tree size 255.
  - At tree size 63, the top 32 experts hold 93% of routing weight.
  - Some expert pairs co-activate 10-33x (mean 23x) more often than uniform. — [MoE-Spec](https://arxiv.org/html/2602.16052v1)
- Cascade (Mixtral, top-2 of 8): for K=7 (8 tokens), uniform routing would give more than 7 unique experts (3.5x data movement). Math prompts gave about 3x, which the authors attribute to "expert affinity". — [Cascade](https://arxiv.org/html/2506.20675v1)
- DraftExpert (draft side, K=4, unique routed layer-experts drafted): top-1/2/3 routes give 84/183/244 for the "DS" model and 86/186/248 for "ML". — [DraftExpert](https://arxiv.org/html/2607.24434v1)

### Inferences
- Measured against the uniform formula (my calculation):

  | Source and model | Tokens | Measured | Uniform | Shrink |
  |---|---|---|---|---|
  | 2609.22156, Qwen3-Coder-30B-A3B | 2 | 12.1 | 15.5 | -22% |
  | 2609.22156, Qwen3-Coder-30B-A3B | 4 | 18.2 | 29.1 | -37% |
  | 2609.22156, Qwen3-Coder-30B-A3B | 6 | 22.6 | 41.1 | -45% |
  | 2609.22156, Qwen3-Coder-30B-A3B | 9 | 27.5 | 56.4 | -51% |
  | 2609.22156, Qwen3-Coder-30B-A3B | 11 | 30.2 | 65.1 | -54% |
  | 2610.34440, Qwen3-30B-A3B | 8 | 27.5 | 51.6 | -47% |
  | 2610.34440, OLMoE | 8 | 32.0 | 42.0 | -24% |
  | 2610.34440, gpt-oss-20b | 8 | 11.5 | 21.0 | -45% |
  | MoE-Spec, OLMoE tree | 31 | 39 | 63.0 | -38% |
  | MoE-Spec, OLMoE tree | 127 | 54 | about 64 | -16% (near saturation) |
  | EcoSpec, Qwen3-235B | 4 | 23.7 | 29.1 | -19% |
  | EcoSpec, GPT-OSS-120B | 4 | 11.6 | 15.3 | -24% |
  | EcoSpec, Qwen3-235B, BS4 | 16 (4 requests) | 54.9 | 82.4 | -33% |
  | EcoSpec, GPT-OSS-120B, BS4 | 16 (4 requests) | 32.6 | 51.0 | -36% |
  | EcoSpec, DeepSeek-V3.1 | 4 | 31.4 | 30.5 | +3% (no shrink) |

  So "20-31%" is a 4-token figure. The shrink against uniform keeps growing with depth in a chain.
- 20-31% basis (my arithmetic): Cohere's own K=3 numbers give 19.8% below independence and 30.0% below uniform. The post's exact basis for the 31% end is not clear in the extracted text.
- Marginal cost of a deeper chain (Qwen3-Coder, from the 2609.22156 table): each added token brings 3.4 new experts per layer at m=2 to 3, about 2.0 at m=5 to 6, and 1.2-1.5 at m=8 to 11. By about 10 tokens, roughly 81-85% of a new token's 8 experts are already loaded. Uniform routing would add about 4.2 new experts per token at that depth.
- The DeepSeek-V3.1 result (no shrink at 4 tokens) is a warning not to assume Qwen-like correlation for 256-expert models. The cause (group-limited sigmoid routing? MTP drafts?) is untested.
- Rough upper-bound estimate for a 32-token tree (my estimate; assumptions stated). EVICT's 1-to-32-token step adds 4.2 ms on an A100. At about 1.7-2.0 TB/s that buys at most 7-8.5 GB of extra bf16 expert reads. One Qwen3-30B-A3B expert across all 48 layers is about 453 MB (hidden 2048, expert FFN 768, from the public config, not fetched here). So the extra reads cover at most 16-19 experts per layer, and the union of EVICT's 32-token EAGLE-3 tree is at most about 24-27 experts per layer. Uniform routing predicts about 112. Tree siblings may route more alike than chain tokens.
- Extrapolation, untested: if the roughly 50% shrink seen at 8-11 tokens held for a 512-expert top-10 model, a 16-token block would touch about 70 experts per layer (uniform: about 139).

### Gaps
- No primary paper tables union size against block length for 16-32 tokens on 128-512-expert models. The Cohere NeurIPS full text might.
- No study separates overlap in tree blocks (siblings) from chain blocks (depth). The meaning of Cohere's "step 4" (lag or draft position) is unclear in the extracted text.
- 256-expert, DeepSeek-V3-class models at 16-32 tokens: the only data is XShare's formula estimate (about 163 experts at 32 tokens), said to match observation for plain batches, not speculative blocks.

## Q4. What is the best verify cost per token, or per pass relative to a one-token decode step, reported for MoE, and on what hardware?

### Takeaway
The best primary-source figure is EVICT's profile on Qwen3-30B-A3B (A100 SXM4 80GB, SGLang, BS1). Verifying 32 tokens costs 1.62x a one-token pass, about 0.05 decode steps per verified token. 16 tokens cost 1.47-1.52x and 8 tokens 1.34-1.39x. Cohere reports 4 tokens at 1.25x at BS1 on an unnamed 128-expert model and GPU. Bad cases are much worse:
- Mixtral 8-token verify costs 2.3-3x.
- Offloaded K=4 verify costs 2.35-3.30x before mitigations.
- On llama.cpp Metal (M4 Max), real speculation on Qwen3-30B-A3B was a net slowdown.

### Cited Findings
- EVICT Table 4 (Qwen3-30B-A3B, A100 SXM4 80GB, SGLang, BS1), mean verify latency:

  | Verified tokens k | QA (ms) | HumanEval (ms) | Ratio to k=1 |
  |---|---|---|---|
  | 1 | 6.83 | 7.04 | 1.00 |
  | 8 | 9.17 | 9.75 | 1.34 / 1.39 |
  | 16 | 10.01 | 10.70 | 1.47 / 1.52 |
  | 32 | 11.06 | 11.40 | 1.620 / 1.619 |

  — [EVICT](https://arxiv.org/html/2605.00342)
- Cohere (vLLM, unnamed 128/8 MoE, unnamed GPU):
  - Target time T(4)/T(1) is about 1.25 at BS1 and 1.54 at BS2.
  - BS1 speculative speedup 1.95x with acceptance length 2.73.
  - Drafting costs about 15% of one target step (14.3% profiled).
  - Non-expert kernels take 2-10 microseconds each at BS1, of which 2-3 microseconds is launch overhead. — [Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding)
- 2609.22156 (Qwen3-Coder-30B-A3B, A100, BS1, about 600-token context): median verify time is 8.98 ms at m=2 and 13.67 ms at m=11 (1.52x). No m=1 row is reported. — [2609.22156](https://arxiv.org/html/2609.22156)
- EcoSpec (Qwen3-235B-A22B, 8x H200, HF prototype, gamma = 4): T_AR/T_verify is 0.59 at BS1, so verify costs about 1.7x an AR step. The ratio falls to 0.32 at BS8. — [EcoSpec](https://arxiv.org/html/2607.12696)
- Cascade (Mixtral FP8, RTX 6000 Ada, vLLM, BS1): at K=7, verify overhead is about 3x (math), 2.3x (code) and 2.6x (extraction). For dense LLaMA-3-8B it is at most 3%. — [Cascade](https://arxiv.org/html/2506.20675v1)
- DraftExpert (offloaded, RTX 4090 host memory or phone flash): fixed K=4 verify costs 2.35-3.30x a one-token AR decode. Its best pipeline brings that to 1.25-1.40x. — [DraftExpert](https://arxiv.org/html/2607.24434v1)
- AcceptMoE (offloaded, RTX 5090): standard SD moves 2.7x the host-to-device bytes per output token of plain AR (2394 vs 888 MiB, Qwen3-Instruct BF16). AcceptMoE (lossy) moves 633 MiB, 0.71x of AR. — [AcceptMoE](https://arxiv.org/html/2608.02989)
- 2610.34440, Apple M4 Max, llama.cpp Metal, Qwen3-30B-A3B Q4_K_M with a Qwen3-0.6B draft, 37 held-out MT-Bench prompts:
  - Median speedup 0.877 at k=1 and 0.517 at k=8. Plain decoding was fastest on 33 of 37 prompts.
  - At n=8 on Metal with Qwen3, the per-slot kernel ran at 280 GB/s and the dedup kernel at 98 GB/s.
  - Re-reading an already-read expert costs 0.88-1.04 of a fresh read on Metal, 0.27-0.40 on MLX-LM and 0.18-0.34 on CUDA. — [2610.34440](https://arxiv.org/html/2610.34440)
- 2610.34440, CUDA (RTX 4070 Ti SUPER): per-slot effective bandwidth reaches 933-1018 GB/s at n=8. It exceeds the 672 GB/s DRAM peak from n=4, because the 48 MiB L2 serves the repeat reads. — [2610.34440](https://arxiv.org/html/2610.34440)

### Inferences
- Verify cost per verified token, in one-token-step units (my calculation): EVICT 32 tokens about 0.051; EVICT 16 tokens about 0.092-0.095; Cohere 4 tokens about 0.31. In EVICT, tokens 17 to 32 cost about 0.066 ms each, about 1% of a decode step.
- These low ratios hold for small-active (about 3B) MoEs on HBM GPUs at BS1. There, a one-token step is far from bandwidth-bound: EVICT's 6.83 ms compares with about 3.2 ms to stream about 6.6 GB of bf16 active weights at 2.04 TB/s (my estimate). A verify block mostly fills idle bandwidth. Where the step is already near the byte limit (offloading, larger experts, bandwidth-saturated serving), published ratios are 1.7-3.3x.
- On Apple Metal per-slot kernels, a re-read costs about as much as a fresh read (0.88-1.04), so verify cost grows almost linearly in n. That is the stated mechanism behind the net slowdown there. Good gather/dedup kernels on unified-memory machines are therefore a lever in their own right.
- For SSD- or host-offloaded verification, the evidence says to price the pass on unique slow-link bytes per accepted token. Prefetch plus truncation, both lossless, took fixed-K verify from about 2.4-3.3x down to 1.25-1.40x of an AR step.

### Gaps
- I found no primary measurement of verify cost at 16-32 tokens for 256-512-expert models. For Apple MLX the data stops at n=9.
- I collected no public per-pass verify ratios for DeepSeek-V3/R1 MTP in production expert-parallel serving.
- Cohere's GPU is not named, and its 1.25x figure comes from a blog, not a paper.

## Q5. Is there a proof that an N-token verify pass cannot cost less than reading the union of its experts, and does anything beat it?

### Takeaway
I found no formal I/O lower-bound proof for MoE verification. The union floor appears only as a definitional bound: in 2610.34440, Lean-checked positivity gives V_dedup(k) >= E_dedup(k+1) >= mW and E_slot >= E_dedup. The paper also measures that a dedup kernel's DRAM reads equal the union bytes within 0.2-2.5%. Nothing exact goes below the union's DRAM bytes. Methods get under it only by:
- verifying fewer experts (lossy);
- verifying different or fewer tokens (lossless, but that is a different union);
- keeping experts resident across passes (cuts slow-link bytes, not the union);
- storing fewer bytes per expert (quantization).

### Cited Findings
- 2610.34440, floors: the union of k+1 sets contains the first token's m experts, so E_dedup(k+1) >= mW > 0. Also V(0) >= mW/beta(1), V(k) >= mW/B_k, V_dedup(k) >= E_dedup(k+1) >= mW, and E_slot - E_dedup >= 0 (reuse can only reduce requested bytes). Lean modules: Positivity, ReuseUnion, MoreReuse. — [2610.34440](https://arxiv.org/html/2610.34440)
- 2610.34440, measurement: over 70 Nsight Compute profiles on CUDA, the dedup kernel reads the union bytes U·b from DRAM within 0.2-2.5%. The per-slot kernel reads between U·b and 2.35·U·b. The author says the union "is as good as the routing it is given". — [2610.34440](https://arxiv.org/html/2610.34440)
- 2610.34440: even so, per-slot kernels beat dedup kernels in every measured pair at n=2 and n=8 (r = 0.32-0.75). On CUDA, L2 pushes per-slot effective bandwidth above the DRAM peak. Requested bytes are not DRAM bytes. — [2610.34440](https://arxiv.org/html/2610.34440)
- EcoSpec: the verify footprint at layer l is the union of S_l(x_t) over the block, "not the token count". Verify latency scales "linearly with the number of active experts" (Fig. 1a caption). — [EcoSpec](https://arxiv.org/html/2607.12696)
- 2609.22156: latency fitted on the layer-averaged union reaches R^2 up to 0.9991. The authors treat the union as an observed signal, not a causally identified traffic volume. — [2609.22156](https://arxiv.org/html/2609.22156)
- Going below the natural union by changing the computation (lossy): MoE-Spec's budget B ([MoE-Spec](https://arxiv.org/html/2602.16052v1)), AcceptMoE's self-sized set ([AcceptMoE](https://arxiv.org/html/2608.02989)) and XShare's batch or speculative selection ([XShare](https://arxiv.org/html/2602.07265)).
- Residency instead of restriction: AcceptMoE's residency-aware pruning under offloading cuts expert-weight traffic 38.6-48.6%, raises the hit rate 6.9-8.2 points and throughput 4.6-15.1%, for -0.27 points accuracy. — [AcceptMoE](https://arxiv.org/html/2608.02989)
- Lossless prefetch: DraftExpert hits 86-88% ([DraftExpert](https://arxiv.org/html/2607.24434v1)); SP-MoE predicts the top-1 expert about 88-89% of the time ([SP-MoE](https://arxiv.org/html/2510.10302v2)).
- Fewer bytes per expert: AcceptMoE notes GPT-OSS-120B ships packed MXFP4 and Qwen ships BF16, so their byte counts are not comparable. One DeepSeek-V3.1 FP8 expert is 44.04 MB (EcoSpec). — [AcceptMoE](https://arxiv.org/html/2608.02989); [EcoSpec](https://arxiv.org/html/2607.12696)

### Inferences
- For exact verification with weights not already on chip, the slow-memory floor per pass is the union bytes of the experts the block actually routes to. One pass can go below it only by reusing bytes cached from earlier passes. So the meaningful floor is cumulative unique expert bytes per accepted token over a window of passes, not per pass. This matters for offloaded or SSD-streamed setups with a resident expert cache.
- The time floor is union bytes divided by the bandwidth a dedup or gather kernel actually achieves. Today that bandwidth is 32-75% of per-slot bandwidth, so below 5-8 tokens the union "floor" in time sits above the per-slot time. Better dedup or gather kernels are an open, exact lever.
- Exact ways to reduce union bytes per accepted token:
  - choose drafts that share experts (EcoSpec);
  - drop low-value branches (EVICT);
  - use longer, more correlated chains (the shrink grows with depth, Q3);
  - share passes across requests.
  None goes below the union of the tokens actually verified.

### Gaps
- My searches found no paper with a formal I/O lower bound (for example a red-blue pebble argument) specific to MoE verification, beyond the definitional floor above.
- I found no analysis relating on-chip cache size (GPU L2, Apple system cache) to the union for reuse across passes.
- I found no method that claims exact verification below the union bytes of the verified tokens.
