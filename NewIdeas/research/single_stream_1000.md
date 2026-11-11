# Single-stream ~1,000 tok/s for a 125B MoE on an M5 Pro: literature sweep (2024 to Oct 2026)

τ = accepted tokens per big-model pass. Every figure is the authors' or vendor's own claim unless tagged **[3P]**, meaning a third party measured it. Derived numbers (TFLOP/s, exponents, budgets) are my arithmetic. An arXiv ID's first four digits give the month (2608 = Aug 2026).

## 0. The physics again, plus a second wall

- **Bandwidth.** Bytes per pass ≈ 3 GB + 1.0 GB·u. Here u is the expert-union size as a multiple of one token's expert set, and 1.0 GB ≈ (55 GB at ~3.5 bits − 3 GB dense)/51.2. Plain AR reads 4 GB per token, which caps it at ≤75 tok/s. Reaching 1,000 tok/s means **3.3 accepted tokens per GB read**; AR gets 0.25.
- **Compute.** Each verified row costs ≈24 GFLOP. BaseRT (2607.19438, Jul 2026) measured on an **M5 Pro**:
  - Q4 prefill of dense Qwen3.6-27B: 497 tok/s ≈ 27 TFLOP/s.
  - Qwen3-30B-A3B in 128-token chunks: 2,478 tok/s ≈ 16 TFLOP/s.
  - Decode: 105 tok/s.

  Scaled to a 12B-active model, that is ≈700–1,100 rows/s. So **1,000 tok/s sits on the FP16-path compute roofline even at 100% acceptance.** INT8 tensor paths look about 2× faster; that is an estimate from M5 Max reports.
- **[3P]** NeurIPS 2026 "Evaluating LLM Inference Optimizations on Apple Silicon": every speculative-decoding method it tested *reduced* single-request throughput on Apple.

## 1. Lossy / judged acceptance

| Work | Setting | τ / speed | Quality Δ | Mechanism |
|---|---|---|---|---|
| Judge Decoding 2501.19309 (Jan 2025, ICLR'25) | 8B→405B, 25-token drafts | **τ 19.7** vs 6.3 lossless; 3.9×, 129 tok/s on 8×H100 | "almost exactly preserved" on 5 benchmarks; HumanEval 86.6→80.4 if the judge never saw code | 16k-parameter linear head on target embeddings, trained on 500 labelled examples |
| AutoJudge 2504.20039 (Apr 2025) | 8B→70B | **>40 tokens/pass** on GSM8K; 22–25 on LiveCodeBench | ≤1 pt on GSM8K; ~2 pts on LCB. vLLM: 1.5× over SD at −2 to −4 pts. Offloaded: 1.3–2.0× at −0.6 to −5 pts | mined "important-mismatch" labels feed a logistic classifier; mining took >2,700 GPU-h (per JuDi) |
| JuDi 2601.04766 (Jan 2026, EMNLP'26) | — | matches or beats AutoJudge | — | training-free KL threshold |
| Fuzzy SD 2502.20704 (ACL-F'25) | 8B→70B | GSM8K τ 9.7→13.5; CSQA 2.0→8.1 | −2.0 pts | JS-divergence threshold |
| AdaptiveSpec 2609.02897 | EAGLE-3 on 8B models | τ 2.8→4.3 (up to 5.7) | 91–109% of the original accuracy kept | margin rule plus tree shaping |
| R2R 2505.21600 | R1-1.5B + R1-32B | big model on 11–15% of tokens; 2.8× | average 46 vs 50 | token router |
| SpecReason 2504.07891 | step-level | 1.4–3.0× | +0.4–9% | big model scores the small model's reasoning steps |

**[3P] cautions:**
- "Acceptance Is Not Speed" (2610.23818, Oct 2026) tested 12 target–drafter pairs. Training-free relaxed rules accept **1.34× more tokens per round but produce 1.54× longer completions**, and the rules that do cut verifier rounds typically lose a lot of accuracy.
- "Revisiting Lossy Verification" (2607.26627):
  - Typical acceptance lifts block efficiency 3.76→4.61 at −3.4 pts on MATH.
  - Judges give the best trade-off but behave non-monotonically.
  - For truncation rules, the gap grows with difficulty: +0.4 pp on GSM8K, +6.7 pp on AIME.

Only *trained* judges reach τ≈20–40, and only on math/code. AutoJudge's 40 tokens/pass became ≤1.5× in vLLM (≈2× offloaded), likely because its sequential 8B drafter dominated (my reading).

## 2. Parallel / block drafters

| Work | τ / speed | Mechanism |
|---|---|---|
| DFlash 2602.06036 (Feb 2026, ICML'26) | Qwen3-8B: **τ 6.5** average (7.9 on MATH, 4.2 on MT-Bench) vs EAGLE-3 at 3.0–3.4. Qwen3-Coder-30B-A3B: τ 8.1 on HumanEval, but only 3.5× vs 4.2× for dense 8B. Lossless. | 5-layer drafter, 16-token block in one pass; target features injected into every draft layer's K/V; 800k target-generated samples |
| DDTree 2604.12989 (Apr 2026) | **τ up to 10.7** (Qwen3-8B MATH; Coder-30B-A3B HumanEval), using 128–512-node trees. Its rerun of DFlash got 6.6 vs the paper's 6.5 [3P]. | best-first tree built from one diffusion pass |
| CaDDTree 2606.01813 | τ 10.4 at ≈221 nodes on average | picks the throughput-optimal budget each round |
| DFlare 2606.02091 / DFlash-2 (Inco, Aug 2026) / H-Spec 2609.24197 | +5–11% speed / +21% τ for +1.3% latency [vendor] / +5–13% τ | wider conditioning; path selector; no drafter-side KV cache |
| DSpark 2607.05147 (DeepSeek, Jul 2026) | DeepSeek-V4 in production: +60–85% per-user speed vs MTP-1. For reference, V3's MTP accepted its second token 85–90% of the time | parallel backbone plus a small sequential head; per-request verify length |
| EAGLE-3 2503.01840 | τ 6.2 (8B), 5.9 (70B) with 48–60-node trees | feature fusion |
| Orthrus 2605.12825 | frozen Qwen3-8B: 5.4 tokens/pass, 5.4×, lossless | light diffusion head reads the AR model's KV cache |
| Saguaro 2603.03251; DPara 2609.27396; Mirror-SD 2510.13161 (Apple) | +30% over SD; 3.2–3.5×; 2.8–5.8× on 14–66B | overlap drafting with verification; Mirror-SD splits GPU and NPU |

Lossless τ plateaus near 6–8 per 16-row chain; τ≈10 needs 128–512 rows.

## 3. Converting a pretrained AR model into a parallel decoder

| Work | Cost | Tokens/pass or speedup | Quality vs original |
|---|---|---|---|
| CLLMs 2403.00835 (ICML'24) | consistency fine-tune on Jacobi trajectories | 2.4–3.4× | ≈preserved |
| Jacobi Forcing 2512.14681 | self-distillation on the model's own trajectories | 4.5× tokens per iteration; 3.8–4.0× wall-clock | "minimal loss" |
| Set Block Decoding 2509.04185; "LLM Knows the Future" 2507.11851 | fine-tune / gated-LoRA MTP (8B) | 3–5× fewer passes; ≈5× on code/math, 2.5× on chat | "no loss" |
| DiffuLLaMA 2410.17891; Dream-7B 2508.15487 | <200B / ≈580B tokens | — | "competitive" |
| Fast-dLLM v2 2509.26328 (ICLR'26) | ≈1B tokens | 2.5× | claims parity; **[3P] −11 pts MATH-500** (Orthrus) |
| SDAR 2510.06303 | CPT+SFT, includes 30B-A3B | — | parity on 11 of 18 benchmarks |
| TiDAR 2511.08923 (MLSys'26) | 150B-token CPT (8B) | **8.25 tokens/pass**, 5.9× tok/s | "minimal loss" |
| Nemotron-Labs-Diffusion-8B (HF card, 2026) | — | 5.9× tokens/pass; 850–1,015 vs 253 tok/s (GB200) | "same accuracy" [vendor] |
| LLaDA2.0/2.1-flash 2512.15745 / 2602.08676 | Ling-flash-2.0 → 100B-A6B | 535 vs 256 tok/s; 2.1 adds token editing, 892 tok/s quantized | 73.2 vs its AR parent's 72.2 |
| **DiffusionGemma 2608.00146** (Aug 2026) | Gemma-4 26B-A4B, <10% of the AR training tokens | **19.7 tokens/pass** (256-token canvas, ≈12 steps); 1,479 vs 204 tok/s (303 with MTP), H100 | GSM8K −0.3, HumanEval −3.7, GPQA −6.6, **AIME −15.1**. Each canvas touches **84/128 experts per layer**, and the MoE kernel runs 4.3× slower per step |

**[3P]** In Hao AI Lab's accuracy-under-parallelism study, EAGLE-3 still beat every diffusion LLM.

## 4. Copy-heavy regimes

- **SuffixDecoding 2411.04975 (NeurIPS'25):** τ 6.3 on AgenticSQL (5.3×) and 7.8 on SWE-Bench (2.5×), but only 1.8 on Spec-Bench. Prompt-lookup decoding gets 2.4 / 3.2 on the same tasks.
- **EfficientEdit 2506.02780:** 10.4× on CanItEdit and 13.1× on CodeIF-Bench (reuses unchanged code, edit-specific drafter).
- **Cursor speculative edits:** ≈1,000 tok/s on a 70B model, ≈13× [vendor].
- **Morph:** n-gram drafting with k=64, ≈10,500 tok/s on B200 [vendor].

## 5. MoE verification cost

| Work | Result |
|---|---|
| MoE-Spec 2602.16052 | Fixed per-layer expert budget. On OLMoE a 127-token tree touches **54 of 64** experts per layer. +10–30% over EAGLE-3; quality 0 to −3 pts |
| EVICT 2605.00342 | Truncates the tree before verification. On Qwen3-30B-A3B: −74.7% tokens but only −32.5% experts; 1.21× |
| EcoSpec 2607.12696 | Draft selection that accounts for expert cost: up to 1.62× (DeepSeek-V3.1, Qwen3-235B, GPT-OSS-120B) |
| **AcceptMoE 2608.02989** (Aug 2026) | Commitment-weighted expert set that sizes itself: **23–34 of 128 experts** per layer for a 64-token tree. τ unchanged (4.39); **−0.27 pts** (worst −1.22); 1.29× in GPU memory, 2.06× offloaded. [3P] At the same budget, MoE-Spec loses −3.1% and vote-count selection −9.1%. With offloading, standard SD waited on experts 65–78% of the time and was slower than AR on 7 of 8 Qwen pairs |
| DES 2602.00879 | MoE diffusion LLMs: the expert union grows almost linearly with parallel tokens. One expert set per block cuts unique experts by 55% and latency by 38%, keeping 99% accuracy |
| Cascade 2506.20675 | SD on MoE can be up to 1.5× *slower*; Cascade switches speculation off when it would not pay |
| OEA 2511.02237 / XShare 2602.07265 | Batch-aware re-routing: −39% MoE-layer latency / −30% expert activations |

**Union data points.** OLMoE: 31 rows → 4.9× one token's expert set, 127 rows → 6.75×. DiffusionGemma: 256 contiguous rows → 10.5×. Implied exponents **0.39–0.46** vs your 0.67 (pools partly saturated); measure chains vs trees on your model.

## 6. ≥500 tok/s single-stream, ≥30B, ≤1 TB/s

**No published system meets this.** The nearest data points:

- **M5 Pro, measured (BaseRT):** Qwen3-30B-A3B 105 tok/s, Qwen3.6-35B-A3B 111, dense 27B 18.
- **500+ tok/s, but only on datacenter GPUs (≥3.35 TB/s) or unstated hardware:**
  - DFlash on gpt-oss-120b: 500–600 tok/s per user on B300 [vendor].
  - DiffusionGemma: 1,479 (25B).
  - Cursor: ≈1,000 (70B).
  - Mercury 2: ≈1,009 (model size undisclosed).
  - LLaDA2.0/2.1-flash: 535/892.
- **Negative:** community llama.cpp tests found A3B-MoE speculation slower than AR on an RTX 3090 [3P, anecdotal].

## 7. Skeleton-of-Thought family

| Work | Speedup | Quality |
|---|---|---|
| SoT 2307.15337 (ICLR'24) | >2× on 8 of 12 models (max 2.39×) | better on generic/knowledge/roleplay; worse on writing, Fermi, math, coding; a router recovers |
| APAR 2401.06761 | up to 2× (4× with Medusa) | — |
| PASTA 2502.11517 | 1.21–1.93× | LC win rate +2.2% to −7.1% |
| Hogwild! 2504.06261 | 32B: ≈2× with 2 workers, 3.2–3.6× with 4 | — |
| Multiverse 2506.09991 | up to 2× | AR-level AIME at 32B |
| ThreadWeaver 2512.07843 | up to 1.53× on average (3.56× at best) | 71.9 vs 72.2 |

## PRINCIPLES

1. **Optimise accepted tokens per GB and per TFLOP, not τ.**
2. **With natural routing (k^0.67), no pass width works.** To hit 1,000 tok/s:

   | Rows per pass | Bytes | τ needed |
   |---|---|---|
   | 16 | 9.4 GB | 31 (impossible) |
   | 32 | 13.2 GB | 44 (impossible) |
   | 64 | 19.2 GB | 64 (impossible) |
   | 256 | 44 GB | 147, plus ≈246 ms/pass of FP16-path compute |

3. **The dense part sets a floor.** 3 GB per pass allows at most 100 passes/s, so τ ≥ 10 even with zero expert traffic.
4. **The compute wall:** rows/s ≤ ≈1,000 on the measured FP16 path. That requires INT8-class tensor throughput (or fewer active FLOPs per row), **≤32–40 rows per pass, and ≥50–70% of rows accepted.** DDTree-style trees (τ≈10 at 128 nodes) would deliver ≈100 tok/s here.
5. **The drafter must be one pass per block, or n-gram/suffix.** A 2 ms budget for 32 sequential tokens leaves ≈20 MB of reads per token.
6. **Lossless ceiling here ≈200–400 tok/s** (DFlash-class τ 6.5–8 on 16 rows, uncapped vs capped union). 1,000 tok/s needs **τ≈17–25 on ≤32–40 rows**, which only copy regimes and judged acceptance have shown.
7. **Judge lossy schemes on whole answers:** answer-change rate, length inflation, hard reasoning.

**Worked operating point:**
- 32 rows per pass, union capped at 25 experts/layer (u=2.5): 5.5 GB, 18.3 ms.
- INT8 compute: 15.4 ms, overlapped with the reads.
- Drafter: 2 ms.
- Result: ≈49 passes/s, so **τ=20 gives ≈980 tok/s**.
- Variants: u=2 needs τ 18.7; u=4 needs 25.

## Most promising directions

1. **Verifier-side expert cap tuned for chains (AcceptMoE-style commitment weighting).** The prerequisite: a 32-row pass drops from 13.2 to ≈5–6 GB, likely for ≈0.3–1 pt (AcceptMoE). Measure your union exponent first.
2. **Target-conditioned block drafter (DFlash → DSpark/DFlash-2 lineage)**, block 32, with a semi-AR head and trained on the 125B model's own outputs. Target: lossless τ≈7–9 at ≈2 ms per block, which is ≈340–440 tok/s with the cap.
3. **Learned judge on top (AutoJudge / JuDi KL threshold / Judge-Decoding head), gated by domain**, to lift τ to ≈17–25 on code, math and structured text. Budget ≤1–2 pts, and track length inflation.
4. **Copy/suffix drafts merged into the same verify block.** This is the only regime with lossless τ in the tens; 1,000+ tok/s is plausible for edits and rewrites.
5. **Compute side:** W4A8 matmuls on the M5 neural accelerators, plus optionally a smaller top-k for verification rows. Both cut FLOPs and the expert union.

**De-prioritised:**
- **Converting the whole 125B model to diffusion:** DiffusionGemma-style hard-reasoning loss, a large training bill, and 256-row canvases at ≈6 TFLOP per step.
- **Wide trees.**
- **SoT-style branching:** ≤≈2× with quality risk, and unrelated branches widen the expert union faster than a single chain does.
