# Fastest local (batch-1) decode of large MoE LLMs on Apple Silicon and edge devices: expert offloading to SSD/flash, speculative verification, low-bit verification (2023-2026)

Scope note: research pass done 2026-11-14. "Primary" = paper, first-party repo/README, or vendor research page that ran the test itself. "Secondary" = press, blogs, aggregators, community posts or search-result summaries. Byte-per-token and bandwidth-fraction numbers in the Inferences sections are my own arithmetic, and their assumptions are stated.

Naming note: **MoE-SpeQ and Specter are the same paper** (arXiv 2511.14102). v1 (18 Nov 2025) was titled "MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts". v3 (22 Oct 2026) was retitled "Specter: Fast MoE Inference via Quantization-Enabled Multi-Step Speculative Offloading" and accepted to ASPLOS '27. "SpecMoEOff" is arXiv 2508.21706, "Accelerating MoE Inference by Hiding Offloading Latency with Speculative Decoding". Sources: [arXiv 2511.14102 abs](https://arxiv.org/abs/2511.14102), [arXiv 2511.14102v1 (MoE-SpeQ title)](https://arxiv.org/html/2511.14102v1), [arXiv 2508.21706](https://arxiv.org/abs/2508.21706).

**KEY NUMBERS FIRST** (details, settings and caveats are in the sections below):

- **Fastest single-Mac 100B+ MoE decode, no speculation:**
  - gpt-oss-120b (MXFP4) at **87.9 tok/s** on an M5 Max 128 GB (614 GB/s), MLX, 4K context. [Hardware Corner, secondary](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
  - Same model at **79.68 tok/s** llama.cpp tg128 on an M2 Ultra 192 GB. [llama.cpp #15396, primary](https://github.com/ggml-org/llama.cpp/discussions/15396)
  - Qwen3.5-122B-A10B 4-bit at **65.9 tok/s** on the M5 Max. [Hardware Corner](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
  - DeepSeek-V3/R1 671B 4-bit MLX at about **18-21 tok/s** on an M3 Ultra 512 GB at short context, about 6 tok/s at 13-16K (community). [Hardware Corner](https://hardware-corner.net/studio-m3-ultra-running-deepseek-v3)
- **Mac cluster:** an 816B-total / 42B-active NVFP4 MoE at **36.73 tok/s** on 4× M5 Ultra (tensor parallel, RDMA over Thunderbolt 5). M5 Ultra decode is 1.6× the M3 Ultra (1.2 TB/s vs 819 GB/s). [Apple ML Research, Oct 2026, primary](https://machinelearning.apple.com/research/mlx-distributed-inference)
- **SSD expert streaming (PagedMoE, M5 Pro 48 GB, ~31-33 GB peak):**
  - Kimi-K2 (578 GB) at **8.8 tok/s** with prune 0.8; Qwen3-235B at **7.9** with prune 0.7; Qwen3-Coder-Next at **20.6** exact.
  - A separate draft model was a **2-3× regression** (verify batches thrash the expert cache). n-gram/self-draft verify gave up to **2.26×**.
  - [README, primary, author-reported](https://github.com/noahclark556/paged-moe)
- **Phones:**
  - BigMoMo: **74.1 ms/token (≈13.5 tok/s)** on a OnePlus 13 (Snapdragon 8 Elite, 24 GB) with Q4_0 21-30B MoEs and D=2/B=2 trees. That is 4.83× over on-demand offloading and 1.76-1.82× over the best speculation-aware baseline. [arXiv 2609.14643](https://arxiv.org/html/2609.14643v2)
  - PowerInfer-2: a 47B model at **11.68 tok/s**. [arXiv 2406.06282](https://arxiv.org/abs/2406.06282)
- **Speculation + offloading:**
  - Specter (same paper as MoE-SpeQ): **1.50× average, 1.88× max** TPOT over SpecMoEOff. It uses an INT4 GPTQ self-draft with K=8-16 and **96.63% acceptance** on an A100-40GB over PCIe 4. [arXiv 2511.14102v3](https://arxiv.org/html/2511.14102v3)
  - SpecMoEOff: up to **2.5×**, but in a large-batch setting (Mixtral FP16, EAGLE draft, 1-10 drafts). [arXiv 2508.21706v2](https://arxiv.org/html/2508.21706v2)
  - SP-MoE: **1.07-3.5×** TPOT. [arXiv 2510.10302](https://arxiv.org/abs/2510.10302)
  - SpecMoE: up to **4.30×**. [arXiv 2604.10152](https://arxiv.org/abs/2604.10152)
- **Speculation can hurt MoE decode, and the fixes:**
  - Cascade: up to **1.5× slowdown** from 2-3× more expert data movement. [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
  - MoE-Spec: a fixed per-layer expert budget gives **+10-30% over EAGLE-3**. [arXiv 2602.16052](https://arxiv.org/abs/2602.16052)
  - EcoSpec: cost-aware drafting gives **up to 1.62×** on DeepSeek-V3.1 671B, Qwen3-235B and gpt-oss-120b. [arXiv 2607.12696](https://arxiv.org/abs/2607.12696)
- **Low-bit and quality:**
  - QSpec (W4A4 draft, W4A16 verify): **1.64×** with no quality loss. [arXiv 2410.11305](https://arxiv.org/pdf/2410.11305)
  - QuantSpec: **>90% acceptance**, about **2.5×**. [arXiv 2502.10424](https://arxiv.org/pdf/2502.10424)
  - HOBBIT: **≤1% accuracy loss** with ~30% of experts at INT4/INT2 and 3% skipped. [arXiv 2411.01433v2](https://arxiv.org/html/2411.01433v2)
  - BigMoMo expert pruning: −0.05 to −0.18 points.
  - No paper measures **verifying** a 100B+ MoE with a lower-bit copy.
- **Bandwidth:** no published percent-of-peak for MoE decode on Apple Silicon. My estimates from measured runs: dense decode ≈77-89% of nominal (from Apple's own table), MoE ≈35-60%, tensor-parallel cluster ≈18% per Mac. That leaves roughly 1.5-2.5× of MoE kernel headroom.
- **Above 200 tok/s for a 100B+ model:** only gpt-oss-120b on one RTX PRO 6000 96 GB: **221.3 tok/s** at 4K ([Hardware Corner](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)) and **196.31** llama.cpp tg128 ([#15396](https://github.com/ggml-org/llama.cpp/discussions/15396)). Nothing found on a Mac or phone.

**NOT COVERED in this pass** (time limit; no further deep dives):
- Full texts of SP-MoE, SpecMoE, EcoSpec and MoE-Spec (abstracts only, so no hardware, budgets or draft lengths).
- MoE-SpAc.
- The hardware used in "LLM in a flash".
- Metal-level kernel profiles (MLX gather_qmm, llama.cpp mul_mat_id).
- Measured M5 Ultra single-machine numbers.
- Multi-GPU consumer rigs.
- gpt-oss-120b with EAGLE-3/MTP on consumer GPUs or Macs.
- Primary-source Kimi K2 Mac benchmarks.
- BigMoMo per-model tok/s.

---

## 1. Fastest reported local decode speeds for large MoE models on Apple Silicon (MLX, llama.cpp, PagedMoE, others) and on phones/edge

### Takeaway
On one Mac with the model fully in unified memory, the fastest measured decode for a 100B+ MoE is about **80-88 tok/s**: gpt-oss-120b (MXFP4) on an M5 Max 128 GB (MLX) or an M2 Ultra 192 GB (llama.cpp). Qwen3.5-122B-A10B 4-bit reaches about **66 tok/s** on the M5 Max. DeepSeek-V3/R1 671B 4-bit reaches about **18-21 tok/s** on a 512 GB M3 Ultra. An 816B-A42B NVFP4 MoE reaches **36.7 tok/s** across four M5 Ultras linked by RDMA (Apple, Oct 2026). None of these headline numbers uses speculative decoding.

With SSD expert streaming (PagedMoE on a 48 GB M5 Pro, about 31-33 GB peak), 132-578 GB checkpoints decode at about **6-9 tok/s**, but only with lossy pruning. An 80B-class model decodes at about 20 tok/s exactly.

On phones, the best published MoE decode is about **13.5 tok/s**: BigMoMo, Q4_0 21-30B MoEs on a 24 GB OnePlus 13. PowerInfer-2 runs a 47B sparse Mixtral at **11.68 tok/s**.

### Cited Findings

#### Apple Silicon, whole model in unified memory (single machine)
- **gpt-oss-120b, MXFP4 MoE, llama.cpp Metal** (`llama-bench`, ub 2048, flash-attn on), **M2 Ultra 192 GB, 76-core GPU**:
  - tg128 = **79.68 ± 0.12 tok/s**.
  - pp2048 = 1244.57, pp8192 = 1101.31, pp16384 = 955.41, pp32768 = 752.31 tok/s.
  - No speculative decoding.
  - Source: [llama.cpp discussion #15396, "guide: running gpt-oss with llama.cpp", ggerganov, 18 Aug 2025 (primary)](https://github.com/ggml-org/llama.cpp/discussions/15396)
- **gpt-oss-20b** (below the 30B range; reference point), same harness, tg128:
  - M3 Ultra 512 GB / 80-core GPU: **115.52**
  - M2 Ultra 192 GB / 76-core: **116.08**
  - M4 Max 36 GB: **92.36**
  - M1 Max 64 GB: 75.15
  - M1 Pro 32 GB: 45.68
  - M3 Ultra pp2048 = 2816.47 tok/s.
  - Source: [llama.cpp #15396 (primary)](https://github.com/ggml-org/llama.cpp/discussions/15396)
- **M5 Max MacBook Pro 14"** (128 GB, 40-core GPU, 18-core CPU, **614 GB/s**), MLX `mlx_lm`. Secondary source: the article credits Reddit user cryingneko for the M5 Max runs. Decode / prefill tok/s at 4K, 16K and 32K context:
  - **gpt-oss-120b**, "gpt-oss-120b-MXFP4 in 8-bit quantization" (64 GB): decode **87.9 / 76.0 / 64.5**; prefill 1325 / 2710 / 2537.
  - **Qwen3.5-122B-A10B 4-bit** (69.6 GB): decode **65.9 / 60.6 / 54.9**; prefill 881 / 1239 / 1068.
  - **Qwen3-Coder-Next 8-bit** (84.7 GB): decode **79.3 / 74.3 / 68.6**, and 48.2 at 64K. Peak memory 87.1-92.6 GB. The article does not give parameter counts.
  - No speculative decoding is mentioned.
  - Source: [Hardware Corner, "Apple M5 Max for Local LLMs: First Benchmarks vs RTX Pro 6000 and RTX 5090", 11 Mar 2026 (secondary)](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
- **Community oMLX runs** (MLX-based server; community-submitted; figures taken from search-result summaries, not re-fetched):
  - gpt-oss-120b MXFP4-Q8 on M5 Max 128 GB / 40-core: about **80 tok/s** at 4K context, 1,618 prompt tok/s. [oMLX 5muhzznz](https://omlx.ai/benchmarks/5muhzznz)
  - Same model, another M5 Max run: about 85 tok/s at 1K context, falling to about 71 at 8K. [oMLX khaf683c](https://omlx.ai/benchmarks/khaf683c)
  - gpt-oss-120b on M3 Ultra 80-core / 512 GB: **65.4 tok/s** at 1K context. [oMLX xoeb7x9j](https://omlx.ai/benchmarks/xoeb7x9j)
- **DeepSeek-V3 / R1 671B on M3 Ultra 512 GB** (community and secondary):
  - Hardware Corner: the 4-bit V3 model runs at "> 20 toks/sec" with mlx-lm, with "serious slowdowns" at large contexts such as 32K. [Hardware Corner (secondary)](https://hardware-corner.net/studio-m3-ultra-running-deepseek-v3)
  - Community mlx_lm output for DeepSeek-V3-0324 4-bit:
    - 69-token prompt: 58.08 tok/s prefill, **21.05 tok/s** generation, 380.2 GB peak.
    - About 17.8 tok/s at roughly 1K context.
    - About 5.8 tok/s at roughly 16K context, with about 465 GB peak.
    - Figures are as quoted in search results from r/LocalLLaMA threads. I did not verify which exact post they come from. [r/LocalLLaMA mirror 1](https://lr.eu.psf.lt/r/LocalLLaMA/comments/1jj6i4m/deepseek_v3/mjm5rub/?context=3); [r/LocalLLaMA mirror 2](https://redlib.hackliberty.org/r/LocalLLaMA/comments/1jke5wg/m3_ultra_mac_studio_512gb_prompt_and_write_speeds)
  - DeepSeek-R1 MLX 4-bit, 13,140-token prompt: 59.56 tok/s prefill, **6.385 tok/s** generation, 491 GB peak. A 16K context ran out of memory. [dev.to, SomeOddCodeGuy (community)](https://dev.to/someoddcodeguy/running-deepseek-r1-0528-q4km-and-mlx-4-bit-on-a-mac-studio-m3-le4)
  - llama.cpp, V3 Q4_K_M: about 6.2 tok/s generation and about 9 tok/s prompt processing at roughly 8K context. [dev.to, SomeOddCodeGuy (community)](https://dev.to/someoddcodeguy/m3-ultra-mac-studio-512gb-prompt-and-write-speeds-for-deepseek-v3-0-671b-gguf-q4km-for-those-47hi)
  - A Chinese report gives 19.66 tok/s (MLX) vs 15.78 tok/s (GGUF) for V3 671B on a fully specced M3 Ultra. [kocpc (secondary)](https://en.kocpc.com.tw/?p=10624)
  - The r/LocalLLaMA thread title "M3 Ultra 512GB does 18T/s with Deepseek R1 671B Q4" is listed in search results. [mirror (community)](https://redlib.ssps.io/r/LocalLLaMA/comments/1j8r2nr/m3_ultra_512gb_does_18ts_with_deepseek_r1_671b_q4)
- **Apple M5 (base), MacBook Pro 24 GB** vs M4 (153 GB/s vs 120 GB/s, +28%), `mlx_lm.generate`, 4096-token prompt plus 128 generated tokens:
  - Qwen3-30B-A3B 4-bit (17.31 GB): generation **1.25×** M4, TTFT 3.52× M4.
  - gpt-oss-20b MXFP4: generation 1.24×, TTFT 3.33×.
  - Generation gains across the tested models are 19-27%.
  - A 30B MoE gets "under 3 seconds" TTFT.
  - Neural Accelerator support needs macOS 26.2 or later.
  - Source: [Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU", 19 Nov 2025 (primary, vendor)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

#### Apple Silicon, multi-Mac clusters
- **Apple ML Research, "Local and Distributed Large LLM Inference with MLX and RDMA" (on Apple's Ultra desktop), 15 Oct 2026** (primary, vendor):
  - Desktop Macs with M3 Ultra and **M5 Ultra**, up to 512 GB each, linked by **RDMA over Thunderbolt 5** with two cables per link.
  - Model: **Muse Spark Open, 816B total / 42B active MoE, NVFP4 ≈ 493 GB**.
  - Decode: **15.74 tok/s** on 2 M5 Ultras (pipeline), **24.99** on 2 M5 Ultras (tensor parallel), **36.73** on 4 M5 Ultras (tensor parallel). The text says it exceeds "36 tokens per second" on four Macs at 8,192-token context.
  - Dense references in the same table:
    - Mistral Medium 3.5 (128B dense, FP8): 4.95 (M3 Ultra), 7.70 (M5 Ultra), 13.69 (2× M5 tensor), 23.37 (4× M5 tensor).
    - Qwen 3.8 (27B dense, BF16): 12.46 / 19.69 / 33.27 / 50.89.
  - Bandwidth: M5 Ultra 1.2 TB/s vs M3 Ultra 819 GB/s. M5 Ultra decode is **1.6×** M3 Ultra and TTFT is up to 4.6× faster.
  - Peak power: about 200 W (M3 Ultra) and about 300 W (M5 Ultra).
  - **Speculative decoding is not mentioned.**
  - Source: [Apple ML Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
- **Kimi K2 (1T) on Mac clusters** (secondary; not verified against primary posts):
  - About 21 tok/s on 2× M3 Ultra 512 GB.
  - About 28 tok/s for Kimi K2 Thinking on 4× M3 Ultra (1.5 TB total, under 500 W), attributed to Jeff Geerling, Dec 2025.
  - Another write-up gives about 25 tok/s with RDMA over Thunderbolt 5 vs about 5 tok/s on a standard connection.
  - Sources: [Creative Strategies, a 1T-parameter model on a $40K cluster of Ultra desktop Macs (secondary)](https://creativestrategies.com/?p=10397); [Context Studios (secondary)](https://www.contextstudios.ai/blog/cluster-two-mac-studios-or-buy-512-gb-local-llms-on-the-m5-ultra-fact)
  - A Hugging Face discussion mentions an MLX g32 conversion of "Kimi-K2.7-Code" at 20.1 tok/s average, tensor-parallel on two Ultra desktop Macs. [HF discussion (community)](https://huggingface.co/moonshotai/Kimi-K2.7-Code/discussions/7)
- **DBRX MoE on 2-4 M2 Ultra desktop Macs** with expert parallelism:
  - Expert parallelism significantly cuts inference time.
  - Communication time for exchanging expert outputs is comparable to computation time, so **network latency matters more than bandwidth**.
  - The authors found and removed overhead from Apple's memory-management logic.
  - The cluster is 1.15× more cost-efficient than an H100 system.
  - Source: [arXiv 2506.23635 (RACS '24) (primary)](https://arxiv.org/abs/2506.23635)

#### Apple Silicon, SSD expert streaming (model larger than RAM): PagedMoE
- **System:**
  - MLX/mlx-lm engine, v0.3.0 pre-release, AGPL-3.0.
  - Replaces mlx-lm's `SwitchGLU` with a disk-backed version that reads experts from the unmodified checkpoint. Backbone (attention, embeddings) stays resident.
  - Byte-budgeted expert cache with frequency-aware eviction.
  - **Route prediction**: small resident router matrices run ahead on the current hidden state. A governor keeps it on only when it measures a win.
  - **Slab storage**: fixed GPU slots, reported at +11% decode on Qwen3-235B and +20% on Qwen3-Next.
  - Native C read pool with the GIL released; demand misses are prioritized over speculative reads.
  - Optional online "sidecar" expert predictor.
  - Expert reads bypass the page cache where macOS allows it.
  - Optional **lossy pruning** (`PAGED_MOE_PRUNE`) skips low-weight experts that would need a disk read.
  - Without pruning, output is bit-identical to a fully resident run.
  - Source: [PagedMoE README (primary, author-reported)](https://github.com/noahclark556/paged-moe)
- **Benchmark setup:** **Apple M5 Pro, 48 GB**, internal SSD, RAM budget about 35 GB. Stock mlx-lm cannot load these models on this machine. Cold ~2K-token prompt, 128 warm-up tokens, then 512 timed greedy tokens (wall clock). The README says these are September numbers. Results (same source):

  | Model | Checkpoint | Peak RAM | Prefill | Decode | Setting |
  |---|---|---|---|---|---|
  | Qwen3-Coder-Next 6-bit | 65 GB | 30.7 GB | 235 tok/s | **20.6 tok/s** | exact, no prune |
  | Qwen3-235B-A22B 4-bit | 132 GB | 30.6 GB | 99 tok/s | **7.9 tok/s** | prune 0.7, wait 0.2 |
  | GLM-4.7 4-bit | 199 GB | 31.8 GB | 73 tok/s | **5.9 tok/s** | prune 0.8, wait 0.2 |
  | DeepSeek-V3.2 4-bit | 378 GB | 33.2 GB | 45 tok/s | **5.7 tok/s** | prune 0.8, wait 0.2 |
  | Kimi-K2-Instruct 4-bit | 578 GB | 31.6 GB | 27 tok/s | **8.8 tok/s** | prune 0.8, wait 0.2 |

  Kimi held 10.2 tok/s over the last quarter of the run. Exact (unpruned) mode is slower for the large models.
- **Other PagedMoE numbers** (same source):
  - Qwen3-235B "needs about 8 GB of expert weights per token and misses about 1.5 GB", so decode is SSD-bound.
  - Qwen3-235B history: about 2 tok/s at first, 2.6 after profiling, 6+ later.
  - Qwen3-Coder-480B exact: about 1 tok/s in September.
  - Improved miss reads: MiniMax-M3 4-bit went from 5.65 to 9.31 tok/s.
  - Qwen3-235B, 512 tokens on a seeded KV cache: 9.1 s → 6.0 s with next-layer read prefetch.
  - Prefill on a 31.5K-token cold prompt: about 34 s (937 tok/s, single pass).
  - Thunderbolt drives are "several times slower" than the internal SSD.
- **"Results by Mac" rows** (same source):
  - 16 GB Air: Qwen3-Coder-Next 6-bit at 10.8 tok/s with 8.4 GB peak.
  - 18 GB M3 Pro: Qwen3-235B-A22B 4-bit at 8.6 tok/s with 8.3 GB peak and only a 1.6 GB expert cache.
  - 24 GB M4 Pro: Qwen3-235B at 9.1 tok/s.
  - 48 GB M4 Pro/Max: Qwen3-235B at 9.0 tok/s, Qwen3-Coder-Next at 23.4 tok/s.
  - The README notes the 64 GB and 128 GB rows are IO-scaled estimates, not runs, and that the "GPU is M5 Pro".
- **Speculative decoding in PagedMoE.** It is not a separate draft model. "Exact multi-token decode" drafts from n-gram matches in the context, from self-drafting, or from an MTP head where the checkpoint has one. A governor enables drafting only where it measures a speedup. Results on Qwen3.8-Flash-Next, 256 greedy tokens (same source):

  | Setting | "Edit a file" | "Explain a file" |
  |---|---|---|
  | Plain, no prune | 5.9 tok/s | 7.1 tok/s |
  | Verify + self-draft, no prune | **13.4 tok/s (2.26×)** | 8.3 tok/s (1.16×) |
  | Prune 0.7 + flow, plain | 9.8 tok/s | 12.1 tok/s |
  | Prune 0.7, exact verify | **15.6 tok/s (1.60×), "99% exact tokens"** | 12.0 tok/s (governor kept it off) |

  - "Copy-back" decoding at a prune setting rose from about 15 to **48.8 tok/s**.
  - **A draft-model approach was a 2-3× regression on the 235B, because verify batches thrash the expert cache.**

#### Phones / edge
- **BigMoMo** (arXiv 2609.14643; v1 13 Sep 2026, v2 3 Oct 2026; under review):
  - **Phones:**

    | Phone | SoC | DRAM (available) | Storage |
    |---|---|---|---|
    | OnePlus 15 | Snapdragon 8 Elite Gen 5 | 16 GB (11.52 GB) | UFS 4.1 |
    | OnePlus 13 | Snapdragon 8 Elite | 24 GB (18.23 GB) | UFS 4.0 |
    | OnePlus Ace 5 | Snapdragon 8 Gen 3 | 12 GB (6.00 GB) | UFS 4.0 |

    UFS is "typically around 4 GB/s". Compute runs on the Hexagon NPU.
  - **Targets**, all **Q4_0**, each with a small dense draft:
    - Qwen3-30B-A3B (128 experts, top-8), draft Qwen3-0.6B.
    - LFM2-24B-A2B, draft LFM2.5-350M.
    - Gemma-4-26B-A4B-it, draft "Gemma-4-26B-A4B-it-assistant".
    - ERNIE-4.5-21B-A3B, draft ERNIE-4.5-0.3B.
  - **Default tree:** draft length D = 2, branch width B = 2.
  - **Speed (TPOT):**

    | Phone | BigMoMo | Best baseline | SD-OnDemand |
    |---|---|---|---|
    | OnePlus 13 | **74.1 ms (≈13.5 tok/s)** | 128.2 ms | 161.3 ms |
    | OnePlus 15 | 304.4 ms (≈3.3 tok/s) | 552.6 ms | 979.7 ms |
    | OnePlus Ace 5 | 579.4 ms (≈1.7 tok/s) | 1030.8 ms | 1859.6 ms |

    The tok/s figures are my conversion from TPOT. My extraction did not identify which model these TPOT rows use.
  - **Mean speedups:** 4.83× over on-demand autoregressive offloading, 3.00× over SD-OnDemand, and 1.76× over the best SD-aware baseline (SpecMoEOff, SP-MoE, MoE-SpAc). The abstract says 1.82×.
  - **Per-model speedups** over the best baseline: Qwen3-30B-A3B 1.73×, LFM2 1.47×, Gemma-4 1.80×, ERNIE-4.5 2.03×.
  - **Scaling:** supports MoEs up to 80B (Qwen3-Next-80B-A3B on the OnePlus 15), at 1.4-1.8× over the best SD baseline. No tok/s is given for the 80B model.
  - Sources: [arXiv 2609.14643v2 HTML (primary)](https://arxiv.org/html/2609.14643v2); [abs](https://arxiv.org/abs/2609.14643)
- **PowerInfer-2** (arXiv 2406.06282, v3 12 Dec 2024):
  - Runs "a 47B LLM on a smartphone" at **11.68 tokens/s**, with up to 27.8× over state-of-the-art frameworks and "negligible accuracy loss".
  - Method: neuron-cluster scheduling. Dense clusters run on the NPU and sparse ones on the CPU, with a segmented neuron cache and I/O-compute overlap.
  - [arXiv abs (primary)](https://arxiv.org/abs/2406.06282)
  - A secondary summary identifies the model as TurboSparse-Mixtral-47B (about 3B activated) on OnePlus 12 / Ace 2. It reports 2.13 tok/s with 7 GB of available memory vs 11.68 tok/s at 19 GB. [Liner review (secondary)](https://liner.com/review/powerinfer2-fast-large-language-model-inference-on-smartphone)
- **HOBBIT on Jetson AGX Orin 32 GB** (edge):
  - Mixtral-8x7B (45B total / 14B active) and Phi-MoE (42B / 6.6B), INT8 base with INT2 replacement experts.
  - Decode averages **13.0× and 18.9× over llama.cpp**, and 3.64× and 9.93× over MoE-Infinity.
  - No absolute tok/s in my extraction.
  - Source: [arXiv 2411.01433v2 (primary)](https://arxiv.org/html/2411.01433v2)
- **Qualcomm AI Hub** (dense reference only): Qwen3-8B w4a16 at about 13.7 tok/s on Snapdragon 8 Elite and about 16.1 tok/s on 8 Elite Gen 5. [Qualcomm Qwen3-8B model card (vendor)](https://huggingface.co/qualcomm/Qwen3-8B)
- **Qualcomm marketing** mentions "up to 220 tokens per second" on Snapdragon 8 Elite Gen 5 with gpt-oss-20b as an example. No methodology is given, so treat it as unverified and not batch-1 decode. [Qualcomm mobile AI page (vendor marketing; via search summary)](https://www.qualcomm.com/artificial-intelligence/mobile-ai)

### Inferences
- **Speed tracks active bytes, not total parameters.** On one Mac, the ≥100B ranking is gpt-oss-120b (~5B active, ~80-88 tok/s), then Qwen3.5-122B-A10B (~55-66), then DeepSeek 671B (37B active, ~18-21).
- **Clusters scale sub-linearly.** Tensor-parallel decode over RDMA went from 2 to 4 M5 Ultras at 24.99 → 36.73 tok/s (1.47× for 2× the hardware) on the 816B MoE.
- **M3 Ultra decode was probably not purely bandwidth-bound.** The M5 Ultra / M3 Ultra decode gain (1.6×) is larger than the bandwidth ratio (1.2 TB/s / 819 GB/s ≈ 1.47×). Either M3 Ultra decode was partly latency- or overhead-bound, or M5 brings non-bandwidth improvements.
- **SSD streaming on a 48 GB Mac:**
  - Exact decode of 200B+ MoEs is about 1-6 tok/s. The 6-9 tok/s headlines depend on lossy pruning.
  - Exact decode stays near 20 tok/s only for an 80B-class model with a high cache-hit rate (Qwen3-Coder-Next, 65 GB checkpoint).
  - Speculation helps only with nearly free drafts (n-gram, self-draft, MTP) on copy-heavy tasks (up to 2.26×). A separate draft model's verify batch enlarges the expert union and thrashes the cache (2-3× slower).
- **PagedMoE's per-Mac rows are probably emulations.** The rows for M2/M3/M4 Macs look like RAM-capped runs on the M5 Pro ("GPU is M5 Pro"), not measurements on those Macs.
- **On phones, available DRAM matters more than SoC generation.** The OnePlus 15 has a newer SoC but only 11.5 GB available, and is about 4× slower than the OnePlus 13 with 18.2 GB.
- **Speculation appears only in the offloading systems.** None of the top in-memory Apple numbers uses speculative decoding. It shows up only in SSD/flash offload systems (PagedMoE, BigMoMo), where it amortizes I/O.

### Gaps
- No independent measured decode for gpt-oss-120b or other 100B+ MoEs on a **single M5 Ultra**. Only estimates were found, ranging from about 14 to 115 tok/s; see Q5.
- Kimi K2 1T numbers on one or several M3 Ultras come only from secondhand sources. I found no first-party MLX benchmark for the 2-Mac setup.
- PagedMoE figures are author-reported and not independently replicated. They also include 2026 model names (Qwen3.8-Flash-Next, GLM-5.2, MiniMax-M3, KAT-Coder-V2.5) whose specs I did not verify. The README does not say whether self-drafting was active in its main decode table.
- BigMoMo: my extraction did not get per-model absolute tok/s or the model behind the Table 9 TPOT rows.
- I found no reliable measured Qwen3-30B-A3B (or other 30B+ MoE) decode on a phone outside BigMoMo, and no iPhone/iPad MoE numbers.

---

## 2. Systems that combine speculative decoding with expert offloading: speedups and verify-block sizes

### Takeaway
At least eight 2025-2026 systems combine speculation with expert offloading. The common lesson is that the verify block must be sized and shaped against the expert union it touches.

| System | Reported gain | Verify block / draft | Notes |
|---|---|---|---|
| Specter / MoE-SpeQ | 1.50× average, up to 1.88× TPOT over SpecMoEOff | K = 8-16, INT4 self-draft | 96.6% acceptance; A100-40GB + PCIe 4.0 |
| SpecMoEOff | up to 2.5× throughput | 1-10 drafts | Large batch, not batch 1 |
| SP-MoE | 1.07-3.5× TPOT | not extracted | |
| SpecMoE | up to 4.30× throughput | not extracted | |
| BigMoMo | 4.83× vs on-demand; 1.76-1.82× vs best SD baseline | D = 2, B = 2 trees | Phones |

Three papers add the counter-evidence and fixes:
- **Cascade**: naive speculation can slow MoE decode by up to 1.5×.
- **MoE-Spec**: a fixed expert budget gives +10-30% over EAGLE-3.
- **EcoSpec**: cost-aware draft selection gives up to 1.62× on DeepSeek-V3.1 671B, Qwen3-235B and gpt-oss-120b.

Apple's "LLM in a flash" is flash offloading for dense models via sparsity. It is not speculative and not MoE.

### Cited Findings

#### Specter (arXiv 2511.14102 v3, ASPLOS '27; v1 = MoE-SpeQ)
- **Mechanism:**
  - A GPTQ-quantized copy of the target is the draft: expert linear layers use symmetric INT4, group size 128.
  - Router/gating, attention, norms and shared experts stay FP16 and are shared with the target, along with the KV cache.
  - Multi-step lookahead feeds an Expert Lookahead Buffer (entropy-aware caching, lookahead-aware eviction).
  - A "Speculative Governor" uses an Amortization Roofline Model to pick draft length.
  - A fused MoE kernel speeds up drafting.
  - Sources: [arXiv 2511.14102v3 (primary)](https://arxiv.org/html/2511.14102v3); [v1 abstract via search summary](https://arxiv.org/html/2511.14102v1)
- **Hardware:** one **NVIDIA A100 40 GB**, 24-core Xeon Silver 4310, 256 GB host RAM, **PCIe 4.0 x16 (31.5 GB/s theoretical)**. [v3](https://arxiv.org/html/2511.14102v3)
- **Models:** Phi-3.5-MoE (41.9B / 6.6B active, 16 experts, top-2), Qwen1.5-MoE (14.3B / 2.7B, 60 experts, top-4 + 1 shared), DeepSeek-V2-Lite (15.7B / 2.4B, 64 experts, top-6 + 1 shared). [v3](https://arxiv.org/html/2511.14102v3)
- **Speculative depth and acceptance** ([v3](https://arxiv.org/html/2511.14102v3)):
  - K is chosen offline from {2, 4, 6, 8, 12, 16, 20, 24, 32}. Selected: **K = 16** (DeepSeek-V2-Lite), **K = 8** (Qwen1.5-MoE), **K = 16** (Phi-3.5-MoE).
  - Mean token acceptance is **96.63%**, up to 99% on DeepSeek-V2-Lite. The paper states "an expected 28.7 accepted tokens per window".
- **Single-request TPOT, high-memory setting** ([v3](https://arxiv.org/html/2511.14102v3)):

  | Model | Specter | SpecMoEOff | Mixtral-Offloading (SM) | vLLM | Transformers |
  |---|---|---|---|---|---|
  | Phi-3.5-MoE | **163.3 ms** | 292.1 ms | 295.1 ms | 297.5 ms | 960.6 ms |
  | Qwen1.5-MoE | **35.3 ms** | 66.4 ms | 74.2 ms | 176.6 ms | 258.2 ms |
  | DeepSeek-V2-Lite | **72.2 ms** | – | – | – | 344.9 ms |

  - Average speedup over SpecMoEOff is **1.50× (up to 1.88×)**; up to 5.9× over Transformers on Phi-3.5-MoE.
  - At batch 32, DeepSeek-V2-Lite averages 130.3 tok/s (Specter) vs 40.2 (SpecMoEOff).
  - v1 (MoE-SpeQ) claimed up to 2.34× over state-of-the-art offloading on Phi-MoE (via search summary).
- **Caching and routing fidelity** ([v3](https://arxiv.org/html/2511.14102v3)):
  - Expert cache hit rate on DeepSeek-V2-Lite: 99.85% (16 GB), 98.62% (24 GB), 96.25% (32 GB).
  - Target-to-draft expert recall: 0.930-0.961.
  - The INT4 draft matches the FP16 target's top-4 expert IDs **90.8%** of the time on Qwen1.5-MoE.
- **Memory:** sharing cuts the Qwen1.5-MoE footprint at 12K context from 13.40 GB to 7.68 GB (−43%). [v3](https://arxiv.org/html/2511.14102v3)
- **Code:** github.com/wwwwwf/specter_asplos (per the abstract page). [abs](https://arxiv.org/abs/2511.14102)

#### SpecMoEOff (arXiv 2508.21706, v2 31 Oct 2025)
- **Mechanism:** speculation enlarges each expert's workload per offload. It adds GPU/CPU roofline-based coordination, a CPU chunked-attention verification kernel, draft-model offloading, and a hyperparameter optimizer that combines convex optimization with profiling.
- **Hardware:** A30 + Xeon Gold 6426Y (CPU→GPU 25 GB/s, 250 GB host RAM), and RTX 4090D + Xeon Gold 5418Y (23 GB/s, 190 GB host RAM).
- **Model and draft:** **Mixtral-8x7B FP16** (87 GB) with an **EAGLE** draft under 2 GB.
- **Draft length:** swept 1-10, with 5 for the breakdown; dynamic tuning adds about 2%.
- **Batch:** this is a **large-batch** design (roofline global batch 824 on A30, 515 on 4090D), **not batch 1**.
- **Results:**
  - Up to 2.5× (average 2.1×, range 1.6-2.9×) throughput over MoE-Lightning; 11.3-27.5× over DeepSpeed-ZeRO-Inference.
  - Speculation alone adds 1.53× (APPS) and 1.45× (CNN/DailyMail).
  - Small-batch expert offloading leaves the A30 at 0.76% GPU utilization.
- Sources: [arXiv 2508.21706v2 (primary)](https://arxiv.org/html/2508.21706v2); [abs](https://arxiv.org/abs/2508.21706)

#### Other speculation + offloading systems
- **SP-MoE** (arXiv 2510.10302, v2 6 Nov 2025):
  - Speculative expert prefetching that "exploits structural correspondence between the draft and target models to prefetch likely experts ahead of verification".
  - A cutoff-layer policy, and a pipelined runtime with asynchronous prefetch threads and batched I/O.
  - **1.07-3.5× TPOT** over state-of-the-art methods. The abstract gives no hardware or draft length.
  - [arXiv abs (primary)](https://arxiv.org/abs/2510.10302)
- **SpecMoE** (arXiv 2604.10152, DAC 2026):
  - "Self-assisted" speculative decoding that needs no extra training, for memory-constrained systems compared against CPU-offloaded MoE systems.
  - **Up to 4.30×** throughput. Hardware and draft construction are not in the abstract.
  - [arXiv abs (primary)](https://arxiv.org/abs/2604.10152)
- **BigMoMo** (phones; details in Q1):
  - Uses the verification window to decouple expert movement from single-token execution, giving weight reuse, contiguous flash reads and load/compute overlap.
  - Prunes branches and expert activations by acceptance rate, routing impact and movement cost.
  - Reorganizes on-flash experts by runtime co-loading patterns.
  - Draft nodes that contribute no accepted tokens account for **82.1% of verification compute and 81.8% of expert-movement time**.
  - Default **D = 2, B = 2**; sensitivity tested at (D, B) = (1,1), (2,1), (2,2), (4,2), (4,4).
  - Sources: [arXiv 2609.14643v2 (primary)](https://arxiv.org/html/2609.14643v2); [abs](https://arxiv.org/abs/2609.14643)
- **PagedMoE** (Apple Silicon, SSD; details in Q1): n-gram, self-draft and MTP verify give up to 2.26×. A draft-model approach was a 2-3× regression on Qwen3-235B. [README](https://github.com/noahclark556/paged-moe)
- **LLM in a flash** (Apple, arXiv 2312.11514; ACL 2024; v3 30 Jul 2024):
  - Stores parameters in flash and loads them on demand into DRAM.
  - **Windowing** reuses recently activated neurons; **row-column bundling** gives larger contiguous flash reads.
  - Runs models **up to 2× available DRAM**, **4-5× faster on CPU and 20-25× faster on GPU** than naive loading.
  - Dense models with activation sparsity: **not MoE and not speculative**.
  - [arXiv abs (primary)](https://arxiv.org/abs/2312.11514)

#### Evidence that speculation can hurt MoE decode, and the fixes
- **Cascade, "Utility-Driven Speculative Decoding for MoE"** (arXiv 2506.20675, Jun 2025):
  - Draft tokens activate more experts, "increasing data movement and verification time by 2-3x". Static speculation causes "slowdowns up to 1.5x".
  - Cascade enables speculation selectively and tunes K with a utility metric (token gain / verification cost). In vLLM on 5 MoEs it limits worst-case slowdown to 5% and gains 7-14% over static K.
  - [arXiv abs (primary)](https://arxiv.org/abs/2506.20675)
- **MoE-Spec, "Expert Budgeting for Efficient Speculative Decoding"** (arXiv 2602.16052, Feb 2026):
  - Training-free. Enforces a **fixed per-layer expert capacity at verification** and drops the rarely used tail, so speculation depth is decoupled from memory cost.
  - **10-30% higher throughput than EAGLE-3** at comparable quality. Tighter budgets cut latency at some accuracy cost.
  - [arXiv abs (primary)](https://arxiv.org/abs/2602.16052)
  - Models were OLMoE-1B-7B, Qwen3-30B-A3B and Mixtral-8x7B. Experts outside the budget are handled by truncation or substitution. [arXiv HTML via search summary](https://arxiv.org/html/2602.16052v1)
- **EcoSpec, "Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for MoE"** (arXiv 2607.12696, Jul 2026):
  - Adds a predicted marginal expert-activation cost to draft selection, using an expert predictor and a dynamic expert buffer. It favors draft paths that reuse experts already in the verification set; the verification rule is unchanged.
  - **Up to 1.62×** on DeepSeek-V3.1 (671B), Qwen3-235B-A22B and GPT-OSS-120B.
  - [arXiv abs (primary)](https://arxiv.org/abs/2607.12696)
- **MoESD** (arXiv 2505.19645): at moderate batch sizes, where all experts are already loaded, verifying extra draft tokens adds little expert-loading cost, so speculation can help MoE more than dense models. [arXiv HTML (primary, via search summary)](https://arxiv.org/html/2505.19645v4)
- **Single-user MTP regression on a 397B MoE:**
  - Qwen3.5-397B-A17B NVFP4 on **4× RTX PRO 6000**, vLLM nightly, TP = 4, Marlin W4A16 fallback, no MTP: **50.5 tok/s** at concurrency 1.
  - With MTP = 2: **39.6 tok/s (−22%)** at 61-85% acceptance. The author attributes the drop to draft heads trained on native FP4 activations.
  - [vLLM forum, 11 Apr 2026 (community)](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536)
- **Datacenter reference for block size:** vLLM v0.13.0 with EAGLE-3 for gpt-oss on H200 PCIe, **3 draft tokens**: peak output throughput 2574 vs 2024 tok/s (+27.2%), measured under concurrent load, not batch 1. [Red Hat Developer, 16 Apr 2026 (secondary/vendor)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss)

#### Levers reported to make the verify pass cheaper
- **Cap or shape the verify expert union:**
  - Per-layer expert budget: MoE-Spec, [2602.16052](https://arxiv.org/abs/2602.16052).
  - Expert-cost-aware draft selection: EcoSpec, [2607.12696](https://arxiv.org/abs/2607.12696).
  - Utility-gated K: Cascade, [2506.20675](https://arxiv.org/abs/2506.20675).
  - Branch and expert pruning by acceptance, routing impact and movement cost: BigMoMo, [2609.14643](https://arxiv.org/html/2609.14643v2).
- **Prefetch from lookahead:**
  - Draft-routing lookahead buffer: Specter, [2511.14102v3](https://arxiv.org/html/2511.14102v3).
  - Draft→target structural correspondence: SP-MoE, [2510.10302](https://arxiv.org/abs/2510.10302).
  - Router run-ahead on the current hidden state: PagedMoE, [README](https://github.com/noahclark556/paged-moe).
- **Make expert bytes contiguous and reusable:**
  - On-flash co-location of co-loaded experts: BigMoMo, [2609.14643](https://arxiv.org/html/2609.14643v2).
  - Row-column bundling: LLM in a flash, [2312.11514](https://arxiv.org/abs/2312.11514).
  - Fixed GPU slab slots (+11-20%) and page-cache bypass: PagedMoE, [README](https://github.com/noahclark556/paged-moe).
- **Share non-expert weights and KV between draft and target:** Specter (−43% memory), [2511.14102v3](https://arxiv.org/html/2511.14102v3).
- **Lower the bytes of less-important experts:** HOBBIT, see Q3, [2411.01433v2](https://arxiv.org/html/2411.01433v2).

### Inferences
- **Verify blocks in local/offload systems are small**, typically 2-5 tokens:
  - BigMoMo D = 2 / B = 2; SpecMoEOff 5; vLLM MTP 2; EAGLE-3 3.
  - The outlier is Specter (K = 8-16). It is possible because the draft is an INT4 copy of the target whose routing largely matches the target (90.8% top-4 match; 0.93-0.96 recall), so the experts the verify pass needs are mostly already prefetched.
- **Specter's "28.7 expected accepted tokens"** equals α/(1−α) for α = 0.9663, i.e. an unbounded window.
  - Assuming independent per-token acceptance, the realized tokens per verify pass, (1−α^(K+1))/(1−α), are about 7.9 at K = 8 and about 13.1 at K = 16 (my arithmetic).
- **On SSD/flash the expert-union penalty is far steeper than over PCIe.** PagedMoE's 2-3× regression with a separate draft model and BigMoMo's 82% waste on non-contributing nodes are consistent with that. A miss costs SSD/UFS latency plus about 4-14 GB/s, vs about 25-32 GB/s over PCIe.
- **Gains depend heavily on the baseline.** They are large against naive on-demand offloading (4-5×) and modest against strong speculation-aware baselines (1.5-1.8×).
- **Absolute batch-1 numbers in these papers are low.** Specter on A100 + PCIe gives about 28 tok/s on Qwen1.5-MoE (14B) and about 6 tok/s on Phi-3.5-MoE (42B).
- **None of the papers reports results on Apple Silicon or with 100B+ models.** EcoSpec and MoE-Spec, which use 100B+ models, are not offloading systems.

### Gaps
- SP-MoE and SpecMoE: hardware, models, draft lengths and absolute tok/s were not extracted (abstracts only).
- SpecMoEOff gives absolute tok/s only in figures, and its large-batch setting is not comparable to batch 1.
- MoE-SpAc (a BigMoMo baseline) was not researched.
- EcoSpec's and MoE-Spec's hardware, budget values (B) and acceptance numbers were not extracted.
- I found no peer-reviewed system combining speculation with SSD expert streaming on Apple Silicon. PagedMoE is a README, not a paper.
- I could not verify the hardware used in "LLM in a flash" (the abstract page omits it).

---

## 3. Low-bit / quantized verification: verifying with a lower-bit copy, mixed precision for less important experts, and quality loss

### Takeaway
Published work mostly takes one of two routes.

**Draft with a low-bit copy, verify at higher precision:**
- QSpec (W4A4 → W4A16): up to 1.64× with no quality loss.
- QuantSpec (4-bit weights plus 4-bit KV): over 90% acceptance, about 2.5×.
- Specter (INT4 GPTQ copy of an MoE): 96.6% acceptance.

**Lower precision or skip experts inside the target itself:**
- HOBBIT (about 30% of experts in INT4/INT2, 3% skipped): ≤1% accuracy loss.
- BigMoMo expert pruning: −0.05 to −0.18 points.
- PagedMoE pruning: "99% exact tokens" at prune 0.7, but loops or garbage at aggressive settings.
- MoE-Spec expert budgets: "comparable quality".

I found **no paper that verifies with a lower-bit copy of a 100B+ MoE target and measures how often it diverges from the full-precision target**. Also, local runtimes already verify against a quantized (usually 4-bit) target, so "exact" means exact relative to the quantized model.

### Cited Findings
- **QSpec** (arXiv 2410.11305): drafts with low-precision W4A4 and verifies with higher-precision W4A16, sharing weights and KV cache. Up to **1.64× without quality degradation** vs high-precision baselines, and up to 1.55× over state-of-the-art speculative decoding in batched settings. [arXiv 2410.11305 (primary; via search summary)](https://arxiv.org/pdf/2410.11305)
- **QuantSpec** (Tiwari et al., ICML 2025; arXiv 2502.10424): self-speculative. The draft uses a **hierarchical 4-bit quantized KV cache plus 4-bit weights**, with a double full-precision cache buffer. **Acceptance above 90%**, end-to-end up to about **2.5×**. [arXiv 2502.10424](https://arxiv.org/pdf/2502.10424); [PMLR v267](https://proceedings.mlr.press/v267/tiwari25b.html)
- **"Speculative Decoding Meets Quantization"** (arXiv 2505.22179): the extra compute of tree-style verification "undermines the memory access benefits" of 4-bit weights. A hierarchical framework gets **2.78×** for a W4A16 Llama-3-70B on an A100. [arXiv 2505.22179 (primary; via search summary)](https://arxiv.org/html/2505.22179v1)
- **Specter's INT4 self-draft of an MoE target**: expert layers INT4 g128 GPTQ; router, attention and shared experts FP16. Acceptance **96.63%** on average, up to 99%; draft-vs-target top-4 expert agreement **90.8%**; expert recall 0.930-0.961. The paper reports no task-accuracy benchmarks, because draft-then-verify preserves the target distribution. [arXiv 2511.14102v3](https://arxiv.org/html/2511.14102v3)
- **HOBBIT** (arXiv 2411.01433): mixed-precision expert offloading built on llama.cpp. All details from [arXiv 2411.01433v2 (primary)](https://arxiv.org/html/2411.01433v2):
  - **How it works:** on a cache miss, a less critical expert is loaded at low precision instead of high precision.
    - RTX 4090: FP16 base, **INT4** replacements.
    - Jetson Orin: INT8 base, **INT2** replacements.
  - **How importance is decided:** by gating-weight norm ‖G(x)eᵢ‖, with cumulative thresholds T1 = 0.6 and T2 = 0.9 for Mixtral.
    - Result: **67% high precision, 30% low precision, 3% skipped**.
    - The top-1 expert is always high precision.
    - The cache prioritizes high-precision experts.
  - **Components:** token-level dynamic loading, layer-level adaptive prefetch, and sequence-level multidimensional caching.
  - **Accuracy:**
    - Mixtral GSM8K 0.52 → 0.51 (FP16 + INT4) and 0.52 → 0.51 (INT8 + INT2).
    - Phi-MoE GSM8K 0.83 → 0.82 / 0.83.
    - TruthfulQA changes are of similar size.
    - Overall, accuracy drops "by no more than 1%".
  - **Skipping vs downgrading:**
    - A 10% expert-skip rate raises perplexity by more than 1%.
    - Replacing under 20% of experts with low-precision versions keeps the decline within 1%.
  - **Speed:**
    - RTX 4090 (24 GB, PCIe 4.0, 256 GB host RAM): 3.21-3.29× over MoE-Offloading; 2.30-3.92× over MoE-Infinity.
    - RTX 4090 + CPU: 1.31-1.42× over llama.cpp; about parity with Fiddler on Mixtral (0.99×).
- **BigMoMo pruning quality** (Table 5; no prune → node pruning → node + expert pruning):
  - GSM8K: 90.70 → 90.70 → **90.57**.
  - HumanEval pass@1: 89.60 → 89.60 → **89.55**.
  - MMLU-Pro: 69.70 → 69.70 → **69.52**.
  - Expert pruning costs 0.05-0.18 points.
  - Source: [arXiv 2609.14643v2](https://arxiv.org/html/2609.14643v2)
- **PagedMoE pruning:**
  - Skipping low-weight experts that would need a disk read is "a fidelity trade-off, not a free speedup".
  - At prune 0.7 with exact verify: 15.6 tok/s with "99% exact tokens".
  - The README's "hard parts" section reports loops at aggressive settings on GLM-5.2, and garbage output at aggressive pruning on the 235B in earlier versions.
  - Tool calls and early turn tokens run at stricter settings.
  - Source: [PagedMoE README](https://github.com/noahclark556/paged-moe)
- **MoE-Spec**: tighter per-layer expert budgets at verification "further reduce latency at some cost in accuracy". At its operating point the paper reports "comparable quality" to EAGLE-3. [arXiv 2602.16052](https://arxiv.org/abs/2602.16052)
- **Low-bit target paths can break MTP heads**: on Qwen3.5-397B NVFP4 served through the Marlin W4A16 path, MTP = 2 had 61-85% acceptance and lost 22% of decode speed. [vLLM forum](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536)

### Inferences
- **How often a 4-bit copy disagrees with FP16:** on small MoEs (14-42B), a 4-bit GPTQ copy agrees with its FP16 parent on about 96-99% of tokens and about 91% of top-4 expert choices.
  - So a lossy "verify with the 4-bit copy" scheme would diverge from the FP16 target roughly once every 25-30 tokens on those models.
  - This is an extrapolation from Specter's acceptance rate and is not measured for 100B+ models.
- **Downgrading beats skipping.** Lowering precision of less important experts costs ≤1 point when about 30% of expert loads are downgraded. Skipping experts hurts more per unit of savings (HOBBIT: 10% skips → >1% perplexity; under 20% low-precision replacement → within 1%).
- **These levers compose.** For a memory-bound verify pass, cost ≈ bytes of the expert union + non-expert weights + KV.
  - Low-bit experts shrink bytes per expert (HOBBIT).
  - Budgets cap the union (MoE-Spec).
  - Cost-aware drafting shapes the union (EcoSpec).
  - Pruning with exact verify trades a small token mismatch for I/O (PagedMoE).
- **Wide trees can erase 4-bit savings.** 2505.22179 shows the verify pass's compute can cancel the 4-bit bandwidth win on A100. On Apple GPUs (low compute per byte relative to NVIDIA), the tree width where verification stops being almost free is probably smaller. This is untested here.

### Gaps
- I found no measurement of end-task quality when **verifying** a 100B+ MoE with a 2-3-bit copy instead of its 4/8/16-bit version.
- I found no Apple-Silicon-specific study of low-bit drafting or verification for MoE.
- QSpec, QuantSpec and 2505.22179 test dense models. Their MoE behavior is untested in these sources.
- HOBBIT's absolute tok/s, and its accuracy beyond GSM8K/TruthfulQA, were not extracted.

---

## 4. Memory-bandwidth and kernel-efficiency limits for MoE decode on Apple Silicon

### Takeaway
No source I found reports achieved bandwidth (% of peak) for **MoE** decode on Apple Silicon, or profiles MLX `gather_qmm` / llama.cpp `mul_mat_id` expert kernels. Apple states that generation is bandwidth-bound.

My estimates from measured runs:
- **Dense** decode reaches about **77-89%** of nominal bandwidth on M3/M5 Ultra (from Apple's own table).
- **Large MoE** decode reaches about **35-60%** of peak.
- **Small-active MoE** decode on Ultra chips looks overhead/latency-bound rather than bandwidth-bound.
- **Tensor-parallel clusters** reach about 18% of per-Mac bandwidth.

So MoE-specific overhead (gather/dispatch, many M = 1 matmuls, router synchronization, non-expert bytes) leaves roughly 1.5-2.5× headroom relative to dense efficiency.

### Cited Findings
- **Apple on the M5:** "Generating subsequent tokens is bounded by memory bandwidth, rather than by compute ability"; "TTFT is compute-bound, while generation is memory-bandwidth-bound". M5 153 GB/s vs M4 120 GB/s (+28%) gives a 19-27% generation gain, with Qwen3-30B-A3B 4-bit at +25%. [Apple ML Research, Nov 2025](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- **Apple distributed paper:** M5 Ultra 1.2 TB/s vs M3 Ultra 819 GB/s; M5 Ultra decode 1.6× faster. **No percent-of-peak figures are reported.** Dense decode on one machine: Qwen 3.8 27B BF16 at 12.46 / 19.69 tok/s and Mistral Medium 3.5 128B FP8 at 4.95 / 7.70 tok/s (M3 / M5 Ultra). [Apple ML Research, Oct 2026](https://machinelearning.apple.com/research/mlx-distributed-inference)
- **M5 Max spec:** 614 GB/s. The article reports no achieved bandwidth or efficiency. [Hardware Corner](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
- **M4 Max spec:** 410 GB/s, the binned configuration. [OminiX-MLX performance-tuning docs (secondary)](https://www.mintlify.com/OminiX-ai/OminiX-MLX/advanced/performance-tuning)
- **gpt-oss-20b with llama.cpp Metal, tg128:** M3 Ultra 115.52 ≈ M2 Ultra 116.08 > M4 Max 36 GB 92.36 > M1 Max 75.15. [llama.cpp #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
- **Multi-node MoE on M2 Ultra (DBRX):**
  - Exchanging expert outputs costs about as much as compute, and latency matters more than bandwidth.
  - Apple's memory-management logic added overhead that the authors had to remove.
  - Source: [arXiv 2506.23635](https://arxiv.org/abs/2506.23635)
- **SSD-bound decode (PagedMoE):** Qwen3-235B needs about 8 GB of expert weights per token and misses about 1.5 GB, so "SSD bandwidth sets the pace". The README calls decode "mostly disk-bound". [PagedMoE README](https://github.com/noahclark556/paged-moe)
- **PCIe-offload analogue:** small-batch expert offloading leaves an A30 at 0.76% GPU utilization (Mixtral-8x7B). The MoE layer is limited by CPU→GPU transfer, which is "nearly fully utilized". [SpecMoEOff v2](https://arxiv.org/html/2508.21706v2)
- **No direct measurement found:** a search for MLX MoE decode as a percent of peak bandwidth turned up only projections and an unanswered Apple Developer Forums question asking for Metal profiling of MoE vs dense decode. [Apple Developer Forums thread 831042](https://developer.apple.com/forums/thread/831042)

### Inferences
Assumptions behind all the arithmetic below:
- MLX 4-bit ≈ 4.5 bits per weight (≈0.5625 B/param, group-64 scale and bias).
- MXFP4 ≈ 4.25 bits; NVFP4 ≈ 4.5 bits.
- M2 Ultra peak 800 GB/s and RTX PRO 6000 ≈ 1.79 TB/s are vendor specs I did not re-verify.
- gpt-oss-120b is ≈117B total / ≈5.1B active, with ≈3.6B active expert params and ≈1.5B non-expert params (attention + unembedding).
- KV-cache reads are ignored, so the fractions below are slight underestimates.

**Dense decode efficiency (Apple's own numbers):**
- Qwen 3.8 27B BF16, ≈54 GB/token:
  - M3 Ultra: 12.46 tok/s → ≈673 GB/s ≈ **82%** of 819 GB/s.
  - M5 Ultra: 19.69 tok/s → ≈1.06 TB/s ≈ **89%** of 1.2 TB/s.
- Mistral Medium 3.5 128B FP8, ≈128 GB/token:
  - M3 Ultra: 4.95 tok/s → ≈634 GB/s ≈ **77%**.
  - M5 Ultra: 7.70 tok/s → ≈986 GB/s ≈ **82%**.

**MoE decode efficiency (single Mac):**

| Model | Chip | Est. bytes/token | Decode | Effective bandwidth | % of peak |
|---|---|---|---|---|---|
| Qwen3.5-122B-A10B 4-bit | M5 Max | ≈5.6 GB | 65.9 tok/s | ≈370 GB/s | ≈**60%** of 614 GB/s |
| gpt-oss-120b, MXFP4 experts + 8-bit non-experts | M5 Max | ≈3.5 GB | 87.9 tok/s | ≈310 GB/s | ≈**50%** |
| gpt-oss-120b, llama.cpp, non-experts 8-bit to BF16 | M2 Ultra | ≈3.5-5.0 GB | 79.68 tok/s | ≈280-400 GB/s | ≈**35-50%** of 800 GB/s |
| DeepSeek-V3 671B 4-bit MLX (37B active) | M3 Ultra | ≈20.8 GB | ~21 tok/s | ≈437 GB/s | ≈**53%** of 819 GB/s |

For gpt-oss-120b, the 3.5 GB estimate is ≈1.9 GB of experts plus ≈1.6 GB of non-experts.

**Clusters:**
- Muse Spark Open, 42B active NVFP4 ≈23.6 GB/token.
- 4× M5 Ultra tensor parallel: ≈5.9 GB per Mac per token × 36.73 tok/s ≈ 217 GB/s ≈ **18%** of each Mac's 1.2 TB/s. That points to communication/latency limits, consistent with the DBRX finding.
- 2× M5 Ultra pipeline: 15.74 tok/s ≈ **31%** of one Mac's bandwidth applied sequentially.

**Small-active MoE on Ultra chips:**
- gpt-oss-20b runs at about 116 tok/s on both the M3 Ultra and M2 Ultra.
- The 36 GB M4 Max, with about half the bandwidth (410 GB/s), reaches about 80% of that speed (92 tok/s).
- So small-MoE decode on Ultras is bound by fixed per-layer and per-token overheads (dispatch, synchronization, small kernels), not bandwidth.
- On base chips it is closer to bandwidth-proportional: M5 vs M4 gave +25% decode for +28% bandwidth on Qwen3-30B-A3B.

**Mac vs NVIDIA:**
- RTX PRO 6000: gpt-oss-120b at 221.3 tok/s × ≈3.5 GB ≈ 775 GB/s ≈ 43% of peak. Qwen3.5-122B-A10B at 98.4 × ≈5.6 GB ≈ 551 GB/s ≈ 31%.
- Macs reach a higher fraction of their lower peak, but lose on absolute speed.

**Where the headroom is:**
- Dense ≈ 80-89% vs MoE ≈ 35-60% suggests roughly 1.5-2.5× from MoE-specific kernel and system work on Apple Silicon:
  - Fused or sorted expert gather.
  - Fewer kernel launches per layer.
  - Lower-precision non-expert weights. In gpt-oss, BF16 attention plus the unembedding can rival the expert bytes.
  - Avoiding router host syncs.
- This is an inference, not a measured profile.

**SSD streaming:**
- PagedMoE's ≈1.5 GB of misses per token at the 6-9 tok/s it reports implies about 9-13 GB/s of sustained SSD reads.
- That is near internal Mac SSD limits, so exact SSD-streamed decode of 200B+ models on 48 GB cannot go much faster without fewer bytes per token (pruning, lower-bit experts) or more tokens per expert load (speculation with a small expert union).

### Gaps
- No Metal-level profile of MLX `gather_qmm` / `gather_mm` or llama.cpp `mul_mat_id` for MoE decode, and no published achieved GB/s for MoE decode on any Apple chip.
- The bytes-per-token figures above are estimates: quantization overheads, non-expert precision in each checkpoint, and KV reads are not verified.
- M5 Pro memory bandwidth and Mac internal SSD read speeds were not verified in this pass.

---

## 5. Any reported local decode above ~200 tok/s for a 100B+ total-parameter model on consumer hardware

### Takeaway
Only one case was found: **gpt-oss-120b (≈5B active, MXFP4) on a single 96 GB RTX PRO 6000 Blackwell**, a workstation GPU rather than a consumer card.
- **221.3 tok/s** at 4K context (Hardware Corner, Mar 2026; engine not stated).
- **196.31 tok/s** llama.cpp tg128 (Aug 2025).

No measured result ≥200 tok/s was found for a 100B+ model on any Mac, phone, AMD APU or DGX-class desktop. The best Mac figure is about 88 tok/s (M5 Max). None of the qualifying numbers uses speculative decoding.

### Cited Findings
- **gpt-oss-120b, MXFP4, llama.cpp CUDA** (`llama-bench`, ub 2048):
  - **RTX PRO 6000 96 GB: tg128 = 196.31 ± 0.14 tok/s**, pp2048 = 5518.07.
  - RTX PRO 6000 Max-Q 96 GB: tg128 = 170.62 tok/s.
  - For scale: gpt-oss-20b on RTX 5090 32 GB: tg128 = 282.51; on RTX 4090: 221.95.
  - Source: [llama.cpp #15396 (primary)](https://github.com/ggml-org/llama.cpp/discussions/15396)
- **gpt-oss-120b on RTX Pro 6000 Blackwell 96 GB** (engine not stated): decode **221.3 tok/s at 4K**, 193.2 at 16K, 178.5 at 32K. On the same page, Qwen3.5-122B-A10B 4-bit gets 98.4 / 93.7 / 91.3 tok/s. [Hardware Corner, Mar 2026 (secondary)](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
- **Other RTX PRO 6000 figures:**
  - Ollama, single RTX Pro 6000, 4-bit weights: about 134 tok/s for gpt-oss-120b. [DatabaseMart (secondary)](https://www.databasemart.com/blog/ollama-gpu-benchmark-pro6000)
  - Estimates, not measurements: 202 tok/s at Q4_K_M, 4K context, 1 user ([llmrun.dev (estimate)](https://llmrun.dev/gpu/rtx-pro-6000-blackwell?featured=0)); about 190 tok/s ([gpuprix (estimate)](https://gpuprix.com/italy/best-gpu-for-local-llm/gpt-oss-120b)).
- **Larger-active MoEs are far below 200 even on workstation GPUs:** Qwen3.5-397B-A17B NVFP4 on 4× RTX PRO 6000 (vLLM, TP = 4) gives **50.5 tok/s** at concurrency 1. [vLLM forum (community)](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536)
- **Macs:**
  - Best measured 100B+ result: gpt-oss-120b at 87.9 tok/s on an M5 Max 128 GB (MLX, 4K context). [Hardware Corner](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)
  - M2 Ultra llama.cpp: 79.68 tok/s. [llama.cpp #15396](https://github.com/ggml-org/llama.cpp/discussions/15396)
  - Four-M5-Ultra cluster: 36.73 tok/s for an 816B MoE. [Apple ML Research](https://machinelearning.apple.com/research/mlx-distributed-inference)
  - M5 Ultra gpt-oss-120b figures in circulation are estimates only (about 14-115 tok/s across sites). [runaihome (estimate)](https://runaihome.com/blog/mac-studio-m5-max-vs-m5-ultra-local-ai-2026/); [LLMCheck (estimate)](https://llmcheck.net/models/gpt-oss-120b-on-m5-ultra/)
- **AMD Strix Halo** (Ryzen AI Max+ 395, 128 GB), gpt-oss-120b:
  - About 30 tok/s (LM Studio, ROCm llama.cpp). [MindStudio (secondary)](https://mindstudio.ai/blog/local-llm-benchmarks-ryzen-ai-halo)
  - About 31-46 tok/s on the Radeon 8060S with llama.cpp. [llocalhost (secondary)](https://llocalhost-web.pages.dev/speed-bench/all-results)
- **SSD streaming:** the highest decode found was PagedMoE "copy-back" decoding at **48.8 tok/s**. That is on Qwen3.8-Flash-Next at a prune setting on a 48 GB M5 Pro, not a 100B+ exact run. [PagedMoE README](https://github.com/noahclark556/paged-moe)
- **Speculation on gpt-oss-120b:** NVIDIA's TensorRT-LLM EAGLE-3 guide targets GB200/B200 only. No consumer or Mac numbers for gpt-oss-120b with EAGLE-3 or MTP were found. [NVIDIA TensorRT-LLM tech blog 11](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog11_GPT_OSS_Eagle3.html)

### Inferences
- **The one >200 tok/s 100B+ result needs three conditions at once:**
  - About 5B active parameters (about 3.5 GB/token).
  - About 1.8 TB/s of GDDR7.
  - The whole model resident in 96 GB of VRAM.
  - Even so it reaches only about 43% of peak bandwidth (see Q4). With 10B+ active, the same GPU drops to about 100 tok/s.
- **A Mac would need more than bandwidth to reach 200.**
  - Without speculation, a Mac needs ≥2.3× the best M5 Max result.
  - At the 50-60% MoE efficiency estimated above, gpt-oss-120b-class bytes (≈3.5 GB/token) at 200 tok/s need ≈700 GB/s effective, i.e. about 1.2-1.4 TB/s nominal. That is above the M5 Max (614 GB/s) and at or above the M5 Ultra's 1.2 TB/s.
  - So on Apple Silicon, 200+ tok/s for a 100B+ MoE realistically needs several tokens per verify pass (speculation/MTP with a small expert union), fewer bytes per token (lower-bit experts, expert budgets), or both.
  - That is consistent with the levers in Q2/Q3.
- **Speculation has not yet been shown to push consumer 100B+ MoE decode past 200 tok/s.** The only consumer single-user MTP datapoint found (397B NVFP4) was a 22% slowdown.

### Gaps
- No primary measurement found for gpt-oss-120b (or any 100B+ MoE) with EAGLE-3/MTP/n-gram speculation on consumer GPUs or Macs.
- Hardware Corner does not name the engine for its RTX Pro 6000 numbers, and the original Reddit data was not checked.
- No data found for multi-GPU consumer rigs (e.g. 2-4× RTX 3090/4090/5090) running a 100B+ MoE at batch 1.
- No measured M5 Ultra single-machine numbers for 100B+ MoEs.
