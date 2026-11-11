# External published results: large-MoE batch-1 decode with experts streamed from SSD or offloaded

Compiled 2026-11-04 for the "have we done something never achieved before?" check.
Sources: arXiv abstracts (export API), arXiv HTML full text (sentences and tables extracted locally), GitHub READMEs and docs. No PDFs were read.
Every number below appears on the cited page. My own conversions or arithmetic are marked **(conv.)**.

**Conventions**
- **Fast mem** = GPU VRAM + host RAM used or capped, as the source reports it. **RSS-only** means the source reports process RSS. RSS leaves out the OS page cache that file-backed expert reads fill, so real occupancy can be higher. The Memory-Sovereign paper (2608.23805) makes exactly this point.
- **exact (verified)**: the source shows bit- or token-identity against a reference. **exact (by construction)**: the source says there is no approximation beyond the chosen checkpoint precision but shows no identity check. **lossy**: the system adds expert quantization beyond the checkpoint, skips or prunes experts, lowers top-k, or uses approximate prediction.
- GGUF runs (Q4_K_M, Q4_0, q2_k) are counted at the artifact's own precision. Quantization the system adds itself is counted as lossy.
- Buckets are by reported memory. An RSS-only row sits in the bucket of its RSS, flagged.

---

## 0. Correction to the brief: Memory-Sovereign Inference (arXiv 2608.23805)

- **The 12.0 tok/s MTP run is not Qwen3-Next-80B.** Table 9, campaign C4: "Qwen3.6 temporal MTP", a canonical **Qwen3.6-27B Q4_K_M** target plus its linear MTP block. Prompt 8,192 tokens, 512 outputs. Contract: host 7.5 GiB + GPU 11.5 GiB. Result: **12.016 end-to-end tok/s, 14.599 generation-phase tok/s**, with 723 full logit rows checked bitwise against a same-build oracle. Section C9 calls this Qwen3.6 path "resident". At 27B total it is outside the >= 80B scope.
- **Qwen3-Next-80B-A3B Q4_K_M:** "A 512+64 mechanism diagnostic decodes at **5.037 tokens/s without reuse and 11.636 tokens/s with LRU64**."
  - Resource contract for the Qwen3-Next campaigns: **host hard cgroup 11 GiB + GPU audited 23 GiB** on a physical 24 GiB RTX 3090. That is up to 34 GiB of fast memory, so this run is **not** in the <= 24 GB bucket.
  - Hardware: Ryzen 7 5800X, 2 TB WD_BLACK SN850X.
  - Exactness: all 64 token IDs and every byte of the 64 full logit rows match a zero-cache oracle. This is shown for the principal 32K+64 cell. The paper does not separately state exactness for the 512+64 diagnostic.
- The paper mentions "an external 8.803 tokens/s point" but does not identify it.

---

## 1. >= 80B MoE, total fast memory <= 24 GB

| system | model (total / active) | precision | fast mem used | storage | hardware | decode tok/s (bs 1) | date | link | confidence |
|---|---|---|---|---|---|---|---|---|---|
| Palm-Infra "mollm" (Tencent Youtu Research), SSD expert offload | Qwen3.5-122B-A10B (122B / 10B) | **lossy**: the system's own W4 expert package from the BF16 base; the SSD doc gives no quality number | **14.64 GiB peak RSS** (10 GiB expert RAM cache); **20.60 GiB peak RSS** (16 GiB cache); RSS-only | internal Mac SSD, `pread` workers; dense weights locked in RAM | "48GB Apple Silicon Mac", CPU only. The README's CPU protocol is Apple M5 Pro, 4 threads; the SSD rows do not restate the chip. | **16.19** (10 GiB cache), **16.53** (16 GiB cache); greedy, 16-token prompt, 256 generated, 3-process median | sweep run 2026-07-29 | [README](https://github.com/TencentYoutuResearch/Palm-Infra), docs/ssd-offload.md | **unsure**: RSS excludes the OS file cache on a 48 GB machine; engineering artifact, no paper |
| Palm-Infra | Qwen3.8-Flash-Next (125B / 6B active, plus 51B n-gram tables, per arXiv 2608.30320) | NVFP4 experts; not verified whether this is an official release format; no exactness gate quoted | **22.0 GiB peak RSS** (4 threads, 16 GiB cache); RSS-only | Mac SSD | 48 GB Apple Silicon Mac | **15.74** (5-process median, 52-token prompt, 64 out). Best observed run: 21.8 with 8 threads and a 20 GiB cache; its RSS is not given and is likely above 24 GB | 2026 (performance doc) | [docs/performance.md](https://github.com/TencentYoutuResearch/Palm-Infra) | **unsure**: RSS-only |
| BigMoMo | Qwen3-Next-80B-A3B (80B / 3B), scalability study only | Q4_0 targets with speculative decoding; the paper calls its expert pruning lossy (GSM8K 90.70 to 90.57) | phone DRAM 12, 16 or 24 GB; the text does not say which phone ran the 80B model | UFS 4.x flash | OnePlus 15 / 13 / Ace 5 | **not tabulated**, Fig. 1 only; "1.4x–1.8x over the best speculative MoE baseline from 8B to 80B" | 2026-09-13 | [2609.14643](https://arxiv.org/abs/2609.14643) | **unsure**: figure only |

**Note:**
- **Best lossy:** Palm-Infra, Qwen3.5-122B-A10B W4, **16.19 tok/s at 14.64 GiB peak RSS** (16.53 tok/s at 20.60 GiB). This is RSS-only, on a 48 GB Mac.
- **Best exact:** none found with <= 24 GB of verified, page-cache-inclusive accounting. The nearest exact rows are above the bucket (section 1b): MSI at 11.636 tok/s with 11 + 23 GiB, and Cacheable-by-Design at 0.441 tok/s on an 8 GB GPU + 32 GB RAM machine.

## 2. >= 80B MoE, total fast memory <= 12 GB

| system | model | precision | fast mem used | storage | hardware | decode tok/s | date | link | confidence |
|---|---|---|---|---|---|---|---|---|---|
| Palm-Infra | Qwen3.5-122B-A10B | lossy, W4 | **5.90 GiB peak RSS** (1 GiB expert cache); RSS-only | Mac SSD | 48 GB Apple Silicon Mac | **12.38** (hit rate 47.9%, 1.72 GB/token logical expert loads) | 2026-07-29 | [docs/ssd-offload.md](https://github.com/TencentYoutuResearch/Palm-Infra) | **unsure**: 1.72 GB/token x 12.38 tok/s is about 21 GB/s (conv.). The doc itself warns this counter is not physical traffic. Repeat reads were most likely served by the OS page cache of the 48 GB machine, so this is not a <= 12 GB result under page-cache-inclusive accounting |
| Colibrì (individual GitHub engine) | GLM-5.2 (744B per the README) | int4 container; the README does not say whether that matches the release format | "resident 9.9 GB" (banner; the README does not name the machine) | SSD streaming, CPU | "25 GB laptop" | **0.05–0.1 cold** | log dated 2026-07 | [colibri](https://github.com/JustVugg/colibri) | **unsure**: the banner RSS and the speed may come from different runs; artifact README |
| BigMoMo | Qwen3-Next-80B-A3B | as in section 1 | OnePlus 15: 16 GB DRAM, 11.52 GB available. Ace 5: 12 GB, 6.00 GB available. The text does not say which phone ran the 80B model | UFS | phones | figure only | 2026-09 | [2609.14643](https://arxiv.org/abs/2609.14643) | **unsure** |

**Note:**
- **Best lossy:** Palm-Infra, **12.38 tok/s at 5.90 GiB RSS**, with the page-cache caveat above.
- **Best exact:** none found.

## 1b. >= 80B MoE, memory-constrained but above 24 GB (24–64 GB): the closest competitors

| system | model | precision | fast mem used | storage | hardware | decode tok/s (bs 1) | date | link | confidence |
|---|---|---|---|---|---|---|---|---|---|
| Evolving Agent Teams (agent-optimized MLX MoE offload engine) | Qwen3.8-Flash-Next (125B / 6B active; "about 79 GB of weights plus a 32 GB per-layer embedding table"; 512 experts per layer, top-10 plus shared) | **exact (by construction)**: "no expert skipping or pruning, no lower top-k, no fewer bits". The gates are top-1 agreement and KL against the naive engine, not bit identity | **peak memory <= 37 GiB** (validity gate) | SSD, "streaming experts that do not fit in memory ... through an offload cache" | 48 GB M4 Pro, MLX 4-bit | **30.3** (same-machine re-measure). First run 27.3; naive engine 13.4; starting commit 26.7; solo-agent engine 28.2 | 2026-10-27 | [2610.36848](https://arxiv.org/abs/2610.36848) | **sure** on the number; **unsure** on prompt and output lengths (not given) |
| Memory-Sovereign Inference | Qwen3-Next-80B-A3B (80B / 3B), Q4_K_M | **exact (verified)** on the principal 32K+64 cell, against a zero-cache oracle | host **hard 11 GiB** + GPU **audited 23 GiB** (34 GiB) | WD_BLACK SN850X NVMe | Ryzen 7 5800X + RTX 3090 24 GiB | **11.636** with LRU64 (512+64 diagnostic); 5.037 with no reuse | 2026-08-24 | [2608.23805](https://arxiv.org/abs/2608.23805) | **sure**: the only row with hard, page-cache-inclusive (cgroup) accounting |
| PagedMoE | Qwen3.5-397B-A17B (397B / 17B) | **Uniform4**: 4-bit routed experts, quantized by the system from BF16, so lossy against BF16. **Dynamic Avg2.5**: lossy; GSM8K-CoT 96.89 vs 97.12 for GPTQ-INT4, WikiText-2 perplexity 3.240 vs 3.201 | **24 GiB DRAM expert cache**. Avg2.5 total is "approximately 29.2 GiB" (24 + 5.2 GiB non-cache). The GPU handles dense and always-active parts; its memory is not in that total | 4 TB PCIe 5.0 x4 NVMe | Ryzen 9 9950X, 64 GB DDR5, RTX 5090 32 GiB | Uniform4 **5.71**; Avg2.5 **8.89**. Baseline ik_llama.cpp: 2.16 (24 GiB cache), 2.48 (32 GiB) | 2026-11-02 | [2611.03173](https://arxiv.org/abs/2611.03173) | **sure** on speeds; **unsure** on total fast memory (GPU not counted) |
| PagedMoE | GLM5.2 | same scheme | 24 GiB cache | same | same | Uniform4 **1.58**; Avg2.5 **2.64**; ik_llama.cpp 0.72 / 0.77 | 2026-11 | same | sure |
| SSD-LLaMA | Kimi-K2.7-Code (1T), Q4_0 | **exact (by construction)**: "executes every selected expert without pruning or substitution"; no token-identity check reported | RTX 5090 32 GB + **32 GB RAM** | 4 TB NVMe, up to 9 GiB/s | RTX 5090 | **1.03** | 2026-09-16 | [2609.18110](https://arxiv.org/abs/2609.18110) | **sure** |
| SSD-LLaMA | "DeepSeek" (likely DeepSeek-V4-Flash, 284B / 13B, MXFP4 experts) | exact (by construction) | 22 GiB VRAM expert cache; main testbed is RTX 5090 32 GB + **16 GB DRAM** | same NVMe | Intel Ultra5 230F + RTX 5090 | **148.87 ms TPOT, so 6.72 tok/s (conv.)**, 80.1% VRAM hit rate. Kimi-K3 (2.8T): 0.465 tok/s, configuration not stated | 2026-09 | same | **unsure**: the paper only says "DeepSeek"; the model is inferred |
| RotaryQuant | "Nemotron-H 120B" (hybrid Mamba+MoE; active count not stated) | **lossy**: 2-bit routed experts; no weight-quantization quality delta, only a 12/12 quality gate | "within 32 GB" (measured peak) | LRU disk eviction "where needed" (not confirmed for this model) | Apple M4 Max with 128 GB | **14.85** (Table 4; batch not stated) | 2026-08-08 | [2608.08081](https://arxiv.org/abs/2608.08081) | **unsure** |
| flash-moe (individual GitHub engine) | Qwen3.5-397B-A17B | **lossy**: K=4 routed experts (+1 shared) per the README. Secondary coverage says the model default is 10; the page does not state it. 4-bit experts; the 2-bit variant "breaks JSON/tool calling" | about 6 GB resident + about 35 GB OS page cache on a 48 GB machine | internal SSD (209 GB at 4-bit) | MacBook Pro M3 Max 48 GB | **4.36** (4-bit, FMA kernel); 5.74 (2-bit); 7.05 (2-bit warm-cache single-token peak) | about Mar 2026 (the repo page shows no date; based on secondary coverage) | [flash-moe](https://github.com/danveloper/flash-moe) | **unsure**: date and K default |
| Palm-Infra | DeepSeek-V4-Flash (284B / 13B) | **exact (by construction)**: checkpoint-native FP8/MXFP4, no output check quoted | 16 GiB expert cache + dense weights; **RSS not reported** | Mac SSD | 48 GB Apple Silicon Mac, 4 threads | **5.71** (provisional single-process, pp256 + tg64). llama.cpp: OOM | 2026 | [Palm-Infra](https://github.com/TencentYoutuResearch/Palm-Infra) | **unsure**: memory |
| Palm-Infra | Hy3-295B-A21B | lossy, W4G128 | 16 GiB cache + dense | Mac SSD | 48 GB Mac | **3.41** | 2026 | same | unsure |
| Tied Trit-Planes ("fucina" Zig engine) | DeepSeek-V4-Flash-0731 (284B / 13B) | **mxfp4**: the released bytes repacked "without any quantization step", so exact by construction. **tq2_0_fx4**: lossy, perplexity 5.08 vs 4.39 at 512 tokens. **q4_k**: lossy | M1 Max with 64 GB; the cache budget for Table 3 is not stated. The text reports a knee at a 26–28 GB budget | USB SSD with an NVMe cache tier | M1 Max 64 GB; i9-13950HX 128 GB | M1 Max: mxfp4 **2.48–2.54**, q4_k 2.91–2.95, tq2 3.11–3.12. i9 (NVMe, evicted cache): mxfp4 2.40–2.47 | 2026-08-09 | [2608.08910](https://arxiv.org/abs/2608.08910) | **unsure**: occupied memory |
| Cacheable by Design? | Qwen3-235B-A22B (235B / 22B), Q4_K_M (134 GB) | **exact (by construction)**: llama.cpp with `-ngl 99 -ncmoe 94`, offload only | RTX 3070 8 GB + 32 GB DDR4 (a 40 GB machine; the paper's "18 GB fast-memory budget" is a modelling parameter) | WD SN530 NVMe, about 2.4 GB/s | i9-12900 | **0.441 warm**; 0.128 cold | 2026-08-18 | [2608.18261](https://arxiv.org/abs/2608.18261) | **sure** on speed; **unsure** on occupied memory |
| Colibrì | DeepSeek V4 Flash (284B) | quantization not stated | **32 GB** RAM, CPU only; or RTX 5080 + 32 GB | NVMe | Ryzen 7 5800X; RTX 5080 | **0.93** (32 GB, CPU); 1.5–1.6 (RTX 5080) | 2026 | [colibri](https://github.com/JustVugg/colibri) | **unsure**: artifact README |

**Note:**
- **Fastest >= 80B streaming result under 48 GB:** Evolving Agent Teams, **30.3 tok/s**, Qwen3.8-Flash-Next at <= 37 GiB. Exact by construction, no bit-identity claim.
- **Best verified exact:** MSI, **11.636 tok/s** (Qwen3-Next-80B, 11 + 23 GiB hard/audited).
- **Best lossy:** RotaryQuant at 14.85 (unsure). For a frontier-size model, PagedMoE at 8.89 (Qwen3.5-397B, 2.5-bit average).

## 3. >= 80B MoE, large-DRAM offload (not memory-constrained: the speed ceiling)

| system | model | precision | fast mem | hardware | decode tok/s (bs 1) | date | link | confidence |
|---|---|---|---|---|---|---|---|---|
| Cloud-grade SLOs for local MoE (CPU-GPU hybrid) | DeepSeek-R1/V3 671B (37B active) | FP8 "intact, original-precision": **exact (by construction)** | **1.15 TB DDR5** + 2x RTX 5090 32 GB | 2x EPYC 9355 | **21.5** FP8 (single-stream, short prompts). **28** Q4_K_M (lossy against FP8; 1K–8K context). Kimi-K2: **22.4**. Same-machine baselines: KTransformers about 22, ik_llama.cpp 14 | 2026-06-09 | [2606.10493](https://arxiv.org/abs/2606.10493) | **sure** on numbers; **unsure**: the paper alternates between "V3" and "R1", and it is unclear whether one or both GPUs are used in decode |
| FreeToken | DeepSeek-V4-Flash (284B / 13B) | native MXFP4 experts, consumed "bit-exactly": exact (by construction), no output comparison | RTX 5090 32 GB + 180 GiB (server) or 192 GiB (desktop) DRAM; host-resident expert pool | 2x Xeon Gold 6459C or Ryzen 9 9950X3D | **22–25** ("per-request mean", batch not stated) | 2026-08-17 | [2608.16157](https://arxiv.org/abs/2608.16157) | **unsure**: which 5090 machine, and batch |
| FreeToken | GLM-5.2 (753B / 40B), NVFP4 experts, 433 GB | exact (by construction) | RTX PRO 6000 96 GB (workstation card) + 512 GiB | Xeon Platinum 8559C | **14.9** (llama.cpp 7.3) | 2026-08 | same | sure |
| KTransformers (official DeepSeek tutorial) | DeepSeek-V3/R1 671B, Q4_K_M | **8 experts: exact** at Q4_K_M. **6 experts: lossy** (top-k 8 to 6) | **>= 14 GB VRAM** (4090/4090D 24 GB) + **382 GB DRAM** | 2x Xeon Gold 6454S | 8 experts: 8.73 (32 cores), 11.26 (dual socket), up to 13.4–13.5 (V0.2.1). 6 experts: 13.69, up to 16.8 (V0.2.1). llama.cpp: 4.51 | early 2025 (V0.2 / V0.2.1; no dates on the fetched page) | [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md) | **sure** on numbers; **unsure** which context length each V0.2.1 table column is |
| KTransformers (SOSP'25 paper) | DeepSeek-V3 | Expert Deferral is lossy (<= 0.5% average accuracy drop) | A100 + dual Xeon (datacenter GPU) | — | about 5.87 before Expert Deferral; +33% with it | 2025 | secondary (CSDN write-up) | **unsure**: secondary source only |
| NVIDIA, "Efficient, VRAM-Constrained xLM Inference on Clients" | Qwen3-235B-A22B-Instruct-2507, q2_k (77 GB) | **lossy** (2-bit k-quant; no quality claim read) | cli3: **2 GB VRAM budget** + 256 GB RAM. cli2: RTX 5070 Ti 16 GB + 128 GB RAM | RTX 5090 + EPYC; RTX 5070 Ti + Ryzen 7 | cli3: **7.7** at 2G (1K context), 11.5 at 32G, 5.2 at 2G / 16K. cli2: 7.7 at 16G | 2026-04-29 | [2604.26334](https://arxiv.org/abs/2604.26334) | **sure** on the tables (batch 1 stated); RAM residency is not stated, but the RAM is at least 128 GB |
| Colibrì, "Machine D" | Qwen3.8-Flash-Next IQ2_XS (68 GB) | **lossy** (about 2-bit) | RTX PRO 4000 24 GB + 62 GB RAM (86 GB; the model fits) | i5-13500 | **51.4** (CUDA). Also DeepSeek V4 Flash UD-IQ2_XXS (85 GB, Vulkan): 6.18–6.25 | 2026 | [colibri](https://github.com/JustVugg/colibri) | **unsure**: artifact |
| Integer-Exact Microscaling (resident reference, no offload) | Qwen3.8-Flash-Next | MLX MXFP4 W4A16; the W4A8 system is bit-identical to its quantized reference | 58.9–65.0 GiB weights resident | M5 Max | **42.4** | 2026-10-31 | [2611.00486](https://arxiv.org/abs/2611.00486) | sure; shows the resident ceiling for this model on a Mac |

**Note:**
- **Best exact:** 2606.10493, DeepSeek 671B FP8 at **21.5 tok/s** with 1.15 TB DRAM. FreeToken, DeepSeek-V4-Flash native MXFP4 at **22–25 tok/s** with 180–192 GiB DRAM.
- **Best lossy:** 2606.10493 Q4_K_M at **28 tok/s**. Colibrì IQ2_XS Qwen3.8-Flash-Next at 51.4 (unsure; the model mostly fits in RAM + VRAM).

## 4. Context: smaller MoEs (plus the two dense must-cover papers)

| system | model | precision | fast mem | storage | hardware | decode tok/s (bs 1) | date | link | confidence |
|---|---|---|---|---|---|---|---|---|---|
| Mixtral offloading (Eliseev & Mazur) | Mixtral-8x7B (47B / 13B) | **lossy**: mixed quantization, 2- or 3-bit experts | VRAM of each card + host RAM (amounts not read) | host RAM | RTX 3080 Mobile / RTX 3060 / T4 / A100 | 2-bit: **2.655 / 2.278 / 2.092 / 3.061**. 3-bit: 2.475 / 2.038 / 1.603 / 2.845 | 2023-12-28 | [2312.17238](https://arxiv.org/abs/2312.17238) | sure (Table 2) |
| Fiddler | Mixtral-8x7B, uncompressed (>90 GB) | exact (by construction) | a single 24 GB GPU + host DRAM holding the experts | DRAM | 24 GB GPU | **"over 3"** (v1 abstract). v3: 1.26x over baselines at single batch; absolute numbers only in figures | 2024-02 / v3 2025-05 | [2402.07033](https://arxiv.org/abs/2402.07033) | sure (abstract) |
| MoE-Infinity | DeepSeek-V2-Lite (16B) | exact (by construction) | A5000 24 GB + host DRAM | DRAM | RTX A5000, PCIe 4.0 | TPOT **155 ms, so 6.45 tok/s (conv.)**. vLLM 485 ms, DeepSpeed 737 ms, Ollama 2590 ms | 2024-01 / v3 2025-03 | [2401.14361](https://arxiv.org/abs/2401.14361) | sure (Table 1) |
| HOBBIT | Mixtral-8x7B and others | **lossy**: cache-miss experts replaced by low-precision versions | edge budgets | llama.cpp-based | RTX 4090; Jetson AGX Orin | figures only; "up to 9.93x" decode vs state-of-the-art offloading | 2024-11 | [2411.01433](https://arxiv.org/abs/2411.01433) | **unsure**: no absolute number in the text |
| PowerInfer-2 | TurboSparse-Mixtral-47B | lossy against Mixtral (a different, sparsified model; "negligible accuracy degradation") | **19 GB used gives 11.68; 7 GB available gives 2.13** | UFS flash | OnePlus 12 | **11.68** | 2024-06 | [2406.06282](https://arxiv.org/abs/2406.06282) | sure |
| ProMoE | DeepSeek-V2-Lite, Mixtral, ... | exact (prefetch) | consumer GPU + DRAM | DRAM | RTX 4090 | figures; 2.07x average (up to 5.02x) decode | 2024-10 / v3 2025-09 | [2410.22134](https://arxiv.org/abs/2410.22134) | unsure |
| AdapMoE | Mixtral-8x7B | **lossy**: 25% fewer activated experts; claims "no accuracy degradation" | edge | — | A6000 / RTX 4090 mentioned | 1.35x; Table 2 latency 0.392 to 0.288 s | 2024-08 | [2408.10284](https://arxiv.org/abs/2408.10284) | **unsure**: platform for Table 2 |
| SwapMoE | Switch Transformer | **lossy**: Rouge-2 -0.041 | 14.2 GiB down to 4.7 GiB; with a 2.0 GiB budget, 1.64 GiB peak GPU and 0.82 s latency | — | Jetson Nano / AGX | latency only | 2023-08 / v4 2024-05 | [2308.15030](https://arxiv.org/abs/2308.15030) | sure (abstract) |
| EdgeMoE | Switch, GPTSAN (<= 26B) | **lossy**: expert-wise bit-width adaptation | 1.8–3.2 GB | external storage | Jetson TX2, Raspberry Pi 4B | figures only | 2023-08 / v2 2025-03 | [2308.14352](https://arxiv.org/abs/2308.14352) | unsure |
| DAOP | Mixtral-8x7B; Phi-3.5-MoE | approximate: CPU pre-calculation with "graceful degradation"; claims accuracy is kept | 25% of experts cached on the GPU | DRAM | A6000 + Xeon Gold 6326 | **3.23** (Mixtral); 5.03 (Phi-3.5-MoE) | 2024-12 / v3 2026-08 | [2501.10375](https://arxiv.org/abs/2501.10375) | sure |
| FloE | Mixtral-8x7B | **lossy**: 9.3x expert compression; 4.4–7.6% average degradation | **11 GB VRAM** | host RAM | RTX 3090 | figures; 48.7x vs DeepSpeed-MII | 2025-05 | [2505.05950](https://arxiv.org/abs/2505.05950) | unsure |
| HybriMoE | Mixtral, DeepSeek-V2-Lite, a Qwen2 MoE | exact (scheduling) | A6000 + host | DRAM | A6000, Xeon Gold 5220R | figures; 1.70x decode vs kTransformers | 2025-04 | [2504.05897](https://arxiv.org/abs/2504.05897) | unsure |
| FineMoE (fMoE) | Mixtral, Qwen MoE, Phi-3.5-MoE | exact (prefetch and caching) | six-GPU testbed | DRAM | — | 47% latency reduction (figures) | 2025-02 / v2 2025-10 | [2502.05370](https://arxiv.org/abs/2502.05370) | unsure |
| FlashMoE (SSD) | models not identified in the text read | exact (cache policy) | "limited RAM" | SSD | user-grade desktop (RTX 5070 Ti mentioned) | up to 2.6x; +51% hit rate (figures) | 2026-01 / v2 2026-10 | [2601.17063](https://arxiv.org/abs/2601.17063) | unsure |
| WiSP | Qwen3-30B-A3B BF16 | **exact (verified)**: byte-identical outputs | about 22 GiB GPU budget | host RAM | RTX 3090 | **9.93** (static offload 4.93) | 2026-06 | [2606.21868](https://arxiv.org/abs/2606.21868) | sure |
| Edge0 | Qwen3.6-35B-A3B-derived (K=4, retrained prerouter + LoRA) | **lossy** ("within a few points" of its fp16 teacher) | **2.9 GiB peak active** (24 GB machine) | SSD | Mac mini M4 Pro 24 GB | **20.4** | 2026-09 | [2609.18063](https://arxiv.org/abs/2609.18063) | sure |
| FreeToken | Qwen3.6-35B-A3B NVFP4 | exact (by construction) | 8 GB VRAM + 32 GiB | DRAM | RTX 4060 Laptop | **39.3** | 2026-08 | [2608.16157](https://arxiv.org/abs/2608.16157) | sure |
| DraftExpert | DeepSeek-V2-Lite (16B) Q4_0 | exact speculative | phone memory | flash to NPU | Snapdragon 8 Elite | 10.18 autoregressive; **15.47** with DraftExpert | 2026-07 | [2607.24434](https://arxiv.org/abs/2607.24434) | sure |
| MawForge | Qwen3.6-35B-A3B Q4_K_M | not evaluated | 18 GiB target | SSD | MacBook Pro M5 Pro 24 GB | up to **19.87** | 2026-06 | [2607.09686](https://arxiv.org/abs/2607.09686) | sure |
| Specter | Qwen1.5-MoE; Phi-3.5-MoE | quantized speculative draft (exactness not read) | T4 16 GB + host | DRAM | T4 | 35.3 ms/token, so 28.3 tok/s (conv.); 163.3 ms | 2025-11 | [2511.14102](https://arxiv.org/abs/2511.14102) | sure |
| CPU-GPU collaborative (2512.16473) | Mixtral-8x7B; Phi-3.5-MoE | not read | RTX 4090 + host | DRAM | RTX 4090 | **4.8**; 10.4 | 2025-12 | [2512.16473](https://arxiv.org/abs/2512.16473) | sure |
| PowerInfer (**dense**, not MoE) | OPT / Falcon / LLaMA-ReGLU up to 175B | activation predictor (small accuracy deltas, Table 7) | RTX 4090 24 GB + host | DRAM | PC-High | average 13.20 (quantized), peak 29.08; FP16 average 8.32 | 2023-12 | [2312.12456](https://arxiv.org/abs/2312.12456) | sure; dense |
| LLM in a flash (**dense**, Apple) | OPT-6.7B FP16 | predictor-based sparsity (approximate) | about 6.5 GB DRAM for a 13.4 GB model | flash | M1 Max Metal; RTX 4090 | 565 ms/token, so 1.77 tok/s (conv.) on Metal M1; 84 ms, so 11.9 (conv.) on GPU; 60 ms speculative | 2023-12 / v3 2024-07 | [2312.11514](https://arxiv.org/abs/2312.11514) | sure; dense |

**Note:**
- **Best exact** among memory-constrained context rows: FreeToken, 35B at **39.3 tok/s** on an 8 GB laptop GPU (+32 GiB RAM). Under the tightest memory: WiSP, verified byte-identical, 9.93 at about 22 GiB.
- **Best lossy:** Edge0, **20.4 tok/s at 2.9 GiB** active memory.

## 5. Not batch-1, datacenter-only, or no absolute number

- **Throughput at batch > 1, so not comparable:**
  - MoE-Gen: DeepSeek-R1 671B, 17 tok/s; DeepSeek-V2 236B, 31 tok/s. Llama.cpp baseline 0.9 and 1. The A5000 rig in Table 1 has 512 GB host memory.
  - MoE-Lightning: up to 10.3x on a T4.
  - Klotski: Mixtral-8x22B on an RTX 3090 + 256 GB, 1.325 tok/s multi-batch.
  - ExpertFlow (2410.17954), SpecOffload (2505.10259), DynaExq (2511.15015; Qwen3-80B, batch 32).
- **Datacenter-only:**
  - Pre-gated MoE: A100, Switch, 111 / 42 tok/s throughput.
  - SiDA-MoE; DeepSpeed-MoE and DeepSpeed-Inference / ZeRO-Inference (throughput-oriented NVMe/CPU offload).
  - KTransformers SOSP'25 (A100).
  - WiSP MiniMax-M2 229B FP8 on an H100: 3.70 tok/s.
  - FluxMoE (H20, L40S); MoE-CORE (NPU, DeepSeek-V4-Flash TPOT 38.0–44.8 ms).
  - "Who should own the expert cache?" (GH200); the HBF papers (simulation).
- **Absolute numbers only in figures:** HOBBIT, Fiddler v3, ProMoE, HybriMoE, FloE, FineMoE, EdgeMoE, FlashMoE, BigMoMo (80B), SSD-LLaMA (GLM-5.2, DeepSeek-V4-Flash, RTX 5060 Ti configurations), and ATSInfer (2607.10183). ATSInfer covers Qwen3-Next-80B, Qwen3.5-122B and GPT-OSS-120B on an RTX 4090 + 64 GB DDR4, at batch 1; its maximum decode speedups are 3.12x over llama.cpp on desktop and 3.29x on laptop.
- **No measured >= 80B numbers:**
  - WISP (GitHub `zeroextub-collab/wisp`): its >= 80B rows are "estimated from physics".
  - Unsloth DeepSeek-R1 1.58-bit "20 GB RAM": a community minimum; the speeds found are estimates (secondary sources).
  - MoE-Direct, MESH (KTransformers PR #2003) and DeepNVMe, cited by MSI: not checked.

## Rows marked unsure

1. Palm-Infra, all rows: RSS-only accounting on a 48 GB Mac. This affects the <= 12 GB and <= 24 GB "best" rows.
2. BigMoMo, Qwen3-Next-80B: figure only, phone unknown.
3. Colibrì, all rows: artifact README; the banner and the speeds may come from different runs.
4. Evolving Agent Teams: benchmark lengths unknown; no bit-identity claim.
5. MSI: exactness of the 512+64 diagnostic is not separately stated.
6. PagedMoE: total memory leaves out the RTX 5090 VRAM.
7. SSD-LLaMA: the "DeepSeek" model identity is inferred.
8. RotaryQuant: unclear whether this model streams from disk; no batch stated.
9. flash-moe: date and K default come from secondary sources.
10. Tied Trit-Planes and Cacheable-by-Design: occupied memory not reported.
11. 2606.10493: model naming varies (V3 vs R1) and GPU count is unclear.
12. FreeToken: which RTX 5090 machine, and batch size.
13. KTransformers: V0.2.1 table column mapping; the SOSP numbers come from a secondary source.
14. The context-table rows marked unsure (figures only).

**What this means for a "first" claim (as far as these sources go):**
- **Already published:**
  - >= 80B at more than 10 tok/s in a reported RSS of 15–21 GiB (Palm-Infra, July 2026; lossy W4).
  - 30 tok/s at <= 37 GiB (Evolving Agent Teams, Oct 2026).
  - Exact, hard-capped Qwen3-Next-80B at 11.6 tok/s with 11 GiB host + 23 GiB GPU (MSI, Aug 2026).
- **Not found:** an exact (verified) >= 80B result under a hard, page-cache-inclusive total of 24 GB or 12 GB.
