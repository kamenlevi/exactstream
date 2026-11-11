# Prior art for the 24 GB goal (Oct 19, research-first pass)

Goal (Kamen, 11:35): about today's speed for Qwen3.8-Flash-Next 125B in 24 GB total RAM on the M5 Pro (64 GB, NVMe
~6-7 GB/s), i.e. an expert cache of ~17 GB instead of ~34 GB, at equal quality. Today (BigMoE, exact): ~40 tok/s at a
34 GB cache with MTP, 27-28 tok/s at 24 GB; eager decode (MTP off, our bench): 28.2 / 19.5 tok/s at 34 / 17 GB.
Method: four parallel survey passes (MoE offloading systems; low-bit and nested experts; Mac/consumer runtimes;
cross-field principles), the field's own words, last 12 months, citations both ways. SR = self-reported.

## 1. MoE offloading, expert caching, prefetching and I/O (systems)

| work | what it does | setting | measured | what we take |
|---|---|---|---|---|
| LLM in a flash (ACL'24, 2312.11514) | predicted FFN rows, sliding window, bundled up/down rows, F_NOCACHE + 32 threads | M1 Max/M2 Ultra, OPT-6.7B | I/O 2196 → 87 ms/token; random reads need ≥ 32 KiB chunks | keep reads ≥ tens of KiB (relevant if we split experts) |
| Mixtral-offloading (2312.17238) | per-layer LRU + next-layer gate on the current state (= our router reuse); speculative loads join the LRU only if used | T4-A100, Mixtral | 1.57 → 2.09 tok/s (T4); next-layer guess 84.6 % | speculative fetches never evict |
| EdgeMoE (TMC'25), Fiddler (ICLR'25), MoE-Lightning (ASPLOS'25), KTransformers (SOSP'25), PowerInfer-2 | sensitivity bits; CPU computes misses; batch pipelines; AMX CPU experts + deferral; neuron clusters on UFS | edge/PCIe/phones | 2.6-3.0× (EdgeMoE, lossy); 1.26×; 10× (batch); deferral 1.45× (lossy); 11.7 tok/s at 19 GB (phone) | compute resident experts while misses stream in; deferral only as a lossy option |
| MoE-Infinity (2401.14361) | activation-matrix prefetch, layer-weighted eviction | A5000 | TPOT 155 vs 485 ms (vLLM) | eviction score for the sim |
| ProMoE (2410.22134) | ~2M-param learned predictor per layer, preemptible chunked prefetch, cached experts first | RTX 4090, Qwen2-57B | **router reuse 66.9 % vs learned 84.7 %**; 1.31× vs LRU | learned correction of router reuse; preemptible reads |
| AdapMoE (ICCAD'24), fMoE/FineMoE (EuroSys'26), Pre-gated MoE (ISCA'24), ExpertFlow (DAC'26) | adaptive top-k + per-layer DP budgets; semantic/trajectory maps, prefetch to a probability threshold; fine-tuned pre-gate; all-layer predictor + horizon S = N_e·E_s/(C_s·T_l) | Mixtral, Switch | 1.35× (lossy); TPOT −47 %; 1.5×; hit 91.9 vs 76.6 % | size prefetch by router confidence; the horizon formula |
| HOBBIT (2411.01433) | low-precision or skipped loads for low-gate misses; prefetch 1-3 layers | Orin (NVMe 3 GB/s), 4090 | up to 9.93× vs MoE-Infinity, GSM8K −0.01; 96 %-accurate prefetch adds only 1.01-1.05× | same idea as our planes |
| SliceMoE (DAC'26, 2512.12990), D2MoE (MobiCom'25) | bit-sliced LRU units, 0-2 critical experts get the LSB, prefill warmup; nested INT + trained router | modeled UFS 3.1; RTX 3060 + SSD | 1.81× / 1.64× decode; 1.14-1.39× | prefill-to-decode warmup |
| SmallThinker (2507.20984) | pre-attention router for storage prefetch (retrained) | PC, 8 GiB cap | 30.2 → 20.3 tok/s | needs retraining |
| FlashMoE (2601.17063) | small MLP eviction trained on Belady labels | 5070 Ti + 7.4 GB/s SSD | Belady 86 % vs LRU ~73 % hit; decode +7 % (Qwen3-30B) | realistic eviction gain ~+7 % |
| Cache-conditional experts (TMLR'25, 2412.00099) | routing biased toward cached experts | phones | misses 35 → 16 % at +0.5 % ppl | only inside the draft head (exact) |
| Not All Models Suit Offloading (2505.16056) | routing-locality study of 20 MoEs | — | knee at a cache ~2× the active experts | our misses are the tail far past the knee |
| DynaExq / DyMoE / DPA-MoE / DuoRate (2610.13328) | gate/frequency-driven mixed precision | Qwen3-Next-80B, Qwen3-30B | 77.6 vs 78.1 (int4); DuoRate reads 2.9 bits/weight at 4-bit quality (abstract) | allocation for planes |
| SSD-LLaMA (2609.18110) | one aligned block per expert, io_uring O_DIRECT, GPU rANS decode | 5090 + 9 GiB/s NVMe, Kimi-K2.7 | **77.7 % of SSD peak vs 43.2 % best baseline**; rANS −33 % bytes | one read per expert; entropy coding (our 125B codes: ~6 %) |
| MoE-SpeQ (2511.14102), DraftExpert (2607.24434), SpecFit (2610.15417), BigMoMo, Cascade, AcceptMoE, EVICT, EcoSpec, SPICE, SeqMoE, S2-MoE | drafter forecasts experts; draft truncation when the verifier's expert set grows past a budget; I/O-aware draft depth; Mamba2 routing forecaster (SeqMoE) | A100/4090/5090/Orin | DraftExpert verify 2.35× → 1.25× a step, 1.45× (lossless); SpecFit +7-29 %; Cascade +7-14 %; **MTP verify unions nearly additive** (31.4 of 32 on DeepSeek-V3.1); SeqMoE hit 91.7 vs 74.2 % at 25 % residency | **choose MTP depth by I/O cost; prefetch each draft's experts as soon as it exists** |
| SpecMD (Apple, ICML'26), Who Should Own the Expert Cache? (2608.12103), replay study (2608.07911), ReMoE (ICML'26), Speculating Experts (2603.19289), "Which expert misses are worth waiting for?" (Oct 2026) | least-stale eviction; LRU 44 vs Belady 62 %; per-step replay; router fine-tune for reuse; default-vector correction ~90 % recall; wait only for rank-1 or gate ≥ 0.15 | A100, GH200, Qwen3-Next-80B | substituting a cached expert collapses GSM8K; +5 pp hit (ReMoE, −0.76 GSM8K); 2.5-3.0× at +0.7-1.5 % ppl (abstract, lossy) | keep pending prefetches unevictable; replay fused steps in our simulator |

Closest setups (this model on Apple Silicon): SlotStream (M5 Pro 48 GB: 13.5 / 15.9 / 20.9-23.7 tok/s at 20 / 22 / 33 GB);
npanj llama.cpp fork (M5 Pro 64 GB, Q4_0: 18-18.6 tok/s at a 36 GiB cache, ~27.6 with the draft head); colibri (x86, MTP
acceptance 94-96 %); moe-prefetch (M3 Max). **We are ahead of every verified result for this model** (40 tok/s at 34 GB).


## 2. Low-bit and nested-precision experts (accuracy at 2-3 bits)

| work | method (calibration?) | models | accuracy | what we take |
|---|---|---|---|---|
| QMoE (2310.16795) | GPTQ ternary/2-bit + dictionary, 0.8 bit/param (calibrated) | Switch-c2048 1.6T | C4 loss 1.18 → 1.20 (2-bit) | only works for hugely over-parameterized experts (fits our 1.5-bit failure) |
| MoQE (2310.02410) | channel-wise RTN on experts | 5.3B MT model | int3 −0.3 BLEU; int2 needed QAT | experts tolerate low bits far better than dense layers |
| QuantMoE-Bench (2406.08155) | GPTQ g128 + structure heuristics | Mixtral, DeepSeek-MoE-16B | Mixtral 75.3 fp16 → 70.7 (3-bit) → 40.5 (2-bit) | shared, first-layer and hot experts deserve more bits |
| MC-MoE (ICLR'25, 2410.06270) | ILP bits per expert {1,2,3} + online pruning (calibrated) | Mixtral-8x7B | 2.54 bits: avg 67.5 vs 71.3; **GSM8K 37.7 vs 58.3** | static per-expert allocation at 2.5 bits still loses ~20 GSM8K points |
| MoEQuant (2505.03804) / EAQuant (2506.13329) | expert-balanced calibration, affinity-weighted GPTQ, router alignment | Qwen-MoE-14B, Mixtral | 4-bit GSM8K 58.4 vs 62.6 fp | calibration must cover all experts; AWQ is weak on MoE |
| MxMoE (2505.05799) | ILP per linear block, Hadamard + GPTQ | Qwen1.5-MoE, DeepSeek-V2-Lite, Mixtral | 2.25 bits still −8 to −13 points; 3.25 bits −1.1 to −1.6 | allocate per matrix, not per expert |
| AlphaQ (2606.04980) | spectral (Hill α) per-layer bits, calibration-free allocation | Qwen3-30B-A3B | 2.5 bits: 69.6 vs bf16 72.8 | the allocation metric works on dequantized 4-bit weights |
| TileQ (2605.09281) / KBVQ-MoE (ICLR'26) | shared low-rank tiles + GPTQ/GPTVQ residual (calibrated) | Qwen3-30B-A3B | 2-bit Wiki2 10.1 vs bf16 8.07 (**+25 %**); 3-bit 8.58 | best published 2-bit on a fine-grained Qwen3 MoE is still ~+25 % ppl |
| CCQ (2507.07145) | convolutional-code PTQ, shift decode | DeepSeek-V3, ERNIE-4.5-300B | DeepSeek-V3 2.06 bpw: 87.8 vs 90.0 (GSM8K 94.9 vs 96.2) | ~2 bpw near-equal only at 300-671B scale |
| GGUF data (R1 2505.02390, Unsloth) | k-quants, IQ quants | DeepSeek-R1, Qwen3.5-35B-A3B | R1 UD-Q2_K_XL 2.70 bpw −0.94 % (down_exps at q3_K); Qwen3.5-35B IQ2_XXS +17 % ppl | **down_proj is the most sensitive expert tensor**; calibrated 2-bit ≈ our RTN 2-bit |
| HOBBIT (2411.01433) | on a miss, a separate low-precision copy once higher ranks hold > 0.6 of the gate mass; skip above 0.9; top-1 always high | Mixtral | GSM8K 0.52 → 0.51 (67 % high, 30 % low, 3 % skipped) | cumulative-gate-mass cutoffs (cache-state-dependent) |
| D²MoE / MWQ (2504.15299) | nested 2-bit base + 1-bit residual planes up to 4 bits; trained bit-width router | Mixtral | Wiki2 4.28 (nested 2-4) vs 4.04 int8 | nesting is cheap |
| SliceMoE AMAT (2512.12990) | calibration-free truncation of codes and zero-point; MSB for all, LSB for 0-2 high-gate experts; MSB-only on an LSB miss | DeepSeek-V2-Lite, Qwen1.5-MoE | sliced 2-bit 10.18 vs direct 2-bit 9.46 (4-bit 7.06) | closest prior work to our planes; truncation without the zero-point fix diverges |
| DynaExq (2511.15015) | router-trace hotness picks the high-precision set | **Qwen3-80B-A3B (512 experts)** | avg 77.6 vs int4 78.1, int2 73.1; GSM8K 86.7 vs 87.0 / 81.0 | the only 512-expert data point: static int2 costs 6 GSM8K points |
| Low-rank compensation (2512.17073) | HQQ INT2/3 + per-expert SVD residual (rank 32-64) for the top-n experts | Mixtral | still far from fp (MMLU 47.5 vs 70.0) | our 4→2 residual is ~white noise: little to gain |
| **DyMoE (2603.19172, Mar 2026)** | gate score picks 4-bit / 2-bit / skip per expert, more 4-bit in early layers (separate GPTQ copies) | Qwen3-30B-A3B | **~75 % of active experts at 4-bit, rest 2-bit: GSM8K 90.9 / MMLU 76.3 vs int4 89.1 / 77.1**; rest skipped instead: 91.5 / 77.0 | gate-aware 4/2 matches int4; the bottom quarter at 2-bit ≈ skipping it |
| **DPA-MoE (2610.21378, Oct 2026)** | AnyBCQ bit-planes, knapsack per expert minimizing Σ g²·residual under a plane budget | Qwen3-30B-A3B, OLMoE | 3 planes 66.1 vs uniform 63.4 | the gate score does most of the work |
| Any-Precision LLM (2402.10517) | SqueezeLLM seed, bit-planes, per-bit-width centroid tables | Llama-2-7B | nested 4-bit 5.62 vs separate 5.61 | non-uniform grids nest at no cost; uniform grids don't nest upward |
| MatQuant (2502.06786) / AnyBCQ (2510.10467) | co-trained int8 ⊃ int4 ⊃ int2 / frozen binary planes with refit scales | Gemma, Llama | sliced int2 of a normal int8 model: 37-40 vs direct int2 51-60; AnyBCQ 2-bit MMLU 35.3 vs Any-Precision 24.7 | refit only the low tier's scales on fixed codes |
| Merging / pruning / skipping (REAP 2510.13999, NAEE, MoBE, dynamic top-k) | prune 25-50 % of experts; fewer active experts | Qwen3-30B-A3B, Mixtral | REAP 25 %: math/code kept, knowledge drops; halving k costs 8 GSM8K points | lossy; not for equal accuracy |
| General low-bit (GPTQ, AWQ, SpQR, QuIP#, AQLM, QTIP; MLX DWQ) | — | Mixtral, Llama-2-70B | Mixtral Wiki2 AQLM 1.98-bit 4.61 vs 3.46 fp16; QTIP 3-bit near fp | best 2-bit fidelity but not nested and CUDA-centric; **MLX DWQ distills only scales/biases on fixed codes** |

Best path to ~2-2.5 bits at equal accuracy for us (survey's ranking):
1. Deterministic gate-aware nested planes (ours) with cumulative-gate-mass decisions instead of a fixed rank. Evidence:
   DyMoE, HOBBIT, DPA-MoE, our 35B GSM8K. Note: MSB codes + g64 scale/bias = 2.5 bpw, so 1.8×, not 2×, experts per GB;
   refit g128 scales for the cached MSB tier (2.25 bpw).
2. A better MSB tier so fewer ranks need the LSB: least-squares refit of each group's scale/bias on the fixed MSB codes
   (our mean folding is a special case), activation-weighted, or DWQ on those scales only. Keeps 4-bit bit-exact.
3. Tensor asymmetry: 3 planes for down_proj, 2 for gate/up (R1 UD-Q2_K_XL style).
4. Hotness-aware residency of LSB planes (DynaExq).
Not viable for equal accuracy: pruning/merging, uniform 2-bit, low-rank compensation alone, < 2 bits, halving k.
Unexplored as far as the survey found: requantizing a *shipped 4-bit checkpoint* (no bf16); deterministic,
cache-independent precision with bit-exact top ranks and no extra storage; the 512-expert top-10 regime with gate-aware
precision; refitting only the low tier's affine scales on fixed codes; one policy over gate rank + tensor + depth; real
SSD streaming with an MSB cache; lossless self-speculation with the MSB planes as the draft.

## 3. Mac and consumer runtimes for large MoEs

**We are already ahead of every public number at ≤ 32 GB.** Best published for our exact model at ≤ 32 GB: Slotstream
14.4-15.9 tok/s at 22 GB, 11.8-13.5 at 20 GB (M5 Pro 48 GB, MLX 4-bit + MTP, SR). Strata streamed Q2_0: 14.1 tok/s
on an M4 mini 32 GB (SR) and 18.9-22.1 on an M5 Pro 48 GB at ~38.7 GB (third party). BigMoE: 27-28 at 24 GB (~1.7×).

| project | design | measured | what we take |
|---|---|---|---|
| Slotstream (carloslfu/slotstream, Swift/MLX) | **one slot pool shared by all 48 layers**, CLOCK eviction; parallel pread lanes; next-layer router on the previous post-attention state **corrected by a ~35 MiB learned table** (router agreement 0.62 → 0.80); MTP 2 drafts (86 % accept) with a **private 64-expert draft cache**; 15 s memory governor; on-disk prefix cache | 20.9-23.7 tok/s at 33 GB, 14.4-15.9 at 22 GB, 11.8-13.5 at 20 GB (SR, M5 Pro 48 GB); learned table +3.1 % with 9 % fewer reads; draft cache 1.23× at a 12 GB target | cross-layer pool / uneven budgets; forecast correction; draft-expert cache |
| Strata (Niko1221/Strata, ggml, MIT; **has a Metal backend**) | profile-seeded hot tier (all 24,576 experts ranked) that decays/adapts; **expert-major file layout**; next-layer-router prefetch; own MTP (≤ 3 drafts) + prompt lookup only when it pays; coupled sampling | 94 tok/s RTX 5070 12 GB + 64 GB RAM at Q2_0 (all experts in RAM); M4 mini 32 GB streamed Q2_0 14.1 tok/s; M5 Pro 48 GB Q2_0 18.9-22.1 (3P); expert-major file **+17 % decode, +30 % prefill** (M4) | expert-major layout (1 read per expert); profile warm start; gated drafters |
| Flash-MoE (danveloper) | Qwen3.5-397B with top-4 instead of 10; parallel pread, OS page cache instead of an app LRU (+38 %) | 4.36 tok/s M3 Max 48 GB (4-bit) | **failures**: F_RDADVISE prefetch (SSD DMA slowed the GPU 73 %), temporal prediction −18 %, mmap −5×, MTP break-even; GPU syncs 65 % of token time on M1 |
| mlx-lm issue #1438 / PR #1588 | per-layer LRU offload discussion and measurements | per-layer LRU hit 0.80-0.87 vs Belady 0.90-0.94; depth-2 prefetch read 6.9 bytes per byte saved; Qwen3-235B: 2394 → 798 preads/token gave 1.68 → 2.22 tok/s (**+32 %**); two parallel miss loaders +30 % | Belady replay as the stopping rule for policy work; fewer reads per expert |
| oMLX (jundot/omlx) | per-layer slots, least-used eviction that never evicts the current step's experts (from Strata), prefill borrows the expert cache, next-chunk prefetch | M3 Ultra capped at 24 GB, our model + Lightning MTP, 10 % resident: 6.5 → 7.6-8.2 tok/s; prefill 130 → 320 tok/s (SR) | protect the verify batch's experts |
| llama.cpp PR #25294 | disk expert streaming: per-layer slots, async pread pool, decaying-hotness eviction, O_DIRECT, wave prefill | 2.4× decode vs mmap + --cpu-moe (GLM-5.2 on GB10) | — |
| KTransformers | CPU experts + GPU hot experts | frequency-based placement vs uniform per-layer: 76.2 vs 65.3 tok/s (50 % on GPU), 107.2 vs 81.1 (90 %): **+17-32 %**; expert deferral (lossy) 1.45× | uneven per-layer budgets |
| AtomicChat GGUF (our model) | `down` ~4-bit, `gate/up` IQ2_S-IQ3_S, more bits in first/last layers | KLD 0.084 / top-1 89.5 % vs BF16, close to a 4/8-bit MLX pack (0.081 / 91.3 %); uniform 3-bit 0.163 / 87.7 % | per-projection precision (down needs more bits) |
| REAP-288 (our model) | prune to 288 of 512 experts | HumanEval 93.9 → 91.5 %; top-k cuts much worse (84.8 % at top-6) | pruning is a lossy lever |
| Qualcomm cache-aware routing (arXiv 2412.00099) | bias routing toward cached experts | > 50 % fewer misses at 0.1-3 % ppl | **use it only inside the MTP draft head**: outputs stay exact, only acceptance changes |
| others | ds4 (byte-budget streaming), anemll-flash-mlx, TurboFieldfare (LFU), Colibri (learned hot set, 71.6 % one-layer-ahead predictability, a 2nd SSD +37.5 %), PowerInfer-2 (phones), SmallThinker (pre-attention router, one cross-layer LRU) | — | — |

Ideas from these projects that BigMoE does not have: one read per expert (expert-major layout); uneven per-layer budgets
or a cross-layer pool; forecast correction; a private draft-expert cache; cache-aware routing inside the draft head only;
profile warm start. Hot-expert 4-bit + cold-expert low-bit streaming: **no runtime does it**.

## 4. Principles across fields (what transfers, and why the plane prototype lost)

| principle | key references | insight for us |
|---|---|---|
| Equal-slope bit allocation: precision follows importance | Masnick & Wolf 1967 (unequal error protection); Shoham & Gersho 1988; foveated rendering (Guenter 2012; Albert 2017: milder foveation when gaze tracking lags) | a token's MoE error is Σ g_i·δ_i, so expert i deserves bits ≈ log2(g_i) + const: one bit per doubling of gate weight; ≲ 1/16 of the top gate deserves ~0 bits. In a verify pass the weight is Σ_t g_{t,e}², which dilutes the skew. Foveate less when the prediction is uncertain. |
| Embedded/layered codes cost one request per layer | progressive JPEG (Wallace 1991), JPEG2000 quality layers (Taubman 2000), H.264 SVC (Schwarz 2007); SVC over HTTP needs several requests per segment (van der Hooft 2016); OTT chose independent renditions over SVC | **store [MSB \| LSB] as one contiguous extent per expert** (full = one read, MSB-only = prefix read, upgrade = suffix read). A separately quantized 2-bit copy is the OTT answer if disk allows. **Our prototype paid SVC's per-layer request penalty.** |
| Sub-blocking: smaller units move fewer bytes but cause more miss events | Liptay 1968 (System/360 Model 85 sector cache); Hill & Smith 1984; Smith 1987 (latency favours long lines); Wu, Yu & Wolf 2001 (prefix segment caching, evict the last segment first) | cache planes, but issue demand I/O per expert per layer. LSB only behind its MSB, evicted first. At 17 GB, MSB planes for ~50 % of experts equal today's 34 GB coverage, so partial caching pays more at 17 GB. **At 34 GB one miss (~0.45 ms) is ~60 % of a layer (~0.74 ms): miss events, not bytes, set the time.** |
| Cost-aware replacement and admission | GreedyDual-Size (Cao & Irani 1997); MLP-aware replacement (Qureshi 2006: isolated misses cost more); AdaptSize (Berger 2017); LHD (Beckmann 2018); TinyLFU | price a miss as P(needed) × P(not prefetched) × marginal stall (the first miss in a layer ~0.45 ms, each extra ~0.25 ms): protect experts whose miss would be the layer's only miss. Admit low-gate one-offs MSB-only or not at all. |
| Never stall on refinement (texture streaming) | mipmaps (Williams 1983); id Tech 5 virtual texturing (van Waveren 2009); D3D12 Sampler Feedback (2019: MinMip map + feedback map); Unity SVT (mip bias rises as the cache fills) | compute with the finest resident precision, log the wanted precision, upgrade for later tokens; raise a global precision bias under memory pressure. **Our design made refinement blocking**, the opposite of every texture streamer. (Determinism: decide from the cache state at pass start, so the output is a fixed function of the input and the cache policy.) |
| Choose quality at issue time from the buffer (ABR) / imprecise computation | BBA (Huang 2014), MPC (Yin 2015), BOLA (Spiteri 2016); imprecise computation (Liu 1991); anytime algorithms (Dean & Boddy 1988) | our "buffer" is I/O slack (reads ahead of compute). Map slack to a precision tier at issue time; MSB mandatory, LSB optional in slack only, dropped at the deadline. |
| Integrated prefetching and caching | Cao, Felten, Karlin & Li 1995; TIP (Patterson 1995); Kimbrel 1996 (horizon = read time / consume time; too-early prefetch hurts); feedback-directed prefetching (Srinath 2007); PACMan (Wu 2011); learning-augmented caching (Lykouris & Vassilvitskii 2018) | horizon ≈ 0.45 / 0.74 ms, so 1-2 layers ahead. Per-layer I/O budget ≈ 6.5 GB/s × 0.74 ms ≈ 4.8 MB (1.7 experts or 3.5 MSB prefixes): above that you are bandwidth-bound, below it event-bound. **Measured: 17 GB eager has 1.6 misses/layer (borderline bandwidth-bound), 34 GB has 0.53 (event-bound).** Low-priority insertion of prefetches, cap speculative bytes, full reads for confident candidates and MSB prefixes for the uncertain tail. |
| Deadline I/O scheduling | EDF (Liu & Layland 1973); SCAN-EDF (Reddy & Wyllie 1993); tail at scale (Dean & Barroso 2013: P(stall) = 1 − Π h_i over fan-out); blk-switch (Hwang 2021) | demand reads of layer l before prefetches for l+1/l+2; submit a layer's reads together (concurrency saturates NVMe, not merging); never hold a demand read to merge with a prefetch. **Planes raised the pieces per token-layer from 10 to 14: the fan-out form of "more layers waited".** |

Top transferable rules: (1) never block on the LSB, deterministic by deciding at pass start; (2) at 17 GB cache for coverage
and price misses in stall time (GreedyDual + MLP-aware cost + TinyLFU admission); (3) decide precision at issue time from
slack and gate ratio; (4) one request per expert per deadline, with contiguous [MSB|LSB] extents; (5) prefetch within the
horizon, demand first, capped speculative bytes.

## Conclusion

**Best published method for our setting.** No published system beats our runtime on this model and machine (≈1.7× the best
public number at ≤ 32 GB, ahead of all verified results at ~34 GB), and no published *lossless* technique gives ≥ 2× at a
fixed SSD bandwidth; every ≥ 2× result is lossy. At a 17 GB cache we are at the SSD ceiling: ~73 % LRU hit → ~0.36 GB of
expert reads per token, ≈ 18 tok/s at 6.5 GB/s (we measured 19.5 eager with prefetch). The published levers, ranked:

1. **Saturate the SSD (exact).** Queue depth 8-32, one read per expert, demand reads first, compute cached experts first.
   SlotStream measured an M5 Pro 2 TB SSD at 9.5 / 17.3 / 17.3 GB/s for cold 2.76 MB F_NOCACHE preads at QD 1 / 8 / 32;
   SSD-LLaMA reached 77.7 % of peak vs 43.2 %; one read per expert +17 % (Strata) / +32 % fewer preads (mlx-lm). Our
   loader issues 9 preads per miss (3 code blocks + 6 small scale/bias reads). **If our SSD reaches ~17 GB/s at depth,
   this lever alone lifts the 17 GB ceiling ~2.5× (toward 40 tok/s at 24 GB). Measure first.**
2. **MTP depth chosen by I/O cost, and draft experts prefetched as soon as the draft exists (exact).** SpecFit +7-29 %,
   DraftExpert, Cascade; npanj: depth 3 + a 0.3 confidence floor beats depth 4 by 10 % on this model and machine.
3. **Prefetch precision: a learned correction of router reuse (exact).** SlotStream 0.62 → 0.80 agreement, 1.10×;
   ProMoE 84.7 vs 66.9 %.
4. **Gate-aware mixed precision (equal accuracy, deterministic if rank-based).** DyMoE, DynaExq (512 experts), HOBBIT,
   DPA-MoE; our 35B mix4 GSM8K-equal over 450, 125B −1.3 points. Build it the way the principles say: one read per
   expert ([MSB | LSB] contiguous, prefix reads), never block on refinement, decide precision at issue time.
5. Forecast-driven eviction (+5-10 %), uneven per-layer budgets (KTransformers +17-32 %; slipstream < 0.3 %: measure),
   prefill warmup (SliceMoE PCW), a private draft-expert cache (SlotStream 1.23× at 12 GB).
Lossy, beyond equal quality: substitution or skips gated by rank, cache-aware routing in the main model, pruning, smaller k.

**Real gaps (where we may be first).** Deterministic, cache-independent gate-rank precision on a *shipped* 4-bit
checkpoint, bit-exact top ranks, no extra storage, on a 512-expert top-10 model with real SSD streaming; bias-corrected
truncation and least-squares refits of the low tier on fixed codes; the MSB planes as a lossless self-speculation draft
(all-resident MSB cache drafts, the 4-bit model verifies); SSD streaming at depth on unified memory with MLX.

**Build order.** (1) Measure SSD bandwidth vs queue depth on our machine (cold 2.77 MB F_NOCACHE preads from the expert
files, QD 1-64, plus BigMoE's 9-piece pattern) and BigMoE's achieved bandwidth at 17 GB. If there is ≥ 1.5× headroom:
one read per expert + a deeper queue in the loader (expert-major file; with BigMoE). (2) MTP depth by I/O cost (with
BigMoE/SpecDecode). (3) Gate-aware precision with one read per expert, if (1)+(2) fall short.

## Build step 1 measured (12:55): SSD read bandwidth vs queue depth on this Mac (scripts/ssd_qd.py, data/ssd_qd.txt)
Cold random preads with F_NOCACHE from the 125B expert files (each expert used once):

| QD | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| one 2.77 MB read per expert | 6.95 GB/s | 12.7 | 14.0 | 13.9 | 14.4 | **14.6** | 14.6 |
| BigMoE's 9 pieces per expert (3 × 819 KB + 6 × 51 KB) | 2.0 | 3.9 | 7.0 | 10.1 | 12.2 | 12.2 | 12.4 |

The SSD tops out at ~14.6 GB/s (SlotStream's 17.3 was measured on another unit), more than 2× the ~6.5 GB/s assumed. BigMoE at
a 17 GB cache uses only ~3.5 GB/s on average (77.6 misses/token × 2.77 MB × 16.1 tok/s), so it has **~4× bandwidth headroom:
the 24 GB problem is per-layer demand-miss latency and prefetch coverage, not bandwidth.** At 40 tok/s the read budget is
~365 MB/token against ~215 MB needed at 17 GB, so wide, imprecise prefetch is affordable (cf. mlx-lm's 6.9 bytes read per
byte saved). For one demand miss, 9 parallel pieces (~0.28 ms) beat one 2.77 MB read at QD 1 (~0.40 ms): split latency-critical
reads into parallel chunks, and use whole-expert reads for bandwidth (prefetch) traffic.
Quality evidence for a lower-byte expert format: 125B GSM8K-150 with top-2 ranks exact and ranks 2-9 on the 3-bit MSB plane
(planes125, plan 4433333333): 146/150 vs eager 147; answers equal eager 147/150, the same as the v2a-vs-eager noise arm,
at −18 % expert bytes. The MSB/LSB plane files (67 GB) were deleted at 12:56.

## Round 32 prior-art pass (2026-10-20 ~21:45, account-1; agent sweep, 2026 entries partly abstract-only)
- Low-bit self-spec of the same model: QSpec 2410.11305 (W4A4 draft / W4A16 verify, dense, ~1.6x), ML-SpecQD 2503.13565
  (MXFP4 copy drafts BF16, 2-2.7x, dense CPU), 2505.22179 (2.78x 70B), **MoE-SpeQ 2511.14102** (INT4 copy of the experts
  resident in VRAM drafts, target experts over PCIe: closest to "resident low-bit drafter + streamed verifier"; separate
  copy, 16-42B, not formally lossless), ELMoE-3D 2604.14626 (bit-nested elastic self-spec, hardware), Lynx 2607.01831
  (MSB-anchor speculation for the KV cache, not weights). Nested storage without drafting: Any-Precision 2402.10517,
  MatQuant 2502.06786, GRINQH 2606.23419 (lists spec-decode as future work).
- Bit-sliced expert caches: SliceMoE 2512.12990, D2MoE 2504.15299 (both lossy, no drafting). Spec + offload: SP-MoE
  2510.10302, DraftExpert 2607.24434 (exact, 1.45x, phone flash), S2-MoE 2608.15018, SPICE 2608.21240.
- Bytes per verified token: MoE-Spec 2602.16052 (lossy expert cap), AcceptMoE 2608.02989 (cache-aware verifier, lossy),
  **EcoSpec 2607.12696** (lossless draft-path choice penalizing new experts, Qwen3-235B, HBM only, not cache-state aware).
- Apple/SSD: LLM in a flash 2312.11514; Flash-MoE (GitHub) Qwen3.5-397B on M3 Max 48 GB 4.4 tok/s (prefetch hurt). Nobody
  reports >6 tok/s for a >100B MoE under memory pressure; >100 tok/s only fully resident on 256 GB machines.
- Open per the sweep: (1) nested-plane drafting where the verifier fetches only LSB planes; (2) lossless tree/draft choice
  priced in SSD bytes given the live cache state; (3) draft routed over resident experts only, exact verifier; (4) multi-
  level plane drafting with long blocks (union saturation); (5) margin-gated LSB skipping (empirical exactness only: no).
- Caveat for our goals: MSB-3 of all experts = 52.8 GB, MSB-2 = 37.8 GB, so a resident plane drafter does not fit 6-12 GB;
  ideas (2)-(4) are the ones that touch the 6-12 GB target.
- Out-of-order experts (round 32c): closest is **MoEpic** (2509.08342: experts split into top/bottom segments, cache-hit
  experts' top segments processed early to widen the overlap window); Klotski (ASPLOS'25, multi-batch pipeline),
  ProMoE 2410.22134 and 2603.19289 (speculative expert prefetch + execution, lossy on mispredicts). Plain expert-granular
  "hits first, misses after" with a bitwise fixed-order combine: not found. Novelty: moderate (exactness is ours).
