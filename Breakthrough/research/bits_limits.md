# How few bits? Experts and dense part at Q4-equal quality

2026-10-19. SR = self-reported. "Q4" = our MLX affine 4-bit g64 (4.5 bpw). Unsourced numbers are my arithmetic.

## 1. Evidence

Theory [1]: iid Gaussian (≈ weights after Hadamard) at 2 bits: NMSE 0.118 scalar, 0.089 E8P, 0.069 trellis; bound 0.0625.

| Method | bpw | Model | Quality (delta) | Cost | Apple decode | Src |
|---|---|---|---|---|---|---|
| QTIP trellis + LDLQ + FT | 4/3/2 | Llama-2-70B | ppl +1.3/+4.5/+19 %; Llama-3-70B +6/+23/+92 % (SR) | GPU-hours | CUDA only | 1 |
| IQ2_KT/IQ3_KT trellis | ~2.1/~3.1 | Llama-3.1-8B | 52/40 tok/s on M2 Max (SR) ≈125 of 400 GB/s: ALU-bound | CPU-hours | Metal (ik_llama.cpp) | 2 |
| Unsloth dynamic GGUF | ~3.6/3.0/2.7 | DeepSeek-V3.1 | Aider 68.4/65.8/61.2 vs Q4_K_M 69.7, BF16 71.6; SE ≈3 (SR) | imatrix | Metal | 3 |
| GGUF k/IQ quants | ~4.4/3.7/3.0/2.75/2.1 | Qwen3.5-35B-A3B (presumed) | KLD 0.014/0.031/0.050/0.097/0.185; Q4_0 0.014 (SR) | imatrix | Metal | 4 |
| IQ3_KS/IQ2_KL/IQ2_KS experts | 3.25/2.76/2.26 | Kimi-K2-Thinking | ppl +7.4/+18/+41 % vs Q4_X (SR) | imatrix | Metal (fork) | 5 |
| AlphaQ allocation | 2.5 | Qwen3-30B-A3B | ppl +6.4 %, acc −3.3 (SR) | none | – | 6 |
| PVQ (Opus-codec VQ) | 3.25 | Llama-3-70B | 98 % of FP accuracy (SR) | Hessians | – | 7 |
| ASTC 6×6 + LoRA; 2-bit QAT | 3.56+0.44; 2 | Apple PT-MoE server; Apple 3B | MMLU −0.8; −3.4 (SR) | PTQ + LoRA; full QAT | texture hardware, ~free; yes | 8 |
| EfficientQAT | 2 | Llama-2-70B | acc −2.9 (SR) | 41 h on 1 A100 | – | 9 |
| MoBE (gate/up bases) | −24…30 % params | Qwen3-235B/DS-V3/Kimi-K2/Qwen3-30B | avg −0.6/−1.3/−1.3/−2.8; D²-MoE −12.2 (SR) | data-free | no kernel | 10 |
| REAP pruning | −25 % experts | Qwen3-30B/Coder-480B/Kimi-K2 | knowledge −5.6/−0.2/−0.7, code ≤1.2 (SR) | calibration | n/a | 11 |

## 2. Best estimate

**How much error Q4 tolerates.** MLX Q4 g64 has NMSE ≈0.008 (Gaussian groups, step = range/15); the bound reaches that at ≈3.5 bits. Scalar codes sit 0.45-0.65 bit above the bound at 2-4 bits, E8 about half that, trellis ~0.07. **So today's format spends ~1 bit/weight more than its error requires.** HIGGS's linearity theorem (Δppl ≈ Σ α·NMSE per layer) means equal weighted error gives equal quality. Our MSB test ran ranks 2-9 at ~4× Q4's error (NMSE ≈0.03 if bin-centred) within GSM8K noise.

**Experts: ≈3.25 bpw stored (range 3.0-3.5), using Hadamard + LDLQ + QTIP-style trellis + sensitivity allocation with more bits for down_proj, post-training only. Confidence: medium.** Matching the pack's KLD needs ~3.4-3.5 bits; staying within benchmark noise likely needs only ~3.0.

**Bits read per token can drop to ≈2.7 with a nested code.** Every routed expert reads a 2.5-bit base (NMSE ≈0.034, same error as our tested truncation). A 1-bit refinement is read only for gate ranks 0-1. Confidence: medium.

**At ≤2.25 bits, no published MoE result is Q4-equal, trained or not.** Apple's full 2-bit QAT still loses 3.4 MMLU; ParetoQ finds a learning transition between 2 and 3 bits.

**Dense: 3.4 → ~2.0-2.4 GB.**
- HC mixers: bf16 → 8-bit saves 0.6 GB (high confidence); 3.5-bit trellis saves 1.0 GB (medium).
- Attention/GDN and lm_head: 3.5-bit trellis, no lower. Unsloth finds attn_* and ssm_out the most sensitive Qwen3.5 tensors, and our 3-bit RTN cost +12.5 %.
- A full 2× would need ~3-bit attention with training (low confidence).

**Budget** (4.915 M params/expert, 480 routed reads/token):

| bpw | MB/expert | experts/GB | all experts | GB/token, 0 % hits |
|---|---|---|---|---|
| 4.5 today | 2.76 | 362 | 67.9 GB | 1.33 |
| 3.5 | 2.15 | 465 | 52.8 GB | 1.03 |
| 3.25 | 2.00 | 501 | 49.1 GB | 0.96 |
| 2.7 nested read, 3.5 stored | 1.66 read | 465 | 52.8 GB | 0.80 |
| 2.5 | 1.54 | 651 | 37.7 GB | 0.74 |

Bits alone deliver a 1.4-1.7× SSD cut, against the 5-16× needed. The in-RAM goal (≤45 GB) needs ~2.75 bits stored: frequency allocation (cold experts ~2.5), past Q4-equal. Quantize from bf16, not from the 4-bit pack.

**Apple decode.** Trellis decoding costs ≤4 ALU ops per weight; on M2 Max it is ALU-bound at ~125 GB/s compressed, still ≥8× the SSD, so free while SSD-bound. At ~1000 tok/s, decode each tile once and reuse it for all drafted tokens. Stock MLX is affine-only: trellis needs a custom `mx.fast.metal_kernel`. ASTC decodes free in texture hardware, but ≤96 index bits per 128-bit block (~6 levels per 36 weights) is no better than 3-bit affine.

## 3. Ideas and prior art

1. **Trellis experts at ~3.25 bits.** Prior art: QTIP (3-bit 70B equals FP16 zero-shot), EXL3, ik_llama.cpp *_KT (Metal), Q-Palette (fractional-bit trellis + allocation). Left: no trellis-vs-Q4 test on a 512-expert hybrid MoE; no MLX kernel.
2. **Code nested by gate rank.** Prior art: our MSB truncation (−18 % bytes, GSM8K 427 vs 431), Any-Precision LLM and MatQuant (nested scalar), AQLM additive codebooks. Gaussian sources are successively refinable (Equitz–Cover 1991). Left: nested trellis for experts appears unpublished.
3. **Sensitivity allocation.** Prior art: HIGGS dynamic bitwidth, MxMoE, AlphaQ, BitsMoE (2-bit Qwen3-30B: 24.9 avg uniform vs 64.3 allocated, SR). Left: our α per tensor class, hot vs cold experts.
4. **Structural compression.** Prior art: MoBE, near-lossless on 235B-1T models. On our shape (d 2560, p 640, 512 experts) it gives 35-45 % fewer expert params and halves bytes per expert, but needs ~2.8 GB of resident bases (m=32) and each miss rebuilds f(Σα·B) (~m× matvec compute). REAP pruning loses 5.6 knowledge points at 25 % on Qwen3-30B; merging is worse on knowledge or generation; width-nested experts (MoSE) need pre-training. Left: MoBE + 4-bit on 512 experts is untested.

Trained routes (EfficientQAT, PV-Tuning, ParetoQ) buy ~0.25-0.5 bit for GPU-days (my estimate). BitDistill/ternary is task-specific.

## 4. Cheapest experiments

1. **Noise-injection sweep** (35B, Mac, forward passes only, hours). Add white noise at NMSE 0.0086/0.012/0.017/0.024/0.034 (ideal trellis at 3.5/3.25/3.0/2.75/2.5 bits) to one tensor class at a time; measure KLD and top-1 agreement against bf16 and the pack, plus the pack's real NMSE. Build if experts at 0.012-0.017 add no more KLD than the pack.
2. **Trellis fake-quant** (35B, then 125B experts). Hadamard + LDLQ (routed Hessians, ~1 M tokens) + QTIP trellis (L=16) at 3.25/3.0/2.5 bits, dequantized into the checkpoint; GSM8K-450, knowledge, KLD, answer agreement. A few H100-hours or 1-2 Mac days. Build if 3.25 bits (or rank-nested 2.5/3.5) gives ≥90 % identical answers within noise.
3. **Metal decode microbenchmark.** QTIP 3INST decode + matvec as a Metal kernel on one 2560×640 expert. Build if ≥60 GB/s compressed on M5 Pro, and ≥4× that when one decode serves 8 drafted tokens.

## 5. Sources

1. QTIP https://arxiv.org/abs/2406.11235
2. https://github.com/ikawrakow/ik_llama.cpp/pull/475
3. https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs/unsloth-dynamic-ggufs-on-aider-polyglot
4. https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks
5. https://huggingface.co/ubergarm/Kimi-K2-Thinking-GGUF
6. AlphaQ https://arxiv.org/abs/2606.04980
7. PVQ https://arxiv.org/abs/2410.16926
8. Apple https://arxiv.org/abs/2507.13575
9. EfficientQAT https://arxiv.org/abs/2407.11062 (not reopened)
10. MoBE https://arxiv.org/abs/2508.05257
11. REAP https://arxiv.org/abs/2510.13999
12. HIGGS https://arxiv.org/abs/2411.17525
13. Q-Palette https://arxiv.org/abs/2509.20214
14. BitsMoE https://arxiv.org/abs/2606.00079
15. MoSE https://arxiv.org/abs/2602.06154
16. ParetoQ https://arxiv.org/abs/2502.02631 (not reopened)
