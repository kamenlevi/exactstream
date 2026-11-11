# Sparse consultation of the 125B (research, 2026-10-19)

**Answer.** Partly. 1.5–8B drafters write most tokens near big-model accuracy. But every published scheme still runs the big model over every token, to verify or to catch up by prefill, and none reports whether final answers stay identical. That pass is our SSD bill, and batching cuts it only 1.3× at 16 tokens per pass and 3.6× at 256 (our trace). The lever is how many tokens each 125B pass keeps. SR = self-reported; recalled = not re-fetched.

## 1. Methods (numbers SR)

| Method | Big share or kept/pass | Big passes/token | Quality, speed | Drafter | Catch-up | Src |
|---|---|---|---|---|---|---|
| Judge Decoding '25 | 19.7/pass (8B→405B) | 0.05 | quality kept; 129 vs 59 tok/s (SD) | 8B | verify | 1 |
| AutoJudge '25 | 20–45/pass | 0.02–0.05 | −≤1 pt GSM8K, −2 LCB; offloaded 70B: 1.20→1.98× over SD as GSM8K 95.4→90.4 % | 1B, 8B | verify | 2 |
| Fuzzy SD; typical acceptance; speculative cascades | ~SD | ~SD | +5 tok/s over SD at −2 pts; cascades beat SD | small | verify | 3, 4 |
| BiLD '23 | low-confidence tokens | – | ≤2.12× (recalled) | T5-small | parallel pass over tokens since last call | 5 |
| R2R '25 (CITER similar) | 12.9 % of tokens | 0.13 | avg 46 vs 50 (AIME 55/57, LCB 39/45, GPQA 44/46); 2.76× | 1.5B + 56M router | overlapped prefill; cost unreported | 6, 8 |
| RelayLLM '26 | 1.07 % of tokens | 0.01 | 49.5 % avg, 6 benchmarks | small | prefill on call | 7 |
| SpecReason '25 | 36–80 % of steps | 0.2–0.6 | +0.4–9 % vs 32B base; 1.4–3.0× | 1.5B | step prefill ≈ 1–2 decodes; rejects rewritten | 9 |
| LayerSkip, SWIFT, Kangaroo | 2–3/pass, lossless | ~0.4 | 2.16×, 1.3–1.6×, 1.68× (recalled) | own layers | shared KV | 11 |
| DraftExpert, MoE-Spec, ECOSPEC '26 | 84–87 % acceptance | – | 1.45×; +10–30 % vs EAGLE-3; 1.62× on Qwen3-235B | own experts | same model | 12–14 |
| Cache-Prior '25 | prefers resident experts | 1 | −50 % misses, −<0.1 % accuracy; fails below 5 % misses (SliceMoE) | – | approximate | 15, 16 |
| DFlash '26 | 6.5/pass (Qwen3-8B, block 16; 5.5 at T=1); EAGLE-3 3.0–3.4 | 0.15 | lossless, 4.9× | 5 layers | verify | 17 |

ConfSpec: 2.24× at parity; RelayGen: 2.2× at <2 % loss [7, 10]; DFlash block 16→24: only +0.8 [18].

## 2. Best fit: lazy chunked verification

An in-RAM drafter (≤1.5 GB per token, same tokenizer) writes ahead. One 125B prefill per chunk (≤256 tokens) is both catch-up and judge (AutoJudge-style linear head on hidden states, plus rank/margin). Keep tokens up to the first must-fix one; the 125B's token there is free. Rollback truncates KV in the 12 attention layers and replays DeltaNet from a checkpoint with cached k/v/gates (~0.2 GB per chunk, no experts).

Loads per kept token ≈ S·L(S)/(kept+1), with chunk length S and our 10 %-cache trace L (152/120/79/42 at S=1/16/64/256). F is the mean gap between must-fix tokens; "ideal" means each chunk ends just before one:

| F | Fixed chunks | Ideal stop | SSD tok/s (fixed / ideal) |
|---|---|---|---|
| 40 (AutoJudge-like) | ~135 | ~80 | 39 / 66 |
| 100 (step-judge-like) | ~105 | ~58 | 50 / 91 |
| 250 | ~67 | ~46 | 79 / 115 |
| 1000 | ~47 | ~43 | 112 / 123 |

**Break-even.** An S-row pass beats eager on SSD only if kept/S > L(S)/152 (79 % at S=16, 52 % at 64, 28 % at 256). DFlash keeps 6.5 of 16 rows: ~295 loads per kept token, 2× eager; long lossless drafts help only RAM. Exact catch-up pays the same toll: R2R's 12.9 % routing means ~8-token catch-up chunks, ~130 loads per token vs 152 eager. If Cache-Prior's ~50 % miss cut holds for chunk unions, loads halve (F=100 with a good stop predictor → ~30 loads, ~180 tok/s).

**RAM:** 3.4 GB dense + 6.8 GB experts + 1 GB drafter ≈ 11.5 GB; dense reads amortize to ~15 MB/token; the drafter caps speed at ~200 tok/s (70 % of bandwidth).

**Aggressive variant:** resident-only catch-up; full experts only at flagged tokens (~152/F loads per token). SSD risk becomes quality risk.

**Main risk: F.** Published drafters imply F ≈ 8–45 at token level (R2R routes 11–15 %; AutoJudge keeps 20–45). Lenient step judges imply F ≈ 100–350 (SpecReason rejects 20–64 % of ~70-token steps). Answer identity is unmeasured.

## 3. One stream at 1000 tok/s on this Mac

Budget: 280 MB and 1 ms per token. Walls:
- **(a)** 3.4 GB dense per pass → ≥12 kept tokens per pass, even with free experts.
- **(b)** A 256-row pass touches ~half the experts (~31–36 GB) → ≥130 kept tokens per pass, all experts in RAM (71 GB > 64 GB; SSD caps ~125 tok/s).
- **(c)** ~1 ms per layer → ≥48 kept tokens per pass.
- **(d)** ~17 TFLOP/s compute (inferred).
- **(e)** A parallel drafter (block or diffusion).

**Records for 100B+ MoEs.** Cerebras (on-wafer SRAM): gpt-oss-120B ~2,200 tok/s (AA; vendor ~3,000), Qwen3-235B ~1,400–1,500. NVIDIA 8×B200 (~64 TB/s, 230× ours): Llama 4 Maverick >1,000 tok/s per user (EAGLE-3, 3 drafts, FP8). Below 3.35 TB/s, only small models reach ~1000 tok/s: Mercury Coder Mini (diffusion, many tokens per pass; 1,109 on H100) and Hazy's Llama-1B megakernel (78 % of H100 bandwidth, ~1 ms/token derived). Nothing published nears 1000 tok/s for a 100B+ MoE under 1 TB/s.

**Apple (SR).** No published Metal megakernel. BaseRT fusion: up to 1.56× over llama.cpp, 1.35× over MLX. NeurIPS'26 study: Metal lacks async copy; CUDA serving tricks give ≤1.20×. A community MLX "megakernel" verifies 5 tokens at ~1.5× one decode.

## 4. Ideas and prior art

- **Lazy chunked verification.** Prior art: AutoJudge, Judge Decoding. Left: 256+-token chunks under SSD streaming; answer identity.
- **Stop predictor.** Prior art: R2R's router, GlimpRouter (first-token entropy), BiLD's fallback, CPU branch-confidence gating. Left: train on our must-fix labels, tuned for recall like a Viola–Jones stage (~10 of ~6,000 features evaluated per window, recalled).
- **Top-k correction.** Prior art: Select to Think (a 1.5B's top-8 holds the 32B's pick 95 % of the time). Left: the 125B picks among the drafter's top-k.
- **Cache-aware chunk pass.** Prior art: Cache-Prior, MoE-Spec, ECOSPEC. Left: K=256 unions with 512 experts at 10 % residency.
- **Rollback by replay.** Prior art: optimistic concurrency control (validate at commit) and GGPO netcode (cheap save/restore, bounded re-simulation). Left: DeltaNet replay cost.
- **Output-space fidelity.** Prior art: Nanite's level of detail (≤1-pixel error). Left: the routing mass each substitution drops is a free error bound.

## 5. Cheapest experiments

1. **Measure F offline.** Log the 125B's top-20 logits while it writes 50 reference answers (AIME/MATH/GPQA/LCB). Teacher-force 2–3 same-tokenizer drafters (0.6–4B); log the 125B's rank and margin of each drafter top-1, and drafter entropy. Report F per rule (top-1; top-k+margin; judge head) and stop-predictor AUROC. Build if F ≥ 250, or ≥ 100 with AUROC ≥ 0.9 plus a working cache-aware pass.
2. **Cache-aware replay on CPU over existing traces** (needs all 512 router scores). Substitute non-resident experts scoring within δ of a resident one; recompute L at K=1/16/64/256; teacher-force 20 traces for top-1 agreement and KL. Build if L(256) ≤ 25 and L(1) ≤ 80 at ≥99 % agreement.
3. **Answer identity at small scale.** Drafter plus rule (1) on 50 prompts versus the 125B alone. Proceed only if ≥90 % of answers match and scores stay within noise.

## 6. Sources
[1]arxiv.org/abs/2501.19309 [2]arxiv.org/abs/2504.20039 [3]arxiv.org/abs/2502.20704 [4]arxiv.org/abs/2405.19261 [5]arxiv.org/abs/2302.07863 [6]arxiv.org/abs/2505.21600 [7]api.semanticscholar.org/graph/v1/paper/arXiv:2505.21600/citations(RelayLLM,RelayGen,GlimpRouter,Select-to-Think) [8]arxiv.org/abs/2502.01976 [9]arxiv.org/abs/2504.07891 [10]aclanthology.org/2026.acl-long.1221.pdf [11]arxiv.org/abs/2404.16710,2410.06916,2404.18911 [12]arxiv.org/abs/2607.24434 [13]arxiv.org/abs/2602.16052 [14]arxiv.org/abs/2607.12696 [15]arxiv.org/abs/2412.00099 [16]arxiv.org/abs/2512.12990 [17]arxiv.org/abs/2602.06036 [18]arxiv.org/abs/2608.30427 [NVIDIA]developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick [AA]artificialanalysis.ai/providers/cerebras [Mercury]arxiv.org/abs/2506.17298 [Hazy]hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles [BaseRT]arxiv.org/abs/2607.00501 [Metal]neurips.cc/virtual/2026/171267 [MLX]huggingface.co/RESMP-DEV/gemma-4-E4B-it-qat-mlx-megakernel
