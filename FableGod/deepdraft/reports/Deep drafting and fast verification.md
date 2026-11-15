# 1,000 tok/s needs two unpublished firsts

Neither half of the plan has been published at the level it needs, so the combination is genuinely new, though none of its parts is exotic. The verify rule is already at its proven limit, but nobody has pushed the cost of an MoE check. The best measurement is 32 rows at 1.62x a one-row pass, on a 3B-active model on an A100, and nothing exists at 16-32 rows for a 256-512-expert model. Deeper drafters are well studied, but extra layers buy little, and published per-guess accuracy stops at guess 15, at about 86% (vendor blog). Ours reaches 89.7% at guesses 23-30 on HumanEval+ and holds 97.8% flat when given the target's true state, so its decay comes from drift, which known training methods address. The goal's arithmetic is textbook: 98% per guess over 30 guesses keeps 23.3 tokens per check, or about 930 tok/s at 25 ms (our calculation). The obstacle is our own floor. An exact 32-row check cannot beat 49.1 ms on this 64 GB M5 Pro, which caps a flat-98% drafter near 485 tok/s and even a perfect one near 650 (our calculation). On this Mac, 1,000 tok/s therefore needs one of two things: checks that read under half the exact bytes (lossy or lower-bit, judged on benchmarks), or more memory bandwidth.

*Terms: a check of n rows verifies n tokens in one target pass and keeps at most n. τ = tokens kept per check, counting the target's bonus token. "Ours" marks our measured numbers. "Our calculation" marks arithmetic found in no source. "Secondary" marks press, aggregator or community sources.*

## Short answers: the rule is settled; the depth and the combination are not

### The accept rule is optimal; nobody has pushed the cost of an MoE check

The rule that decides which guesses survive is settled:

- Standard rejection sampling is optimal among unbiased rejection-based verifiers ([Yin et al. 2024](https://arxiv.org/abs/2411.00841), abstract and search summary only).
- Optimal block verification adds only **5-8%** ([Sun et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3e710b42b1a9ed898f607ec0f4fcc971-Abstract-Conference.html), search summary only).
- A "speed-of-light" bound lets tokens per check grow only with log(check width) over the text's entropy ([Pankratov & Alistarh 2025](https://arxiv.org/abs/2512.11718), abstract only). Wider checks pay off slowly; predictable text pays off fast.

The cost of the check is not settled. The only MoE floor anyone states is definitional: an exact pass must read the union of the experts its rows route to. One 2026 paper checks this bound in Lean. Its Nsight profiles show a union kernel reading exactly those bytes, within 0.2-2.5%, and no exact method goes below the union ([Lan 2026](https://arxiv.org/html/2610.34440)).

The published measurements:

- The cheapest measured MoE check is **32 rows at 1.62x a one-row pass** (11.4 ms, Qwen3-30B-A3B on an A100) ([EVICT](https://arxiv.org/html/2605.00342)).
- On llama.cpp's Metal backend (M4 Max), real speculation on the same model ran at **0.877x with one guess and 0.517x with eight** ([Lan 2026](https://arxiv.org/html/2610.34440)).
- The one paper framed as a "limit" caps speedup near 2.34x, but its EAGLE-3 drafter saturates at about 2.1 accepted tokens. What it measures is the drafter's limit, not the check's ([Amankulov & Mamatin 2026](https://arxiv.org/html/2609.22156)).

Our 16-row check (61-65 ms, ours) takes **1.7-1.8x its 36.5 ms exact floor**, so it runs at 56-60% of floor speed (our calculation). That is near the top of the 35-60%-of-peak band estimated for MoE decode on Apple chips, and below the 77-89% that dense decode reaches (our calculation from [Apple](https://machinelearning.apple.com/research/mlx-distributed-inference) and [Hardware Corner](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/), secondary). The reusable pieces are cost-table check lengths, kernels that read each union expert once at full bandwidth, and, for lossy checks, acceptance-weighted expert sets (table below).

### Extra layers help little; published accuracy ends at guess 15

Deeper and retrained drafters are well studied, and depth is the weakest lever:

- DFlash's 3-, 5- and 8-layer drafters reach Math500 τ **5.64, 5.99 and 6.33**, but speed peaks at 5 layers ([DFlash](https://arxiv.org/abs/2602.06036)).
- Fifteen layers instead of five add 5.9 points of accuracy at draft position 6, and a cheap two-tap convolution recovers about 80% of that gain ([DFlash 2](https://inco.ai/blog/dflash2/), vendor blog).
- AngelSpec gained nothing going from 5 to 7 layers ([AngelSpec](https://arxiv.org/abs/2607.25852)).

Training the drafter on its own outputs matters far more. A reused MTP head scores about 70/10/0% over three steps, and 80/56/36% after self-distillation ([FastMTP](https://arxiv.org/abs/2509.18362)).

Published accuracy falls with depth:

- MTP heads decay steeply: 0.92 → 0.68 → 0.38 on GLM-4.5-Air ([Liu et al. 2026](https://arxiv.org/html/2601.11580)).
- The deepest published curve holds **about 86% flat at guesses 9-15**, where DFlash falls to 77.5% (DFlash 2 on Qwen3.5-4B, MATH-500, T=1) ([DFlash 2](https://inco.ai/blog/dflash2/), vendor blog). Nothing is published past guess 15.
- The only published drafter near 97% at up to 16 guesses is an INT4 copy of the target itself, which runs the whole network for every guess. It reaches 96.6% mean acceptance on 14-42B MoEs ([Specter](https://arxiv.org/html/2511.14102v3)).
- The best code result, **HumanEval τ 9.1** on Qwen3.5-27B ([DFlash](https://arxiv.org/abs/2602.06036)), works out to about 0.88-0.92 per guess (our calculation).

Ours scores **97.9% on the first guess and 89.7% at guesses 23-30** on HumanEval+. That is twice as deep, at higher accuracy, as any published curve, though on a different target and setup. MBPP is harder for us (94.0% → 72.1%), as it is for most published drafters; DFlash, for example, gets τ 5.95 on MBPP against 6.50 on HumanEval.

### The combination is new, and our own floor blocks the exact version

No source pairs a ~98%, 30-deep drafter with checks under 25 ms, and to our knowledge nobody has tried. Four reasons stand out:

- **Providers tune for batched throughput, where long drafts waste work.** At batch 64, 21-token trees fall below 1x on every workload tested ([Liu et al. 2026](https://arxiv.org/html/2601.11580)). NVIDIA's 1,038 tok/s record on Llama 4 Maverick used a **3-token** EAGLE-3 draft because speedup peaked there ([NVIDIA](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)).
- **The other ~1,000 tok/s results come from wafer-scale SRAM.** Cerebras does not mention speculation ([Cerebras](https://www.cerebras.ai/blog/llama-405b-inference), vendor claim).
- **No published drafter justifies a 30-deep check.** At the best published rate (~86% per guess), going from 15 to 30 guesses adds about 0.6 tokens per check (our calculation).
- **MoE checks grow with the expert union.** Static speculation slowed **18 of 35** MoE model-task pairs, by up to 1.5x ([Cascade](https://arxiv.org/html/2506.20675v1)). A separate draft model was a 2-3x regression on a 235B MoE streamed from SSD on an M5 Pro ([PagedMoE](https://github.com/noahclark556/paged-moe), author README).

Our numbers add a fifth reason, specific to this Mac. An exact 32-row check cannot beat 49.1 ms, so even a perfect drafter tops out near **650 tok/s** (our calculation). Deeper checks help little, because a 98% drafter's tokens per check level off toward 50. The plan is "nothing exotic" in method and new in result. On this Mac, its exact version is blocked by bytes, not by a missing idea.

## Key numbers against the published frontier

### Our drafter is already past the published depth frontier

| Measure | Best published | Ours (HE+ / MBPP) |
|---|---|---|
| First-guess accuracy | 93%, math, T=1 ([DSpark](https://arxiv.org/abs/2607.05147)) | 97.9% / 94.0% |
| Deepest reported | ~86% at guess 15 ([DFlash 2](https://inco.ai/blog/dflash2/), vendor) | 89.7% / 72.1% at 23-30 |
| τ on code | 9.1, HumanEval ([DFlash](https://arxiv.org/abs/2602.06036)) | 10.5 (16 rows), 13.8 (32 rows), HE+ |
| Implied rate per guess | ~0.88-0.92 (our calc.) | ~0.94, HE+ (our calc.) |
| Given the target's true state | not published at depth | 97.8% / 93.6%, flat to 30 |

This table positions us; it is not a controlled comparison. The published rows use 4-30B targets, mostly at T=1. Our per-guess accuracy is only the closest match to their conditional acceptance, and we assume, as DFlash does, that kept tokens include the bonus token.

The controlled comparison is between our own two measurements. At guesses 23-30, the 8-point gap between accuracy on the drafter's own state (89.7%) and on the target's true state (97.8%) is drift, which training-time-test methods address. MBPP's lower true-state ceiling (93.6%) is the drafter's limit on that text, and fixing drift alone cannot raise it.

### Published MoE checks stop at 32 rows on a 3B-active model

| Setting | Rows | Check cost |
|---|---|---|
| Qwen3-30B-A3B, A100 ([EVICT](https://arxiv.org/html/2605.00342)) | 16 / 32 | 1.47-1.52x / 1.62x one row |
| 128 experts, top-8 ([Cohere blog](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding), secondary) | 4 | 1.25x |
| Qwen3-235B, 8x H200, HF prototype ([EcoSpec](https://arxiv.org/html/2607.12696)) | 4 | ~1.7x |
| Mixtral, RTX 6000 Ada ([Cascade](https://arxiv.org/html/2506.20675v1)) | 8 | 2.3-3x |
| Qwen3-30B-A3B, M4 Max, Metal ([Lan](https://arxiv.org/html/2610.34440)) | 2 / 9 | net 0.877x / 0.517x |
| Ours: 125B, 512 experts, M5 Pro | 16 | 61-65 ms; floor 36.5 ms |
| Ours | 32 | floor 49.1 ms; no exact single pass yet |

GPU checks look cheap for a reason. A 3B-active model's one-row pass on an A100 takes about twice its byte-streaming time, so extra rows mostly use bandwidth that would sit idle. That is why EVICT's step from 16 to 32 rows adds only 7-11% (our calculation). On our floor the same step adds 12.6 ms, about 0.8 ms per extra row against 2.3 ms per row on average at 16 rows (our calculation). Extra rows are cheap here too, but not free.

Textbook union models make deep checks on fine-grained MoEs look worse than measurements show:

- Uniform routing puts a 512-expert top-10 layer at about 139 experts for 16 rows and 240 for 32 (our calculation from the [MoESD](https://arxiv.org/html/2505.19645v4) formula).
- In practice, routing overlap halves the union at 8-11 rows on Qwen-class models ([Amankulov & Mamatin 2026](https://arxiv.org/html/2609.22156)).
- Cohere's NeurIPS 2026 paper finds speculation works better than predicted on fine-grained MoEs ([NeurIPS poster](https://neurips.cc/virtual/2026/poster/150811)).
- The overlap is not universal: DeepSeek-V3.1 showed none at 4 rows ([EcoSpec](https://arxiv.org/html/2607.12696)).

### 1,000 tok/s exists only on wafer chips and 8-GPU servers

| System | Model | tok/s |
|---|---|---|
| Cerebras ([vendor claim](https://www.cerebras.ai/blog/llama-405b-inference)) | Llama 3.1 405B, dense | 969 |
| Cerebras ([secondary](https://getcoai.com/news/ai-chip-startup-cerebras-outperforms-nvidias-blackwell-in-llama-4-test)) | Llama 4 Maverick | 2,522 |
| Cerebras ([secondary](https://insidehpc.com/2025/08/cerebras-reports-3000-tokens-per-second-inference-on-openai-gpt-oss-120b-model/), vendor claim) | gpt-oss-120B | ~3,000 |
| 8x B200, 3-token draft ([NVIDIA](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)) | Llama 4 Maverick | 1,038 |
| RTX PRO 6000 ([llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396); [secondary](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)) | gpt-oss-120b | 196-221 |
| M5 Max 128 GB ([secondary](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/)) | gpt-oss-120b | 87.9 |
| M5 Pro 48 GB, SSD ([PagedMoE](https://github.com/noahclark556/paged-moe)) | Qwen3.8-Flash-Next | 5.9-15.6 |
| Ours: M5 Pro 64 GB, 45 GB | Qwen3.8-Flash-Next 125B | 101.4 exact, 164.0 lossy |
| Ours: 24 GB | same | 130.3 lossy |

Our rows are HumanEval, greedy, 256 tokens, with speculation. On MBPP we get 70.4 exact and 112.7 lossy at 45 GB, and 91.2 lossy at 24 GB. Most other rows are plain decoding on generic prompts, so the fair claim is narrower than the table suggests. Our numbers are above every single-Mac 100B+ figure we found. PagedMoE's author-reported runs of a model with the same name, on the same chip family within a ~35 GB budget, reach 5.9-15.6 tok/s on different tasks (48.8 in a lossy copy-back mode).

The two closest analogues to our plan both run in datacenters:

- Cursor reached ~1,000 tok/s on a 70B model with speculative edits of copy-heavy files, and published no acceptance data ([Cursor](https://cursor.com/blog/instant-apply)).
- MiMo v2.5-Pro-UltraSpeed passes 1,000 tok/s with DFlash, with no drafter details published ([LMSYS](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).

### What 1,000 tok/s requires on this Mac

All values are our calculation from Leviathan's formula ([Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)) and our measured floors, ignoring drafting time. A 32-row check scores 31 guesses.

| Drafter (per guess) | τ, 32 rows | tok/s at floor | Check for 1,000 |
|---|---|---|---|
| Today, HE+ | 13.8 | 281 | ≤ 14 ms |
| Flat 93.6% (MBPP, true state) | 13.7 | 280 | ≤ 14 ms |
| Flat 97.8% (HE+, true state) | 23.2 | 471 | ≤ 23 ms |
| Flat 98% | 23.8 | 485 | ≤ 24 ms |
| Perfect | 32 | 652 | ≤ 32 ms |

Four requirements follow:

1. **Close the drift gap first.** With today's drafter, 32 rows do not beat 16 even at the floor (281 vs 288 tok/s). A drift-free drafter makes 32 rows win (471 vs 373 tok/s), so fixing drift is what makes deep checks pay.
2. **The 32-row check must take at most 24 ms,** under half its exact floor.
3. **Drafting must be nearly free.** If each guess costs 1% or 2% of a target step, the 23.3x gain at 98% over 30 guesses drops to 17.9x or 14.5x (our calculation). Sequential drafting already loses most of its gain at 15 guesses. Native MTP reaches HumanEval τ 6.15 but only 1.48-1.79x speedup across five benchmarks, against 7.72 and 3.40-4.60x for a one-pass DFlash block ([z-lab model card](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)).
4. **98% leaves little slack.** A thousand greedy runs of Qwen3-235B produced 80 distinct completions from batch-dependent numerics alone ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). That is about 99.5-99.7% per-token self-agreement (our calculation, one prompt). On chat, about 5% of a 4B model's steps are low-confidence, with top-1 probability at or below 0.39 ([Sharma 2026](https://arxiv.org/html/2606.30265)).

That slack is enough for HumanEval-like code. MBPP-like text stays out of reach with this drafter design.

## What we could reuse: drift fixes first, byte cuts second

| Lever (source) | Reported effect | Output exact? |
|---|---|---|
| Train on own outputs ([EAGLE-3](https://arxiv.org/abs/2503.01840); [HASS](https://arxiv.org/abs/2408.15766)) | τ 4.05 → 6.13 with fused features; HASS +8-20% | Yes |
| Rollouts, self-distillation ([AngelSpec](https://arxiv.org/abs/2607.25852); [FastMTP](https://arxiv.org/abs/2509.18362)) | 3rd-step rate 0.27 → 0.52; 0 → 36% | Yes |
| Target state in every draft layer ([DFlash](https://arxiv.org/abs/2602.06036)) | block τ 3.5 → 4.2 | Yes |
| In-block dependency heads ([DSpark](https://arxiv.org/abs/2607.05147); [DFlash 2](https://inco.ai/blog/dflash2/); [DSpine](https://arxiv.org/abs/2609.36173)) | +16-30% τ; ~86% flat at guesses 9-15 | Yes |
| Cost-table check length ([EVICT](https://arxiv.org/html/2605.00342); [Lan](https://arxiv.org/html/2610.34440)) | 1.21-1.25x over EAGLE-3 | Yes |
| Union-aware draft choice ([EcoSpec](https://arxiv.org/html/2607.12696)) | 9-14% fewer experts; ~0 on DeepSeek-V3.1 | Yes |
| Acceptance-weighted expert sets ([AcceptMoE](https://arxiv.org/html/2608.02989)) | 1.29x, −0.27 points | No |
| Downgrade weak experts, don't skip ([HOBBIT](https://arxiv.org/html/2411.01433v2)) | ≤1 point, ~30% at INT4/INT2 | No |
| Relaxed acceptance at near-ties ([Sharma](https://arxiv.org/html/2606.30265)) | theory only | No |

### Drafter levers

Drift fixes come first because our own measurement shows what they are worth. Closing the gap between 89.7% and 97.8% at guesses 23-30 raises a 32-row check from 13.8 to about 23 tokens (+68%, our calculation), and output stays exact.

- **Train at the length you check.** A DFlash drafter trained on 16-token blocks keeps its accuracy at 8 (τ 5.09 vs 5.21). One trained on 8 falls short at 16 (5.02 vs 6.33) ([DFlash](https://arxiv.org/abs/2602.06036)).
- **Add capacity with cheap mixing inside the block, not with more layers.** DFlash gained only +0.7 τ going from 3 to 8 layers. A two-tap convolution recovered about 80% of a 15-layer gain for under 1% added latency ([DFlash 2](https://inco.ai/blog/dflash2/), vendor blog).

### Exact check levers

- **Kernels.** Kernel choice matters on Apple chips. Re-reading an expert within one pass costs 0.88-1.04 of a fresh read under llama.cpp Metal, but 0.27-0.40 under MLX-LM. Published union kernels ran at about a third of per-slot bandwidth on Metal (98 vs 280 GB/s) ([Lan](https://arxiv.org/html/2610.34440)).
- **Policy.** A measured cost table can choose 16 or 32 rows for each block ([EVICT](https://arxiv.org/html/2605.00342)). Lan proves the best check length rises monotonically with acceptance ([Lan](https://arxiv.org/html/2610.34440)). A calibrated confidence head can drop weak guesses at the end of a draft before the check ([DSpark](https://arxiv.org/abs/2607.05147)).

### Lossy check levers

- **Selection rule over budget.** The lossy levers are the only published ways to read less than the union. AcceptMoE's head-to-head shows the selection rule matters more than the budget: at the same budget it costs −0.27 points against standard speculation, versus −2.75 for MoE-Spec's router-mass rule ([AcceptMoE](https://arxiv.org/html/2608.02989); [MoE-Spec](https://arxiv.org/html/2602.16052v1)).
- **Lower-bit checks.** The best published hint of what a lower-bit check would change is Specter's INT4 copy of a small MoE. It agrees with its parent on 96.6% of tokens and on 90.8% of top-4 expert choices ([Specter](https://arxiv.org/html/2511.14102v3)).
- **For the 24 GB line.** DraftExpert's truncation plus prefetch cut a 4-guess offloaded check from 2.35-3.30x to 1.25-1.40x of a one-token step ([DraftExpert](https://arxiv.org/html/2607.24434v1)).

## What nobody has done: deep curves, big-MoE checks, fast Macs

We found no published source for any of the following:

- **Accuracy past guess 15.** No trained drafter with a 32-token block reports acceptance at all. DFlash 2's curve is the deepest ([DFlash 2](https://inco.ai/blog/dflash2/), vendor blog).
- **A learned drafter holding 95-98% flat past 16 guesses on open-ended text.** The only evidence near 98% comes from the best code-editing requests, where the output largely copies the input. There, a draft model's best requests keep 18.3 of 21 tokens per check, about 0.986 per guess (our calculation) ([Liu et al. 2026](https://arxiv.org/html/2601.11580)).
- **The drift gap at depth.** EAGLE-3 plots acceptance only against a few self-predicted inputs, with no values tabulated ([EAGLE-3](https://arxiv.org/abs/2503.01840)). Our own-state and true-state curves measure the gap out to guess 30.
- **Check cost or expert-union size at 16-32 rows on a 256-512-expert model.** Apple MLX data stops at 9 rows ([Lan](https://arxiv.org/html/2610.34440)).
- **Verifying a 100B+ MoE with a lower-bit copy of itself,** with divergence or benchmark cost measured.
- **A speculation result that takes a 100B+ MoE past 200 tok/s on a Mac or consumer GPU.** The one consumer single-user MTP data point we found lost 22% on a 397B MoE across four RTX PRO 6000s ([vLLM forum](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536), community).
- **A formal I/O lower bound for MoE verification** beyond the definitional union.

The "nothing exotic" view is right about methods and wrong about results. Block drafters, drift training, union-aware checks, expert budgets and low-bit drafts all exist, and speculation alone has produced ~1,000 tok/s on copy-heavy edits. But every published ~1,000 tok/s figure runs on datacenter hardware, and those that rely on speculation use 3-token drafts or text that mostly copies its input. Both numbers this plan needs are beyond anything reported:

- at least 97.7% per guess held to guess 30, the rate at which all 30 guesses survive half the time (our calculation);
- a 32-row check under 24 ms on a 512-expert 125B MoE.

## Conclusion

This research moves the bottleneck from the drafter to the check. On HumanEval-like code, our drafter given the target's true state is already at the accuracy the goal needs (97.8% vs 98%). So on this Mac the check decides whether 1,000 tok/s is possible, and the exact 32-row floor rules out the exact version. That makes ~1,000 tok/s on a 64 GB M5 Pro a lossy, benchmark-gated milestone by construction, or a hardware one, since the floor is set by bandwidth and Apple lists the M5 Ultra at 1.2 TB/s ([Apple](https://machinelearning.apple.com/research/mlx-distributed-inference)).

For the exact track, the same numbers give a ladder (check time only, HumanEval+, our calculation):

| Step | tok/s |
|---|---|
| Today's drafter and 16-row check | ~160-170 |
| Drift closed; single-pass 32-row check at today's efficiency against its floor | ~265-285 |
| Same, at dense-decode kernel efficiency | ~365-420 |
| Same, at the floor | ~470 |

The other implication is that our own measurements fill gaps in the literature. Per-guess curves to guess 30 with a true-state ceiling, and 16-32-row check costs on a 512-expert MoE on Apple silicon, each fill a gap that every paper we found leaves open. Each is a publishable result before the full combination exists.

## Sources

**Papers (primary unless marked)**
- [Leviathan, Kalman & Matias 2023, speculative decoding](https://arxiv.org/abs/2211.17192)
- [Yin et al. 2024, optimality of speculative decoding](https://arxiv.org/abs/2411.00841) (read via abstract and search summary)
- [Sun et al., ICLR 2025, block verification](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3e710b42b1a9ed898f607ec0f4fcc971-Abstract-Conference.html) (read via search summary)
- [Pankratov & Alistarh 2025, "Speculative Decoding Speed-of-Light"](https://arxiv.org/abs/2512.11718) (abstract only)
- [Sharma 2026, "When Is a Draft Accepted?"](https://arxiv.org/html/2606.30265)
- [Liu et al. 2026, "Speculative Decoding: Performance or Illusion?" (MLSys 2026)](https://arxiv.org/html/2601.11580)
- [DFlash (ICML 2026)](https://arxiv.org/abs/2602.06036)
- [DSpark](https://arxiv.org/abs/2607.05147)
- [DSpine](https://arxiv.org/abs/2609.36173)
- [AngelSpec](https://arxiv.org/abs/2607.25852)
- [FastMTP](https://arxiv.org/abs/2509.18362)
- [EAGLE-3](https://arxiv.org/abs/2503.01840)
- [HASS](https://arxiv.org/abs/2408.15766)
- [EVICT, "Making Every Verified Token Count"](https://arxiv.org/html/2605.00342)
- [Lan 2026, "Expert Bytes, Kernel Bandwidth and the Best Draft Length"](https://arxiv.org/html/2610.34440)
- [Amankulov & Mamatin 2026, "The Limits of Speculation"](https://arxiv.org/html/2609.22156)
- [MoESD (NeurIPS 2025)](https://arxiv.org/html/2505.19645v4)
- [EcoSpec, "Less Experts, Faster Decoding"](https://arxiv.org/html/2607.12696)
- [Cascade, utility-driven speculative decoding for MoE](https://arxiv.org/html/2506.20675v1)
- [AcceptMoE](https://arxiv.org/html/2608.02989)
- [MoE-Spec, expert budgeting](https://arxiv.org/html/2602.16052v1)
- [HOBBIT, mixed-precision expert offloading](https://arxiv.org/html/2411.01433v2)
- [Specter / MoE-SpeQ (ASPLOS '27)](https://arxiv.org/html/2511.14102v3)
- [DraftExpert](https://arxiv.org/html/2607.24434v1)

**Vendor, company and project pages**
- [Inco AI, DFlash 2 blog](https://inco.ai/blog/dflash2/) (vendor, not peer reviewed)
- [z-lab, Qwen3.5-4B-DFlash model card](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash)
- [LMSYS, DFlash v2 / Spec V2 blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [Cohere blog, MoE models get more from speculative decoding](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding) (secondary write-up of the authors' NeurIPS 2026 paper)
- [Ranjan et al., NeurIPS 2026 poster page](https://neurips.cc/virtual/2026/poster/150811)
- [NVIDIA, Blackwell breaks the 1,000 TPS/user barrier](https://developer.nvidia.com/blog/blackwell-breaks-the-1000-tps-user-barrier-with-metas-llama-4-maverick/)
- [Cerebras, Llama 3.1 405B inference](https://www.cerebras.ai/blog/llama-405b-inference) (vendor claim)
- [Cursor, "Editing Files at 1000 Tokens per Second"](https://cursor.com/blog/instant-apply)
- [Thinking Machines, "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [Apple ML Research, distributed MLX inference with RDMA](https://machinelearning.apple.com/research/mlx-distributed-inference)
- [PagedMoE README](https://github.com/noahclark556/paged-moe) (author-reported, not peer reviewed)
- [llama.cpp discussion #15396, gpt-oss benchmarks](https://github.com/ggml-org/llama.cpp/discussions/15396)

**Secondary and community**
- [CO/AI on Cerebras vs Blackwell, Llama 4 Maverick](https://getcoai.com/news/ai-chip-startup-cerebras-outperforms-nvidias-blackwell-in-llama-4-test) (secondary)
- [insideHPC on Cerebras gpt-oss-120B](https://insidehpc.com/2025/08/cerebras-reports-3000-tokens-per-second-inference-on-openai-gpt-oss-120b-model/) (secondary, vendor claim)
- [Hardware Corner, M5 Max local-LLM benchmarks](https://www.hardware-corner.net/m5-max-local-llm-benchmarks-20261233/) (secondary)
- [vLLM forum, Qwen3.5-397B on RTX PRO 6000](https://discuss.vllm.ai/t/sm120-rtx-pro-6000-nvfp4-moe-performance-report-qwen3-5-397b/2536) (community)
