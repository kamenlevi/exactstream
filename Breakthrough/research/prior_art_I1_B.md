# Prior art: deferred (late-added) experts and residency-biased routing (subagent sweep, 2026-10-21 15:20)

## A. Deferred / late-added experts (NewIdeas I1)
Not found post-hoc (a pretrained model, only missed experts, computed on the saved layer-L input, added half a layer or
one layer later). Queries: delayed / deferred / late / lazy expert; stale or asynchronous expert outputs; late fusion;
"one layer later" / "half a layer"; dependency relaxation; reuse of the previous token's expert output; llama.cpp / MLX
streaming repos (moe-stream, emmy, flash-moe, mac-code).
- Pipeline-Native Transformers (arXiv 2608.23841, Aug 2026): "AsyncExperts": experts routed on layer l's normalised
  residual, gate-weighted sum added at l+2; every expert delayed, TRAINED from scratch (TinyStories ppl 6.50 vs 6.52);
  SSD reads issued at the routing layer, consumed at injection: up to 1.68x net.
- ScMoE (2404.05019, ICML 2025): routed experts read the previous block's mid-layer state; trained: GPT3-MoE-XL ppl
  17.52 -> 16.46; 1.66-1.82x faster inference over PCIe.
- Ladder Residual (2501.06589, ICML 2025): every sublayer output added one module late; post-hoc on Llama-3.1-8B-Instruct
  layers 16-32: MMLU 68.1 -> 63.2 but GSM8K 85.0 -> 10.5 (3B tokens of fine-tuning restored it). WARNING: KLD/MMLU can
  stay flat while long generation collapses.
- DICE (2411.16786, ICCV 2025): stale activations hurt expert-parallel diffusion MoE; keeps sensitive layers synchronous.
- Converse (next-layer experts early, on a stale input): DAOP (2501.10375) from block 4 on: MMLU flat, Mixtral GSM8K 58.9
  -> 33.5 with 25 % of experts cached; Speculating Experts (2603.19289): Qwen3-30B-A3B average 0.817 -> 0.641, GPT-OSS
  flat, early layers drift most; Edge0 (2609.18063): trained pre-router + LoRA, -3.9 average, AIME -6.1 on a 35B.
- Checked, none delay an output: Pre-gated MoE, ProMoE, DuoServe, SiDA, Read-ME, ExFlow, Lina (predict / prefetch /
  place); Fiddler, MoE-Lightning (exact elsewhere); AdapMoE (skips), HOBBIT (low-precision copy on a miss); SPICE (merges
  exact CPU results in the same layer).
- Same-shape data point: Qwen3-Next-80B-A3B (512 experts, top-10, 25 % resident; Research Square rs-11268552): fetching
  only misses that were rank-1 or gate >= 0.15 cost +0.35 % ppl; substituting every miss +5.7 %.
Lessons: defer only low-gate misses, wait for rank-1 / high-gate ones; choose per layer (early layers most sensitive;
one layer late gives the read more time than half a layer); judge on long generation (GSM8K / AIME), not KLD or MMLU.

## B. Residency-biased (cache-conditional) routing
Published: Skliar et al., "Mixture of Cache-Conditional Experts" (2412.00099, TMLR 2025), "Cache-Prior":
z' = z + lambda * Delta_avg * mask (Delta_avg = running per-layer mean of the logit range max - min); biased logits only
rank, mixing weights from the original logits; the top-J experts always kept (J = 2 for fine-grained). At lambda 0.5, half
the experts cached, LRU: misses Qwen1.5-MoE 35 -> 16 %, DeepSeek-V2-Lite 28 -> 7 %, Mixtral 40 -> 21 %; ppl +0.5 %,
~+0.1 %, +2.9 %; downstream < 0.1 % drop; > 2x on Snapdragon. Never tested above 64 experts.
Follow-ups: SpecMD (2602.03921): lambda 0.1-0.5 best; Qwen1.5-MoE GSM8K 0.471 -> 0.296 in its replication (mixed with a
4-bit change), others flat; gate-score substitution collapses GSM8K (< 0.08). AcceptMoE (2608.02989): non-resident experts
dropped from the verifier's set: -0.27 pp, traffic -74..77 %, 2.06x under offload. S2-MoE (2608.15018): bias scaled by the
top-1 vs rank-(k+1) logit gap: Qwen3 KL 0.026, top-1 agreement 93 %, ppl x1.012. MoE-CORE (2610.01950): swaps misses
within 10 % of the top-k boundary, keeps the gate weight, no accuracy reported. Trained: Cache-Aware Joint Router
Adaptation (2609.04895), ReMoE (2605.27081: +26 % reuse), Cacheable-by-Design (2608.18261: + training-free rerouting ~80 %
fewer misses at <= 3.4 % ppl). mac-code repo: absolute +1.0 "safe max" on Qwen3-30B / Qwen3.5-35B (unmeasured).
Lessons: normalise the bias per layer (Delta_avg or the k vs k+1 gap); never displace ranks 0-1, bias the selection only;
Qwen is the routing-sensitive family: validate on GSM8K / AIME; report the per-step expert union vs cache capacity
(2608.07911: replay methodology alone inflates recency policies by 27-29 %).

## Combined with speculative decoding / prefetch
B: residency-aware verifiers are the published win (AcceptMoE; MoE-Spec 2602.16052 caps verifier experts, +10-30 % over
EAGLE-3; lossless EcoSpec 2607.12696 up to 1.62x, DraftExpert 2607.24434 1.45x; SpecMD credits cache-aware routing with
10-20 % of its gains). A: not found; Pipeline-Native's delay-aware read schedule is the only analogue. A + B, or deferral
inside a verify pass, appears unpublished.
