# EvalSuite design

Question: is a 125B variant (Qwen3.8-Flash-Next, 4-bit MLX, experts streamed from SSD by BigMoE) "really similar in
intelligence" to the 4-bit reference, rather than "thinking in a different way entirely" (Kamen)? A decode-only
benchmark cannot answer that cheaply: ~270K decode tokens (~3.5 h at ~20 tok/s per model) detect only ~3-point drops.
So the statistics come from prefill-only tests (teacher-forced logits, letter-probability multiple choice), and decode
is spent on small paired capability checks. Every number is read against two anchors. Brief:
~/.local/share/pacer/briefs/quality_suite.md.

## Anchors
- **floor** (two runs of the same weights): the reference weights through MTPLX/mlx-lm's stock top-k combine (experts
  summed in argpartition order) instead of the harness's gate-rank order. Both are exact computations of the same 4-bit
  model; they differ only in bf16 summation order, as two runtimes of the same weights do (BigMoE eager vs v2a gave 9
  GSM8K answer disagreements in 450). Smoke test: KLD 0.018, top-1 96.4 % on prompt text, so this floor is not tiny.
- **accepted build** (ar4): the routed experts of Intel/Qwen3.8-Flash-Next-W4A16-AutoRound (int4, symmetric, group
  128, AutoRound tuning; Intel reports 99.6 % of BF16 on GSM8K/MMLU/PIQA/HellaSwag), converted exactly to MLX affine
  4-bit g64 (scale per 128-group repeated for both 64-halves, bias = -8 scale, scales fp16 -> bf16), on the reference's
  dense trunk. It perturbs exactly what expert-precision variants perturb, by an amount people accept as "the same
  model". 8-bit (~133 GB) does not fit the 100 GB disk floor. Measured once; its files can be deleted afterwards.
  Note: the reference and AutoRound are independent quantizations of BF16, so d(ref, ar4) contains both builds'
  errors; "variant <= accepted" means the variant moves the model no more than switching to another accepted 4-bit build.

## Tests (prefill-only first, decode behind --decode)
1. Teacher-forced on the reference's own outputs (greedy, non-thinking, cached once in cache/ref/tf_gen.jsonl): 60
   GSM8K CoTs, 30 IFEval answers, 24 HumanEval+ completions, 24 prompts in 8 languages; plus 4 x 65,536-token documents
   (English and French novels, Bulgarian Wikipedia, mlx_lm source). Metrics per position, paired with the reference's
   cached top-256 log-probs: KLD(P_ref || Q) (tail outside the top-256 as one bucket: exact when the tail is empty, a
   lower bound otherwise, the same for every variant; the mean tail mass is reported), P99.9 KLD, top-1 agreement,
   router top-10 overlap (all 48 layers), KLD by position (0-1K, 1-4K, 4-16K, 16-64K). After llama.cpp's KLD /
   same-top-p tooling and Dutta et al. 2024 (KLD and flips track each other).
2. Letter-probability multiple choice: MMLU-Pro (100 per subject, 14 subjects) + GPQA-Diamond (198, options shuffled),
   zero-shot, non-thinking template, assistant prefix "The answer is (", full-vocabulary log-probs of the letters at the
   next position (letter mass ~0.99 on the reference). Metrics: accuracy delta (TOST), flips (argmax letter differs from
   the reference), p(correct) delta.
3. Decode sets (flag): IFEval 100 (strict prompt-level, Google's checker vendored), HumanEval+ 60 (evalplus base+plus),
   math 50 (MATH-500 levels 3-5 x 25, GSM8K x 25, \boxed{}), RULER-style 24 at 16K (12 multi-key needles, 12 variable
   tracking). Non-thinking greedy by default (deterministic; MoonShots' GSM8K setting); --thinking uses Qwen's sampler
   with a per-item seed shared by all variants. Behaviour: length ratio, hesitation-word rate, truncations, loops.

Scope: prefill tests apply a variant's numerics to every token. For decode-only variants (MoonShots' plan keeps prefill
exact) this is conservative: it also perturbs the prompt. Decode sets use the deployed semantics.

## Statistics and rules (evalsuite/margins.json, fixed before any variant result)
- Equivalence: TOST at 90 %, +-2 points, on the pooled multiple-choice accuracy (and decode capability deltas).
- Non-inferiority vs the accepted build (90 % CI, cluster bootstrap over sequences / 1,024-token document blocks):
  KLD mean and P99.9 ratio <= 1.10; top-1 agreement and router overlap no more than 0.5 points lower; MC flips no more
  than 0.5 points higher; KLD at 16K+ ratio <= 1.10.
- Decode: IFEval / HumanEval+ delta > -5 points and exact McNemar p >= 0.05; RULER <= 1 net item lost; answer agreement
  with the reference >= the accepted build's; length ratio 0.90-1.10; hesitation ratio 0.85-1.15; no rise in
  truncations or loops.
"Not significant" is never read as "equivalent": every rule is a CI against a fixed margin.

## Engine
Layer-major forward of many independent sequences (MoonShots' lmp sticky staging, generalized from the chunks of one
prompt to many prompts with their own caches), inside MTPLX's attention_phase("prefill") with MTPLX's prefill cache and
per-chunk PLE staging, so each (layer, chunk) step sees what MTPLX's chunk-major prefill would; each layer's experts
stream once per batch (~24K tokens). Chunks <= 2,048 tokens keep MLX 0.32.2's NAX gather at bm 32 (its bm-64 path is
wrong for unaligned row counts, PrefillKernels N3). Tiered prefill masks the full-E scratch once per layer and pass;
decode tiers gather per slot (planes125's method). The harness reference combines the top-k in gate-rank order
(planes125's decode block), so plan variants differ from it only by their tiers.

## Sources
Brief and its references: Dutta et al. 2024 "Accuracy is not all you need" (arXiv 2407.09141); Kurtic et al. 2024
"Give me BF16 or give me death" (2411.02355); Miller 2024 "Adding error bars to evals" (2411.00640); llama.cpp
perplexity/KLD tool; RULER (Hsieh et al. 2024); IFEval (Zhou et al. 2023); EvalPlus (Liu et al. 2023); MMLU-Pro
(Wang et al. 2024); GPQA (Rein et al. 2023).
