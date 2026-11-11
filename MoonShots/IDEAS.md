<!-- OVERNIGHT SUMMARY (updated 2026-10-19 01:42; next update after 07:17) -->
## Overnight summary (Oct 18–19)
- **Mandate (Kamen, 00:00):** MoonShots builds novel, unknown-path ideas end to end; aim for a different class (e.g. 125B in
  ~half the memory at equal quality, or 80 → 120–200 tok/s). Known-path engineering (prefill overheads, PF3, NAX, kernel queue,
  slice prototype) was handed to "Prefill & kernel engineering" (handover sent with paths).
- **Memory-halving hunt:** lossless coding is capped (125B codes 3.70 bits → ~6 %). Fewer resident weights is SSD-bandwidth-capped
  (~0.8 GB/token at 25 % residency). So the route is fewer bits per expert at near-equal quality.
  **First test (01:38, `scripts/expert_bits35.py`, data/night/expert_bits35.txt):** 35B, routed experts only re-quantized
  (RTN, no calibration), 2048 teacher-forced tokens, 4-bit reference ppl 11.366:

  | routed experts | ppl | dNLL | greedy agreement |
  |---|---|---|---|
  | all 3-bit g64 | 11.84 (+4.2 %) | +0.041 | 0.871 |
  | all 2-bit g64 / g32 | 13.44 / 13.07 (+18 / +15 %) | +0.167 / +0.140 | 0.773 / 0.796 |
  | top-1 of 8 gate ranks 4-bit, rest 2-bit g32 | 12.07 (+6.1 %) | +0.060 | 0.849 |
  | top-2 4-bit, rest 2-bit | 11.78 (+3.6 %) | +0.036 | 0.879 |
  | top-4 4-bit, rest 2-bit | 11.53 (+1.5 %) | +0.015 | 0.911 |
  | top-6 4-bit, rest 2-bit | 11.45 (+0.7 %) | +0.007 | 0.932 |
  | all 8 ranks 4-bit via the mix path (harness check) | 11.36 (−0.05 %) | −0.0005 | 0.955 |

  Reading: plain RTN does not reach equal quality at ≤3 bits. Gate-rank-aware precision helps: low-gate experts' errors are
  scaled by their small gates, so top-2-at-4-bit beats all-3-bit. The greedy-agreement noise floor is 95.5 %: the harness
  check differs only in summation order, so judge by dNLL. Prior art: HOBBIT (2024) loads low-precision copies of
  less-important experts on cache misses, so "gate-aware precision" itself is known. Open parts: (1) better quantizers at
  2–3 bits (MSE-optimal clipping, GPTQ, trellis/QTIP) to close the RTN gap; (2) nested bit-planes (2-bit base resident +
  2-bit residual streamed only for top-rank experts), which needs the nested-2-bit quality, i.e. the top 2 bits of the
  existing 4-bit codes, measured; (3) GSM8K plus a broader benchmark on the best candidate.
  **Round 2 (07:30, `scripts/expert_planes35.py`, data/expert_planes35.txt; text changed overnight, ref ppl 11.409):**
  nested bit-planes. Each 4-bit code q = 4m + l is split into an MSB plane m (2-bit, scale 4s, bias b + offset·s) and an
  LSB plane l (2-bit, scale s, bias 0). MSB + LSB is exactly the 4-bit weight (packing test max err 0.012 = bf16).

  | routed experts (35B) | bits/weight | ppl Δ |
  |---|---|---|
  | own MoE path, 4-bit (harness noise) | 4.5 | +0.24 % |
  | MSB plane only, midpoint / per-group mean-LSB offset | 2.5 | +12.0 % / **+11.3 %** |
  | RTN 2-bit, MSE-clipped (g64 / g32) | 2.5 / 3.0 | +20.7 % / +17.3 % |
  | Hadamard-rotated RTN 2-bit / + clipping | 2.5 | +14.7 % / +18.6 % |
  | 3-bit: clipped / rotated | 3.5 | +4.3 % / +3.4 % |
  | top-2 / **top-4** / top-6 gate ranks exact 4-bit, rest MSB plane | ~3.0 / ~3.5 / ~4.0 read | +3.2 % / **+1.0 %** / −0.2 % |
  | top-4 exact 4-bit, rest clipped / rotated 2-bit | | +1.7 % / +2.1 % |

  The MSB plane of the existing codes is the best training-free 2-bit expert representation tested. MSE-optimal weight
  clipping hurts at 2 bits (outlier weights matter); rotation helps a little.
  **GSM8K-150 screen (07:55, `scripts/gsm8k_planes35.py`, greedy CoT, data/gsm8k_planes35_mix4-msb2m.json):** plain 4-bit
  145/150 (96.7 %); mix4-msb2m 144/150 (96.0 %), 147/150 final answers equal to 4-bit; noise arm (4-bit through the same
  custom path, only summation order differs) 143/150 (95.3 %), 148/150 equal. **The variant sits inside the noise floor.**
  Byte model for a streaming runtime: top-4 of 8 on both planes plus 4 MSB-only reads 28 of 36 plane-units (−22 % expert
  bytes per token); a low-rank miss fetches only the MSB plane (56 % of the expert). Day runner (data/day/):
  mix6 146/150 (145/150 answers equal 4-bit); mix2 142/150 (146/150 equal; A-only 3, B-only 0; ppl +3.2 %, borderline);
  mix4 on problems 150–449: 285/300 vs 4-bit 287/300 (293/300 equal). **mix4 over 450 problems: 429 vs 432, 440/450
  equal; discordant 5 vs 2 (McNemar p ≈ 0.45), equal within noise.**
  Permutation-aligned similarity (`scripts/expert_align125.py`, 125B, Git Re-Basin-style greedy matching): same-layer
  0.038–0.045 vs cross-layer null 0.035; held-out up/down 0.018 vs 0.016. **Experts are not permutations of each other;
  delta coding is dead (~0.001 bits/weight).**
- **Round 17 ideas** (IDEAS.md table): novel per prior art are W-TinyLFU admission (4), AIMD prefetch-depth control (4), and the
  concave stall-time slot-allocation objective (3). Trace PoCs queued: admission_sim, stall_alloc_sim.
- **CPU trace PoCs (00:39, niced in the quiet window; data/night/):**
  - *Expert similarity (125B, L0/24/47, 16 experts):* experts are statistically independent in raw coordinates. Pair corr
    +0.000 (max 0.007), corr with the sample mean 0.250 = 1/√16, the null value. Code deltas vs another expert cost 3.95 bits,
    more than the raw 3.70. **Base+delta and shared-basis streaming are dead in raw coordinates.** Untested: similarity after
    permuting the 640 intermediate neurons (MC-SMoE-style alignment).
  - *W-TinyLFU admission:* worse than plain LRU at both budgets: 50 %: 67–73 vs 58.8 misses/pass; 60 %: 41–43.5 vs 38.0.
    LFU eviction is worse too. Expert reuse is recency-driven. **Negative.**
  - *Per-layer slot budgets (same 12288 total slots, 50 %):* misses 58.8 → 44.0/pass (−25 %), stall 19.9 → 15.9 ms/pass
    (−20 %). In-sample (4 prompts), and MoEPrefetch's DP already does this, so it confirms prior work. The stall-time
    (concave cost) objective picks the same split as the miss objective: **no extra gain (negative for the novel part).**
    Belady-style "skip victims the next window needs" (oracle): only −2 % misses.
- **Night runner** (`scripts/night_runner.sh`, detached, GPU only): expert_bits35. It waits in the quiet-lock queue (keepalive
  stamps its own lock; ≥85 % free-memory guard).
- Earlier tonight: prefetch scheduler exact (−27 % verify wait); pack4 v2 down kernel (−2.0 ms/verify pass); LC1 solved (MTPLX QSA
  state reset); runahead: only exact attention + resident MoE reaches 0.999, cheap slices fail, stall-funded bound −54..−63 %;
  PF3 1.43–1.51× on the 35B.

# MoonShots: LLM speed ideas from outside the LLM field

M5 Pro, 64 GB, 2026-10-17. Goal: one genuine, not-yet-found speedup for local MoE LLMs on
Apple Silicon. Numbers below come from the sibling studies: ~/SpecDecode/FINDINGS.md (SD),
~/MoEPrefetch/FINDINGS.md (MP), ~/BigMoE/FINDINGS.md (BM).

Reference numbers used in the estimates:
- Qwen3.6-35B-A3B Q8 reads **3.46 GB/token** (routed experts 1.07, linear-attn projections 1.08,
  lm_head 0.54, full-attn 0.29, shared expert 0.13, KV/state 0.33). Decode runs at **62 %** of the
  280 GB/s roofline (SD §5.1). 4-bit: 2.0 GB/token, lm_head 0.29.
- Speculative decoding (lookup + MTP, 3.1 tok/pass) spends **89–91 %** of its time in the verify pass.
  A 4-token MoE verify costs 1.75x a plain step, the byte ideal is 1.63x (SD §10).
- Dense 27B: verify of S tokens costs 1.44x (S=3), 1.71x (S=5), 2.84x (S=9) a plain step, because
  MLX's quantized matmul isn't flat in rows (SD §2). The byte ideal is ~1.0x.
- Every GPU↔CPU round trip in the middle of a token costs **~150 µs** (BM §4). This kills any
  per-layer CPU/ANE cooperation in decode.
- Qwen3.8-Flash-Next 125B (BigMoE target): 2.46 GB dense + 1.33 GB routed experts per token at 4-bit.

(Ranked table, per-idea sections and measurements follow; filled in as measurements land.)

## Ranked candidates (score = gain × novelty × cheapness, 1–3 each; measurements pending)

| # | idea (source field) | what it speeds up, rough gain | prior art | score | status |
|---|---|---|---|---|---|
| 1 | **Skinny-M "read weights once" quantized matmul** (CUDA Marlin / CPU SIMD register blocking) | verify passes. Dense 27B spec 1.64x → ~2.5x if S=4 verify costs ~1.1x instead of 1.44–1.71x; BigMoE 125B (2/3 dense bytes) MTP +15–20 %. 35B MoE only ~7 % (SD §10) | MLX issue #3661 asks for it (not shipped); llama.cpp `mul_mv_ext`; Gemma-4 Metal megakernel verify | 3×2×3=18 | `scripts/skinny.py` written, waits for GPU |
| 2 | **Screened lm_head + Gumbel-max exact sampling** (ray-tracing culling / exact MIPS, FEXIPRO) | lm_head = 15.6 % of Q8 bytes, 14.5 % of 4-bit. A 2–3-bit screen + per-row bound → ≤ ~10 % plain decode, ~6 % spec | vLLM PR #60566 (FP8 screen, greedy only, says rank-256 SVD screens nothing). New: low-bit screen on Mac, exact *sampling* via Gumbel | 2×2×3=12 | `scripts/screen_head.py` (CPU, 2B proxy with the same 248K×2048 head) |
| 3 | **GPU idle-gap clock penalty** (power management, race-to-idle) | any loop that syncs per pass (spec decode, BigMoE gates). Unknown, could be 0 or 10 %+ | none found | ?×3×3 | `scripts/gapclock.py`, waits for GPU. Mac is in Energy Mode "Automatic" (`powermode 0`) |
| 4 | **GPU TLB / address-translation cost of scattered expert reads** (OS huge pages) | SD saw ~200 GB/s over 256 small matrices vs 223–236 on one; if translation explains it, ~5–10 % on MoE decode | none found | 2×3×3 | `scripts/tlb.py`, waits for GPU |
| 5 | Persistent megakernel (GPU-driven rendering) | up to the 1.3x kernel ceiling (SD §6) | exists on Metal (Gemma-4 megakernel, BaseRT 1.35x vs MLX) | 3×1×1 | not new; big port |
| 6 | ANE + GPU co-prefill (heterogeneous SoC) | prefill (20–28 % of session time) maybe +30 % → ≤ 8 % session | FusionML 2607.22785 (CPU+GPU prefill 1.15–1.38x; says ANE dispatch too costly per layer) | 1×2×1 | later |
| 7 | Early-reject doomed verify rows (GPU early-Z) | 3–6 % | FASER (mid-pass row masking) | 1×1×2 | dropped, not new |
| 8 | ASTC/BC weights decoded by texture units (game engines) | only helps if dequant is the bottleneck; BC4 ≈ 3-bit affine g16 per bit, so no density win | ARM 2017 (CNNs), mlx-lm issue #328 | 1×2×1 | revisit only if #1 shows ALU-bound |
| 9 | ARC/CLOCK-Pro/Belady expert cache | fewer misses | many 2025–26 papers; reproducibility study says the gap isn't reachable online | 1×1×3 | dropped |
| 10 | CPU SME / ANE computing missed experts, delta frames on hidden states, memoizing copied tokens, hardware compression | — | killed by measured facts: 150 µs per mid-token sync (BM §4), 50 % routing match for copies (SD §5.2), weights ≤ 5 % compressible (SD §6) | 0 | dead |

## Next experiments
1. GPU window (quiet lock): `skinny.py check 8`, `skinny.py bench 8` and `bench 4` (27B MLP shape 5120→17408), `gapclock.py`, `tlb.py`.
2. CPU (share lock): `screen_head.py 900`.
3. If #1 is flat to S≈8: hand to SpecDecode (dense 27B spec) and BigMoE (dense-heavy 125B verify).

## Measurements (2026-10-17 23:5x, quiet lock, MLX 0.32.4)

**#1 Skinny-M matmul (`scripts/skinny.py`, `data/gpu_run1.txt`).** Q8, 24 distinct 5120→17408 matrices (2.27 GB/pass):

| rows M | 1 | 2 | 3 | 4 | 5 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| MLX 0.32.4 | 1.00x (238 GB/s) | 1.00 | 0.99 | 0.97 | 1.00 | **1.68** | **2.69** | 1.37 |
| my weight-once kernel | 1.02 | 1.03 | 1.00 | 0.98 | 1.20 | 2.00 | 14.4 | 29.0 |

- **MLX 0.32.4 is already flat up to 5 rows.** SpecDecode's 1.2–1.4x curve was measured on 0.32.0, so the
  27B dense verify problem is mostly solved by the upgrade. My kernel matches MLX up to M=4 and then spills registers.
- **Remaining gap: 8–12 rows cost more than 16** (1.68x / 2.69x vs 1.37x; the kernel switches at 16).
  Free fix: **pad verify passes with 6–15 rows up to 16** (lookup drafts of 8–16 tokens hit exactly this range).
  Expected: a 12-row verify drops from 2.69x to 1.37x of a plain step on dense layers. Not yet tested end to end.
- Verdict: no breakthrough; one cheap trick for SpecDecode (padding rule + re-measure its cost curve on 0.32.4).

**#3 Idle-gap clock penalty (`scripts/gapclock.py`).** 1.89 GB of 4-bit matvecs, 250 GB/s back to back.
A 0.5–20 ms idle gap before the step changes it by ≤ 2 % (all slightly *faster*). **Dead: no DVFS penalty.**

**#2 screened lm_head and #4 TLB:** not run yet (usage limit). `skinny.py bench 4` needs its argument fixed
(`bench 4 5120 17408 20`).

**#4 GPU TLB / scattered reads (`scripts/tlb.py`, `data/tlb.txt`).** One kernel reads 1 GB from a 6 GB random-filled
buffer in chunks of 16 KB–3 MB, in order vs at random offsets. Random/sequential = 0.98–1.03x at every chunk size, about
220–240 GB/s (the printed GB/s in `data/tlb.txt` are 4x too high: the kernel indexed uint32, not uint4; ratios are valid).
**Dead: no address-translation penalty for expert-sized scattered reads.** The 200 vs 230 GB/s gap SpecDecode saw is
kernel-level (launch tails, small grids), i.e. megakernel territory (#5).
Side note: an `mx.zeros` buffer reads just as fast as random data, so zero pages get no shortcut.

## Round 2 fields (2026-10-18 00:1x)

| # | idea (field) | estimate | verdict |
|---|---|---|---|
| 11 | JIT-specialise kernels on the observed hot expert set (compilers, trace JIT) | decode is bytes-bound and dispatch is already hidden (fusion measured 0.98–0.99x, SD §6): ≤ 2 % | dead |
| 12 | Entropy-code activations (information theory) | a token's activations are ~4 KB vs GBs of weights: 0 %. Only the KV cache matters, and only at long context (≈2 GB/step at 100K, SD §14): lossless bf16 KV coding (≈ DFloat11-style exponent coding, ~25–30 %) would give ~12 % at 100K, ~0 at 10K. A Huffman decode inside the attention kernel costs ALU | parked: long-context only, needs a prior-art check |
| 13 | **Online-trained next-token expert predictor** (CPU perceptron/Markov prefetchers) | if recall@16 one *token* ahead rises from 0.51 (training-free, MP step 6) to ≥ 0.75, BigMoE gets a whole token (~10–20 ms) of lead time instead of one layer (~250 µs, too short per BM v2). Could remove most of the 6 ms/token SSD stall time → +20–30 % on BigMoE | **testing**: `scripts/online_pred.py` on existing traces (CPU) |
| 14 | ANN search for routing (vector databases) | the router is a 256×2048 matvec, ~1 MB/layer: ≤ 1 % | dead |
| 15 | CDN/edge prefetch of the next request | covered: SpecDecode's persisted prefix caches; BigMoE's look-ahead | dead |
| 16 | Multigrid / coarse-to-fine (physics) | = drafting with a coarse model, then verifying; MTP already drafts at 91 % | dead (covered by speculative decoding) |

## Round 2 measurements (2026-10-18 02:10, CPU)

**#2 Screened lm_head (`scripts/screen_head.py`, `data/screen_head.txt`).** 900 real hidden states, Qwen3.5-2B head
(248320×2048, the 35B's shape). Top1–top2 logit gap: median 0.93, p10 0.13.
- 2- and 3-bit screens with a provable bound (Cauchy–Schwarz per row, or per-group L1) rule out **nothing**: the
  bound (~row error × ‖h‖) is wider than the whole logit spread. The rank-256/512 SVD screens are the same, which agrees with vLLM PR #60566.
- The "4-bit" rows are invalid: this head *is* 4-bit, so a 4-bit requantization has almost no error.
- Verdict: **dead at ≤3 bits.** Only a screen close to the head's own precision can work (vLLM: FP8 over bf16).
  For the Q8 35B, that means a ~4-bit screen copy (+0.29 GB RAM), saving at most ~0.25 GB/token ≈ 7 % at plain Q8 decode. Not worth it.

**#13 Online-trained next-token expert predictor (`scripts/online_pred.py`, `data/online_pred.txt`).** Predict
token t+1's top-8 per layer from token t's routing at layers L−1..L+1, learned online (no offline training).
Recall at a 16-expert budget:

| set | temporal (MP baseline) | online count/Markov | **online logistic** |
|---|---|---|---|
| code | 0.515 | 0.476 | **0.645** (0.659 with layers L±3) |
| chat | 0.521 | 0.493 | **0.660** |
| bulgarian | 0.672 | 0.542 | **0.729** |
| realcode (22K tokens) | 0.487 | 0.485 | **0.642** |

- Online logistic regression adds **+6 to +16 points** over the best training-free predictor, with no training data.
  That's real, but short of the ≥ 0.75 I wanted for a whole-token prefetch.
- What matters for BigMoE is recall on the ~17 *misses* per token (the 60 % resident cache already holds the easy
  ones), not overall recall. Next step: feed these predictions into BigMoE's `sim_pf2.py`-style simulator as a third
  prefetch source with a whole token of lead time. Handed to BigMoE as an idea; not a breakthrough by itself.
- Prior art: offline-trained predictors (ExpertFlow, PreScope, CasMoE's online non-parametric matcher, MoE-SpeQ).
  A purely online learner seems less explored, but that doesn't matter until the miss-recall test.

## Status (02:15)
No breakthrough survived. What's left, in order of value:
1. (BigMoE) the online predictor's recall on cache misses, in their simulator.
2. (SpecDecode) padding verify passes of 6–15 rows up to 16 on MLX 0.32.4 (already passed on).
3. Parked: lossless KV entropy coding for 100K contexts (#12), ANE co-prefill (#6).

**#13 follow-up: whole-token-ahead prefetch with the online predictor (`scripts/miss_pred.py`, `data/miss_pred.txt`).**
Per-layer LRU at 60 % resident (BigMoE's exact design). After token t, prefetch the predictor's top-P non-resident experts for t+1.

| set | misses/token, LRU | + online prefetch P=16 | prefetch loads/token | used next token |
|---|---|---|---|---|
| chat | 11.4 | 9.7 (−15 %) | 6.3 | 26 % |
| code | 14.8 | 13.0 (−12 %) | 7.5 | 23 % |
| code, P=8 | 14.8 | 13.7 (−7 %) | 2.6 | 40 % |
| realcode | 18.3 | 15.3 (−17 %) | 9.2 | 23 % |

**Verdict: weak.** It removes only 12–17 % of misses, and ~¾ of its extra SSD reads are wasted. BigMoE's mid-token one-layer
look-ahead already cuts full-read stalls by ~67 % (12.3 → 4.1 per token). The online predictor's only advantage is lead time
(a whole token instead of ~250 µs), so at best it's a small add-on for the misses the look-ahead reads too late. Not worth a BigMoE slot
before their depth-2 prefetch (v3) result.

## Round 3 (2026-10-18 midday): a systematic pass. The cost model, then every assumption behind each line item

### Cost of one step on this Mac (Qwen3.6-35B-A3B, 10K context, from SD/MP/BM)
| line item | Q8 bytes / time | assumption the stack makes | who broke it | gap? |
|---|---|---|---|---|
| routed experts | 1.07 GB (31 %); 4-bit 3.0 ms vs 2.0 roofline | every routed expert is read in full, every token | activation sparsity inside experts (TEAL/CATS, lossy); substitution (MP) | lossy only |
| linear-attn (GatedDeltaNet) projections + mixer | 1.08 GB (31 %); 4.4 ms vs 2.5 roofline: **the biggest absolute gap** | 5 separate projections + recurrence kernels | fusion measured 0.98x (SD §6) | kernel tail; megakernel |
| lm_head | 0.54 GB (16 %) | full vocab every token | vLLM screened head (FP8) | measured dead at ≤ 3 bits (#2) |
| full attention + KV | 0.29 GB + 0.33 GB; ~2 GB at 100K | exact dense KV read | KIVI, Quest, sparse indexers | well covered |
| quantization scales/biases | 0.5 bit/weight = 6 % of Q8, 11 % of 4-bit bytes | bf16 scale + bias per 64 weights | mxfp4 / Q4_K super-blocks | known, ~5 % |
| kernel chain | 1,770 kernels; 62 % of roofline | one CPU-launched kernel per op | Gemma-4 Metal megakernel, BaseRT | known on Metal |
| verify pass (spec) | 89–91 % of spec time; S=4 costs 1.75x | drafts come from the model or *this* context | **SuffixDecoding (Snowflake, vLLM/ArcticInference): a suffix tree over *all past outputs*, built for agent loops** | **CUDA-only; not on MLX → test (#17)** |
| SSD stalls (BigMoE) | 6.2 ms/token at 60 % resident | an expert must be resident before its layer runs | look-ahead (BM); substitution (MP) | covered by siblings |
| prefill, MoE part | ~45 % of a Q8 prefill chunk; gather NAX 8 vs 20 TFLOPS dense | chunk-major order: each 2048-token chunk runs all 40 layers, so ~64 rows per expert | MLX #3925 (tile choice); **Layered Prefill (MLSys 2026, arXiv 2510.08055, CUDA serving)** splits by layer groups to cut repeated expert loads | CUDA-only → MLX port candidate (#18) |
| one request at a time | Claude Code's subagents run in parallel | single-stream decode | continuous batching, Sarathi decode+prefill piggyback | known |

Killed by reasoning: **jump-forward on copied spans** (SGLang-style forcing of `old_string` copies). A forced 16-token span costs the
same forward pass as a 17-row lookup verify, so it saves nothing over the lookup drafting SD already does.

### #17 Global suffix-index drafting (SuffixDecoding port; idea from CDN caches)
Today's lookup only searches the current request. Claude Code sessions repeat each other: the same files, the same boilerplate,
the same commands. Index every earlier session's outputs and tool results (on disk in ~/.claude/projects), and draft from the most
recent earlier match whenever the current context has none. Test on Kamen's transcripts (Claude's outputs as a proxy for the
local model's, same task distribution): tokens per decode step for context-only vs context+global lookup. `scripts/suffix_global.py`.

### #18 Layer-major MoE prefill (database: "batch the scan")
Within a layer, the MoE input of chunk c depends only on chunk c's attention at that layer. So run attention chunk by chunk for
layer L, then run layer L's MoE for the *whole prompt* at once: a 16K prompt gives ~512 rows per expert instead of 64, filling the
NAX tiles. If the gather reaches ~16 TFLOPS (dense is 20), the MoE part halves: prefill ~1.3x (1,535 → ~2,000 tok/s), ~5–7 % of a
session, more at session starts and compactions. Activations for 16K tokens are only 64 MB. Needs GPU timing: after BigMoE's 125B run. Prior art: Layered Prefill (up to 39 % less expert traffic, TTFT −70 % on CUDA servers); not on MLX.

### Round 3 literature sweeps (4 subagents, 12:10–12:30; full notes in data/sweep_{a,c,d}.md)
Ranked gaps, all against Q8 35B numbers. *Lossless* = same output as greedy decoding.

| rank | gap | lossless? | who has it | estimate | next step / owner |
|---|---|---|---|---|---|
| 1 | **DFlash block-diffusion drafter for Qwen3.6-35B-A3B already exists with an MLX backend** (z-lab/dflash, HF `z-lab/Qwen3.6-35B-A3B-DFlash`) | yes (it's a drafter) | z-lab; reported 2.2–2.9x | maybe +10–25 % over MTP+lookup on fresh code (Write/prose), where lookup does nothing | SpecDecode: A/B against its MTP head on its 29 samples |
| 2 | Global suffix-index drafting (SuffixDecoding port, #17) | yes | CUDA only (ArcticInference), up to 5.3x on agent benchmarks | to be measured on Kamen's transcripts | me: `scripts/suffix_global.py`, waiting for CPU room |
| 3 | Layer-major MoE prefill (#18; Layered Prefill port) | yes | CUDA only (MLSys 2026) | prefill ~1.3x → session +5–7 % | GPU test after BigMoE |
| 4 | Training-free adaptive top-k (ACE 2609.05228 even names Qwen3.6-35B-A3B; Elbow routing; MoDES) | no | CUDA only | top-8 → ~5 average ≈ −0.4 GB/token ≈ +13 % plain; also shrinks the verify expert union | MoEPrefetch has the perplexity harness: quality test of a router-probability cutoff |
| 5 | TEAL-style activation sparsity on GatedDeltaNet projections (biggest untouched slice, 1.08 GB) | no | TEAL on CUDA; Apple Metal paper 2511.04477 (dense Llama, not MLX); nobody on linear attention | 40 % sparsity ≈ −0.43 GB ≈ +14 % | quality test first (CPU-feasible on the 2B proxy) |
| 6 | Draft head on ANE/SME in parallel with the GPU | yes | Mirror-SD (paper); no measured Mac run | ≤ 6–9 % (drafting is only 6–9 % of spec time, SD §10) | low priority |
| 7 | PASTE-style speculative execution of read-only tools in a Claude Code proxy | n/a | nobody locally | −20–40 % agent wall-clock on Read/Grep-heavy loops (not token speed) | proposal for Kamen |

Ruled out by the sweeps: lossless sparsity (SiLU never gives exact zeros); temporal delta GEMV (consecutive hidden states are far apart);
layer-skip self-speculation on a 3B-active MoE (~1.0–1.2x at best); Lookahead/Jacobi (≤ 5 %); SLC (~24 MB, too small for weights);
Metal 4 residency sets (warm-up only); M5 GPU compression (textures only).

### Round 4 literature sweeps (12:40, notes in data/sweep_r4{a,b,c,d}.md)
Main conclusion: **the lossless headroom inside one decode step is nearly used up.** The verify expert union is a hard floor
(~5–10 % left), kernels have ≤ 1.3x left (megakernel territory), and drafting is ~7 % of spec time.
The remaining lossless wins are *around* the step: prefill, cache structure and draft quality.

| rank | lossless candidate | who has it | estimate on our setup | owner / next step |
|---|---|---|---|---|
| 1 | **DFlash drafter for Qwen3.6-35B-A3B (MLX backend exists)** | z-lab | +10–25 % on fresh code/prose over MTP+lookup | SpecDecode A/B |
| 2 | **BaseRT-class NAX prefill (claims 3.9x MLX prefill on M5 Pro, arXiv 2607.19438)** + layer-major MoE prefill (#18) | BaseRT (Metal, dense); Layered Prefill (CUDA) | prefill 1.3–2x → session +5–15 %; −50 % on session starts and compactions | me: `scripts/layered_prefill.py 8` (one GPU command): µs/token of one MoE layer vs tokens per call |
| 3 | Global suffix-index drafting (#17) | ArcticInference (CUDA) | to be measured | me: `scripts/suffix_global.py` (CPU, one command) |
| 4 | GatedDeltaNet state checkpoint at exact branch points (Marconi); whole-session SSD tier (oMLX) | CUDA research; oMLX | 2–8 s per branched turn; 30–90 s per session resume | SpecDecode |
| 5 | ZMLX fused gate+up+SwiGLU expert dispatch | ZMLX (custom MLX build) | ~0–6 % | low |
| lossy | adaptive top-k (ACE), SERE verify remap, TEAL on GatedDeltaNet, kv_bits 8, Quest sparse attention | CUDA mostly | +13 % / verify 1.75→1.45x / +14 % / +20–45 % at 100K | only if Kamen relaxes the no-quality-loss rule |

## Round 5: bold, assumption-breaking ideas for MoE on unified memory (2026-10-18 13:0x), with honest estimates
| idea | the assumption it breaks | estimate (35B proxy / 125B) | verdict |
|---|---|---|---|
| **Self-speculation with the cache-resident model as drafter + routing oracle.** Draft k tokens with MP's substitution model (no SSD stalls); its routing for those tokens predicts the exact verify pass's experts almost perfectly, so all verify misses are prefetched a whole draft ahead | "prefetch can only see one layer ahead" | each draft token costs a full step (~10.5 ms on 35B); with ~95 % agreement and k=8: ~6.8 tokens per 119 ms cycle ≈ 57 tok/s vs BigMoE's exact streamed MTP k=3 at 80.8 | **dead**: the drafter isn't cheap enough. Only the oracle part survives, and only if a cheap drafter could also route |
| **SMT for MoE streaming**: interleave two decode streams (main agent + subagent/classifier) layer by layer, so one stream's SSD stall is filled with the other's compute (CPU hyper-threading) | "one stream owns the GPU" | stalls are ~6 of ~45 ms/token on the 125B, so ≤ 13 % aggregate, 0 % single-stream | small; only with parallel subagents |
| **Lossless fixed-rate exponent packing of every bf16 tensor read per token** (HC mixers 1.27 GB/token; norms, router, indexer, GDN small tensors) | "bf16 is incompressible at decode speed" | ~25 % of the bf16 bytes → ~0.3 GB/token on the 125B ≈ +6–8 % exact | **alive**: check script ready (`patches/hc_exp_pack/`), CPU, ~1 min, blocked by quiet locks |
| Neuron-level partial residency of experts (hot neurons resident, cold streamed) | "experts are atomic" | exactness needs every neuron (SiLU never exactly 0) | dead for exact |
| Compute missed experts on CPU SME as their bytes arrive | "the GPU computes everything" | doesn't remove SSD latency; adds a sync | dead |
| GPU demand paging from mmap'd weights | "residency is managed by the app" | Metal makes resources resident before commit; no GPU faulting from disk | dead |
| Shared-basis exact factorization of a layer's experts | "experts are stored independently" | quantized experts are full rank; entropy ≤ 5 % (SD §6) | dead |

Honest bottom line so far: on this hardware the exact, single-stream budget is bounded by bytes per accepted token. The
remaining big levers are already being pulled (MTP depth, residency, fusion). A >2x breakthrough would need either a
much cheaper exact drafter that also predicts routing (none exists), or relaxing exactness (MP's substitution, AcceptMoE).

### Round 5 sweeps (13:1x)
- **Routing-predicting drafter:** a real gap for *native MTP* heads (DraftExpert, MoE-SpeQ and SP-MoE use their own aligned drafters).
  But on the 125B, SSD stalls are only ~6 of ~45 ms/token now, so even perfect pass-ahead prefetch is ≤ ~13 %. It only becomes a lever after the overheads are fixed.
- **Lossless in-kernel codecs:** ZipServ/ZipGEMM (arXiv 2603.17435, CUDA Tensor Cores) is prior art for a *fixed-length* bf16 exponent
  format decoded inside the GEMM (~11.3 bits/weight, kernel 2.2x faster than cuBLAS). So row 15a = a Metal port of that idea. DFloat11 (Huffman) is ~2x slower at batch 1.
  For int4/int8 codes, MLX issue #3043 claims only 2.17 bits of entropy per 4-bit weight, against SD's measured order-0 3.6–3.7. If neighbouring codes carry
  structure, a fused rANS matvec could cut *all* weight bytes. **Cheap test:** `scripts/code_entropy.py` (order-1 conditional entropy along rows and columns; CPU, mmap).
- **Radical execution orders:** nothing published does exact "deferred MoE" or wavefront tree verification expert by expert. The combination
  "draft predicts k tokens' experts → SSD prefetch → expert-major verify of the window, with experts laid out by co-activation" is estimated at 1.4–1.8x
  on the 125B, and only if stalls are fully hidden. Our measured union (8 → 12.9 → 17.0 → 20.8 experts for 1–4 tokens on the proxy) is ~2.6 tokens' worth at S=4,
  above the ~1.5 needed for > 2x. **No > 2x exact scheme survives the arithmetic.**

### Measured (13:5x, CPU, read-only on the 125B bare pack): lossless bf16 exponent packing (`data/hc_exponents.txt`, `data/bf16_exponents.txt`)
- **Hyper-connection mixers (387 tensors, 1.28 GB, all read every token):** exponent entropy 2.86 bits. **99.5 % of 64-weight groups fit a
  4-bit exponent offset** → 12.16 bits/weight (76 % of bf16), bit-identical. **Saves 0.30 GB/token** of the 125B's ~4.7 GB → ceiling 59 → ~63 tok/s,
  ~+6 % on the exact path once the fused HC read decodes it. (Huffman bound 10.9 bits; fixed 4-bit offset gets most of it with a trivial decoder.)
- **Expert scales and biases are bf16 too:** 6 families × 1.26 GB = 7.5 GB stored. 100 % of groups fit even a 3-bit offset → 11.1 bits (69 %).
  That frees ~2.3 GB of RAM (≈ 5 % more experts resident at the same budget) and cuts ~3–4 % of the bytes per streamed expert. Exact.
- **4-bit weight codes, order-1 context (`data/code_entropy.txt`, 35B, layer 0):** neighbour codes barely predict each other (H drops 0.00–0.05 bits;
  in_proj_a 0.35). Only the layer-0 expert down_proj sits at 3.0–3.2 bits. MLX #3043's "2.17 bits" claim doesn't hold here. **Dead for int4 codes.**

## Round 6 (14:2x): what the new exactness rule unlocks (our own deterministic fused kernels are the reference)
Map: BigMoE 125B at 55 % resident, ~44 ms/token = GPU 26 ms (0.55 ms/layer vs a 0.35 floor) + SSD waits 9.5 ms + token-to-token 8.4 ms.

| idea | attacks | honest estimate | novelty on Metal/MLX | cheapest test |
|---|---|---|---|---|
| **Close the token-to-token gap**: build token t+1's MLX graph while token t runs (lazy input = on-GPU argmax, `async_eval`), then fetch the n-gram (PLE) rows for the top-k candidates / MTP draft while lm_head runs, so the CPU never sits between tokens | 8.4 ms of CPU-side sync (graph build, lm_head, sampling, n-gram) | if ~6 of 8.4 ms hide: −6 ms → **+15 %** | async_eval pipelining is standard in mlx-lm; *speculative PLE row fetch* for candidate tokens is new | BigMoE's cProfile split of the 8.4 ms; then time the Python graph build alone (no GPU) |
| **Intra-layer megakernel between gates**: everything from the gate release to the next router (HC write, HC read, GDN/attn, HC write, HC read, router + routing row) as 1–3 dispatches instead of ~25–35 | 0.2 ms/layer of hop latency (~4.8 µs/hop measured on the 35B) | −5 to −7 ms → **+12–16 %** | Gemma-4 Metal megakernel covers dense layers; none for hybrid GDN/MoE with a CPU residency gate | count kernels per layer (mx.export_to_dot); if ~30 hops × 4.8 µs ≈ the gap, build |
| **Split-phase MoE, now cleanly exact**: on a stalled layer, compute resident experts into per-expert slots while misses load, then reduce in a fixed order (deterministic, batch-invariant) | SSD waits 9.5 ms | wait = max(0, load − resident compute) → ~25–40 % of waits: −2.5 to −4 ms → **+6–9 %** | ExpertFlow/DALI on CUDA; new with an exact fixed-order reduction on Metal | BigMoE trace: per stalled layer, resident-compute time vs miss-load time |
| Expert-major verify of an MTP window | verify reads per (token, expert) | ~4–5 % (dedup kernel measured, SD §11) | measured already | — |
| Persistent GPU-resident decode loop (no CPU between tokens) | token-to-token gap | blocked: the residency gate and the n-gram fetch need the CPU, and Metal has no in-kernel CPU signalling (BM §4) | — | only for a fully resident model |
| Attention + GDN in one pass; cross-layer pipelining | — | layer dependencies forbid it; only look-ahead prefetch crosses layers | — | dead |

Combined, the top three ≈ 1.35–1.45x on plain exact decode (22 → ~30–32 tok/s) before MTP. The first one is the cheapest and probably the largest.

### Round 6 feasibility sweeps (14:4x)
- **Megakernel on M5:** Metal guarantees no co-residency or forward progress across threadgroups. It can work only with ~1 threadgroup
  per core, `coherent(device)` buffers or atomics, and an atomic work queue for MoE (Cohere's CUDA megakernel style). No published
  Apple grid-barrier cost exists. Mirage MPK / Cohere / Hazy on CUDA: 1.0–1.7x at batch 1. BaseRT on M5 Pro: 1.33x over MLX
  with fusion alone, not a megakernel. Realistic saving between gates: ~0.08–0.12 ms/layer ≈ 4–6 ms/token *if* a barrier costs
  1–3 µs. **Test:** `scripts/gridbarrier.py` (µs per dependent hop vs per grid barrier, G = 1..128 threadgroups, bounded spin, so
  no GPU hang). This is the first published number either way.
- **Host loop:** nobody runs zero-host-sync decode on Apple. Graph and command-buffer replay saves only ~0.1–0.2 ms/token
  (llama.cpp #14482). The win is overlap: `async_eval` double-buffering, a GPU argmax with a 4-byte readback, n-gram rows
  pre-read for the candidates, and `mx.compile` with scalars kept as arrays (MLX #3964 pitfall). Consistent with M6a.

### Measured (15:0x, quiet lock): grid barrier vs kernel hop on the M5 Pro (`scripts/gridbarrier.py`, `data/gridbarrier.txt`)
- Dependent kernel hop (400 chained tiny MLX kernels): **5.4–5.9 µs/hop**.
- Grid-wide barrier inside one dispatch (device-atomic counter + bounded spin), 400 phases: **1.5–2.7 µs/barrier for 1–64 threadgroups
  × 256 threads, 3.6 µs at 128**. Every configuration completed (counter = G × phases, no timeouts), so 128 × 256 = 32K threads
  were co-resident in practice, even though Metal doesn't guarantee it.
- What it means for BigMoE's 0.2 ms/layer gap (~30 hops × 5.4 µs ≈ 0.16 ms, i.e. nearly all of it): a megakernel between gates
  with ~30 barriers saves ~0.09 ms/layer (−4.2 ms/token, ~+10 %). Merging independent ops into ~10–12 phases saves ~0.13 ms/layer
  (−6 ms/token, ~+14 %). The first measured number for this on Apple GPUs that I could find.
- n-gram gather (`data/ngram_gather.txt`): 0.5 ms cold per 16-row call and < 1 ms under expert-streaming SSD load, so BigMoE's
  "4.3 ms PLE gather" is a GPU wait (`np.asarray` on the previous token's arrays inside `stage()`), not I/O.

### Measured (15:3x): can a running Metal kernel publish to the CPU mid-kernel? No (`scripts/publish.py`, `data/publish.txt`)
- A store made at kernel start becomes visible to a spinning CPU thread only about 35 µs before MLX reports completion, i.e. at
  command-buffer end. That held for 0–256 MB of full-GPU streaming reads after the store, and for a threadgroup that stores and exits
  immediately. Partial early visibility appeared only after ≥ 256 MB of traffic (reads don't seem to evict the dirty line).
- Consequence: a persistent per-token kernel whose CPU loader learns the routing mid-kernel is **not possible on current Metal**.
  The CPU gate must stay a command-buffer boundary, so megakernels can only span between gates (BigMoE M6b). Candidate (3) is dead.
- Re-scored by BigMoE's count: ~85 compute kernels per layer × 5.4 µs/hop ≈ 0.46 ms, nearly the whole 0.55 ms/layer. A between-gates megakernel
  (2 per layer) with ~10–20 phase barriers at ~2.5 µs could bring a layer near the 0.35 ms floor: up to −9 ms/token (44 → ~35 ms, **~+25 %**).
- MTLIO and Metal 4 sparse buffers for experts: no prior LLM use found, but both still need the CPU to read routing first, and the
  slot lookup they'd replace is already ~16 µs/layer. Low value.

### Measured (15:4x): GDN-layer megakernel timing skeleton (`scripts/mega_skeleton.py`, `data/mega_skeleton.txt`). The payoff is small.
- Real 125B GDN-layer shapes, 70 MB of weights + state per layer, 8 distinct layers: **megakernel (1 dispatch, 9 grid barriers)
  0.419 ms/layer vs eager MLX chain 0.433 ms/layer (+3 %)**. Both run at ~165 GB/s, against a floor of 0.251 ms at 280 GB/s.
- So the gap to the floor is **not hop latency**. MLX already overlaps most kernel boundaries. It is the bandwidth efficiency of
  decode-sized matvecs: the HC 320×10240 / 10240×320 bf16 shapes reach ~200 GB/s even in isolation, the router 512×2560 only 133 GB/s
  (`data/hc_shapes.txt`). A megakernel that keeps each phase's row-per-simdgroup partitioning inherits the same efficiency.
- Revised estimate for BigMoE M6b: ~3–5 % unless each phase gets better work partitioning (split-K for few-row/long-K shapes, or
  staging the next phase's weights during barrier waits). Not worth an exact full-layer implementation now.
  The "85 kernels × 5.4 µs" arithmetic overstated the hop cost, because independent and overlapping kernels don't pay the full hop.

### Measured (16:2x): split-K decode matvecs vs MLX on the 125B's real shapes (`scripts/splitk.py`, `data/splitk_m1.txt`). Dead.
Dependent chains of 16 distinct matrices, M = 1. After fixing two kernel bugs (a preprocessor `#if` on a template parameter; fences
on every writer and reader), the split-K kernel is correct (rel. err ~2e-3) and deterministic, but **slower than MLX on every shape**
(−0.3 to −1.0 ms/token each). MLX's matvecs already reach 217–282 GB/s on the big shapes (HC 320×10240 / 10240×320 bf16: 230 / 217 GB/s;
in_proj_qkv 270; q_proj 282). The real losers are **tiny, latency-bound ops**: the HC inject (4×10240 bf16) takes 7 µs per call
(12 GB/s) × 96/token ≈ 0.7 ms, and the q8 router (512×2560) runs at 120 GB/s. The lever there is folding tiny matrices into neighbours
(e.g. inject rows into the HC down matrix, as MTPLX's v3 kernel does: ~0.6 ms/token), not split-K.

### Measured (17:xx): 15a HC pack vs BigMoE's compiled dup-row reference, and a better exact alternative
- **15a (exponent pack + MLX GemvWide clone): bit-identical but not faster.** `PackedLinear` matches the compiled dup-row reference
  bitwise at S=1/3/6 (`data/test_packed_hc.txt`), and its lane counts are probed against MLX itself (`data/gemv_probe.txt`). But at the exact
  GemvWide lane schedule the HC matmuls are latency-bound (~145 GB/s; 320 rows × 32 lanes ≈ 10K threads with a fixed per-lane order),
  so 22 % fewer bytes buys nothing and the decode adds latency: down projection 63–84 µs packed vs 47–52 bf16 (`data/time_gemv.txt`).
  **Parked.** It only pays if the kernel becomes bandwidth-bound, which needs a different lane schedule, i.e. a new reference.
- **MTPLX's `hc_verify_read.read_rows` (3 dispatches instead of ~11): bit-identical to BigMoE's compiled dup-row reference** at
  S=1 ([x; x]), 3 and 6, read and write+read, attn and mlp, on real layer-0 weights. **~10–12 % faster per HC read** (101 → 90 µs at S=1,
  97 → 87 at S=3, 16 real modules chained; `data/test_verify_read.txt`). × 97 reads/token ≈ −1.1 ms/token (~2.5 %), exact, already shipped.

### Measured (18:xx): reference-v2 packed HC read with its own schedule (`patches/hc_v2/`). No net gain; dropped.
- v2 (3 dispatches: norm, split-K packed down+inject with fixed-order reduce, packed up+gate+mean): deterministic, batch-invariant,
  rel. err ~5e-3 vs the reference (bf16 rounding level), lossless weights (78 % of bytes). **S=1: 80 µs per HC read vs 87–90 for
  hc_verify_read (−8–10 %); S=3: 97 vs 87 µs (slower).** After hoisting loads, kernel B reaches 152 GB/s, but the read is three
  half-bandwidth kernels in series (norm 17 + down 34 + up 34 µs).
- v3 (2 dispatches, norm fused into every down threadgroup): 90 µs at S=1, 108 µs at S=3. The redundant norm work costs more than the dispatch it saves.
- Verdict: the HC read (13 MB bf16, ~97 per token) is occupancy/latency-bound at ~85–100 µs no matter how its bytes are stored.
  Verify passes (S=3) dominate speculative decoding, so v2 is not a net win. `hc_verify_read` (exact, adopted by BigMoE, +3 % end to end) stays.

## Round 7 (15:xx–16:xx): verify-pass SSD streaming on the 125B
- **(a) Why ~1 ms per stalled layer (`scripts/stall_io.py`, `data/stall_io.txt`):** SSD physics. BigMoE's loader already has every
  (expert × piece) read in flight. Cold F_NOCACHE replay of that pattern: 1 / 2 / 3 / 4 experts → 0.45 / 0.71 / 0.99 / 1.17 ms
  (6.1 → 9.5 GB/s). Contiguous blobs are only ~10 % faster; serial issue would be 3–4× slower. 13–14 GB/s needs ≳5 experts in flight.
  So the only levers are *earlier* and *more concurrent* reads, or fewer misses.
- **Model (`scripts/sim_verify.py`):** a FIFO SSD + LRU event model on 35B proxy traces **does not reproduce BigMoE's measured points**
  (none 27 vs 40.5 ms; v1 predicted worse instead of better), so its policy ranking is not used. Its oracle bound (all of a pass's
  misses known at the start → ~5 ms of streaming per pass vs ~26 measured) says early knowledge is worth up to ~20 ms per verify pass.
- **Constraint (BigMoE, spec125 k=3 at 50 %):** spares come out of residency. 12/12 → 36.8 tok/s is best; wider B with more spares doesn't pay.
- **Ranked levers that need no extra slots:** (1) prefetch into the LRU victim's slot, inserted at the LRU end (low-priority prefetch
  insertion, as in CPU caches: PACMan/RRIP style). Wrong guesses only churn the cache tail, so spares can go to 0 and B can grow.
  (2) **Runahead execution** for expert streaming: during a stall the idle GPU runs the next layer's attention + router on a scratch
  state with substitute experts, learns its near-exact routing ~0.4 ms into the stall, and issues the true misses early. It reads no wrong
  experts and uses no slots. No MoE prior art found for this discard-everything-but-the-prefetches scheme; nearest are MoE-SpAc and
  "Speculating Experts" (2603.19289), which *execute* guessed experts (lossy).
- **(c) Self-speculation with the resident-expert subset as drafter:** dead on cost. Each draft token costs a full dense pass (~27 ms on the
  125B, no SSD), against ~1 ms for an MTP draft. Its routing would be near-exact for prefetch, but k drafts cost k × 27 ms.
- **(d) Expert-major verify:** already true, since a verify pass loads each union expert once for all rows. Co-activation disk layout: ≤ ~10 %
  (the contiguous-read gain) and only when two misses are neighbours.

### Measured (16:xx): runahead execution predicts the next layers' experts almost perfectly (`scripts/runahead_recall.py`, `data/runahead_recall.txt`)
Qwen3.6-35B-A3B 4-bit, real weights, 900 tokens of real text (teacher-forced), per-layer LRU at 55 % resident (21 % of tokens
miss at a given layer). On the tokens that stall at layer L, recall of layer L+1's true top-8:

| predictor | @8 | @12 | @16 |
|---|---|---|---|
| look-ahead router on L's pre-MLP residual (BigMoE prefetch v1, MoEPrefetch pmid1) | 0.754 | 0.866 | 0.909 |
| runahead without attention (L's MoE from resident experts, then L+1's router) | 0.789 | 0.895 | 0.935 |
| **runahead (L's MoE from resident experts, then L+1's attention/GDN, then L+1's router)** | **0.961** | **0.999** | **1.000** |
| look-ahead 2 layers ahead (L+2's router on L's residual) | 0.646 | 0.761 | 0.814 |
| **runahead 2 layers ahead** | **0.960** | **0.998** | **1.000** |

- The attention step is what makes it work: without it runahead barely beats look-ahead. The measurement is pessimistic (the
  scratch run also perturbs earlier tokens' K/V; real runahead perturbs only the current rows).
- Mechanism and prior art: CPU runahead execution (Dundas & Mudge 1997; Mutlu, Stark, Wilkerson & Patt, HPCA 2003). On a
  long-latency miss, keep executing speculatively only to generate prefetches, then discard the results. Applied here: during an
  SSD stall the idle GPU computes the next layer's attention + router on a scratch state with substitute experts.
  MoE prior art executes predicted experts (lossy: "Speculating Experts" 2603.19289) or uses drafts as a sensor (MoE-SpAc);
  nothing found that runs ahead and discards everything but the prefetches.
- Estimate on BigMoE (verify S=3: 65 ms = 38 compute + ~26 streaming after prefetch v1; 27 stalled layers): stalls that follow
  a stalled layer (most of them in verify) get exact predictions with ~1 ms of lead, so they mostly disappear. Stalls after a
  non-stalled layer keep v1's behaviour. Expected streaming −50–60 % → verify ~51 ms → spec125 k=3 36.8 → ~45 tok/s (**+15–30 %**).
  The oracle bound (all misses known at pass start) is ~5 ms of streaming.

### Round 8 (17:xx): CPU memory-latency toolbox applied to expert streaming, first measurements
- **Cheaper runahead slices (35B, `data/runahead_recall_v2.txt`)**, recall of L+1's top-8 on stalling tokens, @8/@12/@16:
  attention-only (skip L's routed experts) 0.836 / 0.943 / 0.971; attention + shared expert 0.866 / 0.963 / 0.984;
  full runahead (resident experts) 0.961 / 0.999 / 1.000; look-ahead 0.754 / 0.866 / 0.909. The routed experts matter;
  the full slice is the one to build.
- **Equilibrium model (`scripts/sim_runahead_depth.py`; proxy traces, demand-first SSD, prefetch cap 6/layer, LRU-end insertion):**
  no prefetch 28.7 stalled layers / 27.8 ms per pass; look-ahead B=12 18.7 / 14.6 ms (−47 %, cf. BigMoE's measured −36 %).
  **Adding runahead on top changes ~nothing** (d=1..3: 14.7–15.4 ms). Once look-ahead is on, the remaining stalls are mostly short
  (one late expert, < 0.6 ms), so they end before runahead knows anything. Runahead is triggered by long stalls, and those are the ones look-ahead already removes.
  Caveat: the proxy's look-ahead recall (0.87 @12) is higher than the 125B's (0.75), which understates runahead there.
  BigMoE is running the real replay oracle on the 125B; that decides it.
- So "runahead 2 layers ahead makes streaming nearly free" is **not supported** by the model. Reads issued 2 layers early would
  cover the ~1 ms latency, but runahead only gets to run during long stalls, and look-ahead already removes those.
- **Prior-art sweep (subagent, abstracts only):** runahead itself (stall-time pre-execution with substitute experts, results discarded,
  prefetches kept) has **no MoE prior art**. The nearest are Pre-gated MoE (ISCA'24, trained gate, lossy), DraftExpert/SP-MoE/MoE-SpeQ
  (drafters drive prefetch), and DAOP/SPICE/BuddyMoE (keep the approximate results, lossy). Also unexplored for MoE: Srinath-style feedback
  throttling (runtime accuracy/lateness/pollution counters; APEX and ProMoE set budgets offline), criticality-ordered SSD reads (ProMoE
  only has current-layer-first preemption), low-bit "shadow" copies used *only* as runahead substitutes (exact), and RRIP/PACMan/Hawkeye-style
  insertion/replacement for expert caches (SeqMoE gets near Belady with *predicted* activations). Partial-residency prior art: MoEpic
  (hot experts' top segments resident), Diff-MoE (tiers).
- **Policy simulation on the real 125B pass unions (`scripts/sim125.py`, `data/sim125.txt`): uncalibrated, not used.** "No prefetch"
  comes out at 15.9 stalled layers / 13.7 ms per pass vs BigMoE's measured 36.5 / 40.5 ms. The recording is per gate chunk, so unions
  are smaller. And in both the 125B and 35B runs the look-ahead baseline gets *worse* than no prefetch (my SSD model saturates under
  prefetch volume), which contradicts the hardware. So its large FDP and criticality "gains" mostly undo that artefact. The trustworthy numbers come from BigMoE's hardware A/Bs.
- **Measured on hardware (BigMoE LEDGER V2, 04b610e): victim-slot prefetch (MoonShots idea 1) = +5 % on spec k=3, exact, tokens identical**
  (victim B=12 36.3 avg vs 12 spares 34.2 in the same window). Runahead replay oracle (V3 bound) is next on BigMoE's list.
- How the new techniques relate to runahead: **FDP throttling stacks with it** (it trims wasted reads and victim churn; BigMoE's
  "B=16 worse" says over-prefetch costs). **Criticality-ordered reads stack** (hold prefetch issue while demand reads are outstanding;
  only matters when queued prefetches delay demand). **RRIP-style insertion overlaps** victim-slot insertion: same lever, and my
  (uncalibrated) runs showed no win over LRU-end. All three are cheap hardware flags on BigMoE's loader, so test them there, not in a model.

## Round 9 (18:xx): 125B prefill: layer-major + full-layer double-buffered prefetch (`patches/layer_major_prefill/DESIGN.md`)
- Chunk-major prefill re-streams ~34 GB of non-resident experts *per 2,048-token chunk*: a chunk's union is ~all 512 experts. Today
  ~6.7–10 s per chunk.
- Exact fixes: (1) prefetch ALL of layer L+1's non-resident experts while L computes (no prediction needed in prefill; double-buffered
  1.4 GB pools, SSD at full bandwidth): short prompts ~7–10 s → ~2.8 s. (2) Layer-major order: experts loaded once per prompt; each (layer,
  chunk) step unchanged → bit-identical KV/state. Estimated 16K-token prompt ~70 s → ~16 s, 32K ~140 → ~32 s.
- Measured compute side (`data/layered_prefill.txt`): the 125B MoE layer costs 8.39 µs/token at 2,048 tokens per call, 5.31 at 8,192,
  4.80 at 16,384 (11.7 → 18.5 → 20.5 TFLOPS). With layer-major, bigger chunks fit; that's a reference-v2 option (chunk boundaries change bits).

### Measured on hardware (BigMoE 8835881, LEDGER V4): runahead does NOT pay on the 125B. Dropped.
Replay oracle (true L+1 ids into victim slots while L stalls; spec k=3, 50 % resident; tokens identical everywhere):
12/12 spares 34.0–34.8 tok/s → oracle with reads at once **38.7–39.5 (+13 %, the ideal)** → oracle with the realistic 0.6 ms
runahead delay **30.6–31.1 (−10 %)**: the late reads land after L+1 needs them and tie up victim slots. Look-ahead victim prefetch (V2)
already gets ~+5 %, ~40 % of the ideal.
Lesson: the 0.96–1.00 recall was real, but **lead time, not recall, is what pays**. Verify passes on the 125B are ~58 % compute
(38 of ~65 ms) and stalls already overlap, so even perfect next-layer knowledge is worth only ~+13 %, and runahead's own latency
eats all of it. (The equilibrium model in round 8 predicted "adds little on top of look-ahead"; the hardware agrees and is harsher.)

## Round 10 (19:xx): cutting 125B verify compute (+10.7 ms for S=3). Nothing exact beats MLX here.
- **Where it goes (`scripts/verify_gather.py`, `data/verify_gather.txt`):** one 125B MoE layer, MLX gather chain, realistic row overlap:
  1 / 2 / 3 / 4 rows → 10 / 16 / 21.8 / 27 unique experts, 0.226 / 0.278 / 0.344 / 0.427 ms per layer = 122 / 159 / 175 / 175 GB/s of
  unique expert bytes. Verify's extra expert time ≈ +0.12 ms/layer × 48 ≈ +5.7 ms of the +10.7 ms. Even plain decode's 10 experts are
  latency-bound (122 GB/s; ideal ~0.11 ms/layer).
- **MTPLX's fused MoE kernels (moe_glu_decode/verify, 2 dispatches, batch-invariant; `data/fused_moe_125b.txt`) are slower than MLX's
  gather chain at every width** (0.90x at 1 row, 0.68–0.75x at 2–4 rows, because they read experts per row). Not a v2-batch win.
- CPU SME / ANE co-execution: needs a second CPU↔GPU sync per layer (~70 µs × 48 ≈ 3.4 ms/pass) on top of BigMoE's gate, and shares
  DRAM bandwidth with an already bandwidth-bound workload. Dead, consistent with FusionML ("decode can't benefit").
- Overlap-aware draft shapes: a draft's experts are unknown until verify, so choosing drafts by overlap needs a routing predictor
  (look-ahead recall ~0.75 on the 125B), and SD showed expert-aware truncation never beats adaptive drafting. Dead.
- Remaining compute lever: **MLX 0.32.4** ("faster quantized MoE matmuls" in its release notes; SD measured +17 % decode on the 35B). It's
  already v2-batch item v2-5, and the cheapest test of the gather efficiency above.

## Round 11 (19:xx): 125B at long context (estimate from shapes + code reading; measurement handed to BigMoE)
- KV: 12 full-attention layers × 2 KV heads × 256 dims × (K, V) bf16 = **24 KB/token**: 0.4 GB at 16K, 0.8 GB at 32K, 2 GB at 80K. That
  comes out of the expert budget: −1 / −2 / −5 % residency → roughly −1 / −2 / −5 % tok/s (BigMoE's residency table: 50 → 55 % = +4.7 %).
- QSA sparse attention: each query attends to its top 512 four-token blocks (2,048 tokens) + the tail. MTPLX's default decode lane builds a
  dense [S, T] bool mask and calls fused SDPA. **MLX's `sdpa_vector` reads K/V only for unmasked keys** (`if (use_key) {...}`), so attention
  reads only the ~2,048 selected tokens (~50 MB/token over 12 layers, constant), plus a T-byte mask and an O(T) mask-check loop.
  (My "skip masked keys" idea turned out to be already in MLX. Checking first avoided a false claim.)
- The indexer scores every compressed key: 768 B × T per token (12 layers) → 12 MB at 16K, 25 MB at 32K, 61 MB at 80K. Small.
- GDN state: constant (~113 MB/token read+write over 36 layers), independent of context.
- MTP head: one full-attention QSA layer with its own cache, so the same pattern per draft step.
- **Estimate:** decode −1–3 % at 16K, −3–6 % at 32K, −8–12 % at 80K, mostly the residency lost to KV, not attention bandwidth. Opt-in
  MTPLX lanes (`MTPLX_QSA_FLASH` for decode, `MTPLX_QSA_GATHER` for verify at ≥16K) skip even the O(T) mask work, but change
  summation order (reference-v2 batch). Expected gain small given the above. Measure first.

## Round 12 (20:xx): layer-major prefill built and measured (PF2, `patches/layer_major_prefill/`)
- **Streamed 35B (BigMoE ExpertStreamer + PF1 staging, 55 % resident, 8,192-token prompt, 2,048-token chunks; `data/lmp35.txt`):**
  chunk-major (PF1) 9.3 s (880 tok/s), 24,040 experts staged → **layer-major 5.7 s (1,441 tok/s), 10,240 staged (each expert once):
  1.63x**. **Bit-identical**: all 80 cache arrays (KV + GDN states) equal, and 32 greedy tokens have the same sha.
- Implementation: `lmp.sticky_staging()` keeps PF1's per-layer staging buffer for all chunks of a layer and holds `ext.adopt` off
  during prefill (a runtime monkeypatch of bigmoe.stream; nothing in ~/BigMoE edited). `prefill_mlxlm` (35B) and `prefill_qwen4`
  (125B, mirrors bigmoe.decode.greedy's prefill incl. the PLE stage and rope scope) swap the loops; `greedy_lmp` = greedy + layer-major.
- 125B one-command A/B for BigMoE: `bench_lmp125.py 0.5 4096` / `0.5 16384` (ref greedy vs layer-major, token sha must match).

## Round 13 (20:xx): our own routed-expert kernel for 1–6 rows (`patches/expert_kernel/`). Target ≥ 240 GB/s **not reached**.
- xk v1 (3 dispatches: gate/up with expert ownership dedup, down, fixed-order combine; weight words hoisted): correct (rel. diff
  ~5e-3 vs mlx-lm SwitchGLU on real 125B weights), deterministic, **batch-invariant** (row 0 at S=4 == S=1 bitwise). But a flat ~154 GB/s
  of unique expert bytes: 1.04x MLX at 1 row, 0.92x at 3, 0.79x at 6 (`data/bench_xk.txt`).
- xk v2 (scales/biases hoisted too; one K pass serving all rows of a shared expert): more register pressure, lower occupancy. 1.27x at
  1 row with RJ=1 (MLX itself varied 0.19–0.24 ms run to run), 0.57–0.89x at 2–6 rows (`data/bench_xk_v2.txt`).
- MLX's SwitchGLU gather (148–175 GB/s at 1–6 rows) is at the practical ceiling I can reach for 10–40 small per-expert matvecs on the
  M5 Pro, consistent with SpecDecode's ~200 GB/s over many small matrices. Dedup saves bytes but costs owner serialization or
  registers, which is more than it saves. **No v2-batch row.**

## Round 14 (17:xx): verify-pass prefetch scheduler, flag-gated patch for BigMoE (`patches/prefetch_sched/`)
- **Key change, just-in-time table publication (`BIGMOE_PF_JIT=1`).** Today a layer's table for pass s+1 is published as soon as the
  layer is served in pass s. From then on its slots are frozen except while the GPU is held at an earlier gate, so victim
  prefetch only runs at stalls (27/48 layers per verify pass). Publishing X's table right after X-1 is served and released
  leaves X's slots free for the whole rest of the cycle. Prefetches can then evict and recycle at any layer, and the
  slot invariant ("a real demand miss ⇒ the GPU took the slow path") holds with no marker protocol. Cost: layer 0 always
  takes the gate.
- On top of JIT: reserve a slot when the prediction arrives (d = 1/2/3, router reuse), then issue reads nearest layer first with
  at most QD in flight. Demand reads never queue, and nothing new starts during them. The GPU posts ranked unique lists
  (`BIGMOE_PF_RANKED`). Feedback comes from per-class precision (distance × confidence) with a threshold θ, plus a per-pass
  budget. There is a full-pass/lead-time oracle bound (`BIGMOE_ORACLE_DIST`).
- MoEPrefetch's predictor numbers are in: trained probes and co-occurrence tables lose to router reuse at d=2/3 (FINDINGS step 17).
  So the scheduler ships with router reuse alone.
- Found while reading the loader: same-pass prefetch recycling plus `ready`-before-`table_add` can put a refilled slot in a published
  table. This is dormant, because verify rows leave no slots for d=2 and `BIGMOE_PREFETCH2` is off.
- **Measured (35B, streamed, 21 runs, all identical tokens):** on BigMoE's 61-slot row, the verify wait goes 25.1 → 18.3 ms/pass (−27 %)
  with JIT + QD 4 + cap 24, and plain decode drops 32 % (12/6/4). An oracle with true ids 3 layers ahead reaches 4.5 ms (−82 %), so the
  remaining lever is recall. Router reuse at L+2/L+3 and wide ranked lists do not help (6–13 % precision). Handed to BigMoE for the 125B
  A/B (J1/J2/ar/O-3 first; `patches/prefetch_sched/README.md`).

## Round 15 (20:xx–22:xx): host build, expert kernels, LC1, decoupled runahead
- Host build (35B SpecDecode loop): fully exposed (+4 ms injected → +4.2–4.9 ms/pass) but only 1.6 ms/pass. C++ gate glue (`ext.gate_glue`)
  is exact and saves 0.1 ms/pass (Python glue ≈ 6 µs/layer). The 125B MTPLX loop's exposure is unmeasured (`patches/host_build/host125.py`).
- Expert kernels, exact: xk2 (dedup) and xk3 (shared dequant) are bit-identical to MLX at S=1..6 and not faster. MLX gate/up gather runs at
  224 GB/s distinct; down (K=640, qmv_impl 256-value blocks with a half-lane tail) at 160 GB/s.
  v2: `pack4` (thread owns 4 rows, repacked layout) gives S=4 0.158 → 0.116 ms/layer: **−2.0 ms/verify pass**, −3.7 at S=6;
  deterministic + batch-invariant. Handed to BigMoE as a v2-batch row.
- LC1 (32K spec ≠ plain): in-process state, not a kernel. Fresh processes agree (lc1_fresh.py). My 35B "drift" was SpecDecode
  prefill not resetting the cache. Allocator poisoning leaves the streamed 35B exact, so the shared streamer path is clean.
  `lc1_leak.py` (clear/poison) is with BigMoE.
- GDN: MTPLX_FUSED_GDN_INPROJ is bit-identical (M=1..8) and saves 0.3–0.43 ms/pass; it needs a dup-row wrap at S=1.
- Decoupled runahead, measurement 1 (`scripts/overlap2q.py`): while the main MLX stream spins (gate stall), a second stream's
  slice runs fully in parallel (0.397 vs 0.401 ms alone). When the main stream is bandwidth-busy there is no overlap (both 2.59 vs
  2.34 ms serial). The slice is only free in stall time. A d=2/d=3 slice costs ~0.14/0.28 ms on the 125B (one or two GDN
  attentions, ~35 MB each). Round-8 recall applies: attention + shared expert 0.963@12 at d=1 past the stall; attention + MoE
  from resident experts 0.999@12, 2 layers: 0.998@12. Caveat: a slice that only runs in stall time protects the 2 layers after
  each stall, so it is self-limiting (equilibrium). Predicting every layer 3 ahead would cost ~+0.3 ms/layer when not stalled.

## Round 16 (23:xx): back to idea hunting (engineering handed to "Prefill & kernel engineering")
| idea | field | potential | exact? | status |
|---|---|---|---|---|
| Entropy-coded 4-bit expert codes (Huffman/ANS, fused GPU decode) for SSD bytes + residency | info theory | **6 %** | yes | measured, negative: 125B routed codes H0 3.70 bits, Huffman 3.73 (data/code_entropy125.txt) |
| Stripe expert reads over internal SSD + a Thunderbolt-5 NVMe (mirrored expert files) | storage / RAID-0 | +60–70 % read bandwidth → stall time −30–40 % → ~+13–20 % decode tok/s (125B, 50 %) | yes | no external drive attached; needs Kamen (hardware ~EUR 250) |
| Persistent prefix cache of post-prefill state (system prompt / tool defs / repeated files) on SSD | databases (materialized views) | TTFT of repeated prefixes → state load (~1 GB at 32K: QSA KV + GDN states) ≈ 0.15 s vs 80 s | yes (same state) | known technique; check MTPLX session snapshots; engineering → BigMoE |
| ANE as a second matmul engine for prefill MoE (affine 4-bit = grouped palettization, exact representation) | heterogeneous compute | +30–50 % prefill GEMM throughput once overheads are fixed | deterministic (v2) | prior art + feasibility to check |
| SLC warming (prefetch the next layer's experts into the system-level cache during idle phases) | CPU cache warming | unknown; DRAM-bound decode leaves little idle bandwidth | yes | cheap bandwidth-vs-working-set probe pending |

## Round 17 (Oct 19, 00:00): ideas from caching theory, storage, networking, game engines, microarchitecture
Bottlenecks: decode SSD stalls ~26–34 ms/verify pass; verify compute 48.7 ms; prefill 42 s at 16K; residency 50–60 %.
Novelty 1–5 from a prior-art sweep (≤3 searches each); PoC = CPU trace simulations in night_runner (data/night/).
| # | idea (field, era) | mechanism | bottleneck | napkin | prior art | novelty | PoC |
|---|---|---|---|---|---|---|---|
| 1 | **Stall-time-aware slot allocation** (cost-aware caching, Cao & Irani 1997) | a layer's simultaneous misses share one stall (measured t(m) = 0.45/0.71/0.99/1.17 ms for 1–4 misses), so allocate slots to minimise Σ t(misses), not misses | decode stalls | −5–10 % stall ms | AdapMoE/MoEpic per-layer budgets (load cost, not concave stall); MLP-aware replacement (ISCA'06) | 3 | stall_alloc_sim |
| 2 | Belady-informed eviction from draft look-ahead (MIN 1966 × spec decoding) | skip victims the next verify window needs | stalls | oracle bound pending | MoE-SpeQ, SP-MoE, SeqMoE (2025–26) | 2 | stall_alloc_sim (oracle) |
| 3 | **W-TinyLFU admission** (databases/caching, Einziger 2015) | a missed expert is computed from scratch and only admitted if its frequency beats the victim's | stalls, residency | pending | none in MoE; a replay study puts admission at only 3–16 % of the LRU→OPT gap | 4 | admission_sim |
| 4 | Stripe expert reads over internal SSD + Thunderbolt-5 NVMe (RAID-0, 1988) | mirrored expert files, split reads by expert parity / queue | stalls | +60–70 % read bandwidth → −30–40 % stall | DeltaFin blog (4 SSDs, M5 Max, Sep 2026) | 2 | needs hardware (Kamen) |
| 5 | Persistent prefix-state cache for hybrid models (databases: materialised views) | reload KV + GDN state for repeated system prompts/tool defs | TTFT | 80 s → ~0.2 s for repeated 32K prefixes | Marconi (MLSys'25), CachyLLama | 1 | → BigMoE (engineering) |
| 6 | ANE as a 2nd GEMM engine for prefill, affine int4 = grouped palettization (Cell SPE-style offload) | split prefill MoE GEMMs GPU+ANE | prefill | +30–50 % GEMM after overheads | NPUMoE (2026, MoE prefill on ANE); FusionML rules ANE out (dispatch) | 3 | weak: M5 GPU already has NAX |
| 7 | Early rejection of verify rows by logit lens (branch-confidence squashing) | drop rows predicted rejected from the remaining layers, exact fallback | verify compute | −3–5 % verify | ProPD (2024), HiSpec, SMoLPU | 2 | — |
| 8 | Gate-independent work (shared expert) inside stalls (game-engine async compute) | run it during the spin | stalls | −0.5 ms/pass | ScMoE, KTransformers | 1 | — |
| 9 | **AIMD congestion control for the speculative SSD queue depth** (networking/TCP) | grow QD while demand latency is fine, halve on a late demand read | stalls | small–moderate | none in MoE (FDP throttling HPCA'07 is adjacent) | 4 | → prefill/kernel session (scheduler) |
| 10 | Virtual-texturing feedback-buffer streaming (game engines) | GPU-written usage feedback drives priority streaming | stalls | ≈ existing scheduler | pieces exist (EStream, HOBBIT) | 3 | — |

Round 17 PoC results (night, data/night/): #1 W-TinyLFU admission NEGATIVE (more misses than LRU at 50 % and 60 %);
#3 stall-time allocation NEGATIVE for the novel part (same split as miss-optimal; per-layer budgets −25 % misses = known DP);
#2 Belady-informed eviction −2 % only; #6/#7 (Scout): SLC warming and logit-lens early rejection DEAD.

## Round 18 (Oct 19, 09:00): nested bit-planes as a storage and caching hierarchy (memory-halving hunt)
Starting point (this morning, 35B): the MSB plane of the existing 4-bit codes (q>>2, 2.5 bits/weight) is the best training-free
2-bit expert form tested (+11.3 % ppl vs RTN +18 %, clipped +20.7 %, rotated +14.7 %), and gate-rank-aware use (top-4 of 8
exact = MSB+LSB, rest MSB) is GSM8K-equal on 450 problems (429 vs 432, 440/450 answers equal; within noise).
Prior art for the parts (checked 09:00): nested bit-planes from MSB truncation = Any-Precision LLM (2024) and AnyBCQ (ICLR
2026); top-n experts per token get a correction, the rest stay low-bit = "Bandwidth-Efficient Adaptive MoE via Low-Rank
Compensation" (arXiv 2512.17073, low-rank factors not planes); hotness-driven precision tiers = DynaExq (2511.15015), HOBBIT
(2411.01433), MC-MoE; intra-expert sparsity streaming = FloE (2505.05950); ASTC weights with hardware decode = Apple's 2025
foundation-model report (server model, 6x6 = 3.56 bits/weight), MLX issue #2418 (not implemented). So the morning's
"not found" claim was too strong: the parts are known; what follows are policies on top of them.

| # | idea (field, era) | mechanism | bottleneck | napkin | prior art | novelty | PoC |
|---|---|---|---|---|---|---|---|
| 1 | **Plane-granular expert cache** (OS compression cache, Douglis 1993; zswap) | MSB and LSB planes are separate cache units; low-rank uses need only MSB, so cold LSB planes leave first and more experts keep a usable form | SSD stalls, residency | **measured: −41 % demand MB/pass at today's 34 GB** (155.5 → 92.4) | DynaExq/HOBBIT keep separate low-precision copies, not plane units | 3 | planes_cache_sim ✓ |
| 2 | **Never-stall LSB fallback** (game-engine texture streaming: draw the resident mip, never stall the frame) | a top-rank use whose LSB plane is late runs MSB-only; the LSB streams in the background | SSD stalls → only MSB misses stall | **measured: −62 % demand MB/pass at 34 GB** (58.5), 3.9 % of top-4 uses degraded; at 24 GB 127 MB/pass (< today's 155.5 at 34 GB), 7.5 % degraded | HOBBIT picks low precision by importance, not by deadline | 3 | sim ✓; quality = expert_tiers35 "fallback p" (queued) |
| 3 | **Embedded 3-tier planes** (successive refinement, Equitz & Cover 1991; JPEG2000 bit-planes) | MSB-1 (1.5 bits/weight) of every expert resident (125B ≈ 23 GB), MSB-2 bit and LSB plane stream for rank tiers | memory: the 125B toward ~24–28 GB | quality pending (T(4,6), T(2,4)...) | Any-Precision (dense, no MoE tiers) | 2 | expert_tiers35 (queued) |
| 4 | RD-optimal, residency-aware plane use (JPEG2000 PCRD-opt 2000; reverse water-filling 1949) | use a resident LSB at any rank (free quality); fetch a missing LSB only when gate² × ΔD > λ × bytes | SSD bytes at equal quality | between planes-R and fallback-R | 2512.17073 (rank cut only) | 3 | sim variant next |
| 5 | MSB-3 dense layers (Max 1960: interior-level quantizers) | attention/GDN/shared-expert linears at their own MSB-3 prefix (3.5 bits) | verify compute (dense = 65 % of 125B bytes/token) | −22 % dense bytes ≈ −12 % verify time if ppl holds | Any-Precision (dense truncation) | 1 | expert_tiers35 "dense-msb3" (queued) |
| 6 | ASTC hardware-decoded experts on the M5 GPU (texture compression) | experts as ASTC 6x6 (3.56 bits) decoded by the texture units for free | bytes and memory −21 % vs 4.5 bits | quality unknown; needs an ASTC encoder (astcenc, a download) | Apple 2025 report (known), MLX #2418 | 2 (port) | needs OK to fetch astcenc |
| 7 | Exact filter-and-refine lm_head (spatial DB filter/refine 1986; MIPS reranking) | MSB-plane logits + per-row error bound, exact refine of the few rows that can still win | verify lm_head bytes | −40 % lm_head bytes ≈ −1–1.5 % per pass, exact | ScaNN-style reranking (known for MIPS) | 2 | — |
| 8 | Neuron-sector streaming for misses (IBM 360/85 sectored cache 1968) | fetch gate rows, then only the up/down rows of strong neurons | miss bytes −30 % | + one dependent SSD latency per miss | FloE (2025) | 1 | — |
| 9 | MSB-plane self-draft (Wilkinson's mixed-precision iterative refinement 1948) | the MSB-only model drafts, the exact model verifies, zero extra memory | tokens/pass | draft ≥ 50 % of a target step: loses to MTP | QSpec, ML-SpecQD | 1 | — |
| 10 | ASTC/BC for 8x8 blocks (2 bits) | — | — | dead by napkin: 128 bits per 64 uncorrelated weights leave ~1 bit of index per weight | — | — | — |

Disk-light variant of #1 (`planes_cache_sim.py msbfull`, data/planes_cache_sim_msbfull.txt): only MSB plane files are
written (~40 GB); top-rank uses read the whole expert from the original 4-bit files. Demand MB/pass at 24/34/45 GB:
221.0/117.8/96.1 vs whole-expert 331.6/155.5/85.3, i.e. −33 %/−24 %/+13 %. Separate LSB files are worth it: −41 % at
34 GB. But MSB + LSB plane files for the 125B take ~70 GB, which leaves ~192 GB free, below the 200 GB floor.
PoC-1 (`scripts/expert_tiers35.py`, data/day2/expert_tiers35.txt; 35B; text changed again, ref ppl 11.861, own-path
harness −0.51 %):

| routed experts by gate rank | ppl Δ | expert bytes read vs 4-bit |
|---|---|---|
| MSB-3 (q>>1, 3.5 bits) for all ranks | +1.56 % | −22 % |
| **top-2 exact, ranks 2–7 MSB-3** | **−0.13 %** (noise) | −17 % |
| top-4 exact, ranks 4–7 MSB-3 | −0.85 % (noise) | −11 % |
| top-4 exact, ranks 4–7 MSB-2 (mix4-msb2m, GSM8K-equal) | +0.83 % | −22 % |
| mix4-msb2m + never-stall fallback, 10 % / 25 % of top-4 uses MSB-2 | +1.91 % / +3.73 % | — |
| MSB-1 (1.5 bits) for all | **+230 %** | −67 % |
| T(4,6): MSB-1 only for ranks 6–7 / T(2,4) / T(3,6) / T(2,6) | +3.5 % / +11.2 % / +3.7 % / +5.8 % | — |
| dense (310 non-expert linears) at MSB-3 / + mix4-msb2m | **+12.5 %** / +15.3 % | — |

Verdicts: #3 (MSB-1 resident tier toward ~23 GB) **dead**, even for the two lowest ranks. #5 (MSB-3 dense) **dead**: no gate
scaling protects the dense part. The fallback costs ~+1.1 % ppl per 10 % of top-4 uses degraded. At the sim's 3.9 %
(34 GB) that is ~+0.4 %, but it is also nondeterministic. New: a 3-bit low-rank tier is free (mix2-msb3 ≈ 0 %). A
2+1+1-bit plane hierarchy is the next quality/bytes frontier: top-2 exact, ranks 2–3 MSB-3, 4–7 MSB-2 ≈ −28 % bytes.
125B quality check (data/day2/planes125_sanity.txt): mix4-msb2m (top-4 of 10 exact, ranks 4–9 MSB plane, decode)
6/6 GSM8K, answers equal to BigMoE's eager and v2a on all 6. The full 150 is queued in day_runner2.

## Round 19 (Oct 19, ~10:40–12:00): 2+1+1 bit-planes, memory −30 % at equal SSD traffic?
PoC A (`scripts/planes3_sim.py`, data/planes3_sim.txt; 125B traces, deterministic, no prefetch, no fallback): each expert
stored as 3 cache units: P2 (MSB-2 + scale/bias, 1.54 MB), B3 (3rd bit, 0.69 MB), B4 (last bit, 0.69 MB). A plan gives
the bits per gate rank (0–9). Demand MB/pass:

| budget | whole-expert LRU (today) | 4444222222 | 4433322222 | **4333322222** | 4433222222 |
|---|---|---|---|---|---|
| 20 GB | 445.7 | 272.1 | 247.6 | 220.8 | 229.7 |
| 24 GB | 331.6 | 195.2 | 176.1 | **155.1** | 162.3 |
| 28 GB | 243.7 | 141.5 | 127.7 | 112.0 | 117.6 |
| 34 GB | **155.5** | 95.2 | 87.0 | 78.7 | 82.1 |

**Plan 4333322222 at 24 GB has today's 34 GB demand traffic: −30 % memory at equal SSD bytes**, if its quality holds. The
plan is top-1 exact, ranks 1–4 MSB-3, ranks 5–9 MSB-2. Quality of the 8-rank analogue (43332222) and of 44332222 is queued
in expert_tiers35b (day_runner3), then GSM8K.

**125B GSM8K-150 (11:10, patches/planes125, data/w125_gsm_planes_mix4.json):** mix4-msb2m (decode; top-4 of 10 exact,
ranks 4–9 MSB-2) 145/150 vs BigMoE eager 147 / v2a 147. Answers equal eager 145/150 (noise arm v2a-vs-eager 147/150);
eager-only #12 and #62, mix4-only none. Passes the screen (−1.3 points), but with the 35B's 5:2 the combined discordance is
7:2 (p ≈ 0.18): a slight negative lean. Next quality arm: 3-bit low ranks (free on the 35B).

Round 19 candidates (prior art checked 10:40, ≤ 1 search each; LIGHT pacing):

| # | idea (field, era) | mechanism | bottleneck | napkin / result | prior art | novelty | PoC |
|---|---|---|---|---|---|---|---|
| 1 | **2+1+1 bit-plane cache** (successive refinement, Equitz & Cover 1991; compression cache, Douglis 1993) | 3 nested planes per expert as separate LRU units; each gate rank fetches the planes its tier needs | SSD stalls, memory | **measured: plan 4333322222 at 24 GB = today's 34 GB demand (155 MB/pass); −49 % at 34 GB** | Any-Precision (planes, dense), DynaExq (tiers) | 3 | planes3_sim ✓; quality queued |
| 2 | Gate-mass-adaptive plans (reverse water-filling, Shannon 1949) | a token's gate mass decides each rank's tier: peaked gates spend fewer bits | bytes at equal quality | queued | HOBBIT (gate thresholds, separate copies) | 2 | expert_tiers35c (queued) |
| 3 | Miss-aware draft length (economics: marginal cost pricing) | stop drafting when the next token's expected expert misses cost more than its acceptance | stalls/pass | BigMoE's MTP accepts ~1.6 tok/pass: little to trade | MoE-SpeQ (amortization-roofline governor, 2511.14102), S2-MoE, 2605.00342 | 1 | dropped |
| 4 | Salient-group refinement (EZW/SPIHT significance coding 1993–96) | keep refinement bits only for input groups with large activation energy | refinement bytes | moderate | SpQR, SqueezeLLM, AWQ | 1–2 | — |
| 5 | Plane-aware prefetch (progressive JPEG, 1990s web) | prefetch only the MSB-2 plane for predicted experts: 1.8× more predictions per byte | stalls | needs prediction traces | HOBBIT-style low-precision prefetch | 2 | — |
| 6 | ARC / LRU-K / 2Q plane caches (databases 1993–2003) | frequency-aware replacement | stalls | TinyLFU and LFU already lost to LRU here (recency-driven) → ≈ 0 | classic | 1 | — |
| 7 | Certified-exact approximate greedy (interval arithmetic, Moore 1966) | bound the logit change from low-rank planes; argmax exact if the margin exceeds the bound | exactness with planes | **dead by napkin**: bounds through 48 layers are orders too loose | — | — | — |
| 8 | Cascaded verification with the resident MSB model (multigrid, Brandt 1977) | cheap verify first, exact verify only the surviving prefix | verify compute | **dead by napkin**: with ~1.6 tokens/pass the exact pass runs every pass anyway | ML-SpecQD | 1 | — |
| 9 | Entropy-coded LSB planes | — | — | **dead**: LSB bits are ~uniform (2 bits of entropy per weight) | — | — | — |

Round 19 top 3: #1 (memory −30 % at equal traffic, quality pending), #2 (pending), and the 125B GSM8K of round 18's plan
(145 vs 147, answers 145/150 equal vs noise 147/150).

**Round 19 PoC A quality (expert_tiers35b, data/day3/; 35B, harness path4 = −0.51 %, so read ppl relative to it):**
44333333 +0.38 % at −17 % expert bytes; 44443322 +0.42 % (−17 %); 44333322 +0.90 % (−22 %); 44433222 +1.07 % (−22 %);
43333333 +1.11 % (−19 %); 44442222 (mix4) +1.34 % (−22 %); 44332222 +1.67 % (−28 %); **43332222 (the 35B analogue of the
125B's 4333322222) +2.46 % (−31 %)**; 33332222 +3.26 % (−33 %).
Verdict: the "−30 % memory at equal traffic" plan costs ~2.5 % ppl, so it is **not** equal quality. At ~equal quality
(4433333333 on the 125B: +0.4 % on the 35B), the 3-plane cache needs 110.5 MB/pass at 34 GB vs 155.5 today (−29 %
SSD demand), a speed lever, but more traffic than today at 24–28 GB (237 / 172). Memory halving at equal quality is not
reachable by rank tiers of the existing 4-bit codes; it would need a better 2–3-bit representation (round 20 target).
The 125B GSM8K for 4433333333 is queued (day_runner3, data/next_plan.txt).

### Prior art for the plane track (orchestrator, 11:40): closer than round 18 said
- **SliceMoE** (arXiv 2512.12990, Dec 2025, "Bit-Sliced Expert Caching under Miss-Rate Constraints"). AMAT: training-free
  truncation of existing codes into MSB/LSB slices (q and zero-point floor-divided by 2^shift; G32 asymmetric). DBSC: one
  LRU shared across layers, MSB and LSB slices as separate units, LSB slices lowest priority ("aggressively evicted after
  initial access"). Gate-score sharpness marks 0–2 "critical" experts per token, which request MSB+LSB; a missing LSB →
  MSB-only. PCW: cache warm-up from prefill statistics. Simulated mobile XPU; DeepSeek-V2-Lite and Qwen1.5-MoE; GSM8K 5-shot;
  up to 1.81× decode speed. **This is our plane-granular cache (round 18 #1/#2) plus a gate-sharpness R.**
- **D2MoE** (arXiv 2504.15299, MobiCom 2025): nested bit-width ("matryoshka") expert weights, per-expert bit-width at
  runtime, hottest-expert-bit-first I/O scheduling under a memory budget; edge devices; −53 % peak memory, 1.39×.
- Earlier: Any-Precision LLM / AnyBCQ (nested bit-planes, dense), 2512.17073 (top-n residuals), HOBBIT, DynaExq.

**What is still ours, as far as we know:** (1) deterministic exact top-R: the LSB of a top-rank use is always waited for,
so the result is a fixed function of the input, compatible with exact speculative verification (SliceMoE falls back to
MSB-only on an LSB miss, so quality depends on cache state); (2) bias-corrected truncation: the per-group mean of the
dropped bits is folded into the MSB bias (msb2m +11.3 % ppl vs +12.0 % with the bin midpoint; plain floor truncation
as in AMAT is biased by ~1.5 s); (3) the 2+1+1 hierarchy with a 3-bit tier, which is nearly free for mid ranks; (4) a
real 125B on Apple Silicon: MLX on an M5 Pro, BigMoE's NVMe streamer with MSB/LSB as virtual layers (no C++ changes),
GSM8K-150 on the real model. Policies to borrow: SliceMoE's unified LRU with LSB-first eviction (vs our fixed Cm/Cl
split), gate-sharpness critical count (our expert_tiers35c tests the same thing as gate-mass plans), and PCW warm-up.
Kamen's target for this track (11:35): about today's speed in 24 GB total RAM, i.e. an expert cache of ~17 GB (half of
today's 34 GB).

### Plane-streaming prototype benches (patches/planes125, data/day4 + data/day5; eager decode, MTP off, 128 tokens)
Unit test (layer 0): MSB+LSB = the 4-bit experts within bf16 noise (1.1e-2 decode, 5.9e-3 prefill); MSB-only exact (4e-4).

| expert cache | BigMoE, no prefetch | planes best split (MSB/LSB slots) | other splits | BigMoE production (prefetch, JIT, fusions) |
|---|---|---|---|---|
| 34 GB (C=256) | **23.0 tok/s** (16.9 stalled layers/tok) | 19.8 (340/135; 23.6 stalled) | 17.7 (380/90), 16.9 (420/46) | 28.2 (9.4 stalled) |
| 17 GB (C=128) | **16.1** (33.3 stalled), peak 23.4 GB | 14.8 (150/89; 55.0 stalled) | 14.3 (110/134), 13.7 (130/112) | 19.5 (21.5 stalled), peak 23.4 GB |

**Negative.** Planes read fewer bytes (≈47 vs 70 MB/token at 34 GB) but are 14–26 % slower. A top-rank use needs
two separate reads from two fixed-size pools, so more layers stall; there are also two matmuls per projection. The sim's
demand-bytes metric missed stall events and the fixed split (SliceMoE uses one LRU with LSB-first eviction).
Gate-mass-adaptive plans (expert_tiers35c, 35B): t4=0.5/t3=0.7 +0.17 % ppl at −22 % expert bytes, vs fixed plans
+0.6–0.8 % at −22 %; t4=0.3/t3=0.85 −0.14 % at −21 % (harness −0.51 %). Adaptive beats fixed by ~0.5 pp at equal bytes,
which confirms SliceMoE's gate-sharpness rule (prior art). Next: the prior-art survey (PRIOR_ART.md), then build the best
published method.

## Round 20 (Oct 19, ~13:10): MSB-3 expert cache + LSB-1 sidecar (orchestrator's suggestion (b)); research-first
Prior art: SliceMoE (MSB slices cached, LSB slices for 0-2 critical experts per token, LRU with LSB-first eviction;
simulated, cache-state-dependent precision). Ours would be: a 3+1 split (equal accuracy on the 125B: plan 4433333333,
GSM8K 146 vs 147, answers equal 147/150 = noise arm), exact top-2 always waited for, and one contiguous record per expert.
Sim (`scripts/msb3_cache_sim.py`, data/msb3_cache_sim.txt; 125B traces, eager, 1 token per pass). Calibration: the sim's
whole-expert LRU at 17 GB gives 68.3 misses / 28.8 stalled layers per token vs BigMoE measured 77.6 / 33.3 (sim ~13 %
optimistic). Stalled layers per token / MB read per token:

| budget | whole-expert LRU | MSB-3 + 10 % LSB-1 sidecar | unified LRU, LSB-first |
|---|---|---|---|
| 17 GB | 28.8 / 189 | 29.9 / 136 | 36.5 / 142 |
| 24 GB | 21.3 / 113 | 23.1 / 79 | 28.1 / 83 |
| 34 GB | 14.0 / 61 | 16.9 / 45 | 18.3 / 46 |

Stall cost model fitted to BigMoE's two eager benches (34 GB: 43.5 ms/token, 16.9 stalled layers, 70 MB; 17 GB:
62.1 ms, 33.3, 215 MB), assuming ~12 GB/s for the 9-piece reads: **~0.40 ms fixed per stalled layer + ~0.083 ms per MB,
~31 ms of compute per token.** At 17 GB, half the token time is stall, and stall events, not bytes, dominate it.
MSB-3 cache at 17 GB under that model: 29.9 × 0.40 + 136 × 0.083 = 23.3 ms vs 27.2 ms → **−14 % stall, ~+7 % tok/s**.
Real but incremental, and it needs 3-bit kernels plus two-part reads for the top-2. **Not building it now**: the larger
levers for 24 GB are stall events (prefetch coverage, MoEPrefetch) and hit rate toward Belady (LRU ~73 % vs Belady ~87 %
at 25 % residency). The per-event fixed cost (~0.40 ms vs ~0.05 ms of NVMe command latency) is also worth a breakdown
(gate spin, GPU drain/refill, loader wake-up).

### Round 20 topic: the fixed per-stall cost (~0.40 ms per stalled layer vs ~0.05-0.1 ms of NVMe command latency)
Orchestrator (13:15): MoEPrefetch's split-phase (resident experts computed while misses load; exact, now in BigMoE) gained
only +3-5 % at 50 % and ~+3 % at 25 %: the breakdown should explain why so little of the stall hides.
BigMoE's gate path (ext loader_main): GPU post_ids writes [seq, n, ids] → one P-core loader thread spinning over all layers'
rows sees it (cpu_wait) → ensure() → blocking parallel preads for the misses (io) → slot ids written into 128 fresh
cache-line pairs of the (layer, pass) region → the GPU spin kernel polls fresh lines until seq appears → expert compute.
Measurement queued (`scripts/stall_breakdown.sh`): C=128 / C=256 eager with full ext.stats, and BIGMOE_SKIPSPIN=1
(compute-only wall time); stall = wall − compute, compared with io per token.
Prior art for a CPU-light I/O path (research-first, 4 of 6 searches so far): GPU-initiated storage access = BaM (ASPLOS'23,
GPU threads issue NVMe commands through high-throughput queues; CPU-initiated access is unsuited to fine-grained
data-dependent reads because of CPU-GPU synchronization overhead) and its product NVIDIA SCADA (CUDA only). On Apple:
Metal 3 fast resource loading (MTLIOCommandQueue: file → MTLBuffer loads, priorities, asynchronous and batched, synchronized
with GPU work through shared events). No LLM or MoE use, and no benchmark, found. MTLSharedEvent cost: Apple says only "may
add overhead"; GPU waits on events only between passes.
**Breakdown measured (13:02-13:12, `scripts/stall_breakdown.sh`, data/stall_c128.txt, stall_c256.txt; BigMoE eager, no prefetch):**

| cache | wall ms/token | loader I/O (pread) ms/token | stalled layers/token | I/O per stalled layer | achieved GB/s during I/O |
|---|---|---|---|---|---|
| 17 GB (C=128) | 62.1 | 28.4 | 33.3 | **0.85 ms** (2.3 misses = 6.4 MB) | 7.6 |
| 34 GB (C=256) | 44.1 | 11.5 | 16.9 | **0.68 ms** (1.5 misses = 4.2 MB) | 6.1 |

The wall-time difference between the two budgets (18.0 ms) equals the I/O difference (16.9 ms), so **the stall is the
loader's read phase**, not gate signalling or GPU drain. (BIGMOE_SKIPSPIN compute-only runs crash: wrong slots give NaN
logits, which mtplx's sampler rejects; not needed.) Why split-phase hides only 3-5 %: a layer's resident-expert compute is
~0.1-0.2 ms against ~0.7-0.85 ms of reads, so at most ~20 % of each stall can overlap. **The lever: the reads in a burst
run at 6-7.6 GB/s while the SSD sustains 12-14.6 GB/s at that queue depth (scripts/ssd_qd.py).** About 40 % of each stall
is loader inefficiency (9 pieces per expert incl. 51 KB reads, burst ramp-up and drain, GCD dispatch wake-ups). Halving it
toward the device rate would give ~+20-25 % at 17 GB, on top of prefetch coverage; BigMoE's record path (one read per
expert) should capture much of it. Next: a burst-latency microbenchmark (bursts of 1-3 experts: 9 pieces vs whole vs
4-8 chunks; GCD vs pre-spawned spinning threads; into wired memory) to put numbers on each cause.

## Round 21 (Oct 19, ~13:10-): the MTP acceptance gap on the 125B (orchestrator: diagnose; fix goes to SpecDecode + BigMoE)
Gap: oMLX reports 96.8 % acceptance (0.6.3, oQ4e, M3 Ultra, 4K context, adaptive depth <= 3; 58.1 vs 22.2 tok/s) and SGLang
an accept length of 3.3 incl. the bonus token (NVFP4, B200, bs 1); BigMoE sees ~1.6 tokens per pass.
**(a) Code read: MTPLX's head vs oMLX's Lightning MTP and vLLM's official qwen4_exp MTP (docs.vllm.ai, NVIDIA variant).**
Same wiring in all three: the trunk's pre-mixer widened stream [hc*d] → pre_fc_norm_hidden over the whole hc*d width (vLLM:
GemmaRMSNorm(hidden_size * hc_count), i.e. not per substream) → fc_hidden per substream; next-token embedding →
pre_fc_norm_embedding → fc_embedding, added to every substream; one QSA decoder layer; hyper_connection_mixer → shared
lm_head; the head's own pre-mixer output chains to the next draft step. Norm conventions checked against the official
checkpoint (range-fetched 1-D tensors, `scripts/official_peek.py` vs `scripts/st_peek.py`): every hc_norm, q/k_norm and
indexer q/k_layernorm in the bare model = official + 1 (the converter's shift, trunk and head alike); the GDN gated norm is
unshifted (one-centred, correct); the head's pre_fc norms ship raw and MTPLX's attach_mtp adds the +1 at load. No defect.
Precision: the bare sidecar head is 4-bit (experts g32, attention g64), 8-bit router and indexer, bf16 fc/hc weights.
**Suspect found in the code: MTP history policy.** `generate_mtpk(...)`'s Python default is `mtp_history_policy="cycle"`:
the head's KV cache never sees the prompt (and the head attends only to rows past it). MTPLX's server and CLI use
"committed" (the head's cache primed with the prompt and extended with every committed token), and MTPLX's knob registry
records why (hy3 2026-07-24, mtpk d1, 1k-ctx real code: "committed history accept 0.903 vs 0.262 unprimed; draft is
context-starved without prompt priming"). SGLang's draft-extend over each iteration's accepted tokens and oMLX's
warm-prefix "Lightning MTP history" (0.6.4) are the same thing. BigMoE's bench.py / mtp_diag.py / diverge125.py and our
mtp_accept125.py all call generate_mtpk without the argument, so they all ran with "cycle".
**(b)+(c) Measured (13:49-13:59, `scripts/mtp_ab125.py` via `scripts/mtp_ab_run.sh`, data/mtp_ab125.{txt,json}).** The 125B
on BigMoE's streamed runtime at C=128 (17 GB of experts), 3 prompts (code, GSM8K #0, prose) × 128 tokens, one load, the
head swapped in place. Conditional acceptance by draft position, pooled over the prompts; MTP tokens per round =
1 + accepted drafts (context-copy rounds excluded):

| head | history | greedy, depth 3 | MTP tok/round | tok/s | t=1.0, depth 3 | MTP tok/round | tok/s | greedy depth 1 |
|---|---|---|---|---|---|---|---|---|
| 4-bit sidecar (BigMoE today) | cycle | 0.776 / 0.586 / 0.385 | 2.41 | 18.5 | 0.691 / 0.579 / 0.530 | 2.30 | 15.0 | 0.805 |
| 4-bit sidecar | **committed** | 0.790 / 0.714 / 0.771 | **2.79** | **22.2** | 0.793 / 0.677 / 0.769 | **2.74** | **18.5** | 0.810 |
| same sidecar re-attached (control) | committed | identical to the row above | 2.79 | 22.3 | identical | 2.74 | 18.8 | 0.810 |
| sidecar dequantized, dense bf16 (control) | committed | 0.800 / 0.700 / 0.729 | 2.77 | 21.4 | 0.793 / 0.740 / 0.676 | 2.78 | 17.3 | 0.819 |
| official bf16 head (+3.5 GB) | committed | 0.803 / 0.724 / 0.775 | 2.84 | 21.9 | 0.839 / 0.769 / 0.825 | 3.02 | 19.3 | 0.810 |
| official bf16 head | cycle | 0.779 / 0.587 / 0.500 | 2.46 | 18.8 | 0.742 / 0.562 / 0.456 | 2.35 | 14.8 | 0.745 |

Per prompt, greedy depth 3, cycle → committed: code 0.778/0.607/0.471 (2.47) → 0.862/0.880/0.682 (3.14); math
0.913/0.643/0.481 (2.78) → 0.892/0.909/0.933 (3.46); prose 0.672/0.512/0.19 (2.08) → 0.69/0.45/0.611 (2.19). SGLang's
3.3 accept length (incl. bonus) is matched on code and math once the history is on.
**Cause: the MTP history policy (the head's own KV cache never holds the prompt or the committed tokens under "cycle").
Fix: `generate_mtpk(..., mtp_history_policy="committed")`, MTPLX's product default: +20 % decode tok/s greedy, +23 % at
t=1.0, at 17 GB, outputs unchanged in kind (exact verify).** Depth-1 acceptance barely moves (0.805 → 0.810): the history
matters for positions 2-3, where the head drafts from its own output. Precision is second order: the official bf16 head
adds +2 % tokens per round greedy (+10 % at t=1.0, one seed), no tok/s gain, +3.5 GB resident; the dequantized control
equals the 4-bit head, so it is the 4-bit values, not the kernels. The head's mapping was proven first
(`scripts/mtp_bf16_check.py`, data/mtp_bf16/check.txt): every bf16 tensor bit-identical to the official one, and MLX RTN
of the official weights reproduces the sidecar (98.6-99.4 % of packed words, same error: 4-bit g32 experts 0.083 rel.,
4-bit g64 attention 0.10-0.13, 8-bit router and indexer 0.011-0.014).
Greedy outputs are identical across heads for a given history policy and depth, and differ across policies and depths
(verify-width numerics on near-ties, BigMoE's known effect), so cycle and committed are compared over slightly different
texts; the direction is the same on all three prompts at t=1.0 and on code and math greedy.
Utilization (C=128, greedy depth 3, pooled): cycle 18.5 tok/s, loader reads 26.3 ms/token = 49 % of wall, 4.8 GB/s from
SSD (33 % of 14.6); committed 22.2 tok/s, 21.6 ms = 48 %, 4.7 GB/s (32 %), 13.7 stalled layers/token (16.1). GPU busy and
memory GB/s not measured in this run. The biggest gap is still the SSD: half the wall time waiting at a third of its rate.
Download (orchestrator OK 13:25): data/mtp_bf16/mtp_official_bf16.safetensors, 5.21 GB, the 31 `mtp.*` bf16 tensors
range-fetched from 28 of the 131 shards of Qwen/Qwen3.8-Flash-Next (360 GB repo) in 587 s; disk free after: 257 GB.
Own bug on the way: mtp_bf16_check.py set MLX's default device to the CPU at import, so the first A/B run (which imports
it) deadlocked in BigMoE's gate with the GPU idle; fixed (device set only when run standalone), rerun clean.

## Round 22 (Oct 19, ~17:10-18:10): do experts loaded only for rejected draft rows pollute the expert cache? **No.**
Prior art (research-first, LIGHT pacing: 3 searches): speculative decoding on offloaded MoE is crowded (SpecMoEOff,
MoE-SpeQ, SP-MoE, MoE-SpAc, SpecMoE, S2-MoE, MoE-Spec, EVICT, EcoSpec, DraftExpert); SP-MoE notes that loads for
rejected tokens compete for I/O, and one summary (2608.21240) that aggressive lookahead evicts useful experts, but no
paper changes the cache policy for wrong-path experts. CPU principle: wrong-path memory references are often useful
prefetches (Mutlu et al. 2005), so demotion could go either way.
Trace (`scripts/verify_trace125.py`, data/verify_trace125.{npz,json}; day_runner7 18:04): the 125B, MTPLX committed history,
depth 3, greedy, 5 prompts (code, 2 GSM8K, prose, a TCP explainer), 1222 tokens, 401 verify passes of 4 rows, routing
per layer and row; rows labelled from the tokens committed per round (labels match MTPLX's accepted_by_depth within 1
per prompt). Wrong-path rows: 358 of 1592 (22.5 %). Trace-driven LRU per layer (`scripts/wrongpath_sim.py`,
data/wrongpath_sim.txt; exact, routing never depends on the cache), misses / stalled layers / est. stall ms per token:

| experts | today (all MRU) | demote wrong-path-only | recency only from valid rows | bound: no wrong-path rows |
|---|---|---|---|---|
| 13 GB (C=96) | 113.1 / 14.7 / 31.9 | 145.2 / 15.5 / 39.6 | 124.5 / 14.9 / 34.6 | 97.9 / 14.0 / 28.1 |
| 17 GB (C=128) | 83.2 / 14.0 / 24.7 | 116.6 / 15.4 / 33.0 | 92.1 / 14.4 / 26.9 | 72.5 / 13.2 / 21.9 |
| 21 GB (C=160) | 63.2 / 13.1 / 19.8 | 97.5 / 15.1 / 28.5 | 70.7 / 13.7 / 21.8 | 55.8 / 12.3 / 17.7 |
| 26 GB (C=192) | 49.0 / 12.2 / 16.1 | 83.1 / 14.9 / 25.0 | 55.3 / 12.9 / 17.9 | 43.7 / 11.3 / 14.6 |

**Negative: the experts that rejected drafts bring in are useful prefetches for the next rounds** (demoting them costs
+40 % misses at 17 GB, keeping only valid-row recency +11 %); today's policy is the best of these. The bound (rejected
rows never verified) is -13 % misses / -11 % stall time at 17 GB: that is the whole cost of verifying rejected rows, the
most that draft truncation (EVICT / DraftExpert style) could recover, before paying for the tokens it gives up.

### Round 22b: MTP depth and expert-cache size with committed history (day_runner8, 18:06-18:19; data/day8, mtp_sweep125_C*.json)
125B, greedy, 192 tokens × 3 prompts (code, GSM8K #0, prose), committed MTP history; per-position conditional acceptance
pooled. Depth sweep at 17 GB (C=128):

| depth | cond. accept | MTP tok/round | tok/s | loader-read share | SSD GB/s | stalled layers/tok |
|---|---|---|---|---|---|---|
| 0 (eager) | - | 1.00 | 17.3 | 43 % | 3.19 | 29.5 |
| 1 | 0.83 | 1.83 | 20.8 | 47 % | 4.06 | 19.1 |
| 2 | 0.88 0.79 | 2.58 | **23.7** | 49 % | 4.61 | 15.0 |
| 3 | 0.79 0.73 0.80 | 2.83 | 22.6 | 50 % | 4.85 | 13.9 |
| 4 | 0.79 0.78 0.77 0.76 | 3.26 | 10.5 | 24 % | 2.33 | 12.6 |
| 5 | ... | 3.49 | 10.2 | 24 % | 2.45 | 11.8 |
| 6 | ... | 3.87 | 10.1 | 24 % | 2.44 | 10.8 |

Depth >= 4 collapses because a verify pass then carries > 40 routing ids: BigMoE's sync-free gate row holds
BIGMOE_MAXK = 40 by default (up to 92 with the env), so those passes take the synchronous path. Depth 2 vs 3 is
prompt-dependent (code 22.6 vs 21.7, math 24.0 vs 25.7, prose 24.4 vs 20.6): an adaptive depth wins over a fixed one.
Speed vs expert cache (eager / MTP depth 3 committed, tok/s; loader-read share at depth 3):
13 GB (C=96) 15.1 / 19.0 (56 %); 17 GB 17.3 / 22.6 (50 %); 21 GB 19.2 / 25.8 (44 %); 26 GB 20.7 / 28.3 (39 %);
34 GB (C=256) 23.1 / 32.0 (31 %). At 17 GB of experts (24 GB total) the best measured setting (depth 2, 23.7 tok/s)
reaches 74 % of 34 GB at depth 3; the loader-read share (half of the wall time at 4.6 GB/s, a third of the SSD rate)
is still the gap.

### Round 22c (day_runner9, 18:27-18:36; data/day9, mtp_sweep125_C128_maxk70.json, mtp_longctx125_C128.json)
**Depth >= 4 with BIGMOE_MAXK=70** (17 GB, same 3 prompts × 192 tokens): depth 3 22.5 tok/s (2.83 MTP tok/round), depth 4
22.7 (3.26), depth 5 21.3 (3.49), depth 6 20.4 (3.65). The collapse at the default MAXK=40 was the gate path; with room
in the row, deeper drafts still don't pay at 17 GB (each extra row's misses cost what its accepted tokens save); depth
2 (23.7) remains the best fixed depth here.
**~4K-token context, depth 3, 17 GB, cycle → committed history:** code module + task (4068 prompt tokens): conditional
acceptance 0.738/0.452/0.500 → 0.836/0.804/0.878, MTP tokens/round 2.24 → 3.10, decode 14.7 → 19.4 tok/s (+32 %);
research note + summary (4048): 0.549/0.420/0.571 → 0.770/0.809/0.763, 1.91 → 2.87, 12.6 → 17.7 tok/s (+41 %).
At long context the history also lifts position 1 (0.55-0.74 → 0.77-0.84), unlike at short context: the fix is worth
more the longer the prompt.

## Round 23 (Oct 19, ~18:40-18:55): MTP-head prediction of the next verify pass's experts (cross-pass prefetch). **No.**
Research-first: SP-MoE (arXiv 2510.10302) feeds each layer of a separate, layer-aligned draft model's attention output
to the target's gate of the same layer (top-1 ~88 %, cosine 0.57-0.95) and prefetches during drafting with a cutoff layer;
MoE-SpeQ and DraftExpert predict the verifier's experts from a draft model too. Our drafter is the 125B's 1-layer MTP
head, so the only candidate state is its output stream (draft step j sits at verify row j-1). Measured
(`scripts/mtp_route_pred125.py` via day_runner10, `scripts/route_pred_eval.py`, data/route_pred_eval.txt; 343 verify
passes, 1013 draft steps, committed history, depth 3): every trunk layer's own MoE read (the eager GatedResidual read,
checked against MTPLX's code) and router applied to the MTP state.
Recall@10 of the actual top-10, MTP state vs the previous position's own top-10 (baseline), layers 0-11 / 12-23 / 24-35
/ 36-47: 0.09 vs 0.28 / 0.15 vs 0.36 / 0.14 vs 0.40 / 0.24 vs 0.41 (rows 0-2 alike). At 17 GB (C=128) the MTP top-10
would cover 12 % of misses (top-20: 21 %) at 578 (1199) wasted reads per pass against 251 misses. **Negative: the head's
state is not a usable proxy for the trunk's per-layer streams, even the last layers'**; a layer-aligned drafter
(SP-MoE) is what makes that method work, and the 125B has none.

## Round 24 (Oct 19, ~18:50): Metal fast resource loading (MTLIOCommandQueue) for expert reads. **No.**
Prior art: none for LLM weights (rounds 16/20 searches). Earlier (round 16) it was scored low-value against the slot
lookup; round 20 made the read phase itself the target (bursts at 6-7.6 GB/s vs the SSD's 12-14.6), so measured now:
`tools/mtlio_bench.swift` (data/mtlio_bench.txt): cold expert-sized reads (2,768,896 B, fresh 16 KB-aligned offsets of
the 125B's 17 shards, never re-read), burst of n experts, commit -> all data in memory, median / p90 of 40 bursts, ms:

| n | MTLIO 1 cmd/expert | MTLIO 4 chunks | MTLIO 16 chunks | pread 1/expert (n threads) | pread 4 chunks |
|---|---|---|---|---|---|
| 1 | 0.428 / 0.456 (6.5 GB/s) | 0.423 | 1.142 | **0.336** / 0.363 (8.2 GB/s) | 0.347 |
| 2 | 0.632 (8.8 GB/s) | 0.801 | 2.106 | **0.561** (9.9 GB/s) | 0.584 |
| 4 | 1.028 (10.8 GB/s) | 1.406 | 4.131 | **0.975** (11.4 GB/s) | 1.000 |
| 8 | 1.963 (11.3 GB/s) | 2.681 | 8.379 | 1.790 (12.4 GB/s) | **1.774** |

MTLIO is slower than plain preads at every burst size and its per-command cost (~65 µs) makes chunking much worse;
chunking doesn't help preads either (the SSD serves one expert at ~8 GB/s however it is split). Useful floor: one cold
expert = 0.34 ms, 2 = 0.56, 4 = 0.98 (BigMoE's loader measured 0.68 ms at 1.5 misses and 0.85 ms at 2.3 misses per
stalled layer in round 20, ~40-50 % above this floor: the record path should close most of that). Beyond the floor,
the only read-side levers left are fewer bytes per miss and overlap (prefetch).

## Round 25 (Oct 19, ~18:50): the MSB-3 expert cache under MTP verify passes (`scripts/msb3_mtp_sim.py`, data/msb3_mtp_sim.txt)
Round 20 scored the MSB-3 cache + LSB-1 sidecar at only ~-14 % stall (~+7 %) under eager decode, where stall events
dominated (33 stalled layers/token). With committed MTP (depth 3) a pass commits ~2.8 tokens: stalled layers fall to ~14
per token while misses stay ~80, so bytes per miss carry ~3/4 of the stall time (the round-20 model, 0.40 ms per stalled
layer + 0.083 ms/MB, matches the measured MTP loader time at 17 GB: 23.0 vs 21.6 ms/token). Trace-driven on the round-22
verify trace (top-2 ranks stood in by a fixed hash: the trace has no gate scores), per token:

| budget | whole 4-bit LRU: misses / stalled / MB / est. ms | MSB-3 + 10 % LSB-1 sidecar | unified LRU, LSB-first |
|---|---|---|---|
| 13 GB | 109.9 / 14.7 / 304 / 31.1 | 117.8 / 14.6 / 221 / 24.2 (**-22 %**) | 166.1 / 15.9 / 244 / 26.6 (-15 %) |
| 17 GB | 82.4 / 14.0 / 228 / 24.5 | 87.2 / 13.9 / 162 / 19.0 (**-23 %**) | 138.3 / 15.8 / 190 / 22.1 (-10 %) |
| 21 GB | 63.2 / 13.2 / 175 / 19.8 | 66.5 / 13.0 / 122 / 15.3 (**-23 %**) | 116.2 / 15.7 / 151 / 18.8 (-5 %) |
| 26 GB | 46.5 / 12.0 / 129 / 15.5 | 48.0 / 11.9 / 87 / 12.0 (**-23 %**) | 89.4 / 15.3 / 112 / 15.4 (-1 %) |

Under MTP the 3-bit tier (plan 4433333333, GSM8K 427/450 vs 431/450, n.s.) is worth about -23 % stall time at every
budget, ~+12 % tok/s at 17 GB (45 -> ~40 ms/token), and more once BigMoE's record path shrinks the fixed per-layer cost
(the bytes term is what MSB-3 cuts). A speed prototype needs: plane files (MSB-3 | LSB-1 records, ~67 GB, make_planes.py
pattern), MLX's native 3-bit gather_qmm on MSB-3 codes (scale 2s, bias b + s·mean(lsb) per group: an ordinary affine
3-bit quantization), 4-bit for the top-2 ranks (merge MSB-3 + LSB-1 into a 4-bit slot, or a 3-bit + 2-bit-coded
correction), and BigMoE's slot pools split by precision.
**Kernel check before building (`scripts/qmm3_bench.py`, data/qmm3_bench.txt; MLX gather_qmm, the 125B's expert shapes,
128-slot pools, top-10 per row, ms per layer's routed GLU):** rows 1: 4-bit 0.260, 3-bit 0.305, plan split (2 @ 4-bit +
8 @ 3-bit, two calls) 0.263; rows 2: 0.387 / 0.386 / 0.403 (+4 %); rows 4: 0.593 / 0.763 / **0.859 (+45 %)**. MLX's 3-bit
kernel is no faster than 4-bit despite 25 % fewer bytes, and the split costs +0.27 ms/layer at 4 rows = +12.7 ms per
verify pass, about what the MSB-3 cache saves in stall (~15 ms per pass at 17 GB). **Not worth building on stock MLX
kernels**: it needs a fused mixed-precision (3/4-bit) routed-expert kernel first (PrefillKernels' area). Side note for
them: the 4-bit routed GLU itself runs at 106 / 143 / 187 GB/s of expert bytes for 1 / 2 / 4 rows (0.59 ms/layer at 4
rows = 28 ms per depth-3 verify pass), far from 280-307 GB/s.

## Round 26 (Oct 19, 20:00-): routed-expert GLU kernel at M = 1..4 (orchestrator 19:58: "use everything the hardware gives")
Research-first: MLX #3661 (M1 Pro: quantized_matmul cost ~linear in M at small M, weights re-swept per row; proposes a
multi-row GEMV / split-K; closed without a fix), PR #3863 (BM=16 qmm tile, closed), #3925 (gather_qmm_rhs_nax BM);
llama.cpp b11404 / PR #29869 (mul_mv_mma.metal: few-row kernels sharing one weight read across 2-16 activation rows via
simdgroup-matrix tiles; 3.4-3.6x speculative decode on M3 Ultra; MoE coverage not stated); Marlin (striped partitioning
and several warps per output tile to fill the GPU at batch <= 64, async weight loads that don't pollute L2, weights
pre-shuffled offline for direct dequant); ZMLX (fused MoE gating/combine for MLX, dispatch overhead).
**Baseline, measured right** (`scripts/glu_kernels_bench.py`, data/glu_kernels_bench.txt): 48 GLUs per command buffer over
8 distinct 128-slot pools (2.8 GB), 125B shapes (gate_up fused [1280, 2560], down [2560, 640], 4-bit g64), random top-10
routing, ms per layer (GB/s of unique expert bytes):

| M | MLX stock, chained | MLX stock, independent | MTPLX fused (2 dispatches), chained | pure uint4 read of the same bytes |
|---|---|---|---|---|
| 1 | 0.135 (204) | 0.117 (236) | 0.184 (150) | 0.089 (309) |
| 2 | 0.233 (230) | 0.218 (242) | 0.325 (164) | 0.167 (317) |
| 3 | 0.331 (234) | 0.317 (240) | 0.462 (165) | 0.243 (314) |
| 4 | 0.425 (233) | 0.412 (239) | 0.603 (162) | 0.303 (326) |

My round-25 figures (106 / 143 / 187 GB/s) were a microbench artifact (an mx.eval per GLU); the real gap is 25-35 % to a
read ceiling of 310-326 GB/s (MLX's own 1 GiB reduce: 256 GB/s). MTPLX's fused kernels are slower than stock on this
Mac (BigMoE doesn't use them). Worth ~2 ms/token at M=1 and ~5 ms per 4-row verify pass (+3-5 % decode at 17 GB).

### Round 26b: whole verify-pass profile (`scripts/pass_profile125.py`, data/pass_profile125.{txt,json}; orchestrator 20:12)
125B at C=256 (34 GB), committed MTP depth 3, greedy, code + math prompts, 82 consecutive pass pairs; BigMoE's spin gate
(no command-buffer boundary at the gates) with BIGMOE_TRACE=1: per (layer, pass) the loader logs when it saw the GPU's
routing row and when it released the slots, on CLOCK_UPTIME_RAW. Pass = 96.3 ms gate to gate:

| block | ms per pass | share | bytes | GB/s | % of 318 |
|---|---|---|---|---|---|
| 35 x [MoE(L) + GDN attention(L+1)] | 34.6 (0.99 each) | 36 % | 144 MB each | 146 | 46 % |
| 12 x [MoE(L) + QSA attention(L+1)] | 12.0 (1.00 each) | 12 % | 140 MB each | 139 | 44 % |
| loader stalls, 48 gates | 31.9 (0.67 each) | 33 % | | | |
| inter-pass: layer-47 MoE, mixer, lm_head, sampling/acceptance syncs, history, 3 drafts, layer-0 attention | 17.7 | 18 % | | | |

Per layer resident: attention side 47.7 MB (GDN) / 43.0 MB (QSA), MoE side 17.4 MB, plus ~26 unique routed experts per
layer per 4-row pass (72 MB) and the GDN state (3.15 MB fp32, read + written). **26 MB of every layer's ~65 MB resident
weights are the two hyper-connection mixers in bf16** (input_mix down/up [320 x 10240] x 2). The per-layer compute runs at
~45 % of the ceiling, matching Scout's "memory at ~44 %"; at ~85 % it would drop from 46.6 to ~25 ms/pass (+~30 % at 34 GB).
Next: per-module timing of the real layers' weight-bearing ops at M = 1 and 4 to find which ones are slow.

### Round 26c: per-module timing of the real layers (`scripts/module_bench125.py`, data/module_bench125.txt)
Every weight-bearing op of the 48 layers, real weights, all instances of one op in one command buffer, dependent calls,
ms per call at M=1 / M=4 (GB/s at M=4, % of 318): HC read (attn and mlp, 13.2 MB bf16, x48 each) 0.087 / 0.092 (143,
45 %); GDN in_proj_qkv (14.75 MB q4, x36) 0.072 / 0.088 (168, 53 %); GDN in_proj_z, out_proj (8.85 MB q4, x36) ~0.051 /
~0.063 (~141, 44 %); QSA q_proj (17.7 MB q4, x12) 0.137 / 0.155 (114, 36 %); QSA o_proj (8.85 MB) 0.064 / 0.076 (117);
QSA k/v (0.74 MB) 0.033 / 0.036 (20 GB/s); indexer (1.74 MB q8) 0.035 / 0.037 (47); router (1.39 MB q8, x48) 0.022 /
0.024 (58); shared expert MLP (2.76 MB, x48) 0.034 / 0.041 (67); shared-expert gate 0.018 / 0.022; GDN in_proj_a/b
(0.07 MB) 0.019 / 0.022 (3 GB/s); PLE key/value (52 / 13 MB bf16, x1) 0.47 / 0.52 and 0.21 / 0.23 ms; lm_head (358 MB q4)
1.40 / 1.53 ms (256 / 234 GB/s). **Sum: 3.20 GB of non-expert weights per pass in 30.6 ms at M=4 (105 GB/s)**; with the
routed GLU (~26 unique experts/layer, ~14.5 ms at ~234 GB/s) this is the profile's 46.6 ms of per-layer compute.
Shape of the loss: ~20 µs per dependent op whatever its size, plus bytes at ~200-250 GB/s; M=4 costs only 6-21 % more
than M=1 (MLX doesn't re-sweep weights per row here). PKD1 (PrefillKernels: the small projections fused, exact) gained
~1 % (t_verify 3.60 -> 3.56 s), so dispatch count alone is not the lever; kernel efficiency on 9-17 MB matrices
(50-65 % of peak) and the bf16 hyper-connection bytes (1.25 GB of the 3.2 GB per pass) are.

## Round 27 (Oct 19, ~20:15-20:40): bit-identical faster qmv, kernel count, BigMoE's reference env
- **qmv_hb** (`patches/qmv_hb/`, data/qmv_hb_v1.txt): MLX's qmv_wide_impl (the kernel at 2 <= M <= 5) with K compile-time
  and every group of a lane (weights via 16-byte loads, scales, biases) loaded before any arithmetic; MLX's dequantize
  expressions, sub-chunk order and shuffle ladder kept. **Bit-identical** to mx.quantized_matmul on the real GDN
  in_proj_qkv / z / out_proj, QSA q / o and lm_head at M = 2..4, but **16-47 % slower** (e.g. in_proj_qkv M=4 0.125 vs
  0.087 ms, out_proj 0.106 vs 0.062): the registers it needs cost occupancy. Load scheduling is not what limits MLX here.
- **Dependency latency**: a chain of 1000 dependent tiny elementwise ops costs 6.3 µs per op (mul+add) via MLX on this
  GPU, 13.7 µs at 200 ops (commit overhead); i.e. ~3-7 µs per dependent kernel whatever its size.
- **Kernel count** (`scripts/pass_graph125.py`, data/pass_graph125.txt): a 4-row forward of the 125B is 9,340 graph nodes,
  ~195 per layer, of which ~100 launch kernels: Multiply 19.4, AsType 10.8, QuantizedMatmul 10.0, Sigmoid 6.8, Matmul
  (bf16 HC) 6.1, Sum 5.8, Add 5.8, Divide 5.0, Arange 4.0 (!), RMSNorm 3.6, Concatenate 3.0, GatherQMM 3.0, compiled
  sigmoid*mul ~5.6, Rsqrt 1.5, SliceUpdate 1.2, Softmax, ArgPartition, PostIds, SpinWait, take, SDPA, conv, GDN kernel,
  rope ... At ~3-7 µs each that is ~0.3-0.7 ms of the ~1.0 ms per layer: **kernel count, not bandwidth, is the main
  per-layer loss**; bytes at ~250 GB/s would take ~0.56 ms.
- **BigMoE's reference env** (DUPROW, COMPILE_HC, HC_VREAD, ASYNC_EVERY, PF_JIT, PREFETCH=12, MAXK=80, PF_GROUP=64,
  FRSPEC): with committed MTP depth 3 at C=256 it gives 32.2 tok/s vs 32.1 plain (data/committed_ref_check.txt;
  bench.py with MTPLX's default cycle history: 31.2 vs 30.0); misses and stalled layers per token are identical, so
  the prefetcher changes nothing measurable in these runs. My traced profile under that env (140.9 ms/pass) was
  perturbed by my own hooks and is discarded; the plain-env profile (96.3 ms/pass) matches uninstrumented speed.
- MTPLX's own compile lane (MTPLX_QWEN4EXP_COMPILE = MTPLX_COMPILED_GDN) compiles whole GDN-layer runs including the
  MoE, which BigMoE's per-layer Python gate forbids; cutting the kernel count needs the non-MoE part of each layer
  (HC reads/writes, attention, norms, gates, rope) compiled or fused with the cache state passed explicitly.

## Round 28 (Oct 19, ~20:45-21:10; orchestrator 20:45: (1) bit-identical small-M matvec, (2) 8-bit HC mixers + PLE K/V)
**(1) Bit-identical matvec: no faster kernel is possible this way** (`patches/qmv_hb/`, data/qmv_hb_v2.txt). Besides the
register-prefetch qmv_hb (16-47 % slower, round 27): ck2 / ck4 = MLX's qmv_wide_impl verbatim with compile-time K and
2 / 4 simdgroups per threadgroup: bit-identical on all real projections at M = 2..4, but 5-10 % slower than MLX's
native call (e.g. GDN in_proj_qkv M=4 0.094 vs 0.088 ms, o_proj 0.064 vs 0.059; lm_head -0-1 %). A custom-kernel call
costs more per dispatch than MLX's native one, and MLX's kernel body already streams the big lm_head at 258-285 GB/s;
on the 9-17 MB matrices the loss is the per-call fixed cost, which a kernel body can't remove.
**(2) 8-bit hyper-connection mixers + PLE key/value** (`patches/hc8/hc8.py`; MTPLX's recipe keeps them bf16): 196
matrices, 1,337 MB bf16 -> 710 MB (-627 MB resident). Per HC read (`hc8_op_bench.py`, real weights, 48 dependent reads):
M=1 0.0822 -> 0.0703 ms (+17 %), M=4 0.0887 -> 0.0816 ms (+9 %); the 8-bit matvec on these shapes runs at 85-99 GB/s,
so halving the bytes saves little: ~0.7 ms per 4-row pass (~0.7 %). End to end (`hc8_smoke.py`, C=256, committed MTP
depth 3, 3 prompts x 192 tokens): 31.4 / 37.1 / 26.9 -> 31.2 / 34.7 / 29.3 tok/s (mean -0.2 %, within the noise of
diverging texts); greedy eager output diverges from bf16 at token 3, text coherent. Value: memory only (0.63 GB ~ +3.7 %
more expert slots at 17 GB), not speed.

## Round 29 (Oct 19, ~20:30-21:15): exact kernel cuts, measured (orchestrator 20:50; BigMoE ledger R2 per-flag A/B)
Per-layer map (`scripts/layer_ops125.py`, data/layer_ops125.txt; layer outputs' DOT diffed): GDN layer ~86 kernel nodes
(Multiply 18, QMM 10, AsType 9, Sigmoid 7, bf16 Matmul 6, Sum 6, Divide 5, Add 5, RMSNorm 3, Arange 3, GatherQMM 3 ...),
QSA ~126 (Multiply 23, AsType 16, QMM 10, Concatenate 9, Add 8, Arange 7, SliceUpdate 5, SDPA 4, rope cos/sin/neg 7 ...).
MLX makes no node for a same-dtype astype: every AsType is a real bf16 <-> f32 conversion (no free cast cuts).
**Per-flag A/B** (`scripts/flag_ab125.py` via day_runner11, data/day11/; one process per flag; kernels of a 4-row forward
under attention_phase("decode_verify"), its logits sha, greedy shas and tok/s with committed MTP depth 3 at C=256, 3
prompts x 128 tokens; base 97.1 kernels/layer, logits 517e36370c90, 31.00 tok/s, noise ~0.3 %):

| MTPLX switch | kernels/layer | logits / greedy shas | tok/s |
|---|---|---|---|
| FUSED_QSA_INDEXER | 92.1 (-240/pass) | **identical / identical (exact)** | 30.97 (0 %) |
| QWEN4_OPDIET bank, k20, resid | 97.1 | identical / identical | 30.94-31.04 (no change on this path) |
| QWEN4_OPDIET rope | 97.9 | identical / identical | 30.25 (-2 %) |
| FUSED_CONVNORM_VERIFY | 84.3 (-12.8) | changed | 31.84 (+2.7 %) |
| FUSED_GDN_CONVNORM / GDN_OUT / GDN_STEP | 97.1 | logits same, generation shas changed | 29.4-30.4 |
| FUSED_GDN_INPROJ | 94.8 | changed | 28.78 (-7 %) |
| FUSED_QSA_QKV | 96.6 | changed | 28.44 (-8 %) |
| FUSED_HC_V3 | 97.1 | generation shas changed | 30.95 |
| FUSED_HC | 74.9 (-22) | changed | **11.89 (2.6x slower)** |
| QWEN4_VERIFY_GLUE | - | crashed (installs on the compiled-verify route only) | - |

**Calibration: cutting kernels exactly buys ~nothing** (-240 kernels/pass, 0 %), and the biggest non-exact cut that
doesn't swap in a slow kernel (convnorm_verify, -13 % kernels) buys +2.7 %. Most of the ~100 small kernels per layer
overlap with the big ones (MLX's concurrent encoder); the round-27 estimate (~4 µs x every kernel) overstated the
lever. Expected value of a full per-layer non-MoE compile (v2b): a few %, not +13 %.

## Round 30 (Oct 19, ~21:00-21:35): MLX command-buffer limits (kernel brief #1, mlx#4521). **Breaks BigMoE's spin gate.**
day_runner12 (`flag_ab125.py`, data/day12): MLX_MAX_MB_PER_BUFFER=4096 (+ MLX_MAX_OPS_PER_BUFFER=1000) at C=256 and C=128:
**920-1160 spin-gate timeouts per run ("results of that pass are invalid")**; with MLX_BFS_MAX_WIDTH=80 as well the run
finished at 3.6 tok/s with changed logits and tokens. Baselines: C=256 28.5 tok/s (a repeat 23.4: the code prompt fell
to 10.4 tok/s under contention on the Mac), C=128 21.6. Cause: the spin gate needs the GPU's routing-row write
(PostIds) to reach the CPU loader, and GPU writes become CPU-visible at command-buffer boundaries; today those
boundaries come from MLX's 50 MB rule closing a buffer at each gather_qmm over the big slot pools. (Correction to
round 26b: the spin gate has no gate-made boundaries, but it depends on MLX's.) A controlled version would raise the
limits and have the gate commit explicitly right after each PostIds (one boundary per layer instead of 2-3): est. 1-3 ms
per pass, BigMoE's gate code.

## Round 31 (Oct 19, ~21:40): chunk-streaming small-M matvec (v2b candidate). **Doesn't clear the ~230 GB/s bar; not built.**
`patches/qmv_cs/` (data/qmv_cs_v1.txt): lanes stream each simdgroup's rows as contiguous 16-byte chunks, fp32 partials,
one simd_sum per (row, m); rel. error vs MLX 0.9-2 % (v2b). Real projections, dependent calls, ms (GB/s): M=1 roughly
equal to MLX (in_proj_qkv 0.0818 vs 0.0834, z 0.0568 vs 0.0533, out 0.0534 vs 0.0520, q 0.0894 vs 0.0846, o 0.0560 vs
0.0519, lm_head equal), M=2..4 4-45 % slower as written (dequant redone per input row). **What the data says about the
mid-size matvecs: time ~ bytes / ~300 GB/s + ~20 µs per dependent kernel** (8.85 MB: 30 + 20 = 50 µs, measured
52 µs = 170 GB/s; 17.7 MB: <= ~224 GB/s; lm_head 358 MB: 270-295 GB/s). The kernel body is not the limit; the fixed
cost of each dependent kernel on the critical path is, and no matvec rewrite removes it. With round 29 (exact cuts of
off-critical-path kernels: 0 %) this closes the small-M kernel line for the 125B on stock MLX.

### Round 22d (Oct 20, 08:37-08:42; day_runner13, data/day13): night20's skipped C=192 (26 GB) jobs
Depth-2 curve point: 28.7 tok/s (3 prompts x 192 tok) -> depth-2 curve 13/17/21/26/34 GB: 19.8 / 23.7 / 26.0 / 28.7 / 32.9
tok/s (steady rise, no knee). 6 prompts x 384 tok at C=192 (BIGMOE_MAXK=60): depth 2 25.5, depth 3 25.7, depth 4 26.1 tok/s
(MTP tokens/round 2.58 / 3.01 / 3.46; loader-read share 38-41 %): depth is nearly flat at 26 GB (+2 % for 4 vs 2).

## Round 32 (2026-10-20 ~22:00, account-1): research pass + 3c load tuning
**3c load tuning, negative:** issuing all of a lane's loads before the arithmetic (qmv_hb style; ushort loads, arrays
[rows][blocks]) stays bitwise but is slower: A+B prefetch 0.146/0.245/0.439/0.907 ms (M=1/2/4/8, 3c+comb), B-only
0.118/0.207/0.365/0.712, vs the committed kernel re-measured in the same state **0.104/0.180/0.314/0.560** (stock4
0.134/0.232/0.421/0.776: 3c+comb = -22/-22/-25/-28 %). Likely register pressure / spills; reverted.
(data/planes_engine/bench_tierglu_pf*.txt, bench_tierglu_v1_recheck.txt)

**First-principles SSD floor (scripts/opt_bound125.py, data/opt_bound125.txt; 401 committed verify passes, 4 rows):**
distinct experts per layer per pass vs rows 1/2/3/4: 10 / 16.1 / 21.4 / 26.2. Misses per pass (4-bit MB) LRU vs Belady
OPT (perfect-knowledge eviction = the floor for any replacement or prefetch-by-prediction policy):

| cache | 4.2 GB | 8.5 GB | 12.7 GB | 17 GB | 25.5 GB | 34 GB |
|---|---|---|---|---|---|---|
| LRU MB/pass | 1930 | 1336 | 971 | 722 | 439 | 297 |
| OPT MB/pass | 1691 | 855 | 560 | 403 | 253 | 187 |
| OPT/LRU | 0.88 | 0.64 | 0.58 | 0.56 | 0.58 | 0.63 |

Reading: perfect prediction buys at most ~1.6-1.8x fewer SSD bytes at 8-17 GB; nothing in replacement/prefetch reaches
the 6-12 GB goal alone. Trace averages ~3.1 tokens per pass (1225 tokens / 396 passes), so the pure-SSD ceiling at
12.7 GB is ~80 tok/s with OPT (560 MB / 14.6 GB/s = 38 ms/pass), ~47 with LRU, x1.4 with the 3-bit plan; a 3 GB/s
Mac mini at 50 tok/s needs <= ~180 MB/pass, i.e. near-full residency. => 6-12 GB at 100+ tok/s needs fewer bytes per
expert (~2-bit-class formats at Q4 quality) AND OPT-class prediction AND wider exact verify (bytes grow 2.6x from 1 to 4
rows while tokens grow ~3x). Ranked next tests: (a) practical predictor vs OPT gap with mtp_route_pred125 predictions;
(b) union growth past 4 rows needs a deeper trace (tree drafts); (c) cache-state-aware draft choice (EcoSpec-style but
priced in misses) replayed on this trace. Prior art: PRIOR_ART.md round 32.

**Round 32b (23:35, 1 core): two first-principles answers from the committed-token routing.**
(1) Wide verify is a weak byte lever (scripts/union_growth125.py, data/union_growth125.txt; rows 0..n-1 of each pass =
the committed sequence, 1222 tokens): distinct experts per layer over k consecutive tokens 10 / 16.4 / 26.9 / 43.8 / 68.6
/ 102.3 for k = 1 / 2 / 4 / 8 / 16 / 32, i.e. union ~ k^0.67; cold 4-bit MB per token 1327 / 1088 / 892 / 727 / 569 / 425.
Even a perfect 32-token verify only cuts cold expert bytes per token 3.1x. For ~1000 tok/s the binding term is the
dense part instead (~2.5 GB read per pass at ~300 GB/s = ~8 ms, so >= 8 accepted tokens/pass are needed even with free
experts): only copy-heavy text (lookup drafting) can reach it on this model and machine.
(2) Most of Belady's gain needs long-horizon knowledge (scripts/lookahead_bound125.py, data/lookahead_bound125.txt):
perfect knowledge of the next W passes, MB/pass at 12.7 GB: W=0 (LRU) 971, W=1 901, W=2 844, W=4 759, W=8 658, W=16 586,
OPT 560; at 8.5 GB W=4 981 / W=8 884 vs OPT 855. A 1-2 pass route predictor (MTP head, pre-gating; all MoE prefetch
prior art) can capture at most 7-13 % of bytes; the remaining ~30 % needs knowing reuse 8-16 passes (25-50 tokens)
ahead => a replacement-policy problem, not a prediction problem. Next: learned-reuse replacement (Mockingjay/Hawkeye
from CPU caches) on the trace (scripts/reuse_policy125.py).

**Round 32c (~23:45): cache policies tried online, out-of-order experts, and the ranked list.**
- Reuse-distance prediction (Mockingjay-style, scripts/reuse_policy125.py): **worse** than LRU (+43 % bytes at 12.7 GB):
  expert reuse is bursty, so a burst's short gaps make the policy keep experts that do not come back.
- Scan-resistant insertion (scripts/insert_policy125.py, _sweep.py): LIP-4/8 (new experts start 4-8 passes old) and
  SRRIP-3 cut SSD bytes 3-6 % at 8.5-17 GB, both trace halves; larger k hurts. Exact, trivial in the ext. Small.
- **Out-of-order experts** (data/ooo_bound125.txt from pass_profile125.json, frac 0.5): loader stalls are 31-34 ms of a
  96 ms pass (per-layer p50 0.57-0.65 ms, p90 1.4). Running the layer's resident (hit) experts during the spin and the
  missed ones after (combine in MLX's fixed LY=8 order, so bitwise) hides ~9.4-10.5 ms/pass (30 % of stalls; bounded
  by hit compute ~0.3 ms/layer with 3c). ~+7-10 % tok/s net of 2-3 extra dispatches per layer. Needs: loader publishes
  hit slots at t_seen, GPU runs 3c on hit pairs, spins for the rest, runs the miss pairs, then the combine kernel.
  Shared expert already runs during the spin (Scout's sharedfirst).

**Ranked list (cheapest decisive test each)** — goals: 6-12 GB at 100-150 tok/s, 16 GB mini 50-75, 4 GB 30-50, 1000 tok/s.
Physics first: the dense part (~2.5 GB/pass at 4-bit, cannot go lower: MSB-3 dense +12.5 % ppl) caps a pass at ~8 ms
(125 passes/s); a 34 GB verify pass takes 96 ms = 46 compute (~130 GB/s, 2.3x below the read ceiling) + 32 stalls +
17 between passes. So: G1 is bound by compute efficiency and stalls before SSD bytes; 1000 tok/s needs >= 8 accepted
tokens per pass (copy-heavy text only); 4 GB / 3 GB/s-SSD goals are out of reach for this model by bytes (needs ~10x
fewer expert bytes/token than LRU 4-bit gives; best stack: 2-bit-class experts 2x x prediction 1.1-1.7x x 3-bit tiers).
1. 3c fused tier GLU into BigMoE's engine (built, bitwise): end-to-end tok/s A/B at 17 / 34 GB. [BigMoE run]
2. Out-of-order experts (+7-10 % est.): prototype = 3c on hit pairs, then miss pairs, then the LY=8 combine; bitwise
   test vs 3c; A/B at 34 GB. Needs a partial slot release in the ext. [MoonShots kernel side + BigMoE ext]
3. Pass compute efficiency (46 ms at ~130 GB/s): per-kernel attribution of the dense part at M=4 (GDN, attention,
   shared expert, lm_head) vs bytes/300 GB/s; target the 2-3 worst. [pass_profile at kernel granularity, GPU ~2 min]
4. LIP-4 insertion in the ext's LRU (-3..-6 % SSD bytes). [one-line ext change + A/B]
5. Sibling union: expert overlap of top-4 candidate tokens at one position (tree width cost) vs chain tokens; if
   siblings share >= 80 % of experts, wide trees are byte-cheap -> more tokens/pass. [GPU ~2 min, daytime]
6. Long-horizon residency (the 30 % of OPT that needs 25-50 tokens of foresight): does the prompt's prefill routing
   histogram predict decode reuse 8-16 passes ahead? [prefill routing dump + CPU sim]

**Round 32d (~23:55): first-principles pass model (scripts/pass_model125.py, data/pass_model125.txt).** T = compute (46 ms
at 34 GB, ~20 of it routed GLU) + between passes (17) + visible stalls (0.30 ms per missed expert, LRU misses from the
trace). Calibration: 21.9 tok/s at 17 GB (measured 23.7), 32.6 at 34 GB (32.9). tok/s at 12.7 / 17 / 34 GB:
today 18.4 / 21.9 / 32.6; +3c+3-bit tiers 21.0 / 24.9 / 36.3; +out-of-order+LIP 23.9 / 28.8 / 43.0; + compute at the read
ceiling (17.7 ms) + thin inter-pass (4 ms) 33.3 / 43.6 / 86.8; + all SSD hidden 62.2 / 83.7 / 143.5; + OPT-class caching
103.7 / 143.5 / 143.5. 16 GB mini (3 GB/s) with everything: 9-14 tok/s. More tokens/pass at 12.7 GB (full stack + OPT):
4 -> 113, 6 -> 129, 8 -> 142 tok/s. Reading: the Breakthrough targets are engine-structure problems (compute efficiency
2.6x, inter-pass 4x, total overlap) before they are byte problems; bytes decide only below ~12 GB and on the mini.

**Round 32e (~00:00): the 2+1+1 record format is NOT worth it for the main track** (scripts/planes4_sim.py,
data/planes4_sim.txt; MoEPrefetch's 800-token traces, windows of 4, LRU per layer over cache units). Demand MB/pass
(unit misses/pass) at expert budgets 10 / 13 / 17 / 24 GB:
whole 4-bit 933 / 746 / 556 / 328; records3 + 4433333333 (today's plan) 709 / 554 / 400 / 224 (389 / 308 / 225 / 129
misses); records4 (P2 + B3 + B4) + 4433332222 657 / 513 / 371 / 210 (572 / 450 / 328 / 189 misses); records4 +
4433333333 794 / 627 / 461 / 267 (worse: P2 + B3 are separate units). So 2+1+1 + 2-bit tails saves only 6-7 % of the
bytes records3 already moves, with 46-49 % more separate reads, and stalls are latency-bound (~0.3 ms per read): likely
no speed gain at 13-24 GB. STATE.md's "+20-25 % at 24 GB" was against all-4-bit (records3 already gets -24..-32 %).
It pays only if most ranks can be 2-bit (4222222222-class): those plans' GSM8K is held by the silent night; the byte
side runs next (planes4_sim.py extra). Decision for Wednesday: do not build the 2+1+1 converter unless a 2-bit-heavy
plan passes quality.
Byte side of 2-bit-heavy plans on records4 (data/planes4_sim_extra.txt), MB/pass (misses) at 10 / 13 / 17 / 24 GB:
4422222222 509 / 387 / 268 / 145 (414 / 319 / 225 / 125): -28..-35 % bytes vs records3 + 4433333333, same read count;
**4222222222 419 / 309 / 206 / 107 (320 / 241 / 164 / 88): -41..-52 % bytes, -18..-32 % reads**; 2222222222 289 / 191 /
110 / 50. => records4 is worth building only together with a 4222222222- or 4422222222-class plan that passes the
quality bar (GSM8K + Eval suite, held tonight); then it is also Breakthrough's natural "MSB-2 reads" format.

**Round 32f (~00:05): the 17-21 ms between verify passes (scripts/interpass125.py, data/interpass125.txt).** From the
loader trace + Python timestamps: a = 3.0-5.8 ms verify tail (L47 MoE, lm_head, sampling/acceptance, history), b = 0.3
ms between the 3 lazy draft calls (no syncs between drafts), **c = 14-15 ms from the last draft call to the GPU reaching
layer 0's gate**: the 3 MTP draft steps' GPU work (MTP layer + full-vocab lm_head each) + the verify graph's Python build
and launch + layer 0 attention. No MTP-layer loader events at frac 0.5 (MTP experts not stalling here). Next (daytime):
DRAFT_SYNC=1 PROFILE_OUT=... pass_profile125.py to split c into draft GPU time vs launch. Levers if drafts dominate:
reduced-vocab draft lm_head (FR-Spec 2025 / VocabTrim: argmax over the frequent ~32K tokens; exact, verify unchanged),
a fused 3-step draft kernel chain; if launch dominates: a compiled/cached verify graph (MTPLX_COMPILED_VERIFY).
**Correction to 32d (~00:08):** the "eff" lever was bytes-only (17.7 ms). Round 29 showed cutting overlapped small kernels
buys nothing, so the floor is bytes + ~12 dependent big kernels/layer x ~20 us = 29.2 ms (kcount125: 175 primitives/
layer, ~83 kernels, mostly overlapped). Revised tok/s (12.7 / 17 / 34 GB): +eff+thin 29.6 / 37.5 / 65.6; +hide 62.2 / 83.7
/ 93.7; +opt 93.7 everywhere above ~12 GB. So ~94 tok/s is the 3.1-tokens/pass ceiling of this model on this machine
with every exact lever; >100 needs more accepted tokens per pass (4 -> ~113, 6 -> 129-157, 8 -> 142-194).

**Round 32g (~00:12): LIP is marginal on a second trace set** (scripts/insert_policy800.py, data/insert_policy800.txt;
MoEPrefetch's 4 x 800-token traces, windows of 4): LIP-4/8 and SRRIP-3 -1.2..-2.7 % vs LRU at 8.5-25.5 GB (-3..-6 % on the
verify trace): not worth BigMoE's time. The Belady gap is larger here: OPT -37 / -44 / -48 / -49 % at 8.5 / 12.7 / 17 /
25.5 GB. Online recency/insertion policies don't reach it; next test: context matching as the long-horizon signal
(scripts/pmatch_policy.py: protect the experts that followed the most similar earlier window, past-only).
**Round 32h (~00:18): context matching captures little of the Belady gap** (data/pmatch_policy.txt): protecting the
experts that followed the most similar earlier window (past-only; own-layer or all-layer match, W = 8-32 windows):
best -4..-6 % bytes vs LRU (12.7 GB: 922 vs 971 MB/pass verify trace, 738 vs 786 traces800; 17 GB -4.0 / -5.6 %), vs OPT
-44..-48 %. With LIP (-1..-6 %) and reuse distance (worse than LRU): **online policies from routing history get <= ~6 % of
a 37-49 % gap.** The long-horizon reuse that Belady exploits is not predictable from history at this granularity; exact
gains past ~6 % must come from bytes (tiers, formats) and overlap, not replacement. Cache-policy line closed.
**Round 32i (~00:20): draft depth is not a lever** (scripts/depth_model125.py, data/depth_model125.txt; trace's pooled
conditional acceptance d1 0.854 / d2 0.796 / d3 0.825, extrapolated 0.80 - 0.02 per depth beyond): tok/s vs depth is flat
around D = 3-5 in today's engine (17 GB best D4 22.2 vs D3 21.9; matches the measured flat 2/3/4 at C=192) and in a
levered one (best D4-5, 94-99 tok/s vs 93.5 at D3). MTP's acceptance profile caps tokens/pass at ~3.5-3.9; past ~95 tok/s
the drafter must get better (lookup + MTP, wider trees: sibling-union probe), not deeper.

**Round 32j (~00:50): where the per-pass compute goes** (scripts/layer_law125.py, data/layer_law125*.txt; pass_profile at
0.50). Interval between gates L -> L+1 (L's MoE + L+1's attention side + router): p50 0.89 ms (GDN next) / 0.93 ms (QSA
next) vs bytes at 300 GB/s 0.47 ms (moe_b + att_b + 26-28 unique experts + GDN state): bytes are 50-53 %, the rest is
~21-23 dependent-kernel-equivalents x 20 us per layer. Steady passes: 41-45 ms of compute vs a 22.3 ms byte floor (the 2
outlier passes are warm-up). So the "eff" lever is real (~20 ms/pass) but structural: it is the length of the dependent
kernel chain per layer, not the kernel count (round 29's exact fusions cut off-chain kernels: 0 %). A per-layer
critical-chain map (which ~22 kernels are serial) is the next decisive measurement for it; the pass model's eff
(29.2 ms, a 12-kernel chain) stays the optimistic target.

**Round 32k (~01:00): the verify pass is latency-bound on a ~55-kernel dependent chain per layer** (scripts/
critical_chain125.py on the exported 4-row graph data/pass_graph125.dot; shared primitive objects split per occurrence;
data/critical_chain125*.txt). 4,902 kernels per pass, **critical chain 2,707 (55 per GDN layer, 58 per QSA layer)**:
MoE ~12 (AsType > PostIds > SpinWait > AsType > GatherQMM > act > GatherQMM > Multiply > Sum > Add > Multiply > Add),
**two hyper-connection blocks ~10 each** (RMSNorm > Multiply > Matmul > Divide > SigmoidMul > Matmul > Sigmoid > Multiply >
Sum > Multiply), GDN attention ~20 (QMM > Concatenate > Convolution > silu > AsType > Multiply > Sum > Add > Rsqrt >
Multiply > AsType > Multiply > CustomKernel > RMSNorm > AsType > Multiply > AsType > QMM > Multiply > Add), router 3.
55 x ~7.6 us = the measured 0.42 ms/layer above the byte floor (round 32j). Round 29's exact cuts (-5 kernels in QSA
layers) were off this chain, hence 0 %; round-20's "intra-layer megakernel, +12-16 %" estimate stands, now with the
chain measured. **Next build (exact, my lane): fused HC block** = the 10-kernel chain as 1-2 kernels with MLX's own
arithmetic (as 3c), bitwise test vs the eager block; 2 per layer -> ~-16 chain kernels/layer ~ -5.8 ms/pass (~6 % at 34
GB, ~4 % at 24 GB). Then the GDN chain (~20 -> ~5). Check first: does BigMoE's current reference already compile HC
(BIGMOE_COMPILE_HC) and what does that do to the chain?
**Correction to 32k (~01:10):** an exact fused HC read already exists: `BIGMOE_HC_VREAD=1` (MoonShots' earlier work via
MTPLX hc_verify_read.read_rows: the whole read in 3 dispatches, bitwise to the compiled read at S=1/3/6; SpecDecode's
stack uses it, capped at 5 rows under DUPROW_CHUNK). The Oct-19 graph likely ran without it, so the reference chain is
probably ~14 kernels/layer shorter (~41). Remaining chain targets: GDN attention (~20/layer; MTPLX CONVNORM fusion was
exact only at short context, LEDGER GDN1) and the MoE part (3c+combine already removes ~3-4/layer). day_runner14 job 5
re-measures the chain under the reference flags before any build.

**Round 32l (~01:55): why MTPLX's GDN conv+norm fusion (LEDGER GDN1, +9 %) is not exact, and an exact version.**
mtplx/kernels/gdn_conv_norm.py keeps the conv output and silu in fp32 and rounds once; the eager chain rounds conv1d's
output to bf16, then silu in bf16 (Sigmoid, Multiply), l2norm in fp32 with MLX's own sum order, astype, and multiplies
by bf16(inv_scale) in bf16. Different bits in general; tokens matched at 64-token prompts by luck, diverged at 32K.
An exact fused verify-row prework kernel (conv + silu + split + l2norm + scale, replicating each rounding and MLX's
orders, as 3c does) would keep the ~+9 % with the reference's shas. Step 1: pin MLX's conv1d (depthwise k=4, bf16)
accumulation order and the 128-value row sum's lane mapping (patches/planes_engine/gdn_probe.py, GPU seconds).

**Round 32m (Wed 09:01, day_runner14): measured.**
- **Sibling union** (scripts/sibling_union125.py, data/day14/sibling_union.txt; 3 prompts x 6 positions, greedy): the top-4
  candidate tokens at one position use **21.2** experts/layer vs **26.1** for 4 consecutive committed tokens (1 token: 10).
  A sibling adds ~3.7 new experts vs ~5.4 per chain token: width is ~30 % cheaper than depth, not free; and the top-1
  candidate usually holds p 0.8-1.0, so alternatives rarely get accepted. Wide trees are not the lever past ~95 tok/s.
- **Exact GDN patch at model level** (scripts/gdn_exact_model125.py under BigMoE's exact reference env, 24 GB mode):
  decode-step and 4-row verify logits **bitwise identical** to the eager reference (fbee9e1e.../aa5576c4...); MTPLX's own
  CONVNORM kernel differs (3116dd58.../f3038648...).
- **Out-of-order experts cost** (bench_split.py): post-spin critical path at M=4: fused 0.314 ms/layer -> 0.058 (10 %
  missed pairs) / 0.122 (30 %); hit pairs 0.294 / 0.237 ms run inside the spin (p50 ~0.6 ms): ~0.19-0.26 ms/layer,
  ~9-12 ms/pass when layers stall. M=1: 0.106 -> 0.079 (dispatch overhead ~25 us each). Sent to MoEPrefetch (integrating).
- **3c tuning closed:** ushort loads = byte loads (<= 4 %), RA 8 slightly worse, RB 8 / NSG 4 within noise; 3c+comb stays
  -23..-28 % vs stock 4-bit (0.107 / 0.187 / 0.315 / 0.553 vs 0.141 / 0.244 / 0.423 / 0.768 ms at M = 1 / 2 / 4 / 8).
- Chain under the reference flags: the export only holds the last layer (ASYNC_EVERY=1 cuts the graph per layer);
  rerun with ASYNC_EVERY=0 in day_runner14b.

**Round 32n-o (Wed ~10:15, true reference, 24 GB mode; data/pass_profile125_vt.json, data/verify_start125.txt,
data/mtp_draft_stalls125.txt):**
- Verify-graph building is already pipelined by ASYNC_EVERY=1 (call start -> layer 0 gate ~3 ms; previous gate -> call
  0.9 ms): the ~9.8 ms gap of the non-reference profile is not a lever (retracted).
- **Loader stalls dominate the 24 GB pass:** trunk gates wait on the loader every layer, 70-77 ms of a ~108-116 ms pass
  (~1.5 ms/layer). So out-of-order experts hide the full hit compute (~0.24-0.29 ms/layer at M=4): ~12-14 ms/pass (~+12 %).
- **MTP draft stalls:** with MTP_STREAM=1 each of the 3 draft steps waits on gate layer 48 (p50 1.4-1.8 ms): 4.5-5.1
  ms/pass. Drafts only affect acceptance, so the MTP router can pick among resident experts only (exactness untouched):
  A/B in scripts/mtp_resident_draft125.py (day_runner14c): tokens/pass vs tok/s, identical outputs as a check.
- **Resident-only MTP drafts: negative** (scripts/mtp_resident_draft125.py, data/day14/mtp_resident_draft.txt; true
  reference, 24 GB mode, 3 prompts x 192 tokens): tokens/pass A -> B code 3.62 -> 2.87, math 3.06 -> 2.58, chat 2.54 ->
  1.76; tok/s -13..-25 %; outputs identical (exactness confirmed). With 128 of 512 MTP experts resident the drafter needs
  its true routing; the ~5 ms/pass of draft stalls are worth paying. (A rank-5-10-only mask could save <= ~3 ms/pass at
  best; not pursued.)
- **records4 kernel bitwise** (patches/planes_engine/tierglu4.py, test_tierglu4.py; data/day14/test_tierglu4.txt): tiers 2 /
  3 / 4 from the P2 / B3 / B4 planes, each == its MLX reference (bits=2 on P2; bits=3 on records3's MSB part = v3; bits=4
  on the original expert), plans 4433333333 / 4433332222 / 4422222222 / 4222222222, M = 1-8, all ranks bitwise. With
  make_records4.py (layer 0 verified, 0 mismatches) the records4 path is prototyped end to end on the kernel side.
- **Plan menu on records4** (data/planes4_sim_menu.txt; demand MB/pass at 17 GB vs records3 + 4433333333 = 400):
  4433222222 / 4442222222 322-325 (-19 %, +24 % reads), 4333222222 288 (-28 %), 4422222222 268 (-33 %), 4332222222 263
  (-34 %), **4322222222 237 (-41 %, -14 % reads)**, 4222222222 206 (-49 %, -27 % reads). The ~-45 % that 80 tok/s at 24 GB
  needs is 4222222222 or 4322222222; the latter (rank 2 at 3 bits) is the quality fallback to queue next to 4222222222.
- **records4 kernel speed** (bench_tierglu4.py, data/day14/bench_tierglu4_bt.txt; bit-trick builders, bitwise re-test PASS):
  ms/layer at M=1 / M=8: v3 (4433333333) on records4 0.146 / 0.910 vs records3 0.108 / 0.566 (the tier-3/4 rebuild from
  three planes costs ALU + loads); 4422222222 0.107 / 0.608; 4222222222 0.101 / 0.571 (records3 v3 speed at 31 % fewer
  bytes). GPU time is a wash; records4's win is SSD bytes and cache capacity (planes4_sim). If records4 replaces records3,
  keep a records3-style MSB-3 base for 3-bit-heavy plans or accept ~1.4-1.6x routed-GLU time for them.

### Round 33a (Wed 15:2x, literature subagent): megakernels on Metal, for the speed-of-light layer task
- **M5 Pro probes (lithos-metal, macOS 26.5.1, github.com/lithos-ai/lithos-metal probes/results):** dependent dispatch at the GPU
  1.3-1.6 us (serial / concurrent+barrier / ICB replay); in-kernel grid barrier 1.97 us (relaxed) / 2.47 us (spec-compliant fences);
  1,536 spinning simdgroups all resident, ~half time out at 2,048; one 384-thread threadgroup per core reaches 283-298 GB/s only
  with lane-interleaved 16-byte loads, 4 in flight. With ~30 us of work per phase: in-kernel 81.7-84.7 us/op vs 81.5 one dispatch/op.
- So on M5 a grid barrier is NOT cheaper than a dispatch boundary. Our old "5.4 us/hop" (data/gridbarrier.txt) timed the Python
  graph build too; the GPU-side hop is ~1.5 us. Megakernel wins elsewhere (Hazy No Bubbles 1.5-2.5x, MPK 1.0-1.7x, Cohere MoE
  1.25-1.58x) come from overlapping the next op's weight loads with the previous op's tail and per-dependency counters, not from
  removing launches. lithos-metal and an M5 Max write-up (persistent FFN->GDN-in 2.7 % slower) agree. Metal guarantees no
  co-residency or forward progress across threadgroups (MSL 4.1 spec; OOPSLA'21 2109.06132 shows Apple GPUs violating OBE).
- Consequence for the SoL task: the from-scratch engine's win must come from bandwidth-shaped kernels + fewer, fatter phases +
  CPU-side overhead (Python graph build, MLX encode, gate waits), not from grid barriers. Measured next: GPU-side vs build time
  (sol_layer.py now reports both), dependent vs independent per dispatch, persistent HC pair vs 2 dispatches (persist_pair.py).

### Round 33b (Wed 15:3x): 2-bit-heavy plans, literature (subagent) + a measured refit
- **Predictions (125B, GSM8K-150 ref 147, 4433333333 146):** 4322222222 holds GSM8K ~80 % (145-147; ppl +2.5-4 % by the 35B
  scaling), 4222222222 borderline ~55 % (143-146; ppl +4-7 %). Hard reasoning (MATH-500/GPQA/AIME) loses ~2x what knowledge
  loses (int2 Qwen3-80B 512x top-10, 2511.15015: GSM8K 87->81, MMLU-Pro 75.9->72.8, AIME25 70->63; R1-Distill 3-bit: AIME -10,
  GSM8K -0.1, 2504.04823). GSM8K is our least sensitive test; per-token KL catches flips (2407.09141).
- **Truncation is fine when re-centred** (SliceMoE AMAT 2512.12990: 4->2 truncation with the zero-point carried ~= independent
  2-bit rounding; our b + s*mean(q & 3) is that). Gate-rank precision beats static per-expert precision (DPA-MoE 2610.21378:
  KL 0.514 static vs 0.321 gate-score; HOBBIT 2411.01433 keeps top-1 high, Mixtral GSM8K 0.52->0.51; DyMoE 2603.19172: bottom
  25 % of active experts at int2 costs nothing on Qwen3-30B-A3B GSM8K).
- **Measured (scripts/lsrefit2.py, real 125B experts, layers 0/24/47 x 32 experts, CPU):** per-group least-squares refit of
  the truncated tier's scale+bias vs the stored truncation: tier 2 rel. weight error 0.363 -> 0.349 (MSE -7.6 %), refit
  scale = 0.92 x 4s (p10-p90 0.87-0.97); tier 3 0.161 -> 0.160 (MSE -2.3 %). Smaller than the subagent's Gaussian estimate
  (-10..-22 %, scale 3.2-3.5 s) because min-max g64 codes are spread wider than Gaussian bins. Records4 cost: s2 != 4s breaks
  the exact s3 = s2/2, s = s2/4 derivation, so B3/B4 would store their own scales (+51 KB per projection per sidecar).
  Verdict: not worth a format change by itself; a DWQ-style learned refit (frozen codes, scales/biases distilled to the 4-bit
  outputs, as mlx-lm DWQ) is the version that could matter: untested, 35B first.
- **Cheapest decisive screen (subagent's suggestion, fits EvalSuite's prefill KL runs):** score 4322222222 / 4222222222 by
  per-token KL vs the 4-bit reference on the reference's own 150 GSM8K solutions + ~2k general tokens (prefill only), next to
  4433333333 and 4422222222 (both pass GSM8K). Within ~1.5x of 4422222222's KL -> GSM8K-safe; >= ~2.5x -> visible losses.
  Then spend decode time on ~200 MATH-500/GPQA items, not more GSM8K.

### Round 33c (Wed 15:4x, CPU): activation-aware refit of the truncated 2-bit tier — the version that matters
- Data: NewIdeas' 35B capture (data/cap35/L*.npz: real MoE inputs, routing, outputs; 24,576 tokens per layer). Fit on even
  segments, scored on odd ones. Codes frozen (m2 = q >> 2, nested layout intact); per output row, the 2 x G group scales and
  biases are a closed-form ridge least-squares fit of the row's outputs on the expert's own routed inputs (DWQ-lite, no training loop).
- **Per projection: output MSE x0.64-0.77 on gate/up, x0.74-0.89 on down** (layers 3, 24, 38). Weight-space LS gives only
  x0.94-0.98 and can even hurt (L38 gate/up x1.10-1.13): the objective must be the outputs.
- **Whole MoE block, plan 42222222, all 256 experts: error energy x0.80 (L3), x0.84 (L24), x0.73 (L38)** -> ~-21 % of the
  2-bit ranks' damage, for free at runtime (data/day15/actrefit35.txt).
- Records4 cost: P2 stores the refit alpha/beta instead of 4s / b2 (same bytes); B3 then needs its own s3 (+51 KB per projection,
  RECSZ +5 %, ~+1.5 % SSD bytes for 4222222222). Tier 3 can take the same refit (untested).
- Decisive next test (GPU): 35B model-level KL/ppl for 42222222 trunc vs refit (capture all 40 layers' routed inputs, ~minutes;
  refit every expert; patched-expert eval). If KL drops ~20 %, 4222222222 on the 125B moves toward 4322222222's quality at
  4222222222's bytes, i.e. the 80 tok/s main-track plan gets more margin. 125B needs the same calibration capture (48 layers x 512
  experts; rare experts fall back to truncation through the ridge).

### Round 33d (Wed 16:1x, GPU, quiet lock): model-level refit test on the 35B — mixed, not a format change yet
scripts/refit35.py (data/day15/refit35.txt): calibration 12,288 tokens (gsm 0-399, MoonShots scripts, MoEPrefetch+SpecDecode
findings), ridge 1e-3 x max(1, 512/n), unweighted, all ranks; 67 % of expert projections fitted (rest kept truncated: < 3 x 2G
routed rows). Eval: prefill KL vs the 4-bit reference on gsm 1000-1149 / MoonShots patches code / NewIdeas docs (2,048 tokens each).
| refit / trunc KL | gsm | code | general |
|---|---|---|---|
| 44222222 | 0.847 | 1.124 | 0.945 |
| 42222222 | 0.926 | 1.175 | 0.966 |
| 43222222 | 0.929 | 1.192 | 0.911 |
| 44333333 (tier 3) | 1.023 | 0.752 | 0.837 |
- The 2-bit refit helps gsm/general KL (-3..-15 %) but hurts code KL (+12..+19 %), while its dNLL improves everywhere
  (code 42222222 +0.077 -> +0.022; gsm +0.090 -> +0.062): it moves probability toward the actual tokens but away from the
  reference's distribution, the signature of over-fitting the calibration directions (CPU cross-domain study: in-domain
  -27 %, held-out gsm +15 % at ridge 1e-3; ridge 1 keeps -21 % in-domain and is neutral-to-better out of domain).
- Absolute 35B KLs are large (44333333 0.02-0.05, 42222222 0.09-0.21): the 35B (8 experts, 3B active) is far more tier-sensitive
  than the 125B (EvalSuite: 3-bit cache 0.0017-0.0028).
- Next (prepared, REFIT_LAMBDA / REFIT_WEIGHT=score2 / REFIT_MINRANK=1 / REFIT_PLANS): ridge 0.3-1, gate-score^2 weighted
  rows, fit only on ranks >= 2 (where low tiers run), ~10 min GPU per variant. Not a board item until code KL also drops.

### Round 34 (Thu 06:1x): 2-bit tails fail the formal gate on the 125B (per-part prefill KL)
scripts/kl125_screen.py (the Eval suite's tf machinery, read-only, own-output sets): 4422222222 / 4322222222 / 4222222222 at
2.4-3.9x the per-part targets (1.10 x ar4) on code, gsm, ifeval and multi; 4433332222 over on code / ifeval; 4433333333 passes
(x0.67-0.75). GSM8K accuracy (49/50) was blind to it. Lever 1 (records4 2-bit-heavy tiers) is dead under the formal gate;
only last-rank 2-bit (4433333322-class) might pass, worth ~5 % of expert bytes. stall/REPORT.md has the table.

## Round 35 (Sun 11-01): exact near-ties (M-15) and runahead verification (M-16)
- M-15 EXACT-NT, BETWEEN (data/exactnt.txt). Near-tie acceptance on chat can be made exact as a speculative cascade
  (Narasimhan et al., ICLR 2025): the served model takes the drafter's FRESH depth-1 token where p_125B >= rho x p_top1.
  - Result: chat MOD +14.0 % at rho 0.25, against R-61N's non-exact +22.9 %.
  - What limits it: the block's deep tokens agree with the drafter's own fresh token only 20 % of the time at depth >= 2
    (97 % at depth 1). That is a cross-anchor self-consistency gap, which MTP Head's training could close.
  - It is a served-model change, so Kamen decides.
- M-16 RUNAHEAD: runahead execution (Mutlu et al., HPCA 2003) applied to the MoE verify pass.
  - Mechanism: each pass computes the next block with its non-resident experts SKIPPED and queues their reads. The next pass
    re-checks those rows exactly (all resident by then), so tokens come only from exact rows. This takes the SSD stalls off
    the critical path. The cost is that accepted rows are computed twice, plus a bubble whenever the provisional verdict was
    wrong.
  - Step 1 (CPU) PASS: block error vs REF2-T was 0.6x REF2-T's own tier error (skip) and 0.3x (wait for misses needed at
    rank 0-1).
  - Sensitivity on D-23: the gain hinges on the pure compute per re-checked row. At beta 2.5 ms: +40-80 %. At beta 5 ms:
    +15-25 %.
  - Step 2 runs tonight in NewIdeas' quiet line. Design notes: RUNAHEAD_DESIGN.md.
- Side notes:
  - Trees on top of runahead (verify a tree in the runahead, re-check the chosen branch) do not pay on chat: Breakthrough's
    dynamic trees need ~5 more rows for +19 % kept.
  - The drafter is launch-bound (R-06: 4-bit drafter still 9.6 ms).
  - Rejected-row recycling is dead (M-06).

## Round 36 (Sun 11-01 afternoon): per-layer pool sizes at 45 GB (M-17) and how much RAM 350 needs (CACHE-SIZE45)
- M-17 LAYSZ45 PASS in the replay (data/laysz45.txt). Each layer gets its own base / sidecar pool size (a Lagrangian over the
  layers' miss curves) at the same pool bytes as the preset's uniform split.
  - Held-out: -20.7 % SSD MB per kept token at C 224, -19.3 % at C 256. That is about what +3 GB of uniform pool buys.
  - The gain is the per-layer shape: one re-split for all layers gives only -3 %. Layers 0-2 and 6-7 want ~400 base slots;
    layers 15-16 and 31-32 want the 160 floor.
  - It survives shorter checks (-19 %) and other domains (expA traces: chat -22 %, Bulgarian -33 %, code -19 %, math -25 %).
  - Open: the engine. R-27, the 24 GB version, lost most of its replay gain in BigMoE's engine A/B (b36: -1.5 % vs -8.6 %).
- CACHE-SIZE45 (data/cachesize45.txt): the 350 tok/s SSD budget (37.1 MB per kept token at 13 GB/s) needs C ~350 (54.4 GB)
  with LRU + prior, ~323 (50.8 GB) with per-layer sizes, and ~275 (44.6 GB) even for Belady. At 45 GB the SSD-side ceiling is
  188 tok/s today and 231 with per-layer sizes.
  - The preset's fixed 112-slot sidecar holds LSB misses at ~19 per kept token (~15 MB) whatever C is, so extra RAM should
    not all go to the base pool.
  - Paying only for the kept prefix + 1 (R-62P's oracle) brings the 350 budget to C ~289 (46.5 GB) with per-layer sizes.
- Night 11-01/02 results:
  - M-16 RUNAHEAD step 2 KILL at C 224 (data/runahead/ra_224.txt). The best variant models +0.4 % tok/s. Per-row compute was
    as low as hoped (2.64 ms), but the default is SSD-bandwidth-bound: on code, ~936 reads per check take ~147 ms even at deep
    queue, out of a ~200 ms check. Latency hiding cannot pay at 45 GB; only fewer bytes or more RAM can.
  - M-17 per-layer pool sizes PASS in BigMoE's engine: -17.2 % SSD MB per kept token and +8.3 % steady tok/s at C 224, exact.
- Wed 21:0x-21:2x, two CPU screens for the orchestrator (one core, share lock):
  - M-38 FORESIGHT PASS (scripts/foresight45.py, data/foresight45.txt). On M-14's chained cap45 replay, a forecast of the
    experts each layer needs in the next H checks keeps them from eviction; the rest leave by LRU. At 45 GB (K 266): LRU
    42.9, learned key 39.5 (-7.8 %), OPT 19.2 misses per kept token. Perfect forecast, H 1 / 2 / 4 / 8 / 16: -8 / -14 /
    -23 / -34 / -45 %. Noisy forecast (each future use kept with probability r, the dropped ones replaced by popularity
    false positives): H 8 at r 0.5 / 0.7 / 0.9 = -19 / -26.5 / -32 %; H 16 = -23 / -33 / -42 %. 24 GB (K 147): r 0.7, H 8
    = -27.7 %.
    - At r 0.7, H 8 the gain holds under every noise structure tried: per-look noise -22 %, recall falling 0.9 -> 0.5
      with depth -24.5 %, uniform false positives -25 %, full-horizon checks only -27 %.
    - Controls: a popularity-only forecast is 24 % worse than LRU. With zero recall on LRU's misses (0.9 on its hits) the
      cut is still -19 %. What pays is knowing which residents come back within H checks.
    - What the GPU step must measure: at the eviction decisions ~45 % of residents are used again within 8 checks. r 0.7
      means naming them at recall 0.92 / precision 0.72; recency alone reaches 0.83 / 0.52 at best (used in the last 8).
    - Prior art: MoE-SpeQ / Specter (2511.14102) looks one draft stage ahead (<= 32 tokens, its buffer cleared each stage,
      no foresight eviction); SP-MoE stays within one round with LRU eviction; SPICE adapts its lookahead. Tag: adapted.
      The 8-16-check horizon for eviction, priced under noise, is ours.
    - Fable FAR's follow-up (Wed 22:0x, same frame; ~/FableGod/far/FAR.md, findings note 2026-11-04_2210): forecasters
      at per-use recall ~ precision ~ 0.7 cut misses only 7 % (nearest earlier row) or 14.6 % (a ridge state -> route
      map), not 26.5 %. Their errors repeat across a resident's uses; this model's are independent, so a resident used 3
      times is named with 0.97. The true future tokens reach recall 0.68 at best: the rejected rows hold 32 % of the reuse.
      Perfect 8-check foresight cuts misses 34 % but stalled layers only 5 %. FAR then fit the SSD wait on BigMoE's 106
      logged runs (FAR-WAIT): ~10-16 ms per check + 0.15-0.17 ms per miss (~14 GB/s, the drive's streaming rate), R2
      0.86-1.00. So above the floor the wait is bandwidth-bound and a miss cut maps ~1:1 onto SSD time: -26.5 % misses
      = -20 ms (-14.5 %) of a cap45 check, -6..-9 % at the live 100-200 misses. The floor matches M-28: ~43 stalled
      layers x ~0.28 ms per call ~ 12 ms. The PASS stands only for a forecaster at the per-resident 0.92 / 0.72.
  - M-39 SPANRELAX KILL by the letter (scripts/span_relax_check.py, data/span_relax_check.txt): Fable FULL's bet, relaxed
    acceptance only inside '#' comments, as pre-registered in ~/FableGod/full/FULL.md section 5. The replay reproduces
    RELAX_350's code table (7.78 / 8.61) and Mission 25's table (x1.157, 1.68 %, 55 / 0).
    - Kept clears the lines: rule A 12.65 (x1.200, profile), rule B 12.78 at 0.14 % code-token deviation (prediction 12.1 /
      12.3 / 0.2 %).
    - The disagreement line fires: 4 of rule A's 70 relaxed accepts (5.7 %; B 6.8 %, line 2 %). All 4 are the same
      failure: the 125B opens a comment (argmax ' #'), Mission 25's scan labels the position by its target token, so it
      reads 'comment', and rule A accepts a code draft (' for', ' if').
    - The engine's causal tagger (EvalSuite relax_core.CodeScan + comment_rows) does not make that mistake: A 12.69 with 0
      code drafts accepted, B 12.78 at 0.11 %. Its 5 disagreements are comment continuations. That variant is post hoc
      and needs its own pre-registration.
    - Limits: the 125B's top-8 covers 12 of the 24 windows; the kept numbers are the profile model (Mission 25 halves it);
      length inflation (2610.23818: x1.34 accepted, x1.54 longer answers) is invisible to a walk replay.
- Wed 21:4x M-40 R1CENSUS KILL (scripts/r1_census.py, data/r1_census.txt). This was the first test of Fable FULL's
  runner-up R1: write one answer as parts and decode them as parallel streams in one verify pass. I ran an ast census of
  recorded answers. Parts are top-level defs and classes; an edge is a name reference between two parts. Signatures,
  docstrings, imports, prose and driver code stay sequential.
  - HumanEval+ (24 walk windows) 0 % and MBPP (100) 0 %: one function per answer.
  - Magicoder python (38,284 items): 1.2 % meet the line (>= 2 independent parts holding >= 50 % of the answer); FULL
    predicted 40-60 %. 7.2 % have >= 2 independent parts, but those parts hold only 34 % of their answer.
  - Counting each method of a class as its own part gives 6.3 %; also measuring against the code alone (prose excluded)
    gives 16.8 %. Both are under the 30 % PASS line.
  - The text that could run beside another stream averages 0.8 % of a Magicoder answer (4.6 % with per-method parts), so
    the time R1 could save on this workload is a few percent at most.
  - The d16w agent windows hold no Python (bash, edits, a Swift file).
- Wed 22:24 M-41 SPANRELAX-C, HumanEval+ part (scripts/span_relax_c.py, data/span_relax_c.txt): FULL's rule C on the 12
  fresh HPD-0 windows (Breakthrough's top-8, made with the map harness) is BETWEEN.
  - Kept is x1.074 = 11.32 against the lines PASS >= x1.10 / KILL < x1.05 (prediction x1.13).
  - It is clean: 0 code-token deviations, 0 relaxed accepts outside comments, 0 % disagreement on a re-scan. The data
    lines up: the 125B's top-1 agrees with the walk's own acceptance at 99.79 % of positions.
  - The fresh windows give less to gain: 11.3 % comment rows (15.5 % on the map's 12) and an easier exact profile (10.15
    vs 7.78 kept).
  - Pooled over all 24 windows it reaches x1.137, near FULL's x1.15.
  - MBPP PASS (22:37, 100 windows, 15,351 positions): x1.125 against PASS >= x1.03 (prediction x1.04), again with 0 code
    deviations and 0 % disagreement. MBPP answers hold 19.9 % comment rows (FULL guessed 5-8 %) on a hard exact profile
    (5.01 kept), so the comment rule pays more there than on HumanEval+. The engine number (10.54 x ratio) uses HumanEval+'s
    base; MBPP's own live kept is lower.
- Thu 10:2x-10:3x, four CPU jobs after the records3 conversion (one share lock):
  - R57_RELAX_SPAN tests ALL OK (data/test_relax_span.txt). The patch's own functions, fed text the way r57 feeds its
    comment tagger, reproduce the rule C replay's accept decisions on all 21,504 tested rows: 0 differ. M-41's numbers
    stand (x1.189 / x1.074 / x1.125; the whole-window feed equals the engine's answer feed on every row). BigMoE
    integrated it (fef5586); the engine arm runs Fri ~01:40 after the rx0 gate.
  - M-42 SPLITCHECK CONFIRMED (data/split_route_check.txt). SWEEP rank 5 (drafted 0.3 + bonus gate 0.4) = 52.8 / 62.0 /
    64.2 MB per kept token vs NewIdeas' 53.9 / 62.4 / 64.0, chat x0.71 / 0.73 of the cf 0.5 route, the same best Cl. Far
    stand-ins put chat 3-9 % higher at the same ratio.
  - M-43 KLPERERR CONFIRMED, slightly smaller (data/kl_per_error_check.txt, data/msb2_judge.json).
    - Measured directly on the same rows, with no model in between, the trellis store costs 3.18x / 1.65x / 2.34x the
      KL of MSB-2 truncation per unit of router-weighted expert-output error (code noisy, SE ~+-1.3).
    - The calibration's assumption holds: MSB-2's output error is 4.98-5.10x MSB-3's on real rows.
    - Fit12's layer means understate the store's error: pooled by error it is 1.29-1.82x MSB-3, because late layers
      carry more error. That shrinks NewIdeas' 3.5 / 1.8 / 2.4 to about 3.2 / 1.65 / 2.3.
    - msb2_judge.py rebuilds W4 / MSB-3 / MSB-2 from the original 4-bit safetensors, so it is immune to the records3
      conversion. Its trellis / MSB-3 ratios reproduce Fit12's records3-based ones to 3 decimals.
