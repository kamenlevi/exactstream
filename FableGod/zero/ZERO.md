# ZERO.md — the fastest way to run a 125B fine-grained MoE on one M5 Pro, from first principles

Written 2026-11-04 (evening) by the "zero" session. No prior attempts were read; only the mission brief
(`MISSION_ZERO.md`) and published facts. Tags: **[K]** known/published (link), **[P]** arithmetic from the
givens, **[G]** judgement.

## 0. Verdict in six lines

1. At batch 1 **every token must drag its own ~1 GB of expert weights through DRAM** (480 experts x ~2 MB), and
   the always-on trunk (attention + shared experts + lm_head) adds ~1.8 GB. 286 GB/s therefore caps plain
   token-by-token decoding at ~100 tok/s even with the whole model in RAM. [P]
2. Speculative decoding (the built-in MTP head) amortizes the trunk but **not** the experts: each *drafted*
   token costs ~10 fresh experts per layer. The gain is capped near 3-4x, not k×. EXACT mode tops out at
   **~130-150 tok/s with zero SSD misses, ~60-100 tok/s at realistic miss rates**. [P][G]
3. 300 tok/s is only reachable in FAST mode by **shrinking each token's expert footprint** (3-bit experts,
   gate-threshold top-k, in-expert activation sparsity) — not by a faster cache. [P]
4. The 10 MB/token SSD goal needs a **≥ 99% expert hit rate**; a 24 GB budget holds ~36-43% of the experts, so
   caching alone cannot deliver it. The one mechanism that can is **a cache-aware router**: a DeepSeek-style
   selection bias toward resident experts, tuned until misses ≤ 1%. FAST only. [P][K][G]
5. The 12 GB stretch (13-16% of experts resident) is not reachable at benchmark parity by any mechanism I can
   defend; I do not claim it. [G]
6. **Bet: Design B** (footprint shrink + cache-aware router + MTP speculation + GPU-driven runtime),
   expected 200-270 tok/s on code and ~5-8 MB/token of SSD reads; 300 iff two of four optimistic
   assumptions hold (§3). Design A is the same runtime with the knobs set to EXACT (~60-130 tok/s).

---

## 1. The physics

### 1.1 Sizes from the brief [P]

Note: the spec in the brief (hidden 2560, 512 experts x top-10, expert width 640, 24/2 heads) does **not**
match the public `Qwen/Qwen3.5-122B-A10B` config (3072 hidden, 256 experts x top-8, width 1024, 32 heads)
[K: https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json], so everything below is derived
from the brief's numbers; the GDN head counts are not given and are estimated.

| Item | Arithmetic | Value |
|---|---|---|
| One expert (SwiGLU, 3 matrices) | 3 x 2560 x 640 | 4.92 M params |
| Expert bytes at 3.0 / 3.5 / 4.5 bit | 4.92 M x bits / 8 | **1.84 / 2.15 / 2.76 MB** |
| Routed experts total | 512 x 48 x 4.92 M | 120.8 B params (24,576 experts) |
| Non-routed, implied by 125 B total | 125 − 120.8 | ~4.2 B params |
| Of which: lm_head 248,320 x 2560 | | 0.64 B |
| shared experts 48 x 4.92 M | | 0.24 B |
| attention: 12 full (24x256 q, gated, 2 kv) + 36 GDN | ~50 M + ~42-73 M per layer | 2.1-3.2 B |
| **Dense trunk** (read every token) at 3.5 / 3.0 bit | 4.1 B x bits / 8 | **1.8 / 1.55 GB** |
| **Routed experts per token** at 3.5 / 3.0 bit | 10 x 48 x expert bytes | **1.03 / 0.885 GB** |
| KV cache (12 full-attn layers, 2 KV heads x 256, bf16) | 12 x 2 x 2 x 256 x 2 B | 24.6 KB/token (0.8 GB at 32K) |
| GDN recurrent state | 36 x 32 x 128 x 128 x 4 B | 75 MB total |
| FLOPs per token (active 4.1 B + 2.6 B) | 2 x 6.7 B | ~13 GFLOP |

Machine [given, K]: GPU ~286 GB/s measured (Apple's spec is 307 GB/s
[K: https://www.macrumors.com/2026/03/03/apple-debuts-m5-pro-and-m5-max-chips/]); SSD 12-13 GB/s;
24 GB process budget (12 GB stretch).

### 1.2 Bound 1 — DRAM bytes per generated token (the binding one)

Plain decode, all weights in RAM: 1.8 + 1.03 = 2.83 GB/token → 286/2.83 = **101 tok/s peak, ~70 tok/s at the
80% of peak that streaming quantized matvecs reach on Apple GPUs** [P][G]. Nothing about the SSD; this is
the ceiling with infinite RAM.

Speculative decoding changes the accounting, but differently from a dense model. A verify step over n = k+1
positions reads the trunk once (amortized over A accepted tokens) but reads the **union** of the positions'
experts: U(n) ≈ k_eff + (n−1)·k_eff·(1−ov) per layer, where ov is the fraction of a token's experts already
needed by its neighbours. For 512-expert fine-grained routing with a shared expert the local routing
consistency is low — the ICLR-2026 study of 20 MoEs finds shared experts *harm* consecutive-token
consistency (Qwen3 SRP 54 vs Qwen2-MoE-with-shared 37, F1 at 16-token segments)
[K: https://arxiv.org/abs/2505.16056] — so ov ≈ 0.3 is my central estimate [G]; uniform-random would be
ov ≈ 0.02 (union of 5 tokens = 48 experts) [P].

Consequence [P]: the maximum speed-up speculation can ever give here is
(trunk + experts)/(experts·(1−ov)) = 2.83/(1.03·0.7) ≈ **3.9x** (2.75x at ov=0, 5.5x at ov=0.5),
versus ~k× for a dense model. **Each drafted token costs ~70% of a real token in bytes.** Trees are
counter-productive (every tree node is 10 more experts per layer); draft length must stop where
P(accept) · (step time / A) < cost of one more drafted token (~1-1.5 ms).

Step model used everywhere below (parameters in §A.1):

```
U(n)        = k_eff + (n-1)·k_eff·(1-ov)                      distinct experts per layer, n = k+1 positions
DRAM/step   = D_trunk + k·d_draft + U·48·e·a                   a = in-expert sparsity read factor (1.0 = dense)
T_step      = max( DRAM/(η·286 GB/s) + T_gap ,  m·U·48·e / 12.5 GB/s ) + unpredicted-miss stalls
A           = 1 + Σ_{i=1..k} α^i                               accepted tokens per step (α = per-position acceptance)
tok/s       = A / T_step ;   SSD MB/token = m·U·48·e / A
```

Results of that model (η = 0.8, T_gap = 3 ms, draft 95 MB/token, α = 0.88 on code, ov = 0.3) [P]:

| Case | k | A | U | DRAM/step | tok/s |
|---|---|---|---|---|---|
| EXACT, 3.5-bit, no misses | 4 | 3.9 | 38 | 5.9 GB | **133** |
| EXACT, 3.0-bit, no misses | 4 | 3.9 | 38 | 5.25 GB | **152** |
| EXACT, 3.5-bit, no misses, uniform routing (ov=0) | 4 | 3.9 | 50 | 7.1 GB | 106-117 |
| EXACT, 3.5-bit, ov = 0.5 (optimistic) | 6 | 5.2 | 40 | 6.1 GB | 166 |

So **EXACT cannot reach 300 tok/s on this machine under any caching scheme**; its ceiling is ~130-170.

### 1.3 Bound 2 — SSD bytes and the hit rate

Budget at 24 GB [P]: trunk 1.55-1.8 + MTP block's own 512 experts 0.95-1.1 (resident, see §1.6) + KV at
32K 0.8 + GDN state 0.08 + scratch/runtime ~1.2 → **~19 GB of expert cache = 8,800 experts at 3.5-bit
(36%) or 10,500 at 3.0-bit (43%)**. At 12 GB: ~7 GB → 3,300-4,000 experts (13-16%).

Three facts follow [P]:
- SSD can supply 12.5/229 ≈ **5.5% of the DRAM byte rate**. If more than ~5% of the step's expert bytes
  miss, the step is SSD-bound even with perfect prefetch.
- **10 MB per generated token = 4.7 expert reads (3.5-bit) or 5.4 (3.0-bit) per token.** With k=4 the step
  reads U·48 = 1,824 experts for 3.9 tokens, so the miss rate must be **m ≤ 0.9%** — a 99.1% hit rate.
- A 36% cache gives 99% hits only if the session's routing is extremely skewed. Load-balanced routers are
  trained to be uniform in aggregate; what saves us is *session-level* skew (one user, one codebase).
  MoE-Infinity observed that expert reuse is present at the request level and evens out across requests
  [K: https://arxiv.org/abs/2401.14361]; the ICLR-2026 study's oracle segment-cache hit rates at a cache of
  2x the active experts are only 55-77% over 16-token segments [K: https://arxiv.org/abs/2505.16056]. At 18x
  the active experts (our 36%) I expect 85-95% LRU hits on a code session [G] — i.e. m = 5-15%.

EXACT at those miss rates (k=4, α=0.88, 3.5-bit, 80% of misses prefetched in time) [P]:

| miss rate m | 0 | 1% | 2% | 5% | 10% | 15% | 20% |
|---|---|---|---|---|---|---|---|
| tok/s | 133 | 126 | 120 | 105 | 84 | 57 | 45 |
| SSD MB/token | 0 | 10 | 20 | 50 | 100 | 149 | 199 |

**The whole EXACT story is one number: the miss rate on a real code session.** It is also the cheapest thing
to measure (§2, test T1). Cold misses are not the problem (a 10K-token session touching 80% of experts costs
(19,660 − 8,800) x 2.15 MB / 10K = 2.3 MB/token); capacity misses with long reuse distances are.

### 1.4 Bound 3 — compute (second-order, but not free)

At n = k+1 = 5-7 positions the kernels are small-M GEMMs over 3-3.5-bit weights: arithmetic intensity
2n/(bits/8) = 23-32 FLOP per weight byte, so keeping up with 229 GB/s needs **5-7 TFLOPS sustained
including dequantization**, with M padded to the matrix-unit tile (8) [P]. Apple states M5 Pro/Max have
"over 4x the peak GPU compute for AI" of M4 Pro/Max via a Neural Accelerator per GPU core
[K: MacRumors link above]; no TFLOPS figure is published. Plain ALU dequant of ~15 B weights per step at
~3 ops each is ~45 Gop → ~5 ms if not fused into the matrix path [G]. Verdict: feasible, but the 80%
bandwidth efficiency assumed above is only real if dequant is fused and M is padded — measure early (T2).

Prefill is compute-bound and SSD-bound at once: a 4K-token prompt is ~55 TFLOP and touches essentially all
experts once per chunk, i.e. reads the whole ~50 GB from SSD (≥ 4.2 s) [P]. TTFT ~5-10 s for 4K prompts is
the price of a 24 GB budget; it is outside the per-token goals but must be stated.

### 1.5 Bound 4 — latency and sequential dependence

- 48 layers are strictly sequential, and **routing for layer L is known only after layer L−1**; the step
  budget at 250 tok/s with A ≈ 4-5 is 16-20 ms → ~0.35 ms per layer.
- A 1.84 MB SSD read costs ~0.15 ms transfer + 0.1-0.3 ms latency [G] — about one layer's time. Any miss
  on the critical path must be known ≥ 1-2 layers ahead; the bandwidth overlap (§1.3) needs misses spread
  across the step, i.e. predicted many layers ahead or at draft time. Parameter-free next-layer prediction
  (apply router L+1 to the post-attention residual of layer L) reaches ~90% recall@k on Qwen3-30B-A3B
  [K: https://arxiv.org/abs/2603.19289]; next-two/three-layer top-1 ~90% in HOBBIT's setup
  [K: https://arxiv.org/abs/2411.01433]; a trained cross-layer gate reports 97% prefetch accuracy
  [K: https://arxiv.org/abs/2502.12224]. Multi-layer-ahead recall for 512-expert top-10 is unmeasured
  [G: 70-80% at 3-4 layers ahead].
- Kernel gaps: ~6-12 kernels per layer x ~2-8 µs scheduling gap + tiny-kernel minimums ≈ 50-150 µs/layer
  → **2.5-7 ms per step of dead time** unless fused [G]. A CPU round-trip per layer (0.1-0.3 ms x 48 =
  5-15 ms) would alone halve throughput: routing, gathering and SSD-miss handling must be GPU-driven
  (one command buffer per step, indirect dispatch, the CPU only services the miss queue asynchronously).
- The draft (k x one MTP block) is sequential and runs before the verify: ~0.3-0.4 ms per draft token
  [P: ~95 MB + ~10 kernels]. It can be hidden by drafting the *next* step's tokens on the CPU/ANE during
  the current verify, on the assumption that all current drafts are accepted (true with P = α^k ≈ 0.6 at
  k=4) [G].
- Hidden cost everybody forgets: the **draft's lm_head**. Full vocab at 4-bit is 318 MB per draft token —
  1.3 GB per step at k=4, as much as all the expert reads. Draft with a frequency-ranked sub-vocabulary
  (FR-Spec: draft on a high-frequency subset, verify on the full vocab, no exactness loss, +1.12x on
  EAGLE-2) [K: https://arxiv.org/abs/2502.14856]: 32K x 2560 at 4-bit = 41 MB.

### 1.6 What is always resident (both budgets) [P]

Dense trunk (1.55-1.8 GB), all 48 routers (512 x 2560 x 2 B = 2.6 MB each, 126 MB), the MTP block incl.
its own 512 experts (~1.0-1.2 GB; drafting must never touch the SSD), the draft sub-vocab head (41 MB),
KV/GDN state, scratch. Everything else is the expert cache.

---

## 2. Three designs

### Design A — EXACT: resident trunk + expert cache + MTP speculation + GPU-driven prefetch

**Mechanism.** Experts at 3.0-bit (1.84 MB; 43% of them fit) or 3.5-bit (36%). Cache policy: sequence-level
LFU over gate mass (HOBBIT reports +4.5% hit ratio over plain LFU and 4.7-8.7% lower miss penalty than LRU
[K: https://arxiv.org/abs/2411.01433]; FlashMoE's learned recency+frequency policy claims up to +51% hit
rate over LRU/LFU on edge SSD offloading [K: https://arxiv.org/abs/2601.17063]), warmed during prefill with
per-expert gate-mass counts. MTP draft with adaptive length (stop when the draft's own cumulative
probability falls below the break-even ~0.3), FR-Spec sub-vocab head. Verify = one Metal command buffer per
step: fused norm+router, GPU-side expert gather (sort positions by expert, each expert read once), fused
SwiGLU, GDN chunked scan over the n positions. Misses: at every layer L the GPU applies routers L+1..L+4 to
the current residual for all n positions and writes predicted-miss IDs to a ring buffer; a CPU thread pool
issues F_NOCACHE reads at queue depth ≥ 8 directly into shared MTLBuffers (unified memory, no copy); an
unpredicted miss stalls that layer's expert kernel on a GPU-visible flag.

**Arithmetic** (k=4, α=0.88, A=3.9, U=38, 3.5-bit) [P]: DRAM/step = 1.8 + 0.38 + 38·48·2.15 MB = 6.1 GB
→ 26.7 ms + 3 ms gaps = 29.7 ms → **133 tok/s at m=0**; at m = 5 / 10 / 15%: **105 / 84 / 57 tok/s** and
**50 / 100 / 149 MB/token** from SSD. 3.0-bit: 152 / 119 / 95 / — tok/s. Batch-1 without speculation: 70.
Mode: **EXACT**. Neither goal is met: the design exists because EXACT is required and because B is this
design with three knobs turned.

**Biggest risk.** The miss rate: if the code-session working set is not far more skewed than the training
mixture, m ≥ 10% and the SSD, not the GPU, sets the speed (~60-85 tok/s, ~100-150 MB/token).
Second: multi-layer-ahead prediction recall for 512-way routing is unmeasured; if < 70%, unpredicted stalls
add ~0.3 ms each.

**Cheapest test — T1, routing trace + cache simulation (≈ 4 min of GPU once, then CPU only).**
Run the model slowly (any correct path, e.g. streaming experts from SSD at 5-10 tok/s) on one realistic
coding session: a 1.5K-token repository context + 1.5K generated tokens, greedy. Dump per token, per layer:
the 10 expert IDs, their gate values, and the post-attention residual (bf16; 3K x 48 x 2560 x 2 B = 0.74 GB).
Then on CPU: (a) LRU and gate-mass-LFU hit rate at 8,800 and 10,500 slots over the generated half, (b)
consecutive-token union U(5) per layer → ov, (c) recall of router_{L+d}(h_L) vs true top-10 for d = 1..4.
PASS/KILL (written before the run): **PASS if LRU-or-LFU miss ≤ 3% at 10,500 slots (→ ≥ 115 tok/s EXACT,
≤ 30 MB/token). KILL for the 10 MB goal in EXACT if miss > 2% even with the LFU policy (it will be — then the
10 MB goal belongs to Design B). KILL for the whole "cache" family if miss > 20% at 10,500 slots (SSD-bound
below 50 tok/s; go straight to Design C).**

### Design B — FAST: shrink the per-token footprint and make the router cache-aware (the bet)

**Mechanism** = Design A's runtime plus four knobs, each individually revertible to EXACT:
1. **3.0-bit experts** (trunk at 3.5-4 bit): per-expert 1.84 MB, 43% resident.
2. **Gate-threshold top-k**: keep an expert only if gate ≥ τ·max-gate (cap 10). The paper that studied
   this on Mixtral ("dynamic expert skipping" by routing-weight ratio) got 1.2-1.33x with small losses
   [K: https://arxiv.org/abs/2402.14800]; for a 10-of-512 router the tail experts carry little mass [G],
   target k_eff ≈ 6.5.
3. **In-expert activation sparsity** (TEAL-style magnitude thresholding, training-free, 40-50% model-wide
   sparsity with "minimal degradation" on Llama/Mistral, 1.53-1.8x decode speed-ups
   [K: https://arxiv.org/abs/2408.14690]): read gate_proj fully (1/3 of the expert), then only the up/down
   rows of the ~50% strongest intermediate units → read factor a = 1/3 + 2/3·0.5 = **0.67** of expert bytes
   from DRAM (SSD misses still fetch whole experts). Needs up_proj stored transposed and the quant blocks
   aligned to intermediate units; two dependent kernel phases per layer.
4. **Cache-aware routing**: add a per-expert *selection* bias +β to resident experts before top-k, leaving
   the gate values untouched — exactly the loss-free load-balancing primitive DeepSeek trained V3 with
   (bias "only used to adjust the routing strategy by influencing the top-K selection", not the gating
   value) [K: https://arxiv.org/abs/2408.15664]. β per layer is tuned so that misses ≤ 1%; β = 0 is EXACT
   routing. A miss with gate above a per-layer "must-have" threshold is still fetched (stall); below it,
   the expert is swapped for the best resident one. The resident set is the top of the gate-mass LFU.
Also: residency-aware drafting is unnecessary (the MTP block's experts are all resident).

**Arithmetic** (3.0-bit, k_eff 6.5, a 0.67, m 1%, ov 0.3, η 0.8, fused kernels T_gap 1.5 ms) [P]:
- k=4, α=0.88 → A 3.9, U 25: DRAM/step = 1.55 + 0.34 + 25·48·1.84 MB·0.67 = 3.4 GB → 14.7 + 1.5 = 16.2 ms
  → **230-240 tok/s**, SSD = 0.01·25·48·1.84 MB / 3.9 = **5.7 MB/token**.
- k=6, α=0.90 → A 5.2, U 34: 4.1 GB → 20.6 ms → **254 tok/s**, 5.7 MB/token.
- **300 needs two of:** ov 0.5 (286 alone), η 0.9 (281), a 0.53 i.e. CATS-level 60% in-expert sparsity
  (279), α 0.95 (294); any two → 308-317; all → ~340. [P]
- Without the fused-kernel assumption (T_gap 3 ms): 205-237.
Mode: **FAST**. Both goals are met only in the optimistic corner; the central estimate is 230-270 tok/s
and 5-8 MB/token.

**Biggest risk.** Benchmark parity under four stacked approximations. Each is published as "near-lossless"
on *other* models; nobody has stacked them on a 512-expert fine-grained MoE, and the compounding is
unknown. Mitigation is structural: every knob has a continuous dial (bits, τ, sparsity fraction, β) and the
runtime is identical at the EXACT setting, so parity is bought back with tok/s one dial at a time. Second
risk: the activation-sparsity read pattern (row gathers inside 3-bit packed blocks) may not reach 80% of
DRAM bandwidth; if the kernel hits only 60%, knob 3 is a wash and B lands at ~200.

**Cheapest test — T3, greedy-agreement and ΔNLL on held-out code (≤ 10 min GPU).** Teacher-force 20K tokens
of unseen code (not the T1 session) through EXACT and through B at the central knobs (3.0-bit, τ for
k_eff 6.5, 50% in-expert sparsity, β for ≤ 1% misses measured on the T1 trace). Prefill-style batched
passes read the model ~5 times from SSD (~4 min) plus ~2 min compute. Report per-knob and all-knobs:
greedy top-1 agreement with EXACT and ΔNLL. **PASS: all-knobs agreement ≥ 98% and ΔNLL ≤ +2%
(benchmarks will sit inside run-to-run noise [G]). Dial back any single knob whose own ΔNLL > 1%. KILL
the stacked design if all-knobs agreement < 95%; then B degrades to "A + 3-bit + β" (≈ 150-170 tok/s,
~6 MB/token).** Final arbiter remains a real HumanEval+/MBPP+/GSM8K/IFEval run, which exceeds 10 min.

### Design C — FAST: nothing on the SSD — prune 55-60% of experts, then B's dials

**Mechanism.** One-shot expert pruning with router-weighted activation norms (REAP: near-lossless, Δacc ≤ 2%
on code at 50% pruning of Qwen3-Coder-480B and Kimi-K2; Cerebras reports 97.6% of non-agentic coding and
96.7% of SWE-Bench retained at 50% [K: https://arxiv.org/abs/2510.13999,
https://www.cerebras.ai/blog/reap]). At 60% pruning (205 experts/layer kept): 9,830 x 1.84 MB = 18.1 GB of
experts + 1.55 trunk + 1.1 MTP + 0.9 KV/state + 1.2 scratch = **22.9 GB: the entire model is resident, SSD
reads per token = 0, no cache, no prefetch, no stalls, flat latency.** Then add B's knobs 2-3 (threshold
top-k, in-expert sparsity) on top; knob 4 (β) is moot. Calibration on code + math + chat so the pruned set is
not code-only.

**Arithmetic** (m=0, 3.0-bit, k=6, α=0.9, A 5.2) [P]: plain top-10: U 52, DRAM 6.7 GB → 30.6 ms →
**170 tok/s**; with k_eff 6.5: 221; with k_eff 6.5 and a 0.67: 4.1 GB → 19.3 ms → **271 tok/s, 0 MB/token**.
Same DRAM ceiling as B (pruning removes SSD traffic, not bytes per token); 300 needs the same two-of-four.
12 GB stretch would need ~80% pruning — not credible. Mode: **FAST**.

**Biggest risk.** Binary: REAP's "≤ 2%" is at 50% and on code; 60% and math/chat are off the published
curve (a community 20% prune of a different Qwen3-Coder base lost ~2 pts overall and ~7 pts on math
[K, secondary: https://huggingface.co/cerebras/Qwen3-Coder-REAP-25B-A3B and third-party cards]). If
parity fails there is no dial to turn except pruning less, which puts experts back on the SSD and turns C
into B. Also C can never serve EXACT mode.

**Cheapest test — T4, pruning head-room from the T1 trace (CPU only, minutes).** From the dumped gates and
expert outputs of the T1 session, compute REAP's saliency per expert and the fraction of gate-weighted
output norm carried by the top-40% experts of each layer, on code tokens and (second dump) on a 1K-token
math/chat snippet. **PASS: ≥ 95% of saliency mass in the top 40% in every layer for both domains (→ 60%
pruning plausibly ≤ 2% Δ). KILL: < 85% in any three layers, or code/math top-40% sets overlap < 60%
(the kept set cannot serve both).** Then the real check is T3's agreement test on the pruned model.

### Shared pre-tests (run before any kernel is written)
- **T2 — bandwidth reality (≤ 10 min GPU):** Metal microbenchmark: gather 1,800 random 1.84 MB expert
  blobs from a 19 GB buffer and run 3-bit dequant-GEMV for n = 5 positions, plus one 1.55 GB trunk pass.
  **PASS ≥ 220 GB/s sustained (η ≥ 0.77); KILL < 170 GB/s (then even B tops out ~190 and the first job is
  the kernel, not the architecture).**
- **T5 — MTP acceptance on code (CPU, from the T1 trace):** teacher-force the MTP block at depths 1-6.
  **PASS α₁ ≥ 0.85 and A(k=4) ≥ 3.5; KILL α₁ < 0.75** (published anchors: DeepSeek-V3's MTP second-token
  acceptance 85-90% [K: https://arxiv.org/abs/2412.19437]; Qwen3-Next 0.906 per-position at one draft
  token [K: https://github.com/ROCm/ATOM/pull/239] and mean acceptance length 4.32 on coding at draft
  length 7 [K: https://arxiv.org/pdf/2604.12374]).
- **T6 — SSD (CPU only):** random 1.84 MB F_NOCACHE reads at QD 8-32 from a 50 GB file.
  **PASS ≥ 8 GB/s with p99 ≤ 1.5 ms; KILL < 4 GB/s** (A's miss tolerance halves).

---

## 3. The bet, and why

**Design B.** Reasons, in order:

1. **The physics leaves no EXACT path to 300.** With infinite RAM, EXACT is ~130-170 tok/s because every
   drafted token pulls ~10 experts per layer; the brief's target is therefore a FAST target by necessity,
   and the question is only *which* approximations and in what order.
2. **B's approximations are continuous, separable and revertible.** Bits, τ, sparsity fraction and β are
   four dials; β = 0, τ = 0, a = 1, 3.5-bit *is* Design A, so one runtime serves both modes and parity is
   recovered one dial at a time with a measurable tok/s price each. C's single approximation is a cliff.
3. **β is the only mechanism that can meet the 10 MB/token goal** at 24 GB, because it attacks the miss
   rate at the router instead of hoping for skew; it is the same primitive the model family was trained
   with [K: loss-free balancing], so the perturbation is of a kind the weights have seen.
4. **Expected value:** central 230-270 tok/s and 5-8 MB/token; 300 iff two of {ov ≥ 0.5, η ≥ 0.9,
   60% in-expert sparsity, α ≥ 0.95}. C has the same ceiling with a worse failure mode; A caps at ~130.
5. **Cost of being wrong is small and early:** T1 (one slow 3K-token run, then CPU) decides A vs B vs C in
   under ten minutes of GPU, and T3 decides parity before any runtime work beyond what A needs anyway.

Order of work: T1 → T2 → T5 (one evening, mostly CPU) → build A's runtime (GPU-driven step, gather kernel,
async miss queue) → add the dials → T3 → benchmarks.

What would change my mind: T1 showing LFU miss ≤ 2% at 10,500 slots on a real session (then A alone gives
~120 tok/s EXACT and ~20 MB/token, and B's β is nearly free); or T2 showing η < 0.6 (then the kernel is the
whole project and the designs are moot until it is fixed).

---

## A. Appendix

### A.1 Parameters of the step model [P][G]

| symbol | value | basis |
|---|---|---|
| e | 1.84 / 2.15 MB | 4.92 M params at 3.0 / 3.5 bit [P] |
| D_trunk | 1.55 / 1.8 GB | 4.1 B non-routed params at 3.0 / 3.5 bit [P]; the brief's 125 B total implies the upper GDN head count |
| d_draft | 85-95 MB | MTP block ~30 MB + 10 resident experts 18-22 MB + shared 2 MB + sub-vocab head 41 MB [P] |
| η | 0.80 (0.9 optimistic) | fraction of 286 GB/s reached by streaming quantized small-M kernels on Apple GPUs [G] |
| T_gap | 3 ms plain / 1.5 ms fused | 48 layers x 6-12 kernels x gap + draft kernels [G] |
| α | 0.88 code (0.85-0.95 range) | DeepSeek-V3 MTP 85-90% [K]; Qwen3-Next 0.906 at depth 1 [K]; depth decay [G] |
| ov | 0.3 (0.0-0.5) | low local routing consistency for shared-expert fine-grained MoEs [K: 2505.16056] → [G] |
| r | 0.8 | share of misses prefetched before need; next-layer recall ~90% published [K], multi-layer [G] |
| stall | 0.25 ms per unpredicted-miss layer + bytes/12.5 GB/s | SSD latency + transfer [G] |
| k_eff | 6.5 (FAST) | gate-threshold top-k [G]; 10 in EXACT |
| a | 0.67 (FAST, 50% in-expert sparsity) / 0.53 (60%) | TEAL/CATS-style [K] → read factor [P] |

### A.2 Things I considered and rejected
- **Tree speculation**: every node costs 10 experts/layer; wrong trade here [P].
- **Early-exit pruning of doomed draft positions mid-verify (EXACT-safe)**: saves only the deeper layers of
  positions after the first rejection, ~13-20% of expert bytes at best, needs a trained early acceptance
  classifier; a second-tier optimization, not a design [P][G].
- **2-bit "shadow" copies of the cold tail in RAM (HOBBIT-style substitution)**: at 24 GB there is no room
  (all experts at 2-bit = 30 GB); at 12 GB a 2-bit tail of ~3,000 experts covers 20% total — still far
  from 99% hits [P].
- **Shared-basis / low-rank expert compression**: unproven at the needed ratio; parity risk higher than
  pruning [G].
- **ANE/CPU as extra bandwidth**: same DRAM; at most ~+10% aggregate utilization, and only if the GPU
  alone does not saturate the fabric [G]. Useful only to hide the draft (§1.5).
- **Vocabulary-restricted verify**: EXACT needs full logits; in FAST saves ~0.2 GB/step (~1 ms) — minor [P].
- **12 GB stretch**: needs ~99% hits at 13-16% residency or ~80% pruning; β at that residency displaces
  ≥ 20% of gate mass [G] → fails T3. Not claimed.

### A.3 Sources [K]
- Qwen3.5-122B-A10B config — https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json
- Qwen3-Next model card (512 experts, top-10, 1 shared, MTP) — https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Apple M5 Pro/Max (307 GB/s, "over 4x the peak GPU compute for AI") — https://www.macrumors.com/2026/03/03/apple-debuts-m5-pro-and-m5-max-chips/
- DeepSeek-V3 technical report, MTP acceptance 85-90% — https://arxiv.org/abs/2412.19437
- Nemotron 3 Super (SPEED-Bench acceptance lengths incl. Qwen3-Next coding 4.32) — https://arxiv.org/pdf/2604.12374
- Qwen3-Next per-position acceptance 0.906 (vLLM/ATOM PR) — https://github.com/ROCm/ATOM/pull/239
- Auxiliary-loss-free load balancing (selection bias primitive) — https://arxiv.org/abs/2408.15664
- Not All Models Suit Expert Offloading (local routing consistency, shared experts) — https://arxiv.org/abs/2505.16056
- HOBBIT (mixed-precision substitution, prefetch accuracy, cache policy) — https://arxiv.org/abs/2411.01433
- Speculating Experts (parameter-free next-layer expert prediction, ~90% recall) — https://arxiv.org/abs/2603.19289
- FATE / cross-layer gate (97% prefetch accuracy) — https://arxiv.org/abs/2502.12224
- MoE-Infinity (request-level reuse, evens out across requests) — https://arxiv.org/abs/2401.14361
- FlashMoE (ML cache replacement for SSD expert offloading) — https://arxiv.org/abs/2601.17063
- TEAL, training-free activation sparsity — https://arxiv.org/abs/2408.14690
- REAP, expert pruning — https://arxiv.org/abs/2510.13999 and https://www.cerebras.ai/blog/reap
- Not All Experts are Equal (dynamic expert skipping) — https://arxiv.org/abs/2402.14800
- FR-Spec (sub-vocabulary drafting) — https://arxiv.org/abs/2502.14856
