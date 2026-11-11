# BREAKTHROUGH PUSH: prior art and tags (NewIdeas, Tue 3 Nov 2026, 17:2x)

This covers the four bet families in ~/FableGod/push/BREAKTHROUGH_PUSH.md: staged / cascade verification, lossless weight
compression, expert-major scheduling, and CPU-SME / Neural-Engine co-compute. It also covers two neighbours found on the
way: cache-aware routing and bit-sliced caching.

- Sources: web search 17:1x. IDs marked (mem) come from memory and were not re-checked today.
- Internal results are cited from ~/TRIED/TRIED.md (merged 16:48).
- Tags: published (the method exists as is) / adapted (a published method changed for our setting) / new (no
  published match found).
- Numbers are tagged [K] measured, [P] arithmetic, or [G] judgment.

## The physics every bet below has to pay

Each verify pass costs:
- The dense part, read once per pass: 3.44 GB, so >= ~11.5 ms at ~300 GB/s [P] (TRIED D-07).
- The union of experts the pass touches: ~2.2 MB (4-bit) / ~1.6 MB (3-bit) per distinct expert per layer, in RAM
  bandwidth even when resident.
- SSD misses: 42.0 ms busy per check at C 256 [K].

So any design that adds a pass pays ~11.5 ms of dense reads [P], and any design that widens the rows pays for the union
growth. That is why trees price out today (the push's table). A second pass pays only if it removes far more SSD or
union bytes than ~11.5 ms.

## 1. Staged / cascade verification

### Published

| Work | What it does | Relevance to us |
|---|---|---|
| Staged SD, Spector & Re 2023 (2308.04623) | Tree-shaped speculative batch + a second speculative stage for the DRAFTER (staging is on the draft side) | Tree verify on a dense GPT-2 at batch 1; the target still verifies the whole tree |
| Cascade Speculative Drafting, Chen et al. 2023 (2312.11462, mem) | Vertical cascade (drafters drafted by smaller drafters) + horizontal cascade (cheaper drafters for later, less likely tokens) | Drafter side only |
| Sequoia, Chen et al. 2024 (2402.12374, mem) | Hardware-aware DP for the optimal tree shape; offloading results (Llama2-70B on one L40) | Dense model: with offloading, a bigger tree costs almost nothing. MoE breaks that premise (the union grows with rows) |
| SpecExec, Svirschevski et al. NeurIPS 2024 (2406.02532) | Huge draft trees (top-k parallel search) verified in one target pass on offloaded consumer GPUs; up to ~20 tokens per target pass; 4.6-18.7x vs offloaded AR (Together's write-up) | Same premise as Sequoia: dense weights cost the same for 1 or 1,000 rows. For a 512-expert MoE every extra row pulls new experts, so this does not carry over |
| TriForce, Sun et al. 2024 (2404.11912, mem) | Hierarchical: a tiny model drafts for the target with a retrieval-based partial KV cache, which in turn drafts for the full target | The intermediate stage is the target's own weights with cheap attention. Long-context oriented |
| Self-speculation: Draft & Verify (2309.08168, mem), LayerSkip (2404.16710, mem), Kangaroo (2404.18911, mem), SWIFT (2410.06916, mem) | Early exit / layer skipping of the target as its own drafter | Internal KILLs: D-07 SELF (an AR step reads the 3.44 GB dense part: 15 steps ~170 ms vs a ~7 ms budget); layer-skip on a 3B-active MoE ~1.0-1.2x; L1-09 early-exit verify KILL (exit after 40 disagrees too often) |
| HiSpec, Kumar et al. 2025 (2510.01336) | An early-exit INTERMEDIATE VERIFIER checks drafts cheaply; the full target re-checks periodically (dynamic policy) | The closest published match to "a cheap pass picks, an exact pass checks". Its cheap pass is a layer prefix; ours would be an expert-budgeted full-depth pass |
| HSD, 2025 (2510.19705) | A stack of drafters, each verifying the one below; hierarchy chosen as a shortest path | Drafter side |
| MoE-Spec (2602.16052) | Per-layer expert budget during verification (drop low-gate experts); training-free | The FINAL verify is budgeted (lossy), not a staged pick + exact pass. Our Q1 family is the same idea gated by drafter cf |
| AcceptMoE (2608.02989) | Per-position commitment-weighted verifier expert sets; treats token count, union size and weight traffic as separate costs; transfer cost = non-resident experts | Prices verification by the union and residency, as FLOOR does |
| EcoSpec (2607.12696) | Adds the predicted marginal expert-activation cost to draft selection (up to 1.62x) | Cost-aware tree pruning: a cheap way to keep the tree's union small |
| Utility-Driven SD for MoE (2506.20675); The Limits of Speculation (2609.22156) | The expert union grows with speculation length; verification cost is input-dependent | Background; matches our row-cost finding |

### Internal

- Rnd19#8 "cascaded verification with the resident MSB model, exact verify only the surviving prefix": KILL in the 64
  GB era (napkin). At ~1.6 tokens per pass the exact pass runs every pass anyway.
- T3 at 12 GB is a staged check: pass 1 = top-2 on 16 rows (P1-K1: top-1 + union members, x0.465-0.485 of the bytes);
  certify p1 >= 0.8; pass 2 exact only where uncertified. Certificate PASS [K]: P1-K1 7/2997 = control 7/3019; 3-bit
  pass 1 4/2970.
- The push's TOP3-USE / DEPTH32 / DIVERSE2 / ROUTEPRED results show trees price out at ~7.82 ms per row [K].

### Tag for the push bet

The push bet: "a cheap pass over a wide top-3 tree picks the path, an exact pass checks only that path".
- Tag: ADAPTED. Staged verification with a cheap intermediate verifier is published (HiSpec; TriForce's middle stage),
  and expert budgets in MoE verification are published (MoE-Spec, AcceptMoE).
- New in our version: the cheap pass is the 125B at full depth with an expert budget (top-1 + union members, resident
  only, 3 bits), over a top-3 tree. It only selects; the exact pass verifies one path.
- No published work does path selection over a tree with an expert-budgeted MoE pass followed by an exact pass on the
  chosen path. HiSpec is the nearest.
- The bet must clear two costs: the second dense read (~11.5 ms [P]), and the tree's union in pass 1. The top-1 union
  of a 48-row tree grows sub-linearly; FLOOR's per-block union method prices it.

## 2. Lossless weight compression

### Published

| Work | What it does | Relevance |
|---|---|---|
| DFloat11 (2504.11651, NeurIPS 2025) | Huffman on BF16 exponents, GPU LUT decode in shared memory | BF16 only. Our weights are 3/4-bit codes + bf16 scales |
| EntroLLM (2505.02380) | Huffman on quantized integer weights, parallel decode, edge | The same class as our records |
| ZipNN (2411.05239, mem) | Storage / transfer compression | Decode before use; no inference speedup |
| ECF8 (2025) | Entropy-coded FP8 with GPU decode | FP8 only |
| Tile-level ANS, "Approaching Shannon Bound with Lossless LLM Weight Compression" (2606.15789) | Near-Shannon, random access per tile, parallel GPU decode; up to 1.6x throughput in SGLang | The best published decoder design for our case. Its gain is RAM-bandwidth on a GPU server, not SSD |
| Shared base + delta: D2-MoE (2502.17298, ICML 2025), "MoE trilemma" (2510.02345) | Fisher-weighted / averaged shared base + SVD or low-rank deltas per expert | LOSSY at 40-60 % compression. As lossless storage, a delta only pays if it has lower entropy than the expert |

### Internal (MoonShots)

- 3-bit codes sit at a lossless floor of 2.72 b. The whole 3-bit record floor is 2.98 of 3.5 bits per weight, so at
  most -15 % of bytes [K] (W24-floor).
- Context models gain 0.0035 b (R-23 KILL). Order-1 coding gains 0.00-0.05 b. LSB planes are ~uniform.
- Quantized experts are full rank, and a shared basis gains <= 5 % (napkin KILL).
- GPU rANS fused into the matvec must decode >= 150-200 G symbols/s to break even (C1 / Q1 O1 KILL).
- Net bandwidth estimates: +6.3 % (upper +8.1 %) on the 12 GB AR decode (R-19 BETWEEN).

### Tag and what would be new

- Tag: PUBLISHED. Entropy coding of quantized weights is EntroLLM; tile ANS on a GPU is 2606.15789.
- The only unexplored corner is decoding ONCE per SSD miss into the uncompressed resident slot, not on every use. The
  decoder then needs only the SSD rate: ~11 GB/s compressed, i.e. ~30 G 3-bit symbols/s [P], not 150-200 G.
- Ceiling: -15 % of SSD bytes at the floor [K], about -6 ms of the 42 ms SSD time per check, so <= ~+7 % tok/s
  [P, halved ~+3-4 %]. It is exact, but not a revolution. MoonShots' CPU measurement of records3 settles the floor on
  the real records.

## 3. Expert-major scheduling across checks

### Published

| Work | What it does | Relevance |
|---|---|---|
| FlexGen (2303.06865, mem) | Zig-zag block schedule: run many batches through one layer's weights before moving on | Throughput, many sequences |
| MoE-Lightning (2411.11217, mem) | CPU-GPU-I/O pipeline with paged weights for batch MoE inference | Throughput |
| Klotski (2502.06888) | Expert-aware multi-batch pipeline: more batches lengthen a layer's compute to hide the next layer's loading; correlation-aware prefetch; "near-zero bubble" | The textbook expert-major schedule. It needs several independent batches |
| PEARL (2408.11850, mem) | Draft the next window while the target verifies (pre-verify / post-verify) | The only single-stream way to get a second batch: speculative future checks |
| Parallel parts of one answer: Skeleton-of-Thought (2307.15337, mem), APAR (2401.06761, mem), Hogwild! Inference (2504.06261, mem) | One request decodes several independent parts at once | Our L1-32 intra-request parallelism: BETWEEN (resident; 0.59x of the expert bytes per token; whether that counts as "one chat" is Kamen's call) |

### Internal

- ANCHORCOV KILL (the ANE pre-drafting the next block from 3 anchors).
- EARLYTAP KILL (early-tap drafter for drafting during the check).
- ROUTEPRED (next-check expert prediction): precision 0.07 [K].
- Runahead during stalls KILL (125B replay: realistic -10 %).
- OOO within a check is ADOPTED.

### Tag

- Tag: PUBLISHED for batches (Klotski / FlexGen); ADAPTED for one chat.
- One chat has no second independent batch. Check n+1's rows depend on check n's tokens, and its routing is
  unpredictable (0.07).
- Only two sources of independent rows exist:
  - (a) parallel parts of one answer (L1-32, Hogwild-style). This changes the output, so it is FAST at best;
  - (b) speculative future checks (PEARL), which we have killed twice (ANCHORCOV, EARLYTAP).
- A genuinely new expert-major design must name a third source of independent rows.

## 4. CPU-SME / Neural-Engine co-compute

### Published

| Work | What it does | Relevance |
|---|---|---|
| Hello SME, Remke & Breuer (SC-W 2024; 2409.18779, mem) | Generated SME GEMM kernels on M4; ~2 TFLOPS FP32 per cluster | M4 replaced AMX with SME / SME2; the M5 Pro has it too |
| SMEPilot (2606.16332) | Characterizes and optimizes LLM inference with SME | Published characterization; CPU only |
| Ghidorah (2505.23219) | Speculative decoding + hetero-core model parallelism (CPU + GPU) on a unified-memory device (Jetson NX), up to 7.6x vs sequential | The direct analog: split the verify work across CPU and GPU on unified memory |
| HeteroLLM (2025, mem), PowerInfer-2 (2406.06282, mem), llm.npu (2407.05858, mem) | GPU + NPU / CPU + NPU split on mobile SoCs | Mostly prefill; NPU dispatch costs are the recurring limit |

### Internal

- #6 ANE + GPU co-prefill: PARKED. ANE dispatch costs too much per layer.
- Rnd16 / 17 #6 ANE as a second GEMM engine: KILL (unsure).
- ANCHORCOV / EARLYTAP: KILL.

### Physics and tag

- Decode is memory-bandwidth-bound, and every engine shares the same DRAM. So co-compute adds FLOPs, not bandwidth.
- It can pay in only two places:
  - (a) during the GPU's 42.6 ms of gate spin per check [K], when DRAM is idle. E.g. the CPU (SME) runs work that
    does not depend on the missing experts: the next layer's attention for rows whose experts are all resident,
    or a drafter step;
  - (b) prefill, which is compute-bound.
- Tag: PUBLISHED (Ghidorah is CPU+GPU speculative decoding on unified memory); ADAPTED for "use the SSD-wait gap".
- The new part would be to schedule CPU-SME work only into the gate-spin windows. The bound is what can run without
  the missing experts. That is the same dependency wall as OOO and runahead (KILL, -10 % realistic).

## 5. Neighbours found on the way (relevant to live bets)

- Cache-aware routing (FAST): Mixture of Cache-Conditional Experts (Skliar et al., 2412.00099), BuddyMoE
  (substitution), "Cache-Aware Routing ... RL on hardware features" (NeurIPS 2026). All LOSSY on every token.
  - A 2026 measurement study on Qwen3-30B: every config failed a 1 % perplexity gate; ~80 % miss cut at <= 3.4 % ppl.
  - Our Q1-ON-MISS (QOM, pre-registered 554dfe0) is ADAPTED from these. It does not reroute: only drafter-sure verify
    rows drop rank 2-9 experts that would miss. The map decides.
- Eviction only, routing unchanged (EXACT): Bacon, Calanzone, Larouche (NeurIPS 2026 AXIOM workshop).
  - 2.1x fewer transfers vs LRU, within 3.5 % of the best schedule.
  - BUT it needs future demand: 16 tokens of lookahead recover 96 % of the gain; past-only frequency recovers 27 %.
  - That matches our Belady bound (-33..-44 %) vs learned eviction (-8..-10 %). Future demand is exactly what
    ROUTEPRED could not predict (0.07). Nothing new for us.
- Bit-sliced caching: SliceMoE (2512.12990, DAC 2026). Caches experts as bit slices, assigns precision on demand under a
  miss-rate constraint, uses truncation-based nested quantization (cf. Matryoshka Quantization, 2502.06786, mem), and
  warms the cache in prefill.
  - This is close to our REF2-T tiers (MSB-3 base + LSB sidecar) and to PF_WARM.
  - The next QOM step ("LSB-on-miss": a rank 0-1 at 3 bits when its LSB part misses) is SliceMoE's on-demand
    precision: PUBLISHED.

## 6. Tags for the push bets (summary)

| Bet | Owner | Tag | Nearest prior art | Physics to clear |
|---|---|---|---|---|
| Staged check: cheap pass picks the path in a top-3 tree, exact pass on that path | Breakthrough | adapted | HiSpec, TriForce, MoE-Spec, our T3 | +~11.5 ms dense per extra pass; pass 1's tree union |
| Lossless expert bytes (decode once per miss) | MoonShots | published (+ small adapted corner) | EntroLLM, tile ANS 2606.15789, DFloat11 | Floor -15 % SSD bytes [K] |
| Expert-major scheduling across checks | (thinkers) | published / adapted | Klotski, FlexGen, PEARL, Hogwild! | One chat lacks independent rows |
| CPU-SME / ANE co-compute | (thinkers) | published / adapted | Ghidorah, SMEPilot, HeteroLLM | Shared DRAM; only the gate-spin gap is free |
| Q1-ON-MISS (QOM), FAST | NewIdeas (Fable C's C1-2) | adapted | Cache-conditional experts, Cache-Prior, MoE-Spec | Quality on chat; cache-state dependent outputs |
| Elastic volatile tier (PURGE-1) | BigMoE | adapted | Apple purgeable memory + victim caches | RAM pressure purges |
| LSB-on-miss | (next) | published | SliceMoE | Needs a per-(row, layer) bit mask in the harness |
