# Prior art for Thursday's passes and the lossy-350 leads, written for the paper (MoonShots, Thu 2026-11-05 18:5x EET)

Ask (the orchestrator, widened 18:5x to seven leads; sources ~/FableGod/findings/2026-11-05_1700_lossy350-thinking-hour.md and
the LOSSY350_* / WALL_* files it lists): a web prior-art check of each lead, with the nearest papers or
repos, what is the same, what is new, and a tag (published / adapted / new / unchecked).

Method: web search (standard), abstracts, plus one full-text read (AcceptMoE); this project's own records (TRIED,
NOVELTY.md and its Tuesday sweep). No downloads, no GPU.

## 1. The split form: per-band resident keeps keyed on the drafter's confidence

**What we did** (Fable C's audit test 1, SPLIT90; Breakthrough ~/Breakthrough/keepset/qom_keep.py, strict_rows /
block(split=T)). It is a lossy verifier rule for a 16-row speculative check of an SSD-streamed MoE. Per row and layer,
the rule depends on the drafter's confidence cf in the draft that row verifies:
- **cf >= T (T = 0.9 or 0.99):** the row is computed strictly: its top-2 experts plus ranks 2-9 already in the
  block's loads (the union of the rows' required experts), with no cache-resident extras.
- **0 <= cf < T:** QOM's rule: top-2 plus ranks 2-9 that are in the block's loads OR resident in the layer's pool
  (cache-conditional keeps).
- **No draft (the bonus row without DRAFT17):** exact, all 10 ranks.

Kept experts keep the model's own routing weights. There is no rerouting and no renormalisation. On the decision map,
code passes: 0.92 % flips at T 0.99 and 1.31 % at T 0.9, against the 1.34 % line (twin 0.51 %). Breakthrough
2da53f2 / 4963da6.

**Nearest work**

| work | what it does | same | different |
|---|---|---|---|
| AcceptMoE (arXiv 2608.02989; full text read) | verifier-side expert selection for MoE speculative decoding: ONE eligible expert set per layer and verification block, ranked by target-router scores weighted by offline position-level commitment probabilities; under offloading, nonresident experts are pruned from the shared set; tokens whose experts were removed are rerouted | restricts the verifier's experts; cache residency conditions eligibility; demand weighted by how likely a position is committed | block-wide set, no per-row rule; residency pruning applies to every position; offline position estimates, not the drafter's per-row confidence; reroutes. It has no bands and no strict-vs-resident split (checked in the full text) |
| PSR (arXiv 2610.34721) | routing conditioned on a deterministic logical cache (next-layer router predictions inserted); section 7 (our Tue sweep, NOVELTY.md) weights the verify block's expert demand by the drafter's running confidence and builds one set per block | drafter confidence as a weight; cache-conditioned routing | one set per block, rerouted; confidence is a continuous weight, not a band that switches the residency clause |
| OEA, Opportunistic Expert Activation (arXiv 2511.02237) | each token keeps top-k0 and adds lower ranks only if the batch already loads them | the union step (our strict band = top-2 + the block's loads) | batch-level, no confidence, no residency |
| Mixture of Cache-Conditional Experts (arXiv 2412.00099); Cache-Aware Joint Router Adaptation (arXiv 2609.04895) | boosts the logits of cached experts (training-free) / trains the router for cache coverage | cache-conditional selection | applied to every token, not gated by the drafter's confidence; reroutes |
| Dynamic routing by confidence (Huang et al., ACL 2024 "Harder Task Needs More Experts"); AdapMoE | per-token expert count from the ROUTER's own confidence | confidence-dependent expert count | the router's confidence, not the drafter's; no cache-residency clause |
| LExI (arXiv 2509.02753, ICML 2026) | a static per-layer top-k found offline | fewer active experts | per layer, static; no confidence, no residency |
| DraftExpert (2607.24434), EcoSpec (2607.12696), EVICT (2605.00342), Cascade (2506.20675) | lossless: choose or truncate DRAFTS by expected new verifier experts | drafter confidence and expert cost | the verifier stays exact; nothing is removed from a row's experts |

**Tag: adapted.** Every piece is published: drafter-confidence weighting (PSR section 7), cache-conditional verifier
sets (AcceptMoE, cache-conditional routing), the union step (OEA). What we found nowhere is the per-band clause:
- the drafter's PER-ROW confidence switches the residency clause;
- the surest rows get the strict union, the less sure rows also keep residents, undrafted rows stay exact;
- there is no rerouting.

That clause is the candidate new part. The checks behind "not found" are AcceptMoE's full text, PSR's abstract plus our
earlier full read, and the abstracts of the rest. A full-text pass over cache-conditional routing and MoE-Spec
(2602.16052) would firm it up.

## 2. NAX6-N: a fragment-native, coalesced expert-record layout for Apple's M5 matrix units

**What we did** (Leap N7-13, ~/Leap/data/n7_native.txt; P11 / P11PAD; BigMoE's conversion).
- **Layout:** the 4-bit expert codes (MSB-3 + LSB sidecar) are stored in the order the M5 GPU's matrix units
  ("neural accelerators", reached through Metal 4 tensor ops) read their fragments. Loads are coalesced, and the
  scales are packed to 11 bits.
- **Speed:** +18.7 % per expert-kernel call at 16 rows (quiet, layer 20; +16-18 % at 8-32 rows).
- **Exactness:** bitwise equal to the stock-layout kernel (64 / 64 traces). EvalSuite certified NAX6 as the 45 GB exact
  reference Thursday 14:48.
- **Store:** BigMoE converted the SSD expert store in place to native+p11+pad (4 KB-aligned pieces, md5s logged,
  reversible with --inverse) on Thursday at 10:2x.

**Nearest work**

| work | what it does | same | different |
|---|---|---|---|
| Marlin (IST-DASLab, github.com/IST-DASLab/marlin; Frantar et al. 2024) | 4-bit weights and group scales reshuffled offline into the tensor-core fragment organisation (mma.sync m16n8k16), async coalesced loads | the core idea: store weights in fragment read order, pre-permute scales | NVIDIA tensor cores; weights resident in GPU memory |
| MoE Marlin (vLLM fused_moe marlin; NVIDIA TensorRT-Edge-LLM moeMarlin: expert weights must be pre-swizzled with awq_marlin_repack) | grouped MoE GEMM on pre-swizzled 4-bit expert weights | fragment-native layout for MoE experts | NVIDIA; experts in VRAM, not records streamed from SSD |
| Modular MAX PreShuffledBGroupedGEMM (AMD mxfp4 grouped matmul) | pre-shuffled B for MoE grouped GEMM | the same idea on AMD | AMD; mxfp4 |
| llama.cpp ARM repacking (Q4_0 -> 4x4 / 4x8 / 8x8 interleaved, i8mm / SVE) | weights repacked at load time into the CPU matrix instructions' order | layout for the matrix instruction | CPU, at load time, not stored |
| Apple M5 / Metal 4: MLX's NAX path (0.30.0, PR #2772; split-K PR #3018); WWDC26 session 330 (matmul2d; for custom quantised formats "dequantize into a cooperative tensor"); BaseRT (arXiv 2607.19438: Metal 4 tensor kernels for prefill / MoE GEMM); Cider (INT8 matmul2d prefill); third-party int4 NAX kernels (metaltile-std mt_qmm_nax) | M5 matrix-unit GEMMs, including quantised and MoE | the same hardware path | none describes an offline fragment-native weight layout for the matrix units; Apple's own guidance is to dequantize into a cooperative tensor (abstracts and search only; MLX's NAX kernel source and BaseRT's full text not read) |

**Tag: adapted** (a new setting of published practice). The idea, storing quantised weights in the matrix units'
fragment order, is Marlin's (2024), already used for MoE experts on NVIDIA and AMD. We found no public offline
fragment-native layout for Apple M5's matrix units. Nor did we find one for expert records streamed from an SSD, stored
in that order and converted in place, with the kernel bit-identical to the stock one. The Apple-side "not found" is
from abstracts and search, so it is **unchecked at the source level** (MLX's NAX kernels, BaseRT's full text). The
in-place SSD conversion is engineering.

## 3. GOAL's INERT: wholesale acceptance of drafted comments and docstrings

**The rule** (~/FableGod/goal/LOSSY350_GOAL.md 3.3).
- The causal tagger (R57_RELAX_SPAN's, plus a docstring state) puts a row inside a '#' comment or a docstring. On such a
  row the draft is kept whatever the 125B's argmax is.
- The span's edges stay exact (a drafted '#' or quote, the newline or quote that ends the span, tool comments, `>>>`
  lines).
- Inside a span the check runs past 16 rows, as long as the drafter's block lasts.
- [P], GOAL's replay: HumanEval+ 24 windows tok/s x1.27-1.30 / x1.47-1.53 / x1.50 at 16 / 32 / 64 rows (fresh 12: x1.18 /
  x1.30-1.35); MBPP x1.63-1.65 / x1.79-1.87 / x1.78. The drafter writes 14-40 % (HumanEval+) / 33-45 % (MBPP) of the tokens
  unverified, 0 of them code.

**Nearest work**
- Lossy verification in general: Judge Decoding (ICLR 2025), AutoJudge (2504.20039; it learns which mismatches matter,
  on code too), typical acceptance (Medusa), Fuzzy SD.
- Region- or syntax-aware relaxation: LinguaSpec (ACL 2026 Findings; looser verification for structural words in prose).
- Small-model-generates schemes: BiLD (big-little decoder: the small model's tokens stand unless a fallback or rollback
  fires).
- Our own rule C (M-41: likelihood-gated, comments only).
- The web found no scheme that accepts drafted comment or docstring tokens without the target's verdict.

**Tag: adapted, the span rule new (not found).** Lossy and unverified acceptance is published; skipping the target's
verdict inside syntax-tagged comment and docstring spans, with exact edges and a check that runs to the span's end, was
not found. Risk to state: it changes text the target never vouched for (docstrings are user-visible), and decision 4 is
the judge.

## 4. FULL's comment-free FAST mode, and the drafter-miss share of comments

**The mode** (~/FableGod/full/LOSSY350_FULL.md, Part B): the 125B never opens a '#' comment inside a code block, enforced
by a logit mask.
- **Measured [K]:** comment tokens are 13.3 % of HumanEval+'s generated tokens (MBPP 21.2 %) but carry **52 % of all
  first misses** (hazard 0.169 per comment row vs 0.030 on other code; Mission 25: 44.7 % of the drafter's misses in
  13.5 % of positions).
- **[P]:** kept per check x1.17-1.31, MBPP x1.11.

**Nearest work**
- Banning tokens by a logit mask is published practice: constrained or grammar-constrained decoding (llama.cpp grammars,
  Outlines, guidance). Asking a model for comment-free or concise code is common prompting.
- Acceptance by content: studies stop at the task level (e.g. Acceptance Dynamics Across Cognitive Domains, arXiv
  2604.14682: code / math / logic / chat) or code-vs-prose drafter comparisons. None splits drafter misses by region
  inside code.

**Tags:**
- **The mode: adapted.** The mask is published; using it to remove the drafter's hardest region for speed was not found.
- **The miss-share measurement: new** (no source found). Comments are a small share of code tokens but about half the
  drafter's misses.

## 5. FULL's "two speeds" reading of HumanEval tok/s

**The reading** (LOSSY350_FULL.md section 0, findings note).
- 61-68 % of a HumanEval answer's generated tokens are ONE verbatim copy run of the prompt (the re-emitted signature and
  docstring, exactly one run per answer). MBPP, math and IFEval copy 1-3 %.
- So today's FAST code number is two speeds: ~155 tok/s on the copy and ~72 on the rest. A 350 claim must name its
  workload.

**Nearest work**
- Copying from the prompt is published, with HumanEval among the showcase tasks: prompt-lookup decoding (its README
  notes coding's large gain from copying), CopySpec (EMNLP 2025, evaluated on HumanEval), REST (2.12-2.36x on HumanEval),
  multi-lookup methods (ACL Findings 2025, up to 4.23x on HumanEval), "To Copy or Not to Copy" (arXiv 2609.20186).
- No source found that measures how much of a HumanEval answer is a re-emitted prompt, or reads reported HumanEval tok/s
  as a copy speed plus a decision speed.

**Tag: new as a measurement and reading** (one search; a critique may exist in benchmark-methodology venues:
unchecked there). The copy mechanism behind it is published.

## 6. Never-wait: KTransformers' expert deferral, re-opened at 45 GB

**The idea** (~/FableGod/far/NEVERWAIT_TEST.md; ZERO's LATE-INJECT): a check never waits for an SSD read. A demand-miss
expert runs when its bytes arrive, on the saved layer-L input, and its term enters the residual one layer late.
- At the 45 GB FAST preset only ranks 0-1's demand misses remain on the path (1-2 % of picks).
- [P] 95 -> 112-120 tok/s if the wait is what it costs (FAR-WAIT: ~10-16 ms floor + 0.16 ms per miss).

**Nearest work**
- KTransformers' Expert Deferral (SOSP '25, Chen et al., Tsinghua MADSys; upstreamed to SGLang): deferred experts run
  concurrently with the next layer's computation. The authors report ~1.45x throughput at < 0.5 % average accuracy loss.
- Our own deferral family: B12LL (ADOPTED at 12 GB), DH_W4 (quality PASS at 24 GB), LATE RANK 1 / DHW4 / I4 KILLs.

**Tag: published mechanism, adapted setting** (as FAR tagged it: "adapted, re-opened, not new"). The open part is the
pricing at 45 GB, where the only reads left on the path are rank 0-1 demand misses from an SSD rather than CPU experts.

## 7. Fable C's tie-floor reading of the map's bar

**The reading** (~/FableGod/fablec/WALL_1105.md 2): the 125B's near-tie rows (p1 - p2 <= 0.1: 1.5 / 8.5 / 7.0 % of
code / IFEval / multi rows) flip 20-45 % under every perturbation.
- 4-bit attention alone gives 0.92 / 4.53 / 3.43 % all-row flips against lines of 1.34 / 5.42 / 3.69.
- So all-row flips mostly measure coin-flip rows. The decision map should report non-tie flips beside them (the lines'
  own arm: 0.87 / 2.22 / 1.63 non-tie).

**Nearest work**
- Flips as a compression-quality metric, paired with KL: Dutta et al., "Accuracy is Not All You Need" (arXiv
  2407.09141). Top-token agreement and KL tools: mlx-quant-fidelity and friends, KLD-guided quantisation write-ups (they
  note the damage sits in the tail).
- The fragility of near-tie argmax decisions to tiny numeric changes is known in the nondeterminism / batch-invariance
  literature (the source of REF2-T's batch invariance).
- No source found that stratifies flips by the reference's p1 - p2 margin, or proposes non-tie flips as the bar.

**Tag: adapted.** The flips metric is published and tie fragility is known. The margin-stratified (non-tie) flip rate
as the quality bar is the candidate new part (one search: unchecked beyond it).

## Sources
- AcceptMoE https://arxiv.org/abs/2608.02989 (full text https://arxiv.org/html/2608.02989)
- PSR https://arxiv.org/abs/2610.34721
- OEA https://arxiv.org/abs/2511.02237
- Mixture of Cache-Conditional Experts https://arxiv.org/abs/2412.00099
- Cache-Aware Joint Router Adaptation https://arxiv.org/abs/2609.04895
- Harder Task Needs More Experts https://aclanthology.org/2024.acl-long.696
- LExI https://arxiv.org/abs/2509.02753
- DraftExpert https://arxiv.org/abs/2607.24434 ; EcoSpec https://arxiv.org/abs/2607.12696 ;
  EVICT https://arxiv.org/abs/2605.00342 ; Cascade https://arxiv.org/abs/2506.20675 ; MoE-Spec https://arxiv.org/abs/2602.16052
- Marlin https://github.com/IST-DASLab/marlin
- TensorRT-Edge-LLM MoE Marlin https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/cpp_api/kernels/moeMarlin.html
- Modular MAX https://docs.modular.com/api/kernels/linalg/matmul/gpu/amd/mxfp4_grouped_matmul_amd/PreShuffledBGroupedGEMM/
- MLX NAX https://github.com/ml-explore/mlx/pull/3018 ; Apple MLX on M5 https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- BaseRT https://arxiv.org/abs/2607.19438 ; WWDC26 session 330 summary https://nonstrict.eu/wwdcindex/wwdc2026/330/
- Leads 3-7: Judge Decoding https://iclr.cc/virtual/2025/oral/31778 ; AutoJudge https://arxiv.org/abs/2504.20039 ;
  Acceptance Dynamics https://arxiv.org/abs/2604.14682 ; prompt lookup https://github.com/apoorvumang/prompt-lookup-decoding ;
  CopySpec https://aclanthology.org/2025.emnlp-main.1337 ; REST https://arxiv.org/abs/2311.08252 ; To Copy or Not to Copy
  https://arxiv.org/abs/2609.20186 ; KTransformers SOSP '25 https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/ ;
  Accuracy is Not All You Need https://arxiv.org/abs/2407.09141
