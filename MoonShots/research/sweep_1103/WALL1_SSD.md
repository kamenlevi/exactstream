# WALL 1 (SSD / memory) literature sweep, 2025-2026 (sweep_1103)

Tue 3 Nov 2026, 22:2x-23:0x EET. A MoonShots sub-agent ran it. Method: ~45 web searches (standard and extended).
Only abstracts, HTML pages and READMEs were read: no PDFs were downloaded and nothing was run.

Every item was checked against:
- the mandatory files: ~/TRIED/TRIED.md (1,065 lines at 22:58), ~/TRIED/NOVELTY.md, both PRIOR_ART.md, ~/NewIdeas/DEAD.md and
  ~/NewIdeas/RANKED_45GB.md;
- a wider set: ~/TRIED/*.md, ~/NewIdeas/*.md (incl. QUEUE.md, WALL1_SCOREBOARD.md), ~/NewIdeas/research/, ~/MoonShots/*.md,
  ~/FableGod/push/, ~/Breakthrough/research/, the BigMoE docs, and the decision-map code.

Marks:
- [K] measured, with the source named;
- [P] arithmetic on [K];
- [G] judgement.
Every gain below is an estimate unless it is marked [K].

## Bottom line
- I checked ~60 papers, preprints and write-ups from 2025-2026. For all but four of them, the mechanism is already in the
  files in substance, often with sharper numbers than the paper has.
- For EXACT wall 1, the literature reaches no floor that the team has not already reached. The covered mechanisms include:
  one record per expert, direct reads, demand-first queues, lead-1 forecasts, learned eviction, entropy coding, zero-copy
  slots, keepalive and JOINT.
- The four new items are small or conditional:
  - two are FAST-track: a byte lever and a determinism enabler;
  - two are exact refinements of D2.
  None of them is a +20 % find.
- There are also two novelty-watch notes for the papers (they are not levers) and one gap with no prior art found.

## Ranked items (expected gain x chance)

### 1. Layer-wise expert budgets for UNSURE rows and for the LSB sidecar (LExI / ASET), steered by our own P-frame map
- **Papers:**
  - LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference, arXiv 2509.02753, ICML 2026,
    https://arxiv.org/abs/2509.02753
  - "Faster MoE LLM Inference for Extremely Large Models" (static layer-wise top-k + ASET), ACL Findings 2026,
    https://aclanthology.org/2026.findings-acl.2140/
  - Supporting: EvoESAP (2603.06003) and GRAPE (2604.06542), both on non-uniform layer budgets.
- **Mechanism:** set the active-expert count per layer under one total budget, instead of the same top-k everywhere.
  - LExI is data-free: it scores layer sensitivity from the weights, then runs an evolutionary search with a min / max k
    per layer. On Qwen1.5-MoE it matches expert pruning's H100 throughput at ~10 % better accuracy.
  - ASET's static layer-wise schedules give 10-78 % throughput at minimal loss, and >= 10 % on DeepSeek-V3 with no
    measurable loss.
  - Neither paper covers offloading or verify unions.
- **Why it could move wall 1 here:**
  - The map shows that this 125B's middle layers are an order of magnitude more tolerant [K TRIED 'DM P-frame']. Reusing
    the previous position's MoE output flips code-sure tokens in:
    - 0.14 % of cases in layers 12-23 and 0.10 % in layers 24-35;
    - 22.7 % in layers 0-11 and 18.9 % in layers 36-47.
  - The union stack gives unsure rows one k0 across all 48 layers [K RANKED #32]:
    - unsure k0 8: chat x0.76, code x0.527 of the union;
    - unsure k0 6: chat x0.60, code x0.435, but it is a KILL on multi.
  - Unsure k0 6 only in layers 12-35, with k0 8 elsewhere, takes about half of that step: chat ~x0.68, code ~x0.48. That
    is **~-10 % chat and ~-9 % code SSD MB per kept token vs hyb_0.9_u8** [P, est.; it assumes the middle half carries
    half of the union bytes]. That is about +3-5 % chat tok/s, because chat's bursts are nearly all exposed [G].
  - Sidecar arm: M-26 (no LSB anywhere) gives -47 % code MB per kept token at C 256, but chat gets +4-5 points of flips
    [K TRIED 'M-26']. Dropping the LSB only in layers 12-35 should give ~-20 % code MB [P, est.] for a fraction of those
    flips.
  - The main target is chat, the domain where uniform cuts fail.
- **Already queued nearby:** FLOOR night 1 includes gate_top0_cf0999_L08_35. In that arm, cf >= 0.999 rows skip all
  experts in layers 8-35 (dm_ablations `k_layers`). It covers only the SURE rows. This item covers what no file tests yet:
  the unsure rows' k0 per layer group, and the LSB per layer group.
- **Exact?** No (FAST).
- **Screen:**
  1. CPU, ~5 min: run dm_q1u_masks.py with a per-layer k0 (u6 in 12-35, u8 elsewhere) and with a per-layer t3 mask (no LSB
     in 12-35). Feed both to floor_bytes.py for SSD MB per kept token on the map captures.
  2. Map, ~4 min per domain. `k_layers` exists for the gate; extend it to the unsure rows' k0 and to t3.
  - **PASS:** IFEval and multi all-row flips at paired z <= +1.0 vs hyb_0.9_u8 (or u8 + t3), AND chat SSD MB per kept
    token <= -8 %.
  - **KILL:** z > +1.96 on either chat domain, OR bytes > -4 %.
- **Novelty:** adapted. Layer-wise budgets are published. Applying them to a verify check's unsure rows and to an LSB
  sidecar, steered by a measured reuse map, was not found.
- **Grep evidence:**
  - 'LExI' (case-sensitive) appears only in ~/Breakthrough/research/routing_locality.md. There it is prior art for 12 GB
    "budgeted waiting", marked "Left: layer-weighted budgets at <= 10 %" and never run. '2509.02753' appears in the same
    file only.
  - 'ASET' as a whole word: 0 hits (the case-insensitive hits were "dataset").
  - 'layer-adaptive', 'layer-wise top', 'per-layer top-k', 'EvoESAP', 'Entropy-Penalized': 0 hits anywhere.
  - Depth-dependent BITS appear only as DyMoE's "more 4-bit in early layers" and as the "one policy over gate rank +
    tensor + depth" gap, both in ~/MoonShots/PRIOR_ART.md.

### 2. A deterministic "logical cache" for the cache-conditioned FAST levers (QOM, LSB-on-miss)
- **Paper:** Prefetch-Steered Routing: Deterministic Cache-Aware Expert Selection for Offloaded Mixture-of-Experts
  Decoding (PSR), arXiv 2610.34721, 26 Oct 2026, https://arxiv.org/abs/2610.34721
- **Mechanism:**
  - Cache-conditioned routing decides against a LOGICAL residency set per layer, never against the physical cache.
  - The logical set holds <= C experts. It changes only through routing decisions, plus at most n = 1-2 inserted
    next-layer pre-gate candidates per layer and token.
  - Copies, prefetch timing and queue depth then change only latency, so the outputs are a function of the input and the
    policy alone. Their Table 1: with a physical cache, perplexity shifted with the in-flight copy count; with the logical
    cache, it was identical.
  - On OLMoE at 50 % residency: 10.7 vs 24.7 ms per token against exact routing with LRU, and paired quality differences
    of ~-0.03 points over 8 tasks.
  - Their caveat for speculative decoding: 36 % of committed token-layers ran a different expert set depending on the
    REJECTED drafts. A per-position logical cache with rollback restores equivalence with one-token decoding, at 1.4-4x
    the copies (CPU replay).
- **Why it could move wall 1 here:**
  - It cuts no bytes itself. It removes a reproducibility blocker from QOM. The map prices QOM at -64 / -58 / -62 % SSD MB
> (a quote from the project owner, kept private)
  - The problem appears once FAST runs on top of D2's forecasts, which land at uncertain times, and the PURGE-1 volatile
    tier, which macOS purges under memory pressure. QOM's "would miss" then depends on timing and on the OS, so the same
    prompt can give different tokens from run to run. Same-scores tests and comparisons with the reference then cannot be
    repeated.
  - With a logical cache, an expert that is logically resident but was purged becomes a demand read: it costs latency
    and does not change the tokens.
  - If the logical C is pinned at the 45 GB line, FAST tokens are also identical at 45 GB and at night (C 320
    physical). One EvalSuite same-scores run then covers both lines, and the night's extra RAM only cuts latency. Kamen's
    call.
  - Cost: extra demand reads whenever the physical cache lags the logical one. On a quiet machine that is <= ~1-3 % of
    misses [G].
- **Exact?** No: it applies to FAST levers only. EXACT outputs never depend on cache state, so EXACT is unaffected.
- **Screen (CPU, ~30 min):** replay cap45 through the QOM mask builder twice:
  - (a) against the physical LRU. The volatile tier is purged at random checks (5 seeds; 0 / 10 / 30 % of slabs), and
    forecast landings are jittered by +-0.5 ms;
  - (b) against a logical LRU at the same C.
  - **PASS:** (b)'s masks are bit-identical across all seeds, AND (a) differs in >= 0.1 % of drop decisions (so the
    problem is real), AND (b)'s extra demand MB is <= 5 % of QOM's SSD MB per kept token.
  - **KILL:** (a) is already identical across seeds (nothing to fix), OR (b)'s extra demand MB is > 15 %.
- **Novelty:** adapted. The logical cache is PSR's; the rule "drafter-sure rows drop what would miss" is ours.
- **Grep evidence:**
  - 'Prefetch-Steered', '2610.34721', 'logical cache', 'logical residency', 'PSR', 'copy timing': 0 hits in the mandatory
    files and 0 in the wider set.
  - PSR's other half, a resident bonus λΔ on the router logits, is Cache-Prior / cache-aware routing. That is a KILL in
    TRIED (reads -1-4 %), so only the logical cache is proposed here.

### 3. A trained same-layer (pre-attention) forecast: D2's second chance, and layer 0
- **Paper:** Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models, arXiv
  2511.10676, Nov 2025 (ETH Zurich), https://arxiv.org/abs/2511.10676
- **Mechanism:**
  - Predict layer L's experts from layer L's activations BEFORE its attention block, using 2 linear maps trained with a
    ranking-aware loss (norm and softmax preserve the ranking).
  - Reported accuracy: 93.0 % (DeepSeek-V2-Lite), 94.7 % (Qwen3-30B), 97.6 % (Phi-mini-MoE), ~+15 points over earlier
    predictors.
  - It also covers the first layer.
- **Partly known here:**
  - QUEUE L2-08 measured the UNTRAINED version on the 125B (~/NewIdeas/data/samelayer/results.txt): same-layer top-10
    recall 0.735 (code) and 0.82 (IFEval + multi); top-16 recall 0.857 and 0.929. L2-08 killed it at 12 GB, with a
    ceiling of "the attention half, ~0.45 ms x 48".
  - Only the training is new. LEAD1CORR's ridge raised lead-1 recall from 0.612 to 0.729 [K]. A similar lift at the
    same-layer point would give ~0.85 recall on code and ~0.9 on chat [G].
- **Why it could move wall 1 here:**
  - (a) At recall 0.85, D2's partial-sum forecast still leaves ~15 % of each layer's misses. A second chance ~0.45 ms
    before the router catches most of them.
  - (b) No lead-1 method can forecast layer 0. Its router input is the token embedding plus one block, and the drafted
    tokens fix that embedding.
  - (c) If tonight's pr_160 misses its line (< 0.75), this is a better fallback than the L-input forecast (recall 0.61,
    3.49 reads per miss).
  - Estimates [G]:
    - on top of D1 + D2: <= -1 ms exposed per check;
    - as D2's fallback: the exposure falls from ~12.4 ms (D2 at recall 0.5, exposure_replay) to ~6-9 ms.
- **Exact?** Yes: it changes timing only, and the model's own router still decides.
- **Screen (CPU, ~20 min):** use cap45's captured x_L, which LEAD1CORR already uses. For each layer, fit a ridge map (or
  the paper's 2-map ranking loss) from x_L to layer L's router logits, and measure leave-one-prompt-out recall@10.
  - **PASS:** recall >= 0.85 on code AND >= 0.90 on chat.
  - **KILL:** recall < 0.78 on code (less than +0.05 over the raw router).
  - If it passes, run exposure_replay.py exposed3 with the forecast window set to 0.45 ms.
- **Novelty:** adapted.
- **Grep evidence:**
  - '2511.10676': 1 wider file (~/NewIdeas/research/healing_dataflows.md, in a list of prefetch-only prior art).
  - 'pre-attention': in TRIED only for SmallThinker's retrained router (a KILL), and in QUEUE L2-08 / B8 for the
    untrained same-layer forecast.
  - 'Pre-Attention Expert', 'ranking-aware': 0 hits.

### 4. Re-fit the per-layer pool sizes to the forecast: give slots to the layers the forecast misses (Fate's shallow-favoring cache)
- **Paper:** Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate, arXiv 2502.12224 (v2, May 2025),
  https://arxiv.org/abs/2502.12224
- **Mechanism:**
  - Fate's predictor is weakest in shallow layers, so Fate gives those layers more cache.
  - With that cache, a 97.15 % prefetch accuracy becomes a 99.08 % hit rate.
  - Decode is up to 4.1x / 2.2x faster than load-on-demand / EAP.
- **Why it could move wall 1 here:**
  - The M-17 / M-25b sizes minimize total misses; they even out the misses at 5.5-9.8 per layer [P, SSD_WALL2's
    exposure replay].
  - Once D2 hides forecast misses, the two kinds of miss cost different amounts:
    - a miss the forecast caught costs only hidden bytes;
    - a miss it did not catch costs a demand burst: a fixed 0.12-0.28 ms plus bytes, all exposed.
  - So the objective becomes Σ_L (1 - r_L) x misses_L x burst cost, where r_L is each layer's forecast recall from
    tonight's pr_160 capture.
  - If r_L varies across layers, slots move from layers the forecast covers well to layers it covers badly.
  - Estimate: -0.5..-1.5 ms exposed per check at equal RAM [G]. It refines D2 and is worth nothing without it.
- **Exact?** Yes (a sizing change).
- **Screen (CPU, ~20 min, after pr_160):** take the per-layer r_L, run M-25b's fitter with misses weighted by (1 - r_L),
  then run exposure_replay.py exposed3 with the per-layer r_L.
  - **PASS:** >= 1.0 ms less exposed time per check at equal RAM vs the current joint sizes.
  - **KILL:** < 0.3 ms less.
- **Novelty:** adapted.
- **Grep evidence:**
  - 'shallow-favoring', 'favoring', 'forecast-aware', 'recall-weighted', 'unforecast': 0 hits anywhere.
  - Fate itself is known, but only as a predictor (NOVELTY's LEAD1CORR row; '2502.12224' in 1 wider file). Its cache
    allocation is not in the files.
  - Nearest, but different: "Concave stall-time objective for slot allocation" (a KILL in TRIED) priced misses, not
    forecast coverage.

## A gap: no prior art found, and not from a paper [G]
- **Copy-span routing replay:**
  - Some drafted rows end in k tokens that match a span of the prompt: agent turns re-emit tool output or file text.
  - For those rows, use the routing RECORDED during prefill at the matched prompt position as the row's forecast for all
    48 layers at once.
  - The forecast is long-lead, exists before the check starts, and applies only to copied rows.
- **Prior art:** searches for routing replay or suffix-matched expert prefetch found nothing. ExpertFlow, DoMoE and
  MoE-Infinity match on tokens, semantics or request traces, not on positions.
- **What our files have:** the prompt's footprint used as a set (COLD-CUT) and copy-span drafting (L1-10, MATCH). Neither
  is position-level routing replay.
- **Expected size:** likely small on average (MATCH matches ~5 % of MBPP rows), larger on agent turns.
- **Screen (CPU):** on a capture that includes prefill routing, measure per-layer recall@10 for rows with an 8-gram
  prompt match.
  - **PASS:** >= 0.80 over layers 4-47, AND >= 15 % of agent rows matched.
  - **KILL:** otherwise.

## Novelty watch (for the papers; these are not levers)
- **PSR's §7 (2610.34721)** weights each verify-block position's expert demand by the DRAFTER's confidence: the square
  root of the running product of drafter confidences, with γ 1.05.
  - It builds one expert set per block (AcceptMoE-style) and reroutes inside it.
  - NOVELTY.md calls Q1 NEW partly because "neither uses the drafter's own per-token certainty". That clause has been
    false since 26 Oct.
  - Q1 still differs: each row gets its own expert count, with no reroute and no renormalisation. Cite PSR.
- **Memory-Sovereign Inference: Output-Exact Execution Beyond Full Residency** (arXiv 2608.23805, Aug 2026,
  https://arxiv.org/abs/2608.23805):
  - It streams Qwen3-Next-80B from an SN850X NVMe on an RTX 3090, with all 64 logit rows byte-identical to a zero-cache
    run. It also reports a Qwen3.6 MTP run at 12.0 tok/s.
  - ~/NewIdeas/PRIOR_ART.md's line "the whole stack ... NOT FOUND published" should cite it.
  - What still differs: our 16-row speculative checks are bit-identical to one-token decoding, on MLX / Apple, for a
    125B, with batch-invariant kernels.
- **ExFold** (2608.24938, Aug 2026; same first author as SERE) folds each out-of-pool expert into the "most compatible"
  pool expert with a calibrated scalar. TRIED's line "SERE / SPICE / mean-expert fallback" already KILLs this for the
  125B: the routed experts are mutually orthogonal (best |cos| median 0.024), and the best scaled other expert leaves
  100 % of the missing energy. Not worth testing.

## Already known: in our files in substance (one line each)
- SGLang SSD Expert Pack (LMSYS blog, 29 Aug 2026): one aligned record per expert + O_DIRECT + LFU/LRU, no prefetch. Our
  REC2, F_NOCACHE path and LRU cover all of it.
- SSD-LLaMA (2609.18110): PRIOR_ART. FlashMoE (2601.17063): PRIOR_ART / RANKED #12. Edge0 (2609.18063): TRIED (distill,
  PARKED).
- SeqMoE (2609.12978): TRIED / PRIOR_ART. MoE-CORE (2610.01950): wider files. PreScope (2509.23638): wider file
  (learned predictor + scheduler = the LEAD1CORR / demand-first family).
- SpecPrefetch (2607.24787): a shared-adapter lead-1 predictor, the CORR / LEAD1CORR family. Mira (2609.38090): forecasts
  two layers ahead + quantized transfers (two-layers-ahead KILL; POM).
- ST-MoE (2606.15453): TRIED / NOVELTY. DoMoE (IJCAI 2026): NOVELTY. CasMoE: wider file.
- BigMoMo (2609.14643): PRIOR_ART by name. Its flash re-layout by co-loading is the co-activation layout (napkin KILL)
  plus LSB-ADJACENT (KILL).
- MoE-SpAc (2603.09983): MoonShots IDEAS ("drafts as a sensor"). MoE-SpeQ, SP-MoE, DraftExpert, SPICE, AcceptMoE,
  EcoSpec, EVICT, MoE-Spec, SpecMD, Speculating Experts: PRIOR_ART / NOVELTY.
- OEA, Lynx: NOVELTY. XShare (2602.07265): DEAD.md. SERE (2602.07616): TRIED KILL. Bacon et al. (NeurIPS 2026,
  eviction-only): PUSH_PRIOR_ART.
- ReMoE (2605.27081, ICML 2026): PRIOR_ART; router fine-tuning, so the retraining KILL in TRIED.
- ZipMoE (2601.21198, ICML 2026): codes BF16 exponents with LZ4 / ZSTD on the CPU and partly caches the incompressible
  half. That is our plane pools plus the LOSSLESS line, and our weights are 3/4-bit codes, not BF16.
- LorExperts / BTExperts (2608.07814), LightMoE (2603.12645), the MoE trilemma (2510.02345), D2-MoE: low-rank or
  shared-base experts. TRIED KILL: quantized experts are full rank (L1-16 family).
- MoBiLE (2510.12357): Breakthrough notes. MoDES / ACE / ASET's dynamic part: TRIED, adaptive top-k PARKED / KILL.
- DeepInfer-Mac (2610.36564, 750B on a 24 GB M4): zero-copy slots (= our 1g), LIP insertion of speculative entries
  (TRIED), one global budget. Nothing new.
- streamoe, SwiftLM, mlx-flash (Apple Silicon engines): F_NOCACHE pread, the page cache, a top-k cut. Behind us.
- HybriMoE (DAC 2025): score-based caching + impact-driven prefetch. That is decayed LFU / the popularity prior / ranked
  prefetch lists (TRIED).
- DAOP, CoX-MoE (2605.17889), TriMoE (2603.01058): CPU / AMX / NDP co-compute. That is WHOLE-CHIP (Leap), or it needs
  hardware we lack.
- FreeToken, FluxMoE (2604.02715): elastic GPU memory split between the KV cache and experts. Moot at 45 GB: our KV cache
  is small (12 attention layers).
- SolidAttention (FAST '26), SWARM, DUAL-BLADE: KV cache on the SSD, not experts. 2508.06978: an energy study, not a
  lever.
- "Which expert misses are worth waiting for?" (rs-11268552): PRIOR_ART. The Ingestion Tax (2608.12114): mmap'd weights
  adopted as no-copy GPU buffers; our mmap KILL stands.
- Head-of-record "ping" to warm the drive's map: already killed on the scoreboard as lever 1h (M-31).

## Searched, nothing new (queries run, 3 Nov 22:2x-22:5x)
MoE expert offloading SSD flash consumer 2026 | speculative decoding MoE verification expert union offloading 2026 |
MoE I/O latency NVMe per-request overhead batching 2026 | Apple Silicon unified memory SSD weight streaming MLX 2026 |
cross-layer expert prefetch prediction recall 2026 | batch-aware expert selection union decode training-free 2026 |
shared-base / delta / low-rank residual expert compression 2026 | dynamic expert skipping confident tokens 2026 |
draft tokens predict target experts prefetch 2026 | "expert offloading" NVMe latency-bound prefetch scheduler (extended) |
expert folding / calibrated substitution 2026 | Two-Stage Expert Offloading (DoMoE) | SERE | Qwen3-Next 512-expert
offloading | lossless low-bit weight compression with a fast GPU decoder 2026 | routing prediction from draft-token
identity | arXiv 2610 MoE caching (extended) | 2610 / 2611 lossless verification + cache residency | rejected-draft cache
pollution / expert admission | llama.cpp / MLX SSD expert streaming PRs | MobiSys / MobiCom 2026 on-device MoE |
MLSys / EuroSys / ASPLOS 2026 MoE SSD | FAST / ATC 2026 LLM SSD small reads | NVMe idle-state wake + keepalive |
MoE-SpAc | Jetson unified-memory NVMe zero-copy expert offload | "LLM in a flash" 2026 follow-ups | 2611 arXiv MoE
offloading (extended) | fewer verifier experts while keeping chat quality | Apple SSD F_NOCACHE pread write-ups | Flash-MoE
write-ups | bit-exact deterministic SD + offloading | middle-layer redundancy / per-layer expert count | LExI | coalescing
reads across layers | logical / virtual cache determinism | Apple M4 / M5 MoE SSD papers (extended) | drafter-confidence
verifier expert counts | pre-attention router prediction | DFlash + expert prefetch | n-gram / suffix-matched expert
prefetch | early termination of rejected verify rows | layer-0 prediction from token embeddings | purgeable / volatile
weight caches | elastic expert cache under OS memory pressure.

## Sources
- PSR: https://arxiv.org/abs/2610.34721 ; LExI: https://arxiv.org/abs/2509.02753 ; ASET:
  https://aclanthology.org/2026.findings-acl.2140/ ; Pre-attention prediction: https://arxiv.org/abs/2511.10676 ;
  Fate: https://arxiv.org/abs/2502.12224
- Memory-Sovereign Inference: https://arxiv.org/abs/2608.23805 ; ExFold: https://arxiv.org/abs/2608.24938 ;
  SERE: https://arxiv.org/abs/2602.07616 ; EvoESAP: https://arxiv.org/abs/2603.06003 ; GRAPE:
  https://arxiv.org/abs/2604.06542
- Checked and known: https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/ ; https://arxiv.org/abs/2601.21198 ;
  https://arxiv.org/abs/2607.24787 ; https://arxiv.org/abs/2609.38090 ; https://arxiv.org/html/2610.36564 ;
  https://originshq.com/blog/moe-ssd-expert-serving-runtimes/ ; https://arxiv.org/abs/2605.27081 ;
  https://arxiv.org/abs/2603.09983 ; https://github.com/transybao1393/streamoe ; https://github.com/SharpAI/SwiftLM
