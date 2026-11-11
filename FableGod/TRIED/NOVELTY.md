# NOVELTY: which of our levers are published, adapted or new (NewIdeas, Mon 2 Nov 2026, 22:xx EET)
> (a quote from the project owner, kept private)
before". Every ADOPTED / PASS lever in TRIED.md now ends in "novelty: <tag> (<prior art>)" (novelty.tsv, written by
~/NewIdeas/scripts/novelty_tags.py; tried_merge.py adds it at every re-merge).
- **published**: a published method used as is, ported to MLX and this engine.
- **adapted**: a published idea changed in a way that matters (made bit-exact, a new signal, a new setting).
- **new**: no prior art found in the web checks below.
- **engineering**: an implementation step, a fix, a measurement or a diagnostic, with no method claim.

## The answer, with facts
- **Refresh Fri 6 Nov 21:3x (NewIdeas; tags are judgments on the records, with no new web check):**
  - novelty.tsv now holds 309 tags: 36 published, 91 adapted, 5 new, 177 engineering. Still no lever in a release
    config is tagged new.
  - The comment rule (relaxed acceptance only in code comments) is adapted. Relaxed acceptance is published; the
    comment-only span was found nowhere in two web checks. It is live in RELAX B.
  - The SSD wake penalty + keepalive (MoonShots M-20) is a new finding on a known mechanism. It is now ADOPTED in the
    45 GB EXACT default (Item 5, 4 Nov).
  - QOM, as shipped, runs at thr 0. Every drafted row counts as sure (the bonus row stays exact), so its U3 part (kept
    experts at MSB-3) and its Q1-UNION part act on every drafted row; drafter certainty enters only through the 0.3
    live cut (r57_screen_fast.py:233-240; thresholds <= 0.3 are identical, TRIED CF03-EQ-CF0).
  - Live levers that had no tag:

    | lever | tag | why |
    |---|---|---|
    | NAX6 (exact) / FAST-NAX (its one-call under QOM) | adapted | grouped / gather MoE GEMMs are published (MonoMoE, ONNX Runtime's MoE GEMV); the M5 matrix-unit form and its bit-exactness are ours |
    | DN-KS (ks per shape for narrow GEMVs) | published | split-K GEMV / GEMM (CUTLASS split-K, stream-K) |
    | att_single (one SDPA call, not the 12-chunk fallback) | engineering | using the fused kernel |
    | D7BNAX_32 / QMM32 (17-32-row drafter calls as two 16-row halves / one 32-row kernel) | engineering | |
    | lazy GDN commit (bitwise) | engineering | recurrent-state rollback under speculation is known practice |
    | DRAFT17 (a 17th draft row gated by the next segment's first draft cf) | adapted | the +1 bonus token is standard; the cf gate is ours |
    | HC8 (8-bit hyper-connection mixers) | published | int8 weight-only quantization |
- **215 ADOPTED / PASS levers:** 30 published, 55 adapted, 3 new, 127 engineering.
  - Tue 3 Nov 22:5x: 219 tags (30 published, 56 adapted, 3 new, 130 engineering). The same evening's merge fix
    (tried_merge.py keyed rows on their first word, so a session's sub-ideas under one ID collapsed into the first)
    surfaced 168 more TRIED lines. Tag pass done 22:5x: 293 tags (36 published, 78 adapted, 5 new, 174 engineering); every
    ADOPTED / PASS line in TRIED.md carries one again. The two new 'new' tags are sub-rows of existing new families (DM#10, M-18D).
- **Every lever in the live defaults is published or adapted.** None is new by our checks.
  - 45 GB: REF2-T (Thinking Machines' batch invariance + HOBBIT-style tiers), the sync-free gate (anemll-flash-mlx /
    SlotStream), t 0.3 truncation (EVICT / DraftExpert), OOO (MoE-Infinity / ProMoE overlap, made bit-exact), per-layer
    pools (AdapMoE), DFlash2 (z-lab), the chat switch (WhiFlash's drafter-by-task).
  - 12 GB: its signature trick, cold ranks half a layer late (B12LL, Oct 23), was found here independently, but
    KTransformers published it first as Expert Deferral (SOSP '25).
- **Our own contribution so far:** exactness. We made these published methods bit-identical to one-token decoding on
  Apple's GPU and its matrix units, and stacked them in one MLX engine.
  - Tue 3 Nov 23:1x (MoonShots' sweep): output-exact SSD streaming is itself published. Memory-Sovereign Inference
    (arXiv 2608.23805, Aug 2026) streams Qwen3-Next-80B from an NVMe on an RTX 3090 with all 64 logit rows
    byte-identical to a zero-cache run (and an MTP run at 12 tok/s). What stays ours: 16-row speculative checks
    bit-identical to one-token decoding, batch-invariant kernels on MLX / Apple's GPU and matrix units, at 125B.
- **The three new ideas are all from the last 24 hours, and none is used yet:**
  1. Cut the VERIFIER's experts per row by the drafter's certainty (Q1 / M-18 / DM-Q1): code SSD bytes per kept token
     x0.60, 0.09 % of gated code rows flip. Same-scores test: EvalSuite RELAX arm A, Thu 00:30.
     Tue 03:19 Q1-UNION (a sure row also keeps the ranks the check loads anyway) PASSes at cf 0.9 with lower deviation
     than Q1 at 0.99 everywhere. Its union step is published (OEA, below), so the new part is the certainty gate.
     Tue 3 Nov 23:1x: the certainty gate is published too (PSR §7, arXiv 2610.34721, 26 Oct: the verify block's expert
     demand weighted by the drafter's confidence, one set per block, rerouted). Q1 is now ADAPTED: a per-row count, no
     reroute, no renormalisation. That leaves two new ideas (2 and 3), neither in use.
  2. Reuse the previous position's MoE output in the middle layers (DM P-frame): new as an idea, but KILLED as a lever on
     bytes (Mon 22:5x: the layer-wise map's union is only x0.91).
  3. The comment loophole (relaxed acceptance only inside code comments). It is in the RELAX "B" arm, not yet measured.
     It counts as new only as a span rule; the relaxed-acceptance family is published.

## Prior-art checks of the six "new" candidates (web, Mon 2 Nov 22:xx)
| candidate | verdict | closest prior art |
|---|---|---|
| gate keyed on drafter certainty (Q1) | **adapted** since Tue 3 Nov 23:1x (was NEW: MoonShots' sweep found PSR §7) | PSR (Prefetch-Steered Routing, arXiv 2610.34721, 26 Oct 2026) §7 weights each verify-block position's expert demand by the DRAFTER's confidence (the square root of the running product of drafter confidences, gamma 1.05), builds one expert set per block and reroutes inside it. Q1 still differs: each row gets its own expert count, with no reroute and no renormalisation. Earlier checks: EcoSpec, DraftExpert, EVICT and Utility-Driven SD cut or select DRAFTS by expert cost with a standard verifier. Verifier-side, lossy, per block (re-checked Tue 10:29): AcceptMoE (arXiv 2608.02989) builds one eligible expert set per layer and block from router scores weighted by OFFLINE per-position commitment rates, and every token still computes k experts, rerouted inside the set (-0.27 points vs EAGLE-3); MoE-Spec (2602.16052) caps the block's experts by summed router mass. Neither uses the drafter's own per-token certainty, and neither lets a row compute fewer experts. Ours: each row's expert count is set online by the drafter's certainty for that row; sure rows keep top-2 at the model's weights, with no reroute and no renormalisation |
| comment loophole | **NEW as a span rule** | relaxed / lossy verification in general: MARS (margin-aware), DIVERSED, AdaptiveSpec (margins), "Revisiting Lossy Verification"; none found applies it to code comments |
| COPYGATE (gated match override) | adapted | SAM-Decoding picks retrieval vs an auxiliary drafter by suffix-match length |
| 12 GB two-pass certification | adapted | TriSpec: a lightweight proxy approves easy drafts, a margin rule escalates uncertain tokens to the target. Ours: the proxy is the 125B itself at top-2 experts, plus a drafter-certainty arm (cf 0.99), which is the new part |
| 3-anchor ANE drafting, first-miss anchors | adapted | PEARL (post-verify drafting), SpecBranch (branches at predicted rejection points), Speculative Speculative Decoding and DPara (pre-drafting for many verify outcomes), Mirror-SD (GPU + NPU). The Apple ANE placement is ours |
| NAX hybrid for padded tiles | published practice | MonoMoE (padding waste in decode MoE GEMMs), ONNX Runtime's MoE GEMV fast path for small row counts; the M5 matrix-unit form is ours |
| Q1-UNION (Tue; sure rows keep ranks already in the check's union) | adapted: the union step is published | OEA, Opportunistic Expert Activation (arXiv 2511.02237, ICML 2026): each token keeps top-k0 and adds lower ranks only if the batch already loads them (Qwen3-30B / 235B at batch 16: MoE-layer decode latency -39 % / -15 %); Lynx (2411.08982) batch-aware selection. Ours: the batch is one verify check and k0 is set per row by drafter certainty |
| PURGE-1 elastic volatile expert tier (Leap, Tue) | adapted: published OS practice | Apple's purgeable-memory caches ('Caching and Purgeable Memory'; Metal MTLPurgeableState on buffers / heaps); OS-reclaimable weight caches as mmap'd weights in the page cache (llama.cpp). No LLM / MoE expert tier found (web, Tue 15:1x) |
| P2-PREFETCH (Tue; pass 2's experts read in pass 1's GPU gaps for rows the drafter is unsure of) | adapted | Draft-phase prefetch of the target's experts: MoE-SpeQ (2511.14102), SP-MoE (2510.10302). Ours: the window is the two-pass check's own pass-1 gap, and rows are chosen by drafter certainty |
| 3-bit records on drafter-sure rows (Tue; q1u_cf09_t3: sure rows' ranks 0-1 at 3 bits, the rest 4) | new signal on a published mechanism | Per-token mixed precision: token-level precision restoration (arXiv 2512.17073) keeps a token's top-ranked expert at high precision by ROUTER rank; DynaExq (2511.15015) and DyMoE (2603.19172) set bits per EXPERT. None found keys precision on the drafter's certainty (web, Tue 10:29) |
| P1-OEA (Tue; the 12 GB two-pass check's pass 1 = top-1 + ranks already in the block's union) | adapted: both pieces are published | OEA's union step (arXiv 2511.02237) inside the TriSpec-style proxy pass; using it to halve the proxy's reads (top-1 instead of top-2 at the same mean gate mass) is ours |
| Q1-ON-MISS (QOM; Fable C's C1-2, revived Tue 17:0x on live-cut checks; drafter-sure verify rows drop only the rank 2-9 experts that would miss) | adapted | Cache-aware routing is published and lossy on every token: Mixture of Cache-Conditional Experts (2412.00099), Cache-Prior / BuddyMoE, NeurIPS 2026 cache-aware routing; MoE-Spec (2602.16052) budgets verify experts. New here: no rerouting or substitution, only drafter-sure rows, and only the experts that would miss. Map Tue 22:59: PASS on code / IFEval / multi at cf 0.9 and 0.5 (cf 0.5: 0.36 / 2.68 / 1.63 % flips), the lowest of any FAST arm, at -60 / -37 / -33 % SSD bytes. Closest since 26 Oct: PSR (2610.34721) decides against a deterministic LOGICAL residency set and reroutes within it; its logical cache is also the fix for QOM's run-to-run dependence (MoonShots' Wed screen). The evidence for the 'adapted' tag: the cache-aware papers (PSR included) reroute on every token; keeping every resident expert and gating by drafter certainty is what keeps the flips this low |
| LSB-on-miss (next QOM step: a rank 0-1 served at 3 bits when its LSB slice misses) | published | SliceMoE (2512.12990, DAC 2026): bit-sliced caching with precision assigned on demand under a miss-rate constraint; Matryoshka Quantization (2502.06786) |
| Staged check (BREAKTHROUGH PUSH: a cheap expert-budgeted pass picks the path in a top-3 tree, an exact pass verifies it) | adapted | HiSpec (2510.01336, early-exit intermediate verifier), TriForce (2404.11912), staged SD (2308.04623), MoE-Spec; our T3 two-pass. See ~/NewIdeas/research/PUSH_PRIOR_ART.md |
| LOSSLESS45 (BREAKTHROUGH PUSH: entropy-coded expert bytes, decoded once per SSD miss) | published (the decode-once-per-miss corner is adapted) | EntroLLM (2505.02380), tile-level ANS (2606.15789), DFloat11 (2504.11651); floor -15 % of bytes (MoonShots W24-floor) |
| Expert-major scheduling across checks / CPU-SME or ANE co-compute (BREAKTHROUGH PUSH families) | published / adapted | Klotski (2502.06888), FlexGen, PEARL; Ghidorah (2505.23219, CPU+GPU speculative decoding on unified memory), SMEPilot (2606.16332). One chat lacks independent rows; DRAM is shared |
| COLD-CUT with a per-request table (Breakthrough + Fable C, Tue 20:3x): on drafter-unsure rows drop a rank >= 4 expert unless the corpus table OR the request's own prefill routing knows it (+ the check's union) | new signal on a published mechanism | Skipping with fixed tables: SwapMoE (2308.15030), AdapMoE (2408.10284); the union: OEA (2511.02237). The prefill footprint is used only for placement / prefetch / clustering: ELDR (2607.00466), DoMoE (IJCAI 2026), arXiv 2604.23150, MoE-Infinity (per-request sparsity), DuoServe-MoE (phase split). Two searches (Breakthrough 20:3x, NewIdeas 20:4x) found no cut keyed on the request's own footprint. Map: GENERAL (no cost vs u8 + t3 on code / IFEval / multi) at -3 / -13 / -16 % bytes (SCOREBOARD be67ff0; ~/Breakthrough/research/novelty_cold_cut.md) |
| The prompt footprint alone as the code path's COLD table | new signal on a published mechanism (as above) | Code: no cost at -33 % bytes vs u8 + t3; fails chat (short prompts) |
| SSD wake penalty + keepalive reads (MoonShots M-20 / M-28): the internal SSD charges +44-65 us on the next read after >= 0.25 ms idle; a 4 KB read every <= 270 us removes it | new finding on a known mechanism (proposed 'new' once adopted; tonight's ka1 decides) | NVMe idle-state exit latency is published (APST; AnandTech wake-up tests, which could not trigger Apple SSD sleep states). No published sub-ms numbers for Apple Silicon's internal SSD and no MoE / LLM offload work on it (MoonShots 11-02 22:0x, NewIdeas 20:4x) |
| M-28 per-call decomposition (first byte ~119 us, wake 44, GCD 8, ~96 us outside the I/O path; pieces slower at the loader's queue depth) | engineering (measurements; paper material for the SSD-wall section) | - |
| W1-PICKUP: a coherent(system) spin for the loader -> GPU handoff (Leap, Tue 19:4x): release + pickup ~23-80 us -> ~3-4 us | adapted | MLX 0.32's own Fence fast mode uses coherent(system) seq_cst fences in one-thread kernels; new here as the expert loader's handoff |
| LEAD1CORR: a ridge-corrected lead-1 forecast (L+1's router on L's input + a learned linear correction; MTP Head, Tue 20:3x): recall@10 0.612 -> 0.729 held out | adapted | learned pre-gates: Pre-gated MoE, ProMoE; Fate (adjacent-layer gate inputs) |
| M-29 aligned zero-copy reads into slot memory (MoonShots, Tue 22:33: s / b pieces padded 51,200 -> 53,248 B inside the same part, 4 KB-aligned Metal pool slots) | engineering | Zero-copy direct I/O into device-visible buffers is old: O_DIRECT / F_NOCACHE into page-aligned memory; GPUDirect Storage and BaM on NVIDIA. Ours are measurements: 4 KB alignment works as well as 16 KB on Apple's internal SSD; +5 us vs no copy (no DMA penalty into Metal memory); today's unaligned preadv +843 us |
| M-29 copy tail of the expert loader: 119-122 us per 2.16 MB base record, 170-176 for two side by side, 81 split three ways, sidecar 44.5 | engineering (a measurement; 2x Fable A's estimate) | - |
| M-29 wake-ahead ping: one 4 KB read ~200 us before a predicted burst instead of a keepalive trickle; removes 97.5 % of the +48 us wake (100 us is too late, 400 us too early) | adapted (MoonShots proposed 'new'; its papers sweep may refine the tag) | Anticipatory device wake is known: an early PCIe wake from the downstream port before the doorbell (US patent 11789658); HotOS '15 'Software Engagement with Sleeping CPUs' wakes the CPU ahead of predicted I/O completions; NVMe APST per-state entry / exit latencies (Microsoft's NVMe power docs, AnandTech's wake-up tests). Two searches (NewIdeas Tue 22:5x) found no software-only read timed by a model's router |
| M-30 QUEUEMIX: forecast reads beside demand reads. A busy drive saves the demand only the ~50 us wake, never its first byte; forecasts mixed into a demand burst cost it ~165 us per record, chained after it nothing; IOPOL_THROTTLE shields the demand but the kernel delays the forecasts 1.6-2.5 ms | engineering (measurements; lever 2's design constraints) | I/O priority classes and background reads behind demand are standard (Linux ioprio / BFQ, macOS setiopolicy_np) |
| M-29b zero-copy at the 24 GB line's burst sizes (-21 / -8 / -60 us per call at 4 / 8 / 15 records vs -173 at 2) and M-31 head ping (a 4 KB read of a record's head never shortens its later full read) | engineering (measurements: zero-copy pays only on small bursts; the drive's first byte cannot be pre-warmed per record) | - |
Also checked: cold ranks half a layer late (B12LL, DH_W4) -> KTransformers Expert Deferral. P-frame MoE output reuse ->
nothing found (the closest work caches or prefetches experts across tokens, or skips them per token: ST-MoE, MoDES).

## One new bet in flight per line (Kamen's ask)
- 45 GB: Q1-UNION at cf 0.9, the verifier's experts keyed on drafter certainty (RELAX Thu 00:30, rev 3 proposed);
  + 3-bit sure rows passed the map on 16-row unions (Tue 09:55); with live-length checks Q1-UNION 0.9 passes on code
  only (chat BETWEEN, 10:55), so it is a code lever; REACH-CUT (rows unlikely to be reached keep top-2 + union) adds
  -13 % code union at no measured cost. OEA k8 alone passes everywhere. P-frame reuse KILLED on bytes.
- 24 GB: the union stack at the 24 GB pools (Tue: Q1 0.9 x0.591, OEA hybrid x0.477, + 3-bit sure rows x0.422 SSD MB per
  kept token on the exact v3; PASS on bytes. At 4-row checks the union members nearly vanish, so quality ~ plain Q1:
  0.9 only on code; Q1 0.99 (x0.658) is the safe 24 GB form).
- 12 GB: the two-pass check's drafter-certainty arm (acert_top2_..._cf099; T0 same-scores Tue), and from Tue 09:10
  P1-OEA: pass 1 at top-1 + union members, x0.465 pass-1 reads: the certificate PASSES on code (map Tue 10:18;
  end to end 10:51); timing next (Fit12, 14:3x / 15:07).

## Sources
- EcoSpec: https://arxiv.org/abs/2607.12696 ; DraftExpert: https://arxiv.org/html/2607.24434v1 ;
  EVICT: https://arxiv.org/html/2605.00342v1 ; Utility-Driven SD for MoE: https://arxiv.org/pdf/2506.20675 ;
  Limits of Speculation: https://arxiv.org/html/2609.22156 ; MoE-SpeQ: https://arxiv.org/pdf/2511.14102 ;
  S2-MoE: https://arxiv.org/pdf/2608.15018
- MARS: https://arxiv.org/pdf/2601.15498 ; DIVERSED: https://arxiv.org/pdf/2604.07622 ;
  Revisiting Lossy Verification: https://arxiv.org/pdf/2607.26627 ; AdaptiveSpec: https://arxiv.org/html/2609.02897 ;
  SecureSD: https://arxiv.org/html/2610.08678v1
- SAM-Decoding: https://arxiv.org/abs/2411.10666v3
- TriSpec: https://arxiv.org/pdf/2601.23180 ; SelfJudge: https://arxiv.org/pdf/2510.02329 ;
  Confidence-Modulated SD: https://arxiv.org/pdf/2508.15371
- PEARL: https://arxiv.org/pdf/2408.11850 ; SpecBranch: https://arxiv.org/pdf/2506.01979 ;
  Speculative Speculative Decoding: https://arxiv.org/pdf/2603.03251 ; DPara: https://arxiv.org/pdf/2609.27396
- MonoMoE: https://arxiv.org/pdf/2609.04244 ; ONNX Runtime MoE GEMV path: https://github.com/microsoft/onnxruntime/pull/29628 ;
  MLX discussion: https://github.com/ml-explore/mlx/discussions/3801
- OEA: https://arxiv.org/abs/2511.02237 (ICML 2026 poster: https://icml.cc/virtual/2026/poster/62958) ;
  Lynx: https://arxiv.org/abs/2411.08982
- KTransformers (Expert Deferral, SOSP '25):
  https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
- Cross-token expert reuse / skipping: https://arxiv.org/pdf/2512.16473 ; MoDES: https://arxiv.org/html/2511.15690v2
