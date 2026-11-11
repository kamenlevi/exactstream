# CLOUD TASK: find the drafter method we missed

A closed literature task. You cannot reach our machine, model or files; everything you need about our setup is below.

**Our setup**
- Target: Qwen3.8-Flash-Next, a 125B MoE (48 layers, 512 routed experts, top-10, Gated-DeltaNet + attention layers, hyper-connections, a native MTP head).
- Hardware: one MacBook (M5 Pro, 64 GB, ~270 GB/s memory bandwidth, experts streamed from the SSD), MLX.
- Decoding: speculative and exact (greedy). The target checks a drafted block in one pass and keeps the identical prefix.
- Drafter: DFlash2 (Qwen3.8-Flash-Next-DFlash2), a 1.2 GB block drafter that drafts 16 tokens in one pass from the target's hidden-state taps. Drafting takes 6-12 ms per block.
- Measured on code (HumanEval+, MBPP), per-token agreement 0.909:
  - 0.955 where the target is sure (p1 >= 0.99), 0.66 at its uncertain positions, 0.30 at near-ties
  - by depth: ~0.90-0.95 at depths 1-11, ~0.86-0.89 at depths 12-15
- Kept per 16-token check: 9.6-10.5 on code, 6.6-7.0 on math (GSM8K), 2.0-2.5 on chat (IFEval).
- **What we need:** ~20-22 kept per 32-token check on code (~0.97 per token, flat over depth), with drafting <= ~10 ms per check on the critical path. Our own analysis says drafters under ~1-2B plateau at ~0.93-0.95 per token, and only near-copies of the target reach ~0.98.

**Already tried here and killed or parked.** Don't return these unless a paper shows far bigger code gains at depth >= 8 than this list implies:
- autoregressive heads: EAGLE-3, HASS, GRIFFIN
- semi-autoregressive: Falcon, PARD, Gumiho
- Medusa-style and exact token trees
- bigger drafters (more layers)
- self-speculation / layer skip
- prompt lookup, suffix decoding, copy spans; Lookahead / Jacobi decoding
- causal tails: Domino, DSpark, xPress, DARTree, H-Spec
- LiLiCorr-style selectors; MaskGIT-style refinement
- bandit drafter switching
- more on-policy training data (+0.22 kept per doubling)
- learned acceptance certificates
- FR-Spec reduced vocab; DraftExpert
- Mirror-SD (drafting on the NPU; killed because our small drafter is only 3-9 % of a check and must wait for the check's result)
- lossless speculative sampling; typical / relaxed acceptance (+5-11 % on code at ~1 % token deviation)

**Answer exactly these four questions** (published work 2023 to Nov 2026: arXiv, OpenReview, major labs' blogs and repos):
1. **Deep acceptance on code.** Which methods report >= 0.95 per-token acceptance at draft depth >= 8, or >= 10 accepted tokens per verification step, on code (HumanEval, MBPP, LiveCodeBench)? They must use a drafter <= 2B parameters, a target >= 30B, at temperature 0. For each give:
   - mechanism; drafter and target sizes; block size; accepted length
   - lossless or not
   - training cost and data
   - whether code is available
2. **Learning during a session.** Has any drafter been adapted within a single generation session from the target's verified tokens (test-time training, fast weights, a kNN or cache memory, online mixture weights)? Cross-query online distillation (OSD, 2023) does not count unless it also works within one session. What gains were reported?
3. **A big drafter off the critical path.** Look at systems that draft on another compute unit (NPU, CPU, a second GPU or stream) in parallel with verification, pipelining across checks (e.g. PEARL, Mirror-SD, Dovetail). Did any use a much bigger drafter *because* drafting was hidden, and what did acceptance gain?
4. **Other fields.** Has context mixing (PAQ / cmix), TAGE-style multi-history predictor selection, or a compression-style match model been used to draft, or to mix drafters, in LLM speculative decoding? With what numbers?

**Then translate.** For the best 3 methods:
- convert the reported gain into a ratio over that paper's own baseline
- apply it to our code numbers (kept 10.5 per 16-token check, 0.909 per token)
- estimate our kept per 16- and 32-token check, and mark it as a rough transfer
- rank them by expected kept per check and by cost to try (we may train on free cloud compute but never pay for GPUs; drafters must run on the Mac)

**Output** (the whole answer, <= 1,200 words):
- First, one line per question: YES / NO / PARTLY + the single best citation.
- A table of at most 10 methods: name, arXiv id, mechanism (<= 15 words), drafter / target, code acceptance as reported, lossless?, training cost, our estimated kept per check.
- **Try first:** one method, why, and the cheapest test with its kill line.
- Cite every number to its source. Write "not found" rather than guess.
- **Never an empty answer.** If no method meets a question's bar, give the 3 closest with their numbers. The table always has at least 5 methods, and Try first always names one.

**Budget, so the answer always arrives**
- At most 30 searches and page reads in total, of which at most 8 full papers. For the rest, read the abstract page or the search result.
- Write a complete first answer to ANSWER.md after the first 10 reads, even if parts say "not yet". Update it after every ~10 more.
- When every question is answered or the budget is spent, stop at once and print ANSWER.md as your final message.
- Don't ask questions or wait for approval: make reasonable calls and note each in one line. Don't change or commit anything in the repository.
- If page reads are blocked, work from the search results and say so.
- No model or dataset downloads, no long code; small calculations only.
