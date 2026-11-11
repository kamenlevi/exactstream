# WALL 2 (guesses kept per check): 2025-2026 literature sweep, new-to-us items only (MoonShots, Tue 3 Nov 2026, 22:2x-22:5x EET)

Scope: papers, preprints and write-ups from 2025-2026 that could raise the number of drafted tokens kept per verify check on the 125B
(48 layers, 36 GDN + 12 attention, 512 experts top-10, DFlash2 block drafter with the t 0.3 cut, native MTP head).

How this was done:
- About 45 web searches and about 25 abstract / HTML page reads. No PDFs were downloaded and nothing was run.
- Numbers marked "paper" are the authors' own. Numbers marked "est." are mine and should be halved when deciding (the team's rule).
- Every candidate was grepped (case-insensitive: name, arXiv id, mechanism words) in the 7 mandatory files: TRIED.md,
  NOVELTY.md, NewIdeas/PRIOR_ART.md, MoonShots/PRIOR_ART.md, DEAD.md, RANKED_45GB.md, QUEUE.md.
- As a bonus I also grepped TRIED/*.md, FableGod/ (*.md, push/, fablec/), NewIdeas/research/, GAPS.md, IDEAS.md and FAST_TRACK.md
  Method: `cat <files> | grep -i -c -F '<keyword>'`. Counts below are given as mandatory / extra lines, and every hit was read in context.

Bases used for the estimates, all measured by the team:
- Kept per 16-row check: code 9.57-10.44, math 7.0, chat 2.5-4, agent 4.80 (DFlash2) / 8.20 (MTP K15).
- 7.82 ms per verified row. Gate spin is 42.6 ms of the 84.2 ms check.
- COPYGATE is x1.041 / x1.211 on HumanEval+ (16 / 32 rows), x1.000 on MBPP and x1.032 on agent.

## Verdict

Only four items pass the novelty filter, and two of those are partly known. Most 2025-2026 work falls into families already in TRIED:
- block / diffusion drafters and causal tails
- trees, including GDN tree verification
- MoE expert-cost drafting and verification
- acceptance-targeted losses
- recycling rejected rows
- learned stop heads
- relaxed acceptance

One structural fact explains why the verification-rule papers (UniVer, HSD, SpecTr-GBV, Traversal, TEV) are useless to EXACT mode. Every lossless
verification rule published in 2025-2026 gains only under sampling. At temperature 0 they all reduce to prefix matching.

EXACT therefore gains on WALL 2 only through better drafts, plus more candidates (rows), which TREES already priced out.

| # | item | mechanism (short) | est. gain on kept per check | exact? | chance (G) |
|---|---|---|---|---|---|
| 1 | Test-Time Speculation (TTS), arXiv 2605.09329 | one KL step on the DFlash drafter per check, from the check's own target logits; reset per request | short suites +0-5 %; long answers / agent sessions +5-15 % if the length decay exists here (step 0 decides) | yes | ~30 % |
| 2 | FLy deferred-window rule, arXiv 2511.22972 (name already cited in QUEUE R-61N; its rule never screened) | accept a mismatch only at high target entropy AND if the next W drafts all match | code +3-8 %, chat +5-12 % before pricing the window's extra rows; the value is fewer harmful deviations at equal kept | no (FAST) | ~25 % |
| 3 | ToolSpec schema FSM, arXiv 2604.13519 | a finite-state machine fills the tool schema's fixed tokens inside tool calls | agent +3-8 %, other domains 0 | yes | ~30 % |
| 4 | SwitchSD copy-intent probe, arXiv 2609.20186 | a linear probe on the target state gates copy vs drafter | +0-2 % at 16 rows; +2-5 % on MBPP / agent at 32 rows | yes | ~20 % |

---

## 1. Test-Time Speculation (TTS): adapt the drafter during the request

- **Paper.** Kumar, Sanghavi, Das, "Test-Time Speculation", arXiv 2605.09329 (May 2026). NeurIPS 2026 poster.
  https://arxiv.org/abs/2605.09329 , https://arxiv.org/html/2605.09329
- **Mechanism.**
  - After every check, take one Adam step on the drafter. The loss is a position-weighted KL from the target's distribution to the
    drafter's over the check's block (canvas), plus a proximal term λ·KL(q_t || q).
  - The verify pass already produced these labels, so they are free. K = 1 step per round is best.
  - The update stride S trades gain for throughput: S = 1 gives the best acceptance, S = 5 the best throughput.
  - The drafter resets between requests. Only the drafted (canvas) tokens are trained on, never the prompt.
- **Why it fits here.**
  - The paper tests DFlash on Qwen3.5-122B-A10B. That model is 48 layers, a 3:1 GDN : attention hybrid (12 attention layers),
    256 experts, top-8, in SGLang, which is the nearest published target to our 125B.
  - Results on that model (paper):
    - Acceptance: AIME25 2.9 → 4.9 (+72 %), MATH-500 3.4 → 4.5 (+32 %).
    - Throughput: 1.54x over DFlash at S = 5.
  - Results on Qwen3-8B (paper):
    - LiveCodeBench 2.9 → 4.5 (+55 %).
  - Averages over each table (paper): +41 % over DFlash (Table 1), +34 % over EAGLE-3 (Table 2).
  - The paper's motivation is that offline-trained drafters (DFlash, EAGLE-3, PARD) lose acceptance as the output grows. Gains are
    +15-17 % over the first 10K tokens and +183-245 % at 20-30K (paper, Fig. 5). Nothing under 10K tokens or on short benchmarks is reported.
  - DRAFTER_WALL 2.4 put online adaptation at +1-3 % on code. That estimate rested on repeated mistakes (0.2-0.9 % of misses). TTS
    instead works by gradient generalisation within the request, which is a different mechanism.
- **Estimate (est.).**
  - Today's suites (answers under ~1K tokens):
    - code / math: +0-3 %, e.g. code 9.6 → 9.7-9.9
    - chat: +1-5 %
  - Long answers, or agent sessions where the adapter persists across turns (our variant; TTS resets per request): +5-15 %,
    e.g. agent 4.8 → 5.0-5.5. This holds only if our kept per check actually falls with output position.
  - Mac cost:
    - Use a LoRA, not full Adam: full Adam states for the ~0.5-0.6B bf16 drafter would take ~4-5 GB.
    - One update (forward + backward over 16 rows) is probably launch-bound like the 12.4 ms propose, so ~15-25 ms.
    - At S = 5 that is ~3-5 ms per check (~3-5 % of a ~97 ms cycle) unless it hides in the 42.6 ms gate spin through an MLX
      second stream (unverified).
- **Exact?** Yes. Only the drafter changes; every token is still verified.
- **Variant not found in the literature.** Also take a few steps on the prompt during prefill, which for agent turns is 9-12K tokens
  of in-request text. REGENFREE's failure mode (labels from text the 125B did not write) applies to it.
- **Cheapest screen (pre-registerable).**
  - **Step 0, CPU only, ~minutes:** from existing per-check logs (D-17 / D-23d per-pass data in prompt order, or C45 logs), bin each
    check by the output position of its first row (< 500 / 500-1,500 / > 1,500) and compute kept per check per domain.
    - KILL the length mechanism: every domain's > 1,500 bin is ≥ 0.95x its < 500 bin. Then run step 1 on agent sessions only.
    - PASS to step 1 anywhere: some domain's late bin is ≤ 0.90x its early bin.
  - **Step 1, small training run:**
    - Teacher-forced replay of DFlash2 L1-20 along the 125B's own long windows that have taps + top-32 (d02a / HPD-0 / DLOG formats).
    - LoRA r16 on the 5 layers. One Adam step per check (and S = 5) with TTS's loss on the check's 16 rows. Reset per request
      (and a per-session arm for agent).
    - Read out kept per 16-row check under t 0.3 at output positions ≥ 500, against the frozen drafter on the same checks.
    - PASS: agent or chat ≥ +6 %, and code / math ≥ -0.5 %.
    - KILL: < +2 % on every domain.
- **Grep evidence.**
  - Mandatory files: "Test-Time Speculation", "2605.09329", "test-time", "online distill", "online adapt", "fast weight",
    "OmniDraft", "Online Speculative", "TIDE", "ATLAS", "generation length", "output length", "length decay" all 0 / 0.
    The 15 "online" hits are cache / eviction / pool-sizing / drafter-switch (D-12, D-19) rows; none adapts the drafter's weights at
    test time.
  - Extra files: online adaptation appears only in FableGod/DRAFTER_WALL.md 2.4 (OSD / DVI / OmniDraft, "never tried here",
    est. +1-3 %). It is proposed there, not run.
- **Novelty tag.** Published (TTS). Adapted if run session-level inside the gate spin.

## 2. FLy: entropy gate + deferred window (FAST track)

- **Paper.** Li et al., "Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match",
  arXiv 2511.22972 (Nov 2025, v3 Apr 2026). ICLR 2026. https://arxiv.org/abs/2511.22972 , https://arxiv.org/html/2511.22972v3
- **Mechanism.**
  - At the first mismatch j, look at the target's normalised entropy h_j = H / ln|V|.
  - If h_j < θ (0.3), reject as usual.
  - If h_j ≥ θ, keep the draft token only if the next W = 6 drafted positions all match the target's argmax in the same verify
    pass (those rows are conditioned on the deviant token) and j + W ≤ K. Otherwise take the target token.
  - The deviation is kept only where the target "re-converges" with the draft.
- **Why it fits here.**
  - Results on Llama-3.1-70B at T = 0 (paper):
    - τ: GSM8K 9.87 → 12.57, HumanEval 9.89 → 12.61, MBPP 10.47 → 12.84.
    - Accuracy: -0.76 / 0.00 / -0.27.
  - On 405B the accuracy cost is -0.30 / -0.60 / -0.52 (paper).
  - RELAX_350 already shows what rule-based relaxations buy here (code R2 top-2 at 0.5 p1: +10.9 % at 1.22 % deviation). What none of
    our rules (R1 / R2 / R3 / TYP / rho) has is a harm filter, and the deferred window is one.
- **Estimate (est.).**
  - Before pricing rows: code +3-8 %, chat +5-12 % (with W 2-3; W 6 rarely fires on chat, since the drafter's ~0.6-0.7 per token gives
    P(6 matches) ~0.05-0.12).
  - Two catches here:
    - The window needs W verified rows after the mismatch. Chat checks stop at 3-4 rows under t 0.3, so each fired window costs up to
      W x 7.82 ms plus those rows' misses.
    - θ = 0.3 means H ≥ 3.7 nats at our 248K vocab, which is very strict, so θ must be re-tuned.
  - A variant that avoids the extra rows (ours, not in the paper): accept the mismatch provisionally. Confirm it with the next check's
    first W rows, and roll the KV / GDN state back if they fail.
- **Exact?** No. FAST track, judged by same-scores. Report completion-length inflation as well: "Acceptance Is Not Speed",
  2610.23818, already in our notes, found relaxed rules inflate output length 1.54x on average.
- **Cheapest screen.**
  - A CPU replay on stored checks that include rows after the first miss. Code / math: the live-check rows M-06 RECYCLE used, or the
    map's 16-row blocks. Chat needs one capture with all rows verified (chat rejected-row outputs were never logged).
  - Grid: θ ∈ {0.02, 0.05, 0.1, 0.3}, W ∈ {2, 3, 4, 6}. Price extra rows at 7.82 ms plus misses.
  - PASS: at equal first-order deviation, FLy's kept-vs-deviation curve beats RELAX_350's best rule (R1 / R2) by ≥ +3 % kept on code
    AND chat, net of window rows. Then same-scores with λ (length) ≤ 1.02.
  - KILL: no (θ, W) beats R2 by ≥ +1 % at equal deviation.
- **Grep evidence.**
  - "FLy" has 1 mandatory hit: QUEUE R-61N lists it as published prior art beside Medusa typical acceptance, Judge Decoding and AutoJudge.
  - "2511.22972", "loosely", "entropy gate", "self-correct" are 0 / 0.
  - The 3 / 15 "deferred" hits are deferred MoE / deferred runs.
  - RELAX_350.md, COMMENT_LOOPHOLE.md and SELFCERT.md have no window or re-convergence rule.
  - So the paper is cited by name; its rule was never screened.
- **Novelty tag.** Published (FLy). The cross-check window variant is adapted.

## 3. ToolSpec: schema-aware drafting inside tool calls

- **Paper.** Xia, Li, Du, Song, Li, "ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding",
  arXiv 2604.13519 (Apr 2026, v2 May 2026). https://arxiv.org/abs/2604.13519 , https://arxiv.org/html/2604.13519
- **Mechanism.**
  - A finite-state machine alternates two modes:
    - It deterministically fills schema tokens: keys, punctuation, the argument order fixed by the tool definition.
    - It drafts the variable fields.
  - Separately, it retrieves similar past invocations as drafts.
  - Verification is standard and output-preserving.
- **Why it fits here.**
  - On agent turns our default drafter is DFlash2, trained on code / math windows rather than tool calls. It keeps 4.80 per check where
    the MTP chain keeps 8.20, and agent stays on DFlash2 because of cold experts. Its tool-call structure is a plausible weak spot.
  - Paper ablation (Llama-3.1-8B, API-Bank): full 4.45x, without schema drafting 4.22x (the schema half is worth +5.5 %), without
    retrieval 3.35x.
  - The paper also notes EAGLE-2 / 3 lose to training-free methods on tool-call formats because they were not trained on tool calls.
  - The retrieval half is known in substance here (SUFFIX KILL, MATCH / COPYGATE). The schema half is the new part.
- **Estimate (est.).** Agent kept per check +3-8 % (4.8 → ~4.95-5.2). Zero elsewhere, because the FSM fires only inside
  <tool_call> spans. It can sit beside COPYGATE as one more CPU-side override of the drafted block.
- **Exact?** Yes. The drafts are overridden and the check still decides.
- **Cheapest screen.**
  - CPU only on stored agent walks (HPD-0 agent sessions, the 12 held-out agentic turns, D-23d's agent prompts):
    - (a) the share of DFlash2's first misses on schema-determined tokens: JSON keys, quotes, colons, braces and the closing tag, once
      the tool name is known;
    - (b) a MATCH-style override replay that gives kept per 16-row check.
  - PASS: ≥ 10 % of agent first misses are schema-determined AND agent kept ≥ x1.05.
  - KILL: < 4 % of first misses, or kept < x1.02.
- **Grep evidence.**
  - "ToolSpec", "2604.13519", "finite-state", "tool_call" are 0 / 0.
  - "schema" has 2 mandatory hits: D-08 CALC's one-line remark that calculator drafting "generalizes to other deterministic spans
    (bracket closure, dates, schema keys)", and TYPERULE's prompt keyword rule. Neither was tested on tool calls.
  - "jump-forward" (1 / 2) is SGLang-style forced copy spans, a KILL on napkin, which is a different mechanism.
- **Novelty tag.** Published (ToolSpec). The CPU override beside COPYGATE is adapted.

## 4. SwitchSD: a target-state probe for copy intent, as COPYGATE's gate

- **Paper.** Eisenstadt, Cohen, Cohen-Karlik, Wolf, Zimerman, "To Copy or Not to Copy: Controlling Speculative Decoding via Intrinsic
  Model Signals", arXiv 2609.20186 (Sep 2026). NeurIPS 2026 poster. https://arxiv.org/abs/2609.20186
- **Mechanism.**
  - A linear probe (no bias) on the post-attention residual at one target layer detects "copy mode" with AUC > 0.99. Layer 14 for
    Llama-3.1-8B; the labels are verbatim spans ≥ 5 tokens that occur earlier in the context.
  - Copy runs only when probe ≥ τ (~0.40) and a 5-gram matches. Up to 10 candidates x 20 tokens are verified in one pass.
- **Why it fits here.**
  - HumanEval at T = 0 (paper): copy runs keep 5.88 per round vs EAGLE-3's 2.93, and throughput is +13 % over EAGLE-3.
  - Against CopySpec, copy triggers fall from 27.3 % to 14.2 % of rounds while copy acceptance rises from 1.83 to 5.88.
  - Our COPYGATE gates MATCH with a session duel. Its own analysis (FableGod/copygate/GATE.md) finds that precision at reached
    positions is already high on HumanEval+ (99.1 %) but not on MBPP (76.6 %; 110 harms vs 35 fixes) or agent (91.0 %, x1.032 against
    a line of 1.04).
  - Our 5 taps per position (layers 5 / 15 / 25 / 35 / 45) are already stored, so a probe costs nothing to try.
- **Estimate (est.).** +0-2 % at 16 rows. +2-5 % on MBPP / agent at 32 rows, where copies matter (HumanEval+ x1.204 at 32).
  This is the weakest item: the gate was shown not to be the main limit.
- **Exact?** Yes.
- **Cheapest screen.**
  - CPU: fit a logistic probe on stored taps at match positions. Labels: "copy right at the drafter's first miss", from gate_sim.py.
  - Replay COPYGATE with probe ∧ duel against the duel alone.
  - PASS: MBPP harms down ≥ 50 % while keeping ≥ 90 % of HumanEval+ fixes, and agent ≥ x1.04 at 16 rows.
  - KILL: no gate setting beats COPYGATE's kept on all three sets.
- **Grep evidence.**
  - "SwitchSD", "2609.20186", "copy intent", "copy-intent", "pointer-generator" are 0 / 0.
  - The "probe" hits are SELFCERT (sureness probes) and K-HEAD; neither is a copy gate.
  - CopySpec and Copy-as-Decode appear only in GAPS.md.
- **Novelty tag.** Published signal; using it as COPYGATE's gate is adapted.

---

## Found on the way (other walls)

- **FlexDraft's bonus-guided calibration** (arXiv 2605.20022, May 2026, https://arxiv.org/html/2605.20022) helps the hidden (ANE)
  draft, which is WALL 3, not WALL 2:
  - The block drafter pre-drafts before the bonus token is known. A 2-layer MLP on [mask-position hidden state; embedding of the actual
    bonus token] then adds a logit bias, trained jointly.
  - FableGod/FINDINGS.md item 2 finds our hidden-draft usable share is capped by the bonus guess. It is 0.51 / 0.37 / 0.22 at 3 anchors
    vs 0.72 / 0.62 / 0.76 with a perfect bonus.
  - A small post-check correction (the bias has LM-head width, so ~1-2 ms est.) could move the share toward the cap without
    re-drafting.
  - The calibration's own share of FlexDraft's gain was not in what I read.
  - FlexDraft's main idea, mask rows through the target's last 10 layers, does not fit: mask rows would route experts.
  - Grep: "FlexDraft", "2605.20022", "bonus-guided" are 0 / 0; "bonus guess" has 0 / 7 hits, all in FableGod.

## Already known (in substance) — one line each

- D²SD (2606.04446): a second drafter re-anchors at predicted rejection points, τ 5.31 → 7.05 vs DFlash. It is a sparse tree (SpecBranch / TOP3-USE / D-24); trees priced out (TREES).
- ReTrace (2608.29748): conditions the next DFlash block on the rejected suffix. Recycling family: Carryover 2609.14717 in QUEUE; M-06 RECYCLE KILL.
- D-Loop (2610.06011): a second looped pass re-drafts the suffix from a chosen prefix. Same as SD-semiAR (PARD-style re-drafting) and D-21 REFINE, both KILL.
- DCore, "When Does Better Acceptance Pay?" (2610.24993), controlled run at T = 0, B = 16, Qwen3-8B, low-entropy half: DFlash 6.78, DFlare 7.06, D-PACE 7.46, DCore 7.59, DSpark 8.04, Domino 8.10.
  - These are causal tails (SD-ARtail) and position-weighted losses (SD1 KILL).
  - Worth a 10-minute check: DFlash2's selector-on vs selector-off kept on the HPD-0 walks. If the gap is < +10 % (the study puts a trained tail at ~+19 % over plain DFlash), a trained tail deserves a screen.
- KVShot (2604.26412): the drafter cross-attends to the target's KV. Deep-position retention 73.5 → 77.3 %, but end-to-end MAT is only +0.6 % and drafting latency +5-10 % (paper). Same family as DLite / HCFULL (taps).
- Acceptance-Aware Draft Training (2609.24150): the EAL loss for greedy plus a GRPO stage. Same family as LK losses / Spec-AUF / SD1 (+0-3 % at greedy here).
- PosS (2506.03566), HCSpec (ACL 2026): position-specialist autoregressive drafter layers. Same as SD-AR, KILL on paper.
- SpecLA (2607.16673), STree (2505.14969): tree / chain verification for linear-attention / SSM targets. Same family as Bole / TreeWY; trees priced out on expert cost, not on GDN.
- Oilbird (2608.03839), DReSD, NEST: a draft pool re-keyed by the verifier's hidden state. Same as D-06 PTR (attention as copy pointer, KILL).
- TIDE (2602.05145), Together ATLAS: online drafter adaptation across requests while serving. Same as REGENFREE / L3 / L4 / E2 (own data, +0.22 per doubling).
- PACER (2602.01274), Acceptance-Guided Adaptive SD (ICASSP 2026), AdaEAGLE: trained pre-verification / acceptance heads that set length. Same as STOPCLS / CALSTOP, KILL.
- FlexDraft self-drafting through the target's last layers (2605.20022), "LLM Knows the Future": mask-token drafting through the target. Same as D-07 / GAPS (mask rows cost experts).
- DEER (2512.15176), SpecDiff-2 (2511.00606), DiffuSpec, P-EAGLE (2602.01469), PARD-2: parallel / diffusion drafters. DFlash family.
- UniVer (2605.04543), HSD (2601.05724), SpecTr-GBV (2604.25925), Traversal (2505.12398), TEV (2610.11750): verification rules that gain only under sampling; at T = 0 they are prefix matching.
- Bastion (2605.29727), SpecBlock (2605.07243), CaDDTree, DDTree: trees.
- Make Every Draft Count (2602.21224): recycling plus an autoregressive hidden-state drafter.
- Self-Distillation for MTP (2603.23911), AdaMTP (2608.00434), Nemotron-3 Super's shared MTP heads: MTP head training. Same as D-22 / b5.
- GrowMTP (2609.16648): draft heads trained inside an RL loop. Not for a fixed target.
- Speculative Decoding Across Languages (2605.30580): per-language drafter fine-tunes generalise poorly. Agrees with E1 / D-15a KILL.
- MASSV-style temperature-diverse self-distillation data: drafter data scaling (E2 / DS / DS2, a slow lever).

## Searched but nothing new (queries run, Nov 3; standard mode unless marked)

1. Drafters beyond DFlash / block diffusion, 2026.
2. Hybrid GDN / linear-attention tree verification and state rollback.
3. MoE-target, expert-aware acceptance (lossless).
4. Test-time training of drafters.
5. D²SD.
6. RL-trained drafters for acceptance length.
7. Predicting the rejection position before verification.
8. Acceptance decay with generation length.
9. Multi-draft greedy lossless verification in one pass.
10. Drafters attending to the target KV cache.
11. 2026 code drafters (HumanEval / LiveCodeBench).
12. 2026 MTP-head improvements (Qwen3-Next / Qwen3.5).
13. Prefill-time drafter adaptation (none found).
14. Drafters reusing a hybrid target's SSM state (none found).
15. Conformal / calibrated acceptance prediction.
16. HCSpec.
17. FlexDraft.
18. Target router logits as drafter features (none found).
19. Acceptance theory (2606.30265).
20. Chat / near-tie lossless improvements.
21. Lossy lookahead / window acceptance.
22. Early rejection at intermediate layers in MoE verify (none new).
23. DFlash on hybrid MoE in vLLM / SGLang (model cards only).
24. Draft length predicted from target state.
25. DFlash on MLX / Apple Silicon (community ports only).
26. Training-data diversity for drafters.
27. PosS.
28. SpecDiff-2 self-selection.
29. (extended) October 2026 drafters.
30. (extended) NeurIPS 2026 posters.
31. (extended) Sep-Oct 2026 MoE speculation.
32. Syntax / AST-guided code drafting.
33. Copy-pointer drafters.
34. Agentic / tool-call drafting.
35. Qwen3-Next MTP.
36. kNN hidden-state retrieval drafting.
37. Auxiliary hidden-state losses.
38. Tree Exit Verification.
39. Drafters trained on full precision vs a quantized target.
40. ATLAS / online adaptation.
41. Alignment-augmented SD (EMNLP 2025).
42. Hypernetwork per-prompt drafters (none found).
43. Multilingual drafting.
44. Rejection-prediction heads.
45. SpecLA.

Pages read: TTS (abs + html), TIDE, ReTrace, PACER, Carryover, EAL / GRPO, GrowMTP, KVShot (abs + html), FLy (abs + html),
Acceptance Is Not Speed (html), Practical Investigation of Relaxed SD (abs), SpecDiff-2, FlexDraft (html), D-Loop, DCore (abs + html),
ToolSpec (abs + html), SwitchSD (abs + html), Oilbird, Make Every Draft Count, LongSpec.
