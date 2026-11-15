# LEVERS_A: the ranked GOAL A lever list under the drafter spec (NewIdeas, merged with LEVERS_A_MOONSHOTS.md)

NewIdeas, Sun 15 Nov 2026, 01:1x (`date`). It builds on reports/Deep drafting and fast verification.md and WEEK98.md's log. A
published negative is read as a hypothesis about that paper's setup, not a veto (Kamen 01:0x). Prior-art lines come from
quick searches: "untested" means not found, not proven new.

## STATE Sun 15 Nov 11:47 (`date`): this supersedes the ranking below (kept for the record)
Kamen ~10:4x: chase the EXACT drafter, with the check at equal priority; lossy close-call is the side track.
- **The wall:** kept/32 is lost at depths 0-14 right after a miss. There the 125B itself is unsure, and ~90 % of
  check-start misses sit on its near-ties (MTP Head S6). No reader predicts those.
- **Exact lead = TREES** (MTP Head S9c on BigMoE's real t30b trees): tree-16 x1.136 (MBPP) / x1.060 (HE+) kept vs
  chain-16 at the same rows, net ~+11-13 / +4-5 %. Tree-32 pays once CHECK55 brings R32 to ~1.3. The build lands ~Thu.
- **Sun 21:3x PLAN4D (HE+ deep bands, 4 days):** research/week98/PLAN4D_HEPLUS.md. 98 % per band to 30 is out of
  reach on HE+ this week (true-state head 97.76 flat). Exact ranking for the deep bands: 1 the 2B audit (+4.7 at 23-30),
  2 a full-data distilled auditor (NEW), 3 tree at 0-2 + audited chain (NEW), 4 t30c (Tue night), 5 state arms.
- **Ranking now:** 1 trees; 2 CHECK55; 3 S5b as a screen only; 4 the 2B audit as a tree-branch proposer, only via the
  ANE; 5 lossy close-call (side track).
- **Results of tonight's screens:**
  - S1 DF-TF KILL: strict HE+ 23-30 88.1, TF kept 11.76 vs native 11.46.
  - 1c KILL as screened; data scaling is closed.
  - S5 VOID, S5b building.
  - S4 depth KILL (every init).
  - ALLROWS and kNN closed.
  - Training levers (WALKW, ACCSET, margin loss) closed.
- **Correction:** research/week98/roottree.txt is VOID (the cells' `rank` field is not the head's rank). The margin
  ceilings (margin_ceilings.txt) stand.

## What the spec forces (numbers from WEEK98.md and TRIED)
- **The spec.** >= 98 % per guess to depth 30. Reads <= ~0.5 GB of weights per 32-token block. Compute is hidden or
  trivial, which leaves ~2-4 ms of drafting per block at 400 exact.
- **Today's chain head breaks it on cost, whatever its accuracy.**
  - One MTP step costs ~2.2 ms, of which the lm_head is 1.47 ms (TRIED HEADBENCH, 11-03). 30 steps come to ~66 ms per
    block.
  - The screened 32K head is built (TRIED MTP-STEP-1: 1.257 -> 0.17-0.22 ms per call), which still leaves ~25-30 ms per
    block.
  - WEEK98 records 5.4 GB per block.
  - A sequential drafter fits the spec only if one step reads <= ~17 MB (0.5 GB / 30) and runs in ~0.1 ms.
  - **Measured since (Leap P2 CHAIN-BYTES CLOSED, 6e635f1):** the head alone costs 1.147 ms per step as stored
    (177.6 MB). The best cut is 0.864 ms at 95.1 MB (4-bit Linears + 8K head). The fit is ~0.54 ms fixed per step
    (launches) plus bytes at ~290 GB/s, so a 32-step chain costs >= ~17 ms in MLX's per-op form at any bytes. The
    2.2 ms above is the in-engine step. Only a FUSED-step chain could reopen sequential drafting (Leap's optional
    cache-knee probe).
- **A one-pass block drafter fits on bytes.** DFlash2 reads 0.52 GB per 32-row block. It takes 10.96 ms today, and its
  byte floor at 280 GB/s is ~1.9 ms, so the gap is kernel work.
  - Its accuracy is the problem. At block 32 one pass keeps 10.51 tokens (TRIED L1-20), against the chain head's 13.8 on
    HE+.
- **So under the spec, the Goal A drafter is a block drafter.** That means one pass, or 2-3 refinement passes of
  <= ~0.25-0.5 GB each, holding >= 98 % per position to 30.

## The structural point: drift lives in a carried state, so take the carry out
- **Why the chain decays.** It feeds a 1-layer head's own output back into itself 30 times. Its depth through the block
  is the number of steps, and every step compounds. The CARRY decays within ~4 steps of a refresh (Sandbox98
  KEYFRAMES).
- **A block model has nothing to carry.** It recomputes all 32 positions from TOKENS through K layers, causal within the
  block, reading the 125B's prefix features. At depth j, its accuracy with correct earlier tokens is a fixed function of
  j and K.
  - That accuracy is its teacher-forced accuracy, and it is the fixed point that Jacobi refinement converges to.
  - xPress (arXiv 2608.02438) runs K parallel Jacobi passes over a diffusion drafter's block with the drafter features
    held fixed, which is a fixed-point iteration.
- **It is trained teacher-forced:** parallel and stable, with no TTT chain and no survival weighting (Sandbox98's AUF
  bias).
- **Evidence for:**
  - The head reads a fresh state at 97.8 % flat (HE+), so a good per-position representation is enough.
  - The 2B is a token-composing model with no access to the 125B, and it is already flat at 94.4 % (HE+) and 85.7 % (MBPP)
    untrained (MoonShots).
  - T2: pretrained composition over a local window recovers most of the gap.
- **Unknown, and the first screens:** how high a K-layer, target-fed block model's teacher-forced accuracy stays at
  depths 23-30. One-pass masked (10.51 kept) is not that number.

## Update 01:1x: the audit result turns this into one target
MoonShots' 2B AUDIT on MBPP 100 (a489724; HE+ due tonight): where the 2B disagrees with the head at margin >= tau (tau 4 /
2 / 1 / 0.5 by depth, picked post hoc), the 2B's token stands. 23-30 goes from 72.13 to 85.14, and depth 0 holds at
93.96.
- The drafter becomes FLAT at the auditor's own teacher-forced agreement (MBPP 85.7, HE+ 94.4), and never above it.
- This is the structural point above, measured: a token-composing corrector's fixed point is its teacher-forced
  accuracy.
- **So Goal A reduces to one target: a corrector with >= 98 % teacher-forced agreement with the 125B, at <= ~0.5 GB per
  pass (or on the Neural Engine).**
- Leap P1 (02d58c6): a second GPU stream runs almost serially beside the check (kappa 0.93). So GPU drafting is never
  hidden. A dense drafter on the ANE overlaps at x1.07 on the check.
- Three corrector families, best placed first:
  - (a) **target-fed block models** (DFlash2 class, 0.52 GB): they see the 125B's prefix features, so they could beat any
    standalone small model. Screen S1.
  - (b) **distilled same-family small LMs** (the 2B at 1.06 GB, or Qwen3.5-0.8B at ~0.6 GB, which MoonShots is
    fetching): pretrained composition, no 125B features. Screen S2 = MoonShots' lever 1c.
  - (c) **both at once**: a small pretrained LM with the 125B's prefix features injected into its layers (DFlash-style KV
    injection into the 0.8B / 2B). This is a build. Its sandbox analog (the 0.8B fed the 2B's taps, target = the 2B) is
    S5.

## METRIC RULE (orchestrator 02:2x, after MTP Head caught ALLROWS): judge by kept/32 or the STRICT conditional
Every drafter lever is judged by kept/32 or the strict conditional per-depth accuracy (P(right at j | right at all < j)).
Bands are information only: they count rows past an earlier miss, which the engine never drafts. ALLROWS lifted the bands
by +8.58 / +5.75 and left kept/32 flat (MBPP-50 4.076 -> 4.066, HE-30 5.791 -> 5.687).
- Re-check of this list for band-only evidence:
  - #2, the 2B audit: the bands rose +13.0 (MBPP 23-30) / +4.7 (HE+), but kept rose only x1.09. So "flat at the auditor's
    agreement" is a BAND law, and its deployed worth is +9 % accepted.
  - #1, the corrector framing: it holds for full-context correctors, since teacher forcing = a correct prefix. But its
    target is now strict >= 98 % per depth to 30, i.e. kept/32 ~23.8. S1's lines move to strict + TF kept/32 (Fit12
    asked 02:25). S5's agreement is a per-row proxy; report kept/32 too.
  - #3, depth: the earlier sandbox depth verdicts (HURTS / NO EFFECT) were read on bands. S4 must read kept/32.
  - MoonShots' perfect-chooser bound (96.0 / 99.4 at 23-30) is a band bound.
  - Unaffected: the fresh-state oracle (kept/32 13.77 -> 20.90 HE+, 6.92 -> 12.76 MBPP) and WHERE THE ERRORS LIVE (depth 0,
    where strict = band).

## Ranked list (merged; MoonShots' numbering in brackets)
| # | Lever | Spec fit | Prior art | Screen / owner |
|---|---|---|---|---|
| 1 | **Jacobi block corrector**: a target-fed K-layer block model run to its fixed point (1-3 passes) | bytes fit (<= 0.5 GB per pass); kernels for time | xPress 2608.02438 (Jacobi on diffusion drafters), PARD 2504.18583, DFlash / DFlash2, DFlare 2606.02091, SpecBlock 2605.07243. Ours: run to its fixed point, measured flat-vs-depth to 30 on a 125B MoE, 2B-initialized | S1 DF-TF (Fit12: built, armed; DFlash2 is bidirectional, so one pass per position; starts after Sandbox98's run, ~02:46), S2 = 1c (Sandbox98, MoonShots' lever1c_distill.py), then S3 JACOBI-K |
| 2 | **2B audit / repair + a learned chooser** [1, 7] = #1 with the chain head as the first guess and the untrained 2B as the corrector | 1-3 x ~1.1 GB per block today: over the spec until the 2B shrinks (32K vocab head, 3-bit body, fewer layers; MoonShots' bytes_2b.py) | Staged SD (Spector & Re 2023), Cascade SD (Chen 2023) | MEASURED: MBPP 23-30 72.13 -> 85.14 (a489724), HE+ 23-30 89.65 -> 94.32 (frozen rule; depth 0 -0.05; accepted x1.09), flat at the auditor's 85.7 / 94.4. 1.06 GB per pass, ~+5 % net on the GPU; the ANE route is being priced. Next: S2 = 1c (distil it) |
| 3 | **Depth done fairly** (Kamen's rule): a copied non-zero init, its own higher lr, the full budget, a per-layer target-tap mix, borrowed 2B layers | per step x layers on a chain head (over the spec); free in #1's block model | DFlare 2606.02091 (depth keeps paying with per-layer target fusion + 3x data); SPIRe and 2505.15380 (copy the target's top layers as the drafter's init); FlexDraft vs Mixture of Attentions | S4 DEPTH-FAIR (MTP Head tonight, T30D spec); the block-model version in S3's follow-up |
| ~~4~~ | ~~kNN state memory~~ [3]: **KILL** (Sandbox98 f3335fd: retrieved states are worse than the head's own carry on the cells they replace) | | | |
| 5 | **Code-structure tracker** [4] | ~free (CPU lexer) | DOMINO, Verilog SD: adapted | its 2-min diagnostic needs a per-row dump with token ids (the cells file has none): BigMoE's harness |
| 6 | **Per-answer online adaptation** [6] | ~0.1-0.2 ms per token | OSD 2310.07177 (across requests): adapted | Sandbox98, later |
| 7 | **Cheap head bytes** (screened lm_head, built) | helps the chain head only; it can't reach the spec alone | FR-Spec; TRIED MTP-STEP-1 | parked under the spec |
| 8 | **CPU / ANE concurrency** [5] | hides compute, not bytes (the check is bandwidth-bound) | Mirror-SD 2510.13161; TRIED CPUMTP KILL (CPU lm_head 20 ms) | only if #2 wins |
| 9 | **Close-call acceptance** (lossy) | free | Sharma 2606.30265 (theory) | end piece: +0.2-1.7 pt (CEILING) |
| - | Closed / low: trunk refresh (cost = a verify), ROLLOUT, width / capacity, FRESH-H adapter, prefix KV (GLIDE / KVShot), recurrent memory, post-norm | | | |

## Screens (each <= 15 min; the result -> action line written first)
**S1 DF-TF: DFlash2's teacher-forced ceiling at block 32** (Fit12, its 12 GB engine or the BANK's taps; <= 12 GB, share).
- At each anchor on MBPP 100 + HE+ 24, read DFlash2 (l120_ft)'s top-1 at every block position with the TRUE earlier
  block tokens visible and the later ones masked. If its in-block attention is bidirectional, that means one pass per
  position j with masks at >= j; if causal, one pass. Labels: the 125B's greedy (bank tk[:, 0], or the engine's exact
  greedy).
- Report bands 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30 and kept/32. Also report its native masked one pass on the
  same anchors as the control (it should reproduce L1-20's ~10.5 kept, +/- 0.5).
- **LINE:**
  - PASS if teacher-forced HE+ 23-30 >= 95.0, within 3 pt of its own depth 0, and MBPP 23-30 >= 85 -> DFlash2 is the Goal A
    base: S3 JACOBI-K on it, then a block-32 teacher-forced fine-tune.
  - BETWEEN (HE+ 23-30 90-95) -> it is capacity-limited: a 2B-initialized corrector (S2) or DFlare's per-layer target mix.
  - KILL if < 90 (no better than the chain's 89.7) -> a 5-8-layer target-fed block model doesn't compose 30 tokens;
    borrowed depth decides (S2).

**S2 = lever 1c, 2B-DISTILL: can a same-family small model be distilled toward the 125B?** (OWNER Sandbox98 from 01:2x.
MoonShots' sub1 runs out ~02:15, and it could start only after 04:00. Uses MoonShots' lever7_agree.py. MLX, <= 5 GB,
share.) MoonShots' line stands as its pass line (MBPP agreement >= 89.0). The lines below add the
information reads that decide whether the route reaches the 98-class.
- LoRA (rank 16-32; top layers plus the head) on Qwen3.5-2B, trained on the 125B's greedy answers: the n1112 train
  captures plus the BANK's 360 train files, hard next-token labels (or the top-32 KL where stored). Train <= ~10 min,
  <= 1 epoch.
- Read-out: teacher-forced top-1 agreement with the 125B (bank tk[:, 0]) on HE+ 24 / MBPP 100 / val 40, overall and on
  sure rows (>= 4 nats). Use MoonShots' ~/MoonShots/week98/lever7_agree.py, which gave 94.42 / 85.67 / 87.29 untrained.
- Control: the untrained 2B in the same harness, within +/- 0.1 of those.
- **LINE:**
  - PASS if HE+ >= 97.0 and MBPP >= 91.0, with val up too (not overfitting) -> the distilled 2B is the corrector for S3
    and for MoonShots' audit; then cut its bytes to <= 0.5 GB per pass.
  - BETWEEN (HE+ +1-2.6, MBPP +2-5) -> more data or a full fine-tune before deciding.
  - KILL if < +1 pt on both -> distilling a 2B toward the 125B in our data budget doesn't close the gap; #1 rests on S1.

**S3 JACOBI-K** (after S1 / S2; Fit12 or BigMoE).
- K = 1 / 2 / 3 refinement passes of the best corrector, from (a) its own masked guess and (b) t30b's chain draft.
  Report kept/32 and bands per K.
- **LINE:** PASS if K <= 2 reaches HE+ kept/32 >= 18 (the tap-45 ridge's level; the oracle gives 20.9) at <= ~1 GB per
  block -> Goal A's method is the corrector; train it at block 32, teacher-forced.

**S5 INJECT (sandbox analog of family (c), after 1c; Sandbox98).** Target = the 2B. The 0.8B download needs Kamen's own
OK, so the corrector is the 2B's OWN first 12 layers + its lm_head (pretrained composition, early exit), with a
zero-init adapter that injects the FULL 2B's final prefix states (KV injection). The 0.8B version follows only if Kamen
OKs the download.
- Short LoRA plus adapter training on the sandbox's answers. Read out the 0.8B's teacher-forced agreement with the 2B, on
  MBPP-50 / HE-30, with and without injection.
- **LINE:** PASS if injection adds >= +3 pt of agreement over the plain 0.8B (p < 0.05) -> build family (c) on the 125B
  (the 0.8B / 2B with the 125B's taps); else KILL (c), and the families compete on S1 / S2 alone.

**S4 DEPTH-FAIR** (MTP Head, tonight). T30D per BigMoE's spec with Kamen's rule: a copied (non-zero) init, its own higher
lr, the full budget, read-out stop lines at 420 / 1,050 steps. Untested so far: an UNTIED copied init with its own lr and
a long budget. tie2 (tied) and deep2 (zero-init) were short, re-annealed and converged.

## Sources (quick searches, Sun 15 Nov 01:0x-01:1x)
- xPress, parallel refinement for diffusion drafters: https://arxiv.org/abs/2608.02438
- SpecBlock, block-iterative SD: https://arxiv.org/abs/2605.07243
- PARD, parallel draft model adaptation: https://arxiv.org/abs/2504.18583
- DFlare, scaling draft capacity for block diffusion SD: https://arxiv.org/abs/2606.02091
- SPIRe (draft init from the target's last two blocks): https://matx.com/research/sd
- Speech SD with copied upper layers: https://arxiv.org/abs/2505.15380
- DistillSpec: https://arxiv.org/abs/2310.08461; AdaSPEC: https://arxiv.org/abs/2510.19779; Draft-OPD:
  https://arxiv.org/abs/2605.29343
- Consultant Decoding (Qwen2.5 0.5-7B drafts for 72B): https://arxiv.org/abs/2506.02391

## Pending alignment
EvalSuite is writing GOAL A's certification protocol (CERTIFY98.md: metric, held-out discipline, sample size, decision
rule). Every corrector line above gets re-stated in its metric when it lands.
