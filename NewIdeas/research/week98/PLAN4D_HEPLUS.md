> **CORRECTED Sun 21:5x (`date`): read PLAN4D_REAIM978.md first.** Kamen's target is now 97.8. Item 3 is a KILL on HE+.
> The C2 line below (+2.0 pt in every band 3-30) already FAILS on dev: band 3-6 is +1.2 marginal and +0.4 walk-strict.
> It is replaced by walk-strict 15-22 / 23-30 >= 97.8 point with LB >= 97.0 (P = 2 headline), with kept/32 co-primary.
> Item 1's "expected points" are MARGINAL per-cell bands (unlimited repairs). The gate is walk-strict, where the deep
> bands are already ~97.5-98.3 with any audit and the shortfall is bands 1-14.

# PLAN4D: HE+ deep bands, Mon 16 to Sat 21 Nov (NewIdeas, Goal A lead)

Sun 15 Nov 21:3x (`date`). This answers the orchestrator's 21:2x ask: "a ranked plan for the 4 days left that gets t30b's
own deep accuracy on HE+ as close to 98 % as possible, certifiable Thu-Fri. Also say honestly what MBPP can reach."

Inputs:
- MTP Head's D98 drift brief (TrainLab 72c27e4);
- MoonShots' audit record (LEVERS_A_MOONSHOTS.md, sections B-K);
- PREREG_TTT30B / TTT30C;
- BigMoE's ttt30b run.log and night.log;
- the capture logs (~/FableGod/deepdraft/capture);
- Sandbox98's 1c log (logs/lever1c_a.log, sb/lever1c.py).

Desk only. No GPU was used tonight.

## The answer in four lines
- **98 % per band to depth 30 is out of reach on HE+ this week.**
  - Even a perfect state gives only 97.76 % flat. That oracle is the head with the 125B's true state.
  - The best measured exact fix of the deep bands, the 2B audit, holds HE+ 23-30 at 94.3. The 4B audit holds 95.3.
- **Realistic exact target by Saturday:** HE+ bands 3-30 at ~94.5-95.5 with an audit, and depth 0 at ~98 under the
  tree. t30b alone today sits at 94.5 / 93.1 / 92.2 / 91.0 / 89.7.
- **On HE+, the audited deep bands sit at the auditor's own agreement with the 125B** (MoonShots' law):
  - the 2B agrees 94.42 % and its audited 23-30 is 94.32;
  - the 4B agrees 95.48 % and its audited 23-30 is 95.29.
  - So the cheapest way to raise HE+'s deep bands further is a better auditor, not a better head.
- **MBPP is knowledge-limited:**
  - The true-state head is 93.6 % flat.
  - 98 % would need 93.4 % on the unsure rows, against 79.6 % today.
  - Audited, MBPP 23-30 reaches 85.1 with the 2B (in-sample) and 87.2 with the 4B, against 72.1 for t30b alone.
  - Exact MBPP bands 3-30 can reach ~85-88 this week. 98 is out of reach.

## Ranked levers
Expected points are HE+ bands 3-6 / 7-10 / 11-14 / 15-22 / 23-30 in the band convention, from the records.

The certificate gates walk-strict accuracy, so every lever gets a walk-strict re-read on Monday. ALLROWS showed that bands
can rise while kept stays flat.

Every lever here is exact: it changes only the proposals, and the 125B check verifies every token.

| # | lever | expected (HE+ 3-30, from records) | MBPP 23-30 | build / run | memory, GPU | slot | kill-test Mon by day |
|---|---|---|---|---|---|---|---|
| 1 | **t30b + frozen 2B audit** (tau 4 / 2 / 1 / 0.5, P = 1; P = 2 reported) | +1.2 / +1.7 / +2.5 / +3.6 / **+4.7** (to 95.6 / 94.8 / 94.7 / 94.5 / 94.3), out of sample (HE+ 24); kept/32 +5.0 % (P = 2: +8.9 %) | 72.1 -> 85.1 (in-sample); kept +8.4 % | none for the certificate (MoonShots' join + checkstart.py exist); an EvalSuite amendment by Wed 18:00 | the 2B's teacher-forced pass: GPU minutes at 2-4 GB; join on CPU | none (Thu by day on the fresh cells) | CPU ~10 min: walk-strict per band from checkstart.py on HE+ 24 / MBPP 100. KILL as a per-band candidate if HE+ walk-strict 3-30 gains < +2 pt |
| 1b | the same with the **4B** (high-water mark) | 23-30 **95.29** (+5.6); kept +11.8 % | 87.17; kept +12.0 % | as 1 | 2.37 GB read per pass | none | the same read |
| 2 | **NEW: full-data distilled auditor** (1c rerun at ~8.4x the rows) | lifts the audited level itself: +0.5-1.5 if the 2B's HE+ agreement rises that much (P(PASS) ~0.3) | +1-3 over 85.1 | Sandbox98's sb/lever1c.py on all captured files except the bank's val 40 (3,032 files, ~800k answer rows) instead of the bank's 360 (95k rows); 1 epoch, higher rank | <= 7 GB (1c's guard), ~30 min kill + ~1 h full | none (by day) | ~30 min: read at 5 / 10 / 15 min on HE+ 24 / MBPP 100 / val 40. KILL if HE+ < +1.0 at every read or val falls |
| 3 | **NEW: tree at depth 0-2 + audited chain beyond** | depth 0 at the tree's ~98 (walk: 93.33 -> 98.23), 3-30 at the auditor's level; the two act at different depths (audit: walk-strict d0 only 91.6 -> 92.7) | as 1 | desk join on the existing dumps: tree_t30b_heplus24.npz + rows2b_heplus.npz (NewIdeas) | CPU | none | 1-2 h desk: KILL if the composed kept/32 <= max(tree-16, audited chain) + 2 % on HE+ 24 |
| 4 | **t30c: a second pass from t30b** (fresh cosine) over t30b's split + the **795 files t30b never saw**, minus the bank's val 40 (~1.35x data) | +0.3-0.8 at 3-14, **+0.5-1.5 at 23-30** unaudited; ~0 to +0.5 under an audit (the plateau is the auditor's) | ~+1 unaudited | BigMoE's trainer, ~5,740 steps x ~4.9 s = **~8 h**; eval ~30 min | **15.6 GB MLX** train, 10.1 GB eval | **Tue night only** (over the ~15 GB day limit; needs Kamen's app OK) | MTP Head's sandbox second-pass pre-screen (s30 + pass 2 vs s30), ~40 min, <= 5 GB. Over 15 min, but nothing cheaper tests a second pass. Don't spend Tue night if it gives < +1.0 at 23-30 |
| 5 | carryac / tapmlp at full budget | sandbox +0.6-0.8 at 400 steps; at 125B scale t30f's feature loss was NO EFFECT (-0.12 to -0.93 in every band) | ? | sandbox arms, 20-25 min each | <= 5 GB | competes with 4 for Tue night | only with >= +2 pt in the sandbox (MTP Head's line) |

### Where each number comes from
- **1:**
  - MoonShots' join_heplus.py on MTP Head's cells_t30b_heplus.npz (186,465 cells), schedule fixed on MBPP 100 beforehand.
  - Bytes: 1.06 GB per pass, untimed.
  - At a 52-55 ms check and 4-9 ms per pass it nets -10 % to +1 %. So it is a per-guess lever, not a speed lever at the
    floor.
  - Real-loop timing is a later engine job: it needs the 2B beside the 125B and per-row GDN state checkpoints.
- **2 (primary record):** 1c trained a rank-16 LoRA on the 2B's layer 23 only (639k parameters, cached layer-23 inputs,
  no Gated-DeltaNet backward).
  - Data: the bank's 360 train files, the first 360 of the 795 n1112 files t30b never saw (the next 40 are its val).
  - Peak: HE+ +0.44 / MBPP +0.52 / val +1.43 at 1-2 min.
  - By epoch 4 it overfit (MBPP -0.52), giving KILL against a 89.0 line.
  - A one-epoch run over ~8.4x the rows was never made. WEEK98's 06:3x arithmetic skipped the probe because it was
    aiming at 98 on MBPP's unsure rows.
  - For HE+'s deep bands, the bar is the auditor's agreement: 94.4 now, 95.5 for the 4B.
  - Limit: only layer 23 (and the head) can train in this form, which caps capacity.
- **3:** the tree's depth-0 coverage comes from TREE_LINE / amendment 6; the audit's d0 from MoonShots' checkstart.py.
  - MoonShots: "with a tree, the audit fixes only the path it audits"; this has not been measured.
- **4:**
  - Budget: t30b's s3000 -> final gave +1.0 at 23-30 and +0.3 at 3-6, confounded with the anneal. The loss was flat at
    0.195-0.201 over the last 1,000 steps.
  - Data: ~+1 pt per doubling (t30a -> t30b), so +0.4 for 1.36x.
  - The sandbox's budget +7.1 pt was early on its curve, from native.
  - The 795 files are the Fri 13 Nov 15:08-22:09 capture continuation: magicoder prompts, 207,927 answer tokens. There
    are 3,072 files on disk against t30b's 2,164 train + 113 held out.
  - The bank (1c's 360 train / 40 val) is drawn from these 795. t30c's manifest must hold out the bank's val 40 and any
    file a dev read uses; BigMoE pins it.
  - Steps: (572,381 + ~208k) anchors at t30b's ~136 per step.

### Closed in the records (not ranked)
- capacity: t30x2 -0.79, NO EFFECT; t30x4 -1.51, HURTS;
- the feature loss: t30f -0.93;
- a depth-fair init (S4 KILL);
- DFlash2 TF (S1 KILL);
- ALLROWS, kNN, WALKW, ACCSET, margin loss;
- tree calibration and tree shape (KILL);
- a learned chooser for the audit (adds nothing; the oracle "head OR 2B" at 23-30 is 96.29);
- trunk re-anchoring (a verify each time);
- a new training capture (no night left to train on it before Wed 18:00).

## Calendar (by day: <= ~15 GB of research, >= 5 GB free, share locks, one GPU job > 4 GB at a time)

| when | what | who |
|---|---|---|
| Mon by day | walk-strict re-read of the 2B / 4B audit (CPU, item 1's kill) | MoonShots |
| Mon by day | full-data distilled auditor kill-test (item 2; <= 7 GB; the first > 4 GB GPU job) | Sandbox98 (proposed) |
| Mon by day | sandbox second-pass pre-screen, then state arms if the queue allows (items 4-5; <= 5 GB, after item 2) | MTP Head |
| Mon by day | tree + audit composition join (item 3; CPU) | NewIdeas |
| Mon night | full (M4, E14, close-call); no Goal A GPU | - |
| Tue by day | if 2 passed: full run (~1 h, <= 7 GB), freeze the checkpoint (sha), re-run the join (CPU). If 4's pre-screen passed: pre-register t30c, ask for Kamen's app OK | Sandbox98, MoonShots, MTP Head / BigMoE |
| Tue night | t30c (~8 h, 15.6 GB, alone), if approved and pre-screened; otherwise free | BigMoE |
| Wed by day | t30c eval against t30b in one process (~30 min, 10 GB); TOPK / TREE dumps if it PAYS (5-10 min each); S9c walk | BigMoE, MTP Head |
| Wed ~18:00 | EvalSuite amendment: candidate checkpoints (sha), auditor (sha, tau, P) and lines | EvalSuite, NewIdeas |
| Wed night | the capture (unchanged; the audit needs only the capture's ids, cells and the 125B's top-32) | EvalSuite |
| Thu-Fri | proofs: walk-strict per band, coverage and T per pinned candidate on HE+ 140 / MBPP 300; the auditor's pass on the fresh cells (GPU minutes, 2-4 GB, by day) | EvalSuite |

## Candidates to pin by Wed ~18:00
- **C1:** t30b + EAGLE-2-16, already pinned (amendment 6 / 6a).
- **C2:** t30b + frozen 2B audit, P = 1.
  - Proposed per-guess line: HE+ walk-strict bands 3-30 each >= t30b + 2.0 pt, with the window-clustered 95 % LB above
    t30b.
  - No speed claim.
- **C2b:** the distilled auditor in C2's place, only if item 2 passes by Tue.
- **C3:** t30c in t30b's place, only if it PAYS on dev by Wed ~15:00. Line: HE+ 23-30 >= t30b + 1.0 and no band below
  t30b - 0.3.
- **Information:** the 4B audit, and the tree + audit composition if item 3 passes.
