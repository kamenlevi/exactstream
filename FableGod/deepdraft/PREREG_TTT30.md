# PREREG DEEPDRAFT step 3, first small test (Thu 12 Nov 17:xx, before the run): b5 trained to stay right to depth 30

**Origin.** The orchestrator 17:0x (the new week; Kamen's DEEPDRAFT, ~/FableGod/deepdraft/PLAN.md step 3). The
pre-proof (479f299, KILL) showed that a frozen b5 cannot read a separately predicted trunk state, so the reader itself
must learn to stay right deep. Kamen's model, the 125B, is untouched; only the drafter (the MTP head) trains.

**Setup** (scripts/dd_ttt.py):
- **Recipe, unchanged.** D-22's training-time-test chain and its loss, via a verbatim copy of TrainLab sticky/d22_ttt.py
  (scripts/dd_d22_ttt.py): the per-depth KL to the 125B's top-32 at each row, with Spec-AUF accept-until-fail as b5 was
  trained.
- **Start from b5** (TrainLab ckpt d22_b5: depth 15, wdecay 0.9, 1 epoch, 246 steps).
- **This run:**
  - depth 30, wdecay 0.97 (depth 30 keeps 0.41 of depth 1's weight);
  - cosine lr from 1e-5 (warm-up 20), 600 steps, one window per step;
  - per step, a random slice of at most 192 anchors in the answer, for memory;
  - all 69.1M head parameters trainable.
- **Data.** DEEPDRAFT capture night 1 (Magicoder Python, wd / ids / top-32 / asst): 2,164 files, with the last 5 % (113
  files) never trained on.
- **Memory.** Smoke (3 steps): MLX peak 15.5 GB, 4-8 s per step. A guard stops and saves above 18 GB. Run alone under a
  share lock with MEM. Fans at the minimum, so heat pauses; no timing claims.

**Test, held out (prompt sets never trained on here).** The plain chain as TrainLab d23h_mtp31.py runs it (anchors gen -
2, depth 31), for b5 and the new head t30a in the same process:
- the 24 HumanEval+ windows of data/d16w;
- the 100 MBPP answers of dlog/d1102mbpp.

Agreement by depth band counts valid cells only, as in the pre-proof. Kept per 32-row check comes from the same flags: from
each answer's first anchor, the leading right depths + 1, then advance by that.

**Lines (HumanEval+ code).** The baseline is this eval's own b5; the pre-proof read 97.7 % at depth 0 and 85.1 % at 23-30.
- CONTINUE: t30a reaches >= 90 % at depths 23-30, AND its depth 0 is within 1 point of b5's.
- KILL: t30a under 87 % at depths 23-30.
- Otherwise INCONCLUSIVE.
- Also reported: the bands 7-10 and 15-22, MBPP, and kept per 32-row check, for both heads.
- A note, not a line: the training prompts are Magicoder (different from HumanEval+), which tests transfer.
## RESULT (Thu 12 Nov 19:0x): INCONCLUSIVE (data/deepdraft/ttt30/eval.json)

t30a = b5 + 600 steps of D-22's chain to depth 30 on capture night 1. The run finished all 600 steps in 6,766 s, with heat
pauses under the fan minimum (no timing claim). MLX peak was 15.6 GB in training and 10.1 GB in eval. Both heads were scored
in the same process on the same held-out windows. Agreement with the 125B, in %, by draft depth:

| set | head | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept / 32 |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval+ (24 windows) | b5 | 97.7 | 96.0 | 93.2 | 90.5 | 88.9 | 86.8 | 85.1 | 12.55 |
| HumanEval+ (24 windows) | t30a | 97.6 | 96.0 | 93.5 | 91.7 | 90.6 | 89.1 | 87.3 | 13.12 |
| MBPP (100 answers) | b5 | 93.5 | 87.4 | 80.7 | 74.7 | 70.7 | 66.8 | 63.3 | 6.22 |
| MBPP (100 answers) | t30a | 93.6 | 88.5 | 81.9 | 77.1 | 73.9 | 71.0 | 68.5 | 6.54 |

- **Lines.** Depths 23-30 reached 87.35 % against b5's 85.08 %. That is not >= 90 (CONTINUE) and not < 87 (KILL), so the
  verdict is INCONCLUSIVE. Depth 0 moved -0.09 pt, within the 1-point line.
- **Shape.** The gain grows with depth: +0.0 pt at depth 0, +1.2 at 7-10, +2.2 at 15-22, +2.3 at 23-30. It transfers from
  Magicoder to both held-out sets, and more to MBPP (+5.1 pt at 23-30). Kept per 32-row check rose 4.5 % on HumanEval+ and
  5.1 % on MBPP.
- **Scale.** The run saw at most 115K anchors (600 steps x 192), about 19 % of night 1's 604K answer tokens, once, at
  lr 1e-5 (cosine).
- **Harness.** b5 reproduces the pre-proof's 97.7 / 85.1 (MBPP 93.5 / 63.3) and TrainLab's kept 12.55 (MBPP 6.22 here vs
  6.23 there).
- **Not measured.** The eval stores pooled cells only, so there is no per-window spread. A 2.3-point pooled move on 24
  windows is a direction, not a significance claim.
- **Next (the orchestrator, 19:1x).** The same recipe for about 3,000 steps, one full pass over night 1, with the curve at
  600 / 1,500 / 3,000 steps, pre-registered first. It decides whether more capture nights are worth it. The carried-state
  half of PLAN step 3 comes after that verdict.
