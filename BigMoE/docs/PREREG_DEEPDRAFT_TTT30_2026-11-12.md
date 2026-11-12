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
