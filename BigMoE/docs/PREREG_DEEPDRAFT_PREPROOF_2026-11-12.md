# PREREG DEEPDRAFT pre-proof (Thu 12 Nov 15:1x, before any run): a learned stand-in for the trunk state, read by b5

> (a quote from the project owner, kept private)
pre-proof"). The drift check (data/deepdraft/drift.txt) found that the trunk state's token-specific part decorrelates
within about 5 tokens, while the window-level part persists. So a stand-in must be re-derived per drafted token from the
last exact state.

**Reader.** The MTP head b5 (TrainLab d22_b5, D-22's training-time-test chain), unchanged. In its chain, depth 1 reads
the trunk's exact widened stream W at the anchor's row. Each deeper depth reads the head's own previous output h, which
is the lag that costs 97.7 % -> 86.6 % at depths 23-30 on HumanEval+ (TrainLab d23h).

**The helper (new; the 125B and b5 untouched).**
- A recurrence: z_0 = W_s, the exact stream at anchor s; z_k = z_(k-1) + f(z_(k-1), e(ids[s+k])), where e is the 125B's
  own token embedding (frozen).
- f: a per-stream RMS-normalised z (4 x 2560) projected to 1024, plus e (2560) projected to 1024, a GELU MLP (2048), and
  a projection out to 10240. About 36M parameters, the only trained weights.
- Training: teacher-forced unrolls of K = 8 from random starts in the capture's rows. The loss is the per-stream
  normalised MSE between z_k and the exact W_(s+k): scale-invariant, as the head's pre_fc_norm_hidden reads it. AdamW,
  lr 3e-4, a fixed budget of 2,500 steps x 32 unrolls for every data fraction.
- Training data: DEEPDRAFT capture night 1 (~/FableGod/deepdraft/capture/n1112: Magicoder Python, 2,277 prompts, wd +
  ids). Fractions 1/3, 2/3 and all are nested prefixes in file-name order. The last 5 % of files is the helper's own
  validation set and is never trained on.

**The test (held out: whole prompt sets never seen in training).** b5's chain exactly as d23h_mtp31.py runs it,
teacher-forced to depth 31, with ONE change. At each depth j >= 2 the head reads the helper's z_(j-1) for its anchor in
place of its own previous output h. Keys, values, the TTT attention and the teacher-forced tokens are unchanged.
- Code: the 24 HumanEval+ windows (TrainLab data/d16w, the d23h code set).
- MBPP: the 100 MBPP answers of data/dlog/d1102mbpp (the d23h mbpp set).
- No chat set has the trunk's wd stored; chat is not run.
- Bands as in drafterwall t2_laglaw.py: depth 0 (= chain depth 1), 1-2, 3-6, 7-10, 11-14, 15-22, 23-30.

**Lines (HumanEval+ code, depths 23-30).** Gap recovered g = (acc_helper - acc_b5) / (acc_b5 depth 0 - acc_b5). The b5
figures are this run's own plain chain, which must reproduce d23h's 97.7 / 86.6.
- PASS: g >= 0.40, i.e. about 91.0 % or more.
- KILL: g < 0.15, i.e. under about 88.3 %.
- Otherwise INCONCLUSIVE.
- Also reported: bands 7-10 and 15-22, the MBPP curve, the helper's validation loss, and g for each data fraction (the
  slope).
- Depth 0 is unchanged by construction.

**Controls (same run).** The plain b5 chain (the reproduction check), and the stale stand-in z_k = W_s with no update.

**Limits.** At most 8 GB per GPU process (Kamen's apps are open), under a share lock with MEM. Fans at the minimum,
with heat pauses, so no timing claims. Stop at main 96 %; continue after the 17:00 reset if not done.

## RESULT (Thu 12 Nov 15:3x): KILL (data/deepdraft/preproof/eval.json)

**Per-depth agreement, HumanEval+ code (24 held-out windows).** b5's top-1 equals the 125B's token; depth 0 equals chain
depth 1, which every arm shares.

| arm | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 |
|---|---|---|---|---|---|---|---|
| plain (b5's own chain) | 97.7 | 96.0 | 93.2 | 90.5 | 88.9 | 86.8 | 85.1 |
| stale (z = W_s, no update) | 97.7 | 59.1 | 24.3 | 17.2 | 17.8 | 16.2 | 13.5 |
| helper f1 (1/3 of the data) | 97.7 | 17.4 | 16.8 | 19.5 | 0.0 | 0.0 | 0.0 |
| helper f2 (2/3) | 97.7 | 51.8 | 26.4 | 23.0 | 21.4 | 19.8 | 17.9 |
| helper f3 (all) | 97.7 | 17.6 | 16.5 | 16.8 | 10.1 | 0.0 | 0.0 |

- **Verdict.** The gap recovered at depths 23-30 is g = -6.76 for f3 (f2 -5.34, f1 -6.76) against PASS >= 0.40 and
  KILL < 0.15: KILL.
- **MBPP shows the same pattern.** The plain chain runs 93.5 -> 63.3; f2 gives 49.1 at depths 1-2 and 17.8 at 23-30.
- **Validation (the helper's own).** Per-stream normalised MSE at steps 1 / 8: f2 0.62 / 0.88, f1 0.91 / 0.93, f3 0.88 /
  0.93, against stale 0.86 / 1.19. f1 and f3 barely beat stale and diverge beyond their 8-step training horizon (the 0.0
  past depth 10). No data slope can be read: one seed, and the instability dominates.
- **Reading.** A separately trained small stand-in for the trunk state, read by an unchanged b5, is far worse than b5's
  own carried state. The head needs its input to be the exact trunk state, or its own trained chain state: the predicted
  states (a cosine of about 0.7 at best) are off its training distribution. DEEPDRAFT's route therefore needs the reader
  trained together with its stand-in (PLAN step 3), not a frozen reader with a bolted-on state predictor.
- **Notes.**
  - This run's plain chain reads 85.1 at depths 23-30, against t2_laglaw's 86.6, because that script counts the -1
    cells (beyond the answer) as right. g uses this run's own plain chain, as pre-registered.
  - The eval process peaked at 10.5 GB MLX (M.load at frac 0.03 + 5 arms), over the 8 GB limit set for the day. Swap did
    not grow (4.70 GB).
  - The helpers' weights (111 MB each) are kept outside the repo, in ~/datasets/deepdraft/preproof_helpers/.
