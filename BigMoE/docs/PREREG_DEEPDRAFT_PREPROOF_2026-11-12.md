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
