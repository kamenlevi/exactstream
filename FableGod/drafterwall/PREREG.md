# PREREG — drafterwall (pre-registered lines; each block is committed before its test runs)

Marks: [K] measured, [P] arithmetic on measured numbers, [G] judgment. All tests are CPU only, on recorded data.
Python: `~/BigMoE/.venv/bin/python -I scripts/<name>.py` (numpy only; read-only on every other folder).

## Disclosure: exploratory reads made BEFORE these lines (Wed 11 Nov 2026, 20:1x-20:4x EET)
Three descriptive scripts ran before any line was written (no PASS / KILL line, numbers below are what I saw):
- `scripts/s0_align.py` on `~/TrainLab/data/tree3_map` (12 code / 12 IFEval / 12 multi windows, every position an
  anchor): after the first miss of DFlash2's own 15-draft chain, the later drafts equal the 125B's tokens on
  13.6 % (code) / 7.1 % (IFEval) / 5.2 % (multi) of positions; the next 3 drafts are all right in 10.6 / 2.8 / 2.4 %
  of missed blocks.
- `scripts/s0_class.py` on the same tables: teacher-forced top-1 by the 125B's class and by depth (depth 0 = the first
  draft after the bonus token). Code, all rows: DFlash2 97.4 % and MTP 97.3 % at depth 0, either right 98.6 %, either
  top-3 99.6 %; at depths 11-14 DFlash2 67.5 %, MTP 88.2 %. Code mid rows: 83.7 / 84.7 % at depth 0, 24.5 / 47.8 % at
  depths 11-14.
- `scripts/s0_look.py`: 45 first-miss events printed and read by eye.
Because I saw these, tests T1-T5 below either use files the reads did not touch (T1: the HPD-0 live walks; T2: MBPP,
GSM, the 12 other HumanEval+ windows), or tune on half the windows and report the other half (T4, T5), or test a
derived quantity I had not computed (T3). Where a test repeats a number already seen, it is marked "confirmation".

## T1 SECOND-LOOK: can the rows after a check's first miss be reused? (Wed 11 Nov 2026, 20:4x, before any run; committed 20:50 as 602d66d)
Covers ledger ideas 1-4 and 15 (alignment-conditioned reuse of rejected rows, drafter-initialised Jacobi, the MTP head
on rejected rows' states, unary or soft tails). All of them need the same thing: after the first miss, the rows that
follow must have been fed the RIGHT tokens, so that the 125B's output there is an opinion about the real continuation.
- Data: `~/TrainLab/data/d02b_walk_hpd0_{code,math,mbpp,ifeval,multi}.npz` (the live official loop of DFlash2 L1-20:
  per check the accepted count k, the 15 drafts, and the answer's tokens). Not touched by the exploratory reads.
- Metric: over checks with a first miss at draft k < 14, the share of later drafted positions (k+1 .. 14) whose draft
  equals the 125B's token ("tail alignment"), and the share of such checks whose next 3 drafts are all right.
- PASS (family alive, go to a 125B capture): tail alignment >= 40 % on code, or >= 30 % on IFEval or multi.
- KILL: tail alignment < 25 % on code AND < 15 % on IFEval AND < 15 % on multi.
- Prediction: KILL (the exploratory read on other files said 13.6 / 7.1 / 5.2 %).

## T2 LAG LAW: is the wall the distance from the last exact trunk state, and does it hold on text I have not seen? (20:4x, commit 602d66d)
Claim to test: both existing readers are near 98 % right at the first draft position (two tokens past the last exact
trunk state) and lose their accuracy with every further position, fastest at the 125B's own close calls.
- Data: (a) confirmation: tree3_map code / IFEval / multi; (b) unseen: `~/TrainLab/data/d23h/{code,mbpp}_b5.npz` (the MTP
  head b5 teacher-forced to depth 31 on 24 HumanEval+ and 100 MBPP answers), `~/TrainLab/data/d24/{gsm,mgsm}_b5.npz`.
- Predictions, written before the unseen files are opened:
  - P-a: MTP top-1 at depth 0 >= 0.96 on the 24 HumanEval+ windows, and falls by >= 6 points by depths 7-10.
  - P-b: on MBPP, MTP top-1 at depth 0 is <= 0.95 (MBPP has more close calls), and falls by >= 6 points by depths 7-10.
  - P-c ("lag-2 everywhere"): a hypothetical drafter that is right at a position exactly when the reader is right
    there AT DEPTH 0 would keep >= 20 per 32-row check on HumanEval+ (MTP alone), and >= 25 with the either-of-two
    oracle on the 12 map windows; on MBPP it keeps < 16 with the MTP alone.
- PASS (the law holds as stated): P-a and P-b true, and P-c's HumanEval+ part true.
- KILL (the law is wrong): MTP at depths 7-10 within 3 points of depth 0 on either set, or the lag-2-everywhere
  drafter keeps < 14 per 32-row check on HumanEval+.
- Why it matters: PASS means a 98 % drafter already exists, but only at lag 2, and each sweep of the trunk can give
  lag 2 to one position only; then the wall is causal, not a lack of drafter capacity.

## T3 ROWS BUY THE LOGARITHM: can any tree or picker over the existing readers reach the target? (20:4x, commit 602d66d)
Covers ledger ideas 8, 9, 10, 17 (leaves, trees, both chains as rows, picks).
- Data: tree3_map (both readers' teacher-forced top-3 with probabilities along the true path, depths 0-14).
- Method: (i) ceilings with unlimited rows: kept per 16-row check when the true token only has to be inside a
  candidate set at every depth (DFlash2 top-1 / top-3, MTP top-1 / top-3, union of both top-3); (ii) threshold trees:
  a node is verified when its path probability under the reader is >= tau; kept(tau) from the true path, rows R(tau)
  from the calibration identity R = sum over depths of E[1{P >= tau} / P] on the true path; (iii) the slope of kept
  against ln R.
- PASS (selection can break the wall): union-top-3 ceiling >= 14.5 of 16 on code AND some threshold tree keeps
  >= 13.5 at R <= 128 rows.
- KILL: union-top-3 ceiling < 14.5 on code, or kept at R = 256 rows < 13.
- Prediction: KILL (TrainLab's STAGED put the union oracle at 13.61).

## T4 FUSION: a log-linear mix of the two readers, a handful of weights, held-out (20:4x, commit 602d66d)
Ledger idea 18 (new as a rule: PoE fusion is not in ~/TRIED; the max-confidence pick is, MTPPICK x1.073).
- Data: tree3_map; per domain the 12 windows sorted by name, even indices = tune, odd = held-out.
- Method: the official loop at 16 rows on the teacher-forced tables; at each depth the pick is the argmax over the
  union of both top-3 of w(d) log p_dfl + (1 - w(d)) log p_mtp (floor eps for a token missing from a top-3);
  parameters: w for depth 0, depths 1-2, depths 3+, and eps (4 numbers, grid-tuned on the tune windows).
  Baselines on the same windows: DFlash2 chain, MTP chain, the max-confidence pick, the either-right oracle.
- PASS: held-out kept >= 1.20 x the better single reader on code, IFEval or multi.
- KILL: held-out kept < 1.08 x the better single reader on all three.
- Prediction: BETWEEN / KILL, code about x1.05.

## T5 RUNNER-UP: do the 125B's own second choices at verified rows predict its next close calls? (20:4x, commit 602d66d)
Ledger idea 5 (new: 5 queries in ~/TRIED, no hit). A free by-product of every check: the top-3 at each verified row.
- Data: tree3_map (t125_t3 per row); same tune / held-out split as T4.
- Method: the MTP chain at 16 rows; at a depth where its top-1 probability is < theta, if exactly one of its top-2 /
  top-3 candidates was a runner-up (rank 2-3 of the 125B) at one of the last W verified rows and its top-1 was not,
  switch to it. theta in {0.5, 0.7, 0.9}, W in {8, 32, 128}, tuned on the tune windows.
- PASS: held-out kept >= 1.05 x the MTP chain on any domain. KILL: < 1.02 x on all three.
- Prediction: KILL.

## Results of T1-T5 (Wed 11 Nov 2026, 20:51-20:56; outputs in out/)
T1 KILL (tail alignment 13.2 % code, 7.0 % IFEval, 5.9 % multi). T2 BETWEEN (P-a, P-b true; lag-2 everywhere keeps
20.12 with the MTP alone on HumanEval+ and 12.20 on MBPP; the either-oracle on the 12 map windows 22.75, under my 25).
T3 KILL (union-top-3 ceiling 13.61; best tree 12.37 at <= 128 rows, 12.84 at <= 256). T4 KILL (held-out x1.038 / x1.009 /
x1.016). T5 KILL (x0.973 / x0.978 / x0.970). Descriptive scripts a1-a5 were written after these and carry no line.

## GPU test LAG2-AGENT (pre-registered Wed 11 Nov 2026 21:1x; NOT run: it needs the 125B, outside this session's rule)
- Question: on fresh agent turns and fresh MBPP prompts, with the full context, how much would a reader gain that
  held its depth-0 accuracy at every depth?
- Steps: (1) capture with TrainLab's sticky/dlog_capture.py: 24 agent turns and 50 MBPP prompts not in
  data/dlog/d1102mbpp; (2) gpu_test/g0_mtp31_set.py <set> <folder> (MTP head b5, every anchor, depth 31; compiles, not
  run); (3) TrainLab's every-position DFlash2 walk (TL_WALK_EVERY=1 TL_WALK_CF=1) on the same files; (4)
  gpu_test/g1_lag2_readout.py <set> <walk.npz> <table.npz> (run here on the existing tables: 9.58 / 12.55 / 5.93 / 6.23).
- Metric: per 32-row check, the either-lag-2 drafter's kept divided by the better real chain's kept.
- PASS (the door is worth a training mission): >= x2.0 on agent AND >= x1.7 on fresh MBPP.
- KILL (close the single-stream drafter track): < x1.4 on agent. BETWEEN otherwise.
- Prediction: agent x1.6-2.0, fresh MBPP x2.3-2.7 (the stored MBPP set gave x2.70, HumanEval+ x1.95).
