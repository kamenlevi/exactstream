# PREREG DEEPDRAFT step 3, third test (Fri 13 Nov 11:xx, before any run): the reader's capacity and its carried state, one pass each against t30b

**Origin.** The orchestrator 10:4x, after TTT30B (a855212: INCONCLUSIVE on the lines at 89.7 %, more data PAYS): "Please
pre-register the capacity side now (your capacity sweep / PLAN step 3's carried state), so it's ready for the next free
slot." Its reason: t30a -> t30b was ~5.3x the data for +2.3 pt at depths 23-30, about +1 pt per doubling; at that rate
~97 % would need some 150-200x night 1's data, so data alone will not get there. The arms may run beside the capture if
both fit in memory; no timing claims.

**What PLAN step 3 asks for** (~/FableGod/deepdraft/PLAN.md): a reader that carries a learned stand-in for the trunk
state and updates it step by step, with a feature loss at every depth, and a capacity sweep from 69M to about 300M
trainable (local training only, 1B or less).
- In the D-22 chain, the head's depth-j output h_j already is that stand-in: depth j+1 reads (h_j, token s+j+1) where the
  exact path would read the trunk's widened stream W[s+j]. Today it is trained only through the next-token KL.
- The pre-proof (479f299, KILL) trained a stand-in separately and fed it to the unchanged b5. Here the stand-in and the
  head train together.

**Prior art checked** (~/TRIED, TrainLab's REPORT): no MTP-head capacity or feature-loss test.
- TrainLab P-CAP (11-04): DFlash2 grown to 10 layers by block expansion, on L1-20's data: BETWEEN, worth about 1.5 data
  doublings.
- EAGLE-1 trains its draft head with a feature-regression loss next to the token loss. EAGLE-3 drops it for the
  training-time test and reports that the feature constraint limited scaling with data. So the feature arm can also
  hurt; that is a reason to test it, not to assume it.

## Setup
Every arm is t30b's run with exactly one change, so t30b is the control:
- the pinned split (data/deepdraft/ttt30b/split.json), one pass of 4,206 slices in the seed-1113 order, at most 192
  anchors per step;
- D-22's chain to depth 30, wdecay 0.97, KL to the 125B's top-32 with accept-until-fail;
- Adam, lr 1e-5 cosine to 10 % after a 20-step warm-up; start from b5. The 125B is untouched.

The three arms:
- **t30f, the carried state (69.1M trainable).** A feature loss at every depth: the relative squared error between
  pre_fc_norm_hidden(h_j) and pre_fc_norm_hidden(W[s+j]). That is the next depth's input as the head reads it, against
  what it would read on the exact path. The head's own frozen RMSNorm over all 10,240 channels sets both sides.
  - It applies on the KL's rows and weights: assistant targets, the accept-until-fail support, and wdecay^(j-1),
    normalised the same way.
  - Its weight lambda is fixed before the first update: 0.5 x the KL term / the feature term, as means over the first 20
    slices at b5 (forward only). It is logged and kept on a resume.
- **t30x2, capacity x2.1 (147.7M trainable).** A dense SwiGLU branch beside the head's MoE, added to its output: gate and
  up 2,560 -> 10,240, down 10,240 -> 2,560 (78.6M).
  - The down projection starts at zero, so the head equals b5 at step 0.
  - The branch's learning rate is 10x the rest (1e-4) on the same schedule; the other parameters keep 1e-5.
  - Its init uses seed 1113.
- **t30x4, capacity x4.4 (305.0M trainable).** The same branch at 30,720 wide (235.9M).

Shared with t30b:
- Weight copies after 600, 1,500 and 3,000 steps.
- A share lock that yields to other sessions' quiet locks. SIGTERM saves the weights, Adam's moments (both groups) and
  the step, then exits; the run resumes on the same slice order.
- Heat pauses are allowed; no timing claims.
- Memory guards (MLX peak): t30f 18 GB, t30x2 20 GB, t30x4 24 GB, each to be confirmed by a smoke run before its start.
  An arm starts beside the capture only if NO-SWAP >= its guard + 8 GB.
- Order: t30x2, then t30f, then t30x4 (the largest needs the most room).

**Smoke checks before the first real step, not results.** With lambda = 0 and no branch, the loss must equal t30b's on
the same slice (the code path is unchanged). With a branch, step 0's chain output must equal b5's (zero down projection).
These run first, and each arm's peak is read from a few steps.

## Test
The same held-out windows and harness as t30a and t30b: the 24 HumanEval+ windows of data/d16w and the 100 MBPP answers
of dlog/d1102mbpp; agreement by depth band over valid cells; kept per 32-row check.
- Each finished arm is scored in one process with b5 and t30b. b5 and t30b must re-score as in a855212 (85.1 / 89.7; MBPP
  63.3 / 72.1); a mismatch is a harness bug, to be fixed before any line is read.
- The copies at 600 / 1,500 / 3,000 are compared with t30b's copies at the same steps (the same schedule, so
  like-for-like). They are information only; every line is read after the full pass.

## Lines (HumanEval+ depths 23-30 unless stated; each arm against t30b in the same eval)
- **PAYS:** at least t30b + 1.0 pt (>= 90.66 % if t30b re-scores 89.66), AND MBPP 23-30 not below t30b's, AND depth 0
  within 1 pt of b5.
- **HURTS:** at or below t30b - 1.0 pt, or depth 0 more than 1 pt below b5.
- Otherwise **NO EFFECT** (within a point, the run-to-run noise note of TTT30B).
- **Capacity against data:** points per doubling of trainable parameters, from t30b (69.1M), t30x2 and t30x4, set
  against data's ~+1 pt per doubling (t30a -> t30b). Capacity scales at least as well as data if t30x4 - t30b >= +2.1 pt
  (2.14 doublings).
- **PLAN step 3's lines, reported per arm:** CONTINUE >= 90 % with depth 0 within 1 pt of b5; the proof's PASS line is
  95 %. PLAN's KILL (under 90 % after the planned data and capacity) is not read here: the planned 3-5M tokens are not
  captured yet.
- **Also reported:** the bands 0, 7-10 and 15-22, kept per check on both sets, each arm's training peak, and the feature
  term's curve (t30f).

**What each outcome means.**
- A capacity PAYS argues for a bigger reader: a second head layer by block expansion, up to the ~1B local line.
- A t30f PAYS keeps the feature loss for every later run.
- If nothing pays, the wall is what the reader can see, not its size. The next candidate is EAGLE-3's multi-layer taps:
  the capture already logs DFlash2's 5-layer taps (h), which the MTP head does not read today. It would be
  pre-registered separately.

## Run notes (Fri 13 Nov 15:1x, after the smokes; nothing in the arms or the lines changes)
- **The slot opened at 15:08**, when EvalSuite's 45 GB P2 gate ended. The capture (C 100) started then, and the arms'
  launcher 300 s later.
- **t30x2 smoke, 15:13-15:14:**
  - Both checks EQUAL: b5's loss 0.3548715 equals the lambda-0 path's, and with the zero-init 10,240-wide branch it is
    still 0.3548715.
  - The MLX peak was 22.4 GB at step 1, over the 20 GB guard. The run stopped and the launcher skipped the arm.
  - The guard was an estimate; the measured peak replaces it. The branch adds 6.8 GB over t30b's 15.6 GB at 10,240 wide.
- **t30f smoke, 15:14-15:16:**
  - EQUAL (b5's loss 0.3548715 at lambda 0).
  - lambda = 0.09164, which is 0.5 x KL 0.32056 / feature term 1.74896 (b5, slices 0-19).
  - Peak 15.6 GB, the same as t30b's. It has been training beside the capture since 15:16, and its step 1 equals the
    smoke's.
- **New order: t30f, then t30x2, then t30x4.**
  - Each branch arm's guard is re-set from its own smoke before it starts.
  - t30x2 needs at least its 22.4 GB step-1 peak. t30x4's branch is 3x wider, so a linear projection puts it near
    36 GB. It runs with the capture off, not beside it.

## Result: t30f (Fri 13 Nov 22:07): NO EFFECT, and every band is slightly below t30b
t30f trained 4,206 steps (rc 0 at 21:49) and was scored in one process with b5 and t30b
(data/deepdraft/ttt30c/t30f/eval.json; eval rc 0 at 22:07).
- **Harness:** b5 and t30b re-score bit for bit as in a855212 (HumanEval+ 23-30 85.08 / 89.66; MBPP 63.34 / 72.13).
- **The lines (HumanEval+ 23-30):**
  - t30f scores 88.73 %, against t30b's 89.66 %: -0.93 pt.
  - PAYS needs >= 90.66 %. HURTS is <= 88.66 %, so t30f lands 0.07 pt above the HURTS line.
  - MBPP 23-30 is 71.44 against 72.13 (-0.69). Depth 0 is 97.74 against b5's 97.68 (+0.06, within 1 pt).
  - **Verdict: NO EFFECT** (within a point of t30b).
- **Every band on both sets is below t30b**, by 0.12 to 0.93 pt:

  | band | HumanEval+ b5 / t30b / t30f | delta | MBPP b5 / t30b / t30f | delta |
  |---|---|---|---|---|
  | 0 | 97.68 / 97.90 / 97.74 | -0.16 | 93.46 / 94.01 / 93.89 | -0.12 |
  | 7-10 | 90.51 / 93.10 / 92.51 | -0.59 | 74.70 / 79.61 / 79.00 | -0.61 |
  | 15-22 | 86.85 / 91.04 / 90.38 | -0.66 | 66.78 / 74.38 / 73.89 | -0.49 |
  | 23-30 | 85.08 / 89.66 / 88.73 | -0.93 | 63.34 / 72.13 / 71.44 | -0.69 |
  | kept per 32-row check | 12.55 / 13.83 / 13.20 | -0.63 | 6.22 / 6.92 / 6.82 | -0.11 |

- **The copies against t30b's copies at the same steps (information only), 23-30:**
  - HumanEval+: +0.17 pt at step 600, -0.77 at 1,500, -0.78 at 3,000.
  - MBPP: -0.77, -1.48, -1.15.
  - The gap opens by step 1,500 and stays.
- **The training KL, paired by step** (the same slices: pass seed 1113; t30f's logged 20-step mean total minus lambda x
  its feature term, against t30b's loss):
  - Steps 20-999: even (+0.1 %; t30f above in 27 of 49 logs).
  - Steps 1,000-2,199: +3.7 % (above in 56 of 60).
  - Steps 2,200-4,206: +4.4 % (above in 99 of 101).
  - The orchestrator's own paired check agrees (+2-5 % from step 2,200).
- **PLAN step 3:** 88.7 % is under 90 %, so not CONTINUE. The eval's own field reads INCONCLUSIVE (between 87 and 90).
- **Training:** MLX peak 15.8 GB (t30b 15.6); the eval peaked at 10.1 GB. It ran beside the capture, so no timing is
  claimed.
- **The feature term** (the 20-step mean, at lambda 0.09164):
  - 1.71 at step 1, 1.42 at 600, 1.11 at 1,500, 0.95 at 3,000, 0.93 at 3,500, 0.90 at 4,000 and 0.92 at 4,200.
  - It fell by about half and was flat over the last 1,000 steps.
- **What it means:**
  - The PREREG keeps the feature loss for later runs only on a t30f PAYS, so later runs drop it.
  - The capacity arms t30x2 and t30x4 run next, alone, with the capture paused. The "if nothing pays" reading waits for
    them.
