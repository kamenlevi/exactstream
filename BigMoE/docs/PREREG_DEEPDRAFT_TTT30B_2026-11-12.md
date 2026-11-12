# PREREG DEEPDRAFT step 3, second test (Thu 12 Nov 19:xx, before the run): one full pass over capture night 1

**Origin.** The orchestrator 19:1x, after t30a's INCONCLUSIVE (052bb9d): "your recipe at ~3,000 steps first (one full
pass over night 1). It answers whether depths 23-30 keep climbing, which decides whether more capture nights are worth it
at all." Tonight's GPU order: the 1.2.0 release gate, then the 45 GB P3 + P2 A/B, then this run, then the capture.

**Setup** (scripts/dd_ttt.py; the recipe is t30a's, docs/PREREG_DEEPDRAFT_TTT30_2026-11-12.md):
- **Recipe.** D-22's chain and loss: depth 30, wdecay 0.97, KL to the 125B's top-32 with accept-until-fail. All 69.1M
  head parameters train, starting from b5. Adam, lr 1e-5 cosine to 10 % after a 20-step warm-up, at most 192 anchors per
  step. The 125B is untouched.
- **One full pass instead of random windows.**
  - The split is pinned in data/deepdraft/ttt30b/split.json: 2,164 training files and the same 113 held out as t30a.
    Files the capture adds later change nothing.
  - Each training file's anchors are cut into near-equal consecutive slices of at most 192. The 4,206 slices (572,381
    anchors) are shuffled once (seed 1113), and step i takes slice i.
  - So the run is 4,206 steps, not ~3,000: a file has ~271 anchors (median), so most take two slices.
  - As in t30a, the 30 anchors after a slice also count at shallow depths in that slice's step.
- **The curve.** Weight copies are saved after 600, 1,500 and 3,000 steps; the final head t30b is the one after 4,206
  steps (one pass). The copies sit on this run's cosine schedule, where the learning rate is still high at 600, so they
  are points on one curve, not separate short runs.
- **Sharing.** The run holds a share lock with MEM 18 GB and yields to quiet locks.
  - When another session holds or waits for a quiet lock, it saves the weights, Adam's moments and the step, and stops.
  - It resumes on the same slice order once the slot is free and NO-SWAP >= 24 GB.
  - After a resume the next update is identical to an unbroken run's (checked on toy models, with and without layer
    lists).
- **Memory.** As t30a: MLX peak 15.6 GB, guard 18 GB. Heat pauses are allowed; no timing claims.

**Test.** The same held-out windows and harness as t30a, all heads in one process: b5, t30a, t30b_s600, t30b_s1500,
t30b_s3000 and t30b.
- Sets: the 24 HumanEval+ windows of data/d16w and the 100 MBPP answers of dlog/d1102mbpp.
- Agreement by depth band, over valid cells; kept per 32-row check.

**Lines (HumanEval+; the final head t30b against this eval's b5), as t30a's:**
- CONTINUE: depths 23-30 >= 90 %, AND depth 0 within 1 point of b5.
- KILL: depths 23-30 < 87 %.
- Otherwise INCONCLUSIVE.

**Does more data pay? (the orchestrator's question).** Both heads are fully annealed on the same recipe. t30b saw one pass
(4,206 steps); t30a saw 600 random windows (about 19 % of the anchors).
- PAYS: t30b's HumanEval+ depths 23-30 are at least 1.0 point above t30a's (87.3 %), AND its MBPP 23-30 is not below
  t30a's (68.5 %).
- DOES NOT PAY: under +1.0 point.
- Also reported: the mid-run curve (600 / 1,500 / 3,000 / 4,206) on both sets, the bands 7-10 and 15-22, and kept per
  check.
- A note, not a line: the two runs differ in data order and sampler, so a gap under a point can be run-to-run noise.
  PAYS argues for more capture nights. DOES NOT PAY points to the reader's capacity (PLAN step 3's carried state and the
  capacity sweep), not more of the same data.
