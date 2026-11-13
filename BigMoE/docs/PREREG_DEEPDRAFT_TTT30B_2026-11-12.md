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

## RESULT (Fri 13 Nov 10:38): INCONCLUSIVE on the lines; more data PAYS (data/deepdraft/ttt30b/eval.json)

t30b = b5 + one full pass over capture night 1 on t30a's recipe: 4,206 steps, 572,381 anchors, each once. The run went
02:29-10:08. It stopped once, at step 1,271 (04:27), to move to a corrected night chain: a clean save of the weights,
Adam's moments and the step, then "resume t30b at step 1271" and "optimizer state restored". No quiet-lock yield was
needed. Heat pauses ran throughout, under the fan cap and its morning step-down (no timing claim). MLX peak was 15.6 GB in
training and 10.1 GB in eval. All six heads were scored in one process on t30a's held-out windows. Agreement with the
125B, in %, by draft depth:

| set | head | 0 | 1-2 | 3-6 | 7-10 | 11-14 | 15-22 | 23-30 | kept / 32 |
|---|---|---|---|---|---|---|---|---|---|
| HumanEval+ (24 windows) | b5 | 97.7 | 96.0 | 93.2 | 90.5 | 88.9 | 86.8 | 85.1 | 12.55 |
| HumanEval+ (24 windows) | t30a | 97.6 | 96.0 | 93.5 | 91.7 | 90.6 | 89.1 | 87.3 | 13.12 |
| HumanEval+ (24 windows) | t30b_s600 | 97.6 | 95.7 | 92.6 | 89.9 | 88.1 | 86.1 | 84.2 | 11.81 |
| HumanEval+ (24 windows) | t30b_s1500 | 97.8 | 96.2 | 93.7 | 91.5 | 90.0 | 88.4 | 86.7 | 13.28 |
| HumanEval+ (24 windows) | t30b_s3000 | 97.8 | 96.2 | 94.2 | 92.8 | 91.8 | 90.2 | 88.7 | 13.62 |
| HumanEval+ (24 windows) | **t30b** | 97.9 | 96.4 | 94.5 | 93.1 | 92.2 | 91.0 | **89.7** | 13.83 |
| MBPP (100 answers) | b5 | 93.5 | 87.4 | 80.7 | 74.7 | 70.7 | 66.8 | 63.3 | 6.22 |
| MBPP (100 answers) | t30a | 93.6 | 88.5 | 81.9 | 77.1 | 73.9 | 71.0 | 68.5 | 6.54 |
| MBPP (100 answers) | t30b_s600 | 93.5 | 87.6 | 79.9 | 74.4 | 71.2 | 67.8 | 64.7 | 6.14 |
| MBPP (100 answers) | t30b_s1500 | 93.7 | 88.5 | 81.6 | 76.8 | 74.0 | 71.1 | 68.6 | 6.49 |
| MBPP (100 answers) | t30b_s3000 | 93.9 | 88.9 | 83.0 | 78.7 | 75.9 | 73.2 | 70.9 | 6.76 |
| MBPP (100 answers) | **t30b** | 94.0 | 89.1 | 83.4 | 79.6 | 77.1 | 74.4 | **72.1** | 6.92 |

- **Lines: INCONCLUSIVE.** t30b's HumanEval+ depths 23-30 reached 89.66 % against b5's 85.08 %. That is not >= 90
  (CONTINUE) and not < 87 (KILL); it is 0.34 pt short of CONTINUE. Depth 0 moved +0.22 pt.
- **Does more data pay? PAYS.** t30b is 2.31 pt above t30a at HumanEval+ 23-30 (89.66 vs 87.35; the line is +1.0), and
  its MBPP 23-30 is 72.1 % against t30a's 68.5 % (+3.7).
- **Curve** (one cosine schedule, so the copies are points on one run, not separate short runs). HumanEval+ 23-30: 84.2 /
  86.7 / 88.7 / 89.7 at 600 / 1,500 / 3,000 / 4,206 steps; MBPP 64.7 / 68.6 / 70.9 / 72.1.
  - The 600-step copy is below b5 (-0.9 pt at 23-30) while the learning rate is still high.
  - The last 1,206 steps still add +1.0 pt (HumanEval+) and +1.2 (MBPP), but they also finish the anneal, so the end
    slope is not a clean data effect.
- **Shape.** As in t30a, the gain grows with depth. Against b5, HumanEval+ is +0.2 pt at depth 0, +2.6 at 7-10, +4.2 at
  15-22 and +4.6 at 23-30; MBPP +0.5 / +4.9 / +7.6 / +8.8. Kept per 32-row check: 12.55 -> 13.83 (+10.2 %) on HumanEval+
  and 6.22 -> 6.92 (+11.3 %) on MBPP.
- **Harness.** b5 and t30a re-score exactly as in t30a's eval (85.1 / 87.3; MBPP 63.3 / 68.5; kept 12.55 / 13.12).
- **Not measured.** The eval stores pooled cells only (no per-window spread), and there is one run per arm: t30a and t30b
  differ in data order and sampler as well as in data volume. The +2.3-point gap is above the one-point noise note, but
  it is a direction, not a significance claim.
- **Scale, roughly (the orchestrator, 10:4x).** t30a -> t30b was ~5.3x the anchors for +2.3 pt at 23-30, about +1 pt per
  doubling. If that held, ~97 % at depth 30 would need some 150-200x night 1's data, so data alone will not get there.
- **Next (the orchestrator, Fri 10:4x).** PAYS argues for more capture nights: BigMoE's slot after EvalSuite's 45 GB P2
  gate is the capture continuation, which yields to other sessions' quiet locks. The capacity side (PLAN step 3's carried
  state and the capacity sweep) is pre-registered next, for the next free slot.
