# BREAK98 screening log (NewIdeas, Tue 17 Nov; started 07:3x by date)

The question (~/FableGod/deepdraft/BREAK98.md): deliver the 125B's state at every drafted position to the drafter for a
small fraction of a 125B row's cost. NewIdeas screens the sprint's best ideas by ~12:00; the winner gets the GPU this
afternoon or tonight. GPU rule: a BREAK98 test always goes first. Hard hold for EvalSuite's chains (LIVE 07:53, EvalSuite bbf42c2 + 9633b09; mid-run it yields within 15 s):
/Users/kamen/NewIdeas/research/week98/BREAK98_HOLD (exists = no new EvalSuite step; first line = the test + its expected end).

## Number check of the frame (NewIdeas 07:2x, re-run of Lunatic's drift_rankceil.py on a copy)
- Reproduced exactly, HE+ / MBPP kept per 32-row check: EAGLE-2-31 16.26 / 9.03; ORACLE-31 18.48 / 11.21; K127 23.70 /
  11.28; LAB (true-state chain) 20.77 / 12.72; own chain 13.77 / 6.92.
- ORACLE-31 is a HINDSIGHT bound over the 127 best-first pops. Without hindsight, the head's own information at 32 rows is
  capped at EAGLE-2-31 itself (16.26): the head is calibrated and best-first maximises the expected number of on-true
  nodes. It does not bound a retrained head.
- So 24-26 kept at 32 rows needs true-state-class information (LAB 20.77; LAB + tie siblings 26.79), or ~128 rows of own
  information (K127 23.70) at about the price of 32.

## Ideas in, with their screens
| # | from | idea | its number | screen | NewIdeas verdict |
|---|---|---|---|---|---|
| 2 | FAR | FUTURE-STATE ROWS: k placeholder rows (no token) hung under the end row + lowest-p rows in the check; the next block's head reads their states | ceiling (b98_far.txt, chain-32 HE+): k 8 + 71 % coverage +14 % at G 1, +6 % at G 0.5; k 16 +25 % / +10 %; FAR's pay line +18 % | Sandbox98 Screen 1 dispatched by the orchestrator 07:3x (G(1-8) >= 0.30 + tree-32 kept >= +5 %) | ALIVE, small ceiling at k 8; useful as a probe of placeholder information |
| 3 | ZERO (= NewIdeas' early repair) | IN-CHECK REPAIR (CHECK-AS-TAP-PASS): at layer L the check holds the 125B state at every drafted row; locate the first error, re-predict from the tap, refresh the head, re-draft the tail, catch the new rows up through layers 1..L | bound (b98_repair.txt, perfect location + true-state quality): 1 round 13.77 -> 18.97 (+38 %, 74 % of own -> LAB); 2 rounds 20.83; q 0.5 +22 %, q 0.3 +13 % (1 round) | tonight's mid_*.npz: ZERO's TAP-LAB / FORK-TAP (ridge layer L -> head tap input, frozen head; PASS >= 40 % of the drift gap at L16 bands 3-14) + the readout's error location | TOP LEAD (07:3x); costs per round: head tail re-draft ~2-4 ms + catch-up of layers 1..L (+0.7-1.1 GB, +2.6-3.8 ms) |
| 4 | MoonShots | SESSION VALUE PREDICTION: feed the head the session's true state where the drafted n-gram recurs | hits 9-12 %, R2 <= 0.22; hindsight-best earlier state 0.67-0.76 rad off | 14-s CPU screen on the bank captures | KILL (MoonShots' own) |
| 5 | Breakthrough | LANDMARK: true states by context match; LANDMARK-TAP: free mid-depth taps at ~80 % of code positions | tap 45: matched state no closer than the stale anchor (cos 0.637 vs 0.624; line +0.15); tap 15 / 25 close (0.785 / 0.687 vs 0.405 / 0.485) | screened on TrainLab d17f captures; LANDMARK-TAP: Sandbox98 tapmlp with landmark taps, PASS G >= 0.5 | BETWEEN (KILL side); LANDMARK-TAP small (tapmlp's +5 % ceiling) |
| 6 | Breakthrough (corrected) | LANDMARK-ANS: at drafted positions whose last k tokens recur EARLIER IN THE ANSWER, feed the head that verified row's stored head-input state (a state store + a GPU k-gram hash; free, exact) | coverage 38 % of HE answer positions (49 % past the first third); late-state cos 0.795 to the true state (true state one back 0.798; anchor 0.644). b98_landmark.txt (random refreshes, true-state quality): pi 0.38 -> HE+ 17.83 (+29 %, 58 % of the gap), MBPP +40 % | Sandbox98 SIB harness with this source (PASS G >= 0.5 HE, KILL < 0.1, ~30-50 min) + a 125B confirmation on tonight's capture | ALIVE, best economics so far (07:4x): ZERO check cost, so the kept gain is the tok/s gain; risks: clustered coverage, the head's use of a stale-by-one state |
| 7 | Fable C | IN-CHECK REPAIR on trees, budget-correct; run as an ADDED branch (the original tail stays, so false flags cost rows, never kept); tune the layer-L probe for RECALL; multi-guess correction (head top-2 from the true state + lens top-2) | tree-32 16.30 -> 23.10 one round (x1.42) / 25.81 two (x1.58) HE+, MBPP 9.05 -> 15.01 / 19.45; q 0.5: 20.0 / 20.6; added-branch + recall-tuned ~22 / ~23.5 at q ~0.65 [P]; probe-free refresh at a fixed row DEAD (x1.04-1.16) | on the capture: (1) first-miss recall vs false-flag rate (curve, L 8 / 16 / 24), (2) P(correction in head top-2 from the true state + lens top-2) at the last right row. PASS recall >= 0.8 at <= 20 % false flags AND multi-guess >= 0.75 at L16 | CONFIRMS the lead (07:5x); NewIdeas builds the screen (repair_stage2.py) for the day capture |
| 1 | Fable B | partial-width 125B rows per drafted token (dense-only, shared + top-1, 8 dense layers) | per token 1.6 GB / 1.75 GB / 0.27 GB vs a 3 ms/block budget of ~28 MB per token (57x / 62x / 10x over); batched partial rows before the check = a fork re-anchor, +2.4 % kept per round [measured] | desk (bytes) | KILL as a route (Fable B's own verdict; bytes check below) |

## What a channel must deliver (b98sim.py / b98sim.txt, 07:3x; chain-32 kept, cells only, self-checks exact)
| pattern (true 125B state delivered ...) | HE+ | share of the own -> LAB gain | MBPP |
|---|---|---|---|
| never (the real chain) | 13.77 | 0 % | 6.92 |
| every position (LAB) | 20.77 | 100 % | 12.72 |
| refresh every 2nd depth | 19.20 | 78 % | 10.88 |
| refresh every 4th depth | 17.18 | 49 % | 8.99 |
| refresh every 8th depth | 15.51 | 25 % | 7.67 |
| ONE refresh per block, best depth (D = 4) | 15.32 | 22 % | 7.87 |
| at depths 0-15 only (point) | 18.64 | 70 % | 11.15 |
| at a random 50 % of depths (point) | 16.39 | 37 % | 8.72 |
| at a random 90 % of depths (point) | 19.63 | 84 % | 11.52 |
- READ: the information must be DENSE and early. A channel that refreshes the state once per block (any single partial
  row or one Jacobi round) is worth <= ~22 % of the gap; every 2nd position ~78 %. Depths 0-15 carry 70 %.
- Any idea can be priced here: `mix(rt, src_of_depth)` takes, per depth, which anchor's guess is used.

## IN-CHECK REPAIR priced (07:4x; b98_repairtree.txt, b98_freeride.txt, b98_repaircost.txt)
- TREE version (real 127-node dumps; tree-32 base 16.26 reproduced), HE+: 1 round 22.06 (+36 %, q 1) / 19.20 (q 0.5) /
  17.96 (q 0.3); 2 rounds 24.06 (+48 %, q 1, = the 24 that 550 needs) / 19.98 (q 0.5). MBPP 9.03 -> 14.10 / 17.04 (q 1).
- FREE RIDE (b98_freeride.txt): after a wrong token, the head's old tail is right at the next position only 16 % (HE+) /
  12 % (MBPP); mean ride 1.08 / 0.39 tokens. So a swap-only repair (keep the old tail, no re-draft) is KILLED: the
  tail must be re-drafted.
- COST per round (check paused at L): a readout pass + the tail re-draft (~12 head levels) + the catch-up of layers 1..L
  for the ~16 tail rows (~0.40 ms per layer at the bound). NET tok/s vs tree-32 (551 at the bound, 205 today):
  | tap layer L | ms per round (bound) | 1 round q 1 | 1 round q 0.5 | 2 rounds q 1 |
  |---|---|---|---|---|
  | 4 | 2.9 | +24 % | +8 % | +24 % |
  | 8 | 4.5 | +18 % | +2 % | +13 % |
  | 16 | 7.7 | +8 % | -6 % | -3 % |
  Today's engine: at most +18 % (L 4, q 1).
- READ: the kept bound is large, but after its cost in-check repair is a +10-24 % lever, and only with a SHALLOW tap
  (L 4-8) of high quality. Its cheaper cousin is EARLY DEATH on a big tree: the readout at L SELECTS among pre-drafted
  alternatives (K64 / K127 rows in layers 1..L), with no re-draft pause and no catch-up. Both use the same readout, so
  the day capture (L4 / 8 / 16 / 24) serves both. The capture spec was corrected to add L4 and drop L32 / 40.
- Also KILLED: MoonShots' SESSION VALUE PREDICTION (retrieved states: hits 9-12 %, R2 <= 0.22; the state is long-context).
  MoonShots' side note for the cost model: ~2.7 GB per check of idle DRAM during the compute-bound experts.

## LANDMARK-ANS with the REAL coverage (07:5x; b98_landmark_real.txt) - CORRECTS the +29 % random-coverage bound
A landmark exists iff the k-gram of true tokens ending at the row occurred earlier IN THE ANSWER (bank ids, rows >= gen).
Each landmark refreshes the head with true-state quality (the ceiling statement). HE+ / MBPP chain-32:
| match length k | coverage of walk anchors | REAL landmarks (clustered) | random at the same coverage |
|---|---|---|---|
| 2 | 34 % / 34 % | 17.05 (+24 %, 47 % of the gap) / 8.81 (+27 %) | +27 % / +36 % |
| 3 | 18 % / 20 % | 15.62 (+13 %) / 7.83 (+13 %) | +16 % / +21 % |
| 4 | 11 % / 12 % | 14.52 (+5 %) / 7.49 (+8 %) | +13 % / +15 % |
| 8 | 2 % / 3 % | 13.98 (+2 %) / 7.05 (+2 %) | +2 % / +3 % |
- Breakthrough's 38 % matches k ~2 here (34 %). A 2-token context is exactly where MoonShots' kill bites (retrieval = copy,
  R2 <= 0.22), so the +24 % ceiling needs 2-token landmarks to carry true-state information, which the screen's KILL-COPY and
  stale-by-one lines test. Longer, more trustworthy matches are rare: k 4 gives +5 %, k 8 +2 %.
- The screen must report G BY MATCH LENGTH; the verdict needs G at k <= 3 to pass KILL-COPY.

## LANDMARK-ANS screen 2 on the bank (Breakthrough 6005fc9, 07:5x; pre-reg + addendum 88b0b41)
- HE+, depth 8: coverage 32.9 %; landmark cos 0.725 vs anchor 0.460 on the head's input wd: CONFIRMED on its line
  (cov >= 25 %, +0.10).
- Stale by one: the landmark is CLOSER than the true state one position back (0.725 vs 0.619).
- Within 0.40 rad: landmarks 5.5 % HE+ / 9.4 % MBPP (MoonShots' 3-6 % class); the 1-back true state 0.0 % / 0.2 %. So
  distance alone can't decide; G_stale1 vs G_landmark (Sandbox98, screen 1) does.
- Caveat: half the coverage is 2-token matches (cos 0.69, top-1 agree 39 %); k >= 4 is ~11 % (cos 0.76-0.80). The k >= 4 vs
  k <= 3 split of G and G_copy is in the pre-reg (report-only).

## Capture asks for tonight (to BigMoE by 18:00; collected here)
- Fable B: ablated forwards along the mid-capture rows: shared + top-1 experts (then shared-only / top-2 / top-4),
  dumping the final widened state + top-8 logits. This prices what a cheap proxy must reproduce. Screen: the head's
  walk-strict accuracy at depth 30 from the ablated state, line >= 95 %.

- FAR (a): per every 4th true-text row, 8 placeholder rows' W, the layer-16 residual, top-8 ids / log-probs, top-10 expert
  ids per layer (the union price). FAR (b, optional): layer-16 / 24 attention K/V per true row (prefix-depth inputs).
- ZERO: the 10,240 pre-mixer stream at L8 / 16 / 24 + a dense-only layer-8 state.
- Breakthrough: WITHDRAWN (07:5x). ~/datasets/deepdraft/bank already holds ids + wd (the head's input) for prompt AND answer rows
  of HE+ 24 / MBPP 100, so LANDMARK-ANS needs no capture.

## NewIdeas' own candidates (not yet screened)
- KVX: the drafter cross-attends to the 125B's own KV cache (a few attention layers; context = prompt + accepted
  tokens). It delivers the 125B's view of the context at each drafted position, at KV-read bytes (~MBs, SLC-resident
  across steps), not weight bytes. It does NOT deliver the 125B's state at the drafted positions (the re-plan). No TRIED
  hit; prior art: GLIDE (draft model attends to the target's KV). Screen: a sandbox analog (2B), not a 10-min CPU screen.
