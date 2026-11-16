# IDEAS_1116_acceptance: WALL A, lossy acceptance to 97.8 % per guess to 30 deep on code (Fable C, Mon 16 Nov 2026, 12:0x-12:4x, desk only)

Question (THINK_1116 #1): which lossy acceptance rules reach >= 97.8 % per guess to 30 deep on code while code tests stay
unchanged (EvalSuite's line), and how do we prove the quality by Saturday?

Marks: [K] measured, [P] arithmetic on measured numbers, [G] judgement; modelled gains halved before they are quoted.
Data tonight: one CPU script (`accept_T_1116.py`, 20 s, < 1 GB) on the map's 12 HumanEval code windows (old DFlash2 walks,
the 125B's top-8 log-probs); t30b numbers are the brief's and the records'. Budget terms: 900 = kept / (check + draft);
flat 97.8 % to 30 = 23.15 kept per 32-row check; today tree-32 16.26 exact, 16.91 lossy K*, true state 20.77 (HE+).

## 0. What acceptance can and cannot fix (the decomposition that orders the ideas)

A miss is one of two things, and the check's exact margin at the miss row tells which:
- a TIE miss: the 125B was ~50/50 and the drafter took the other side. The true-state deficit is this part [brief]:
  100 - true state = 6.8 / 4.7 / 3.6 / 2.2 / 1.2 / 0.8 / 0.7 points at bands 0 / 1-2 / 3-6 / 7-10 / 11-14 / 15-22 / 23-30.
  A lossy rule may accept these at little cost to tests: [K] WEEK98-JCC on the 2B: close-call delta 1 nat PASS (pooled
  +0.76 pt, p 0.88, kept +21 %), delta 2 KILL (-6.06, HE+ -9.15). [K] WEEK98-TREE: 142 / 264 answers flipped by the batch
  shape alone, all at near-ties, with identical pass@1.
- a DRIFT miss: the 125B was sure (margin >= 2 nats) and the drafter was wrong. Accepting it is a real output change that
  the lossy gate's tier 0 counts as the bad kind (F_hi, flips at margin >= 2 nats). No acceptance rule may take these on
  CODE rows. They are the chain-to-true-state gap: 1.2 / 3.8 / 4.0 / 3.5 / 2.2 / 2.5 / 1.6 points per band [P from the
  brief's table], and [K] TREES-x-CLOSECALL (MBPP chain-16): of the first misses, TREE rescues 27.2 %, CC 15.3 %, BOTH
  12.1 %, NEITHER 45.4 %. That 45 % is drift outside the head's top-3: problem #4's wall, not WALL A's.
- [P] a perfect tie-forgiveness rule on the real chain gives chain + tie part = 98.8 / 96.2 / 96.0 / 96.5 / 97.8 / 97.5 /
  98.4: bands 1-10 stay under 97.8. So "97.8 per band" is NOT reachable by acceptance on code rows alone, with any rule.
- The way through: the misses are not spread evenly. [K] SLOT PRIOR (HE+): in-comment rows are 13.1 % of answer rows but
  hold 71.2 % of the 125B's near-ties (p_tie 18.9 % vs base 3.5 %); COMMENT-FREE's record: comments carry 52 % of first
  misses. Inside a comment or docstring, a DRIFT miss is also harmless to the tests, as long as the code is unchanged. So
  WALL A's levers are about comment rows: remove them (idea 1) or accept them wholesale (idea 2), and treat code rows with
  the one rule that already passed (idea 3). Idea 4 is the proof.
- [K, tonight] what does NOT work: distribution-exact speculative sampling (accept with min(1, p_T(d) / q(d))) at T 0.3-1.0
  lifts per-band acceptance by only +0.6 .. +2.4 points at bands 0-6 and ~0 at 7-14 on the map's code walks (old
  drafter), the same cancellation Leap's A1 KILL found on IFEval (+0.003 .. +0.014 per token): the drafter's
  whole-distribution mismatch at sure rows eats the tie gain. The forgivable share of misses (draft = the 125B's #2 within
  1 nat) is 41 / 23 / 19 / 6 / 0 % at bands 0 / 1-2 / 3-6 / 7-10 / 11-14: deep misses are drift, not ties.

## Idea 1 (best): COMMENT-FREE as an exact, drafter-independent decoding policy, certified once by a plain-decode run

- Mechanism. Decode with a system instruction that forbids comments and docstrings in code (the product's FAST mode).
  The output is the 125B's own bitwise-exact greedy answer to that prompt; nothing lossy happens in the engine. The rows
  that hold 71 % of HE+'s near-ties and 52 % of its first misses are never generated, so the drafter's walk runs on code
  rows, where it is already near the line, and every answer is ~14 % shorter.
- Numbers. [K] records: comments 13 % of HE+ tokens, 52 % of first misses, 71 % of ties; [P] kept x1.23-1.29 (COMMENT-FREE,
  11-05), answers 14 % shorter. [P] the per-band view: with comment rows gone, the miss rate on the remaining rows is
  0.48 / 0.87 = 0.55x the overall one: chain 92.0 / 91.5 / 92.4 -> ~95.6 / 95.3 / 95.8 at bands 0 / 1-2 / 3-6 before any
  tree or close-call; tree-32 + close-call delta 1 on top [G] ~97-98 at bands 1-6. Budget: tree-32 16.26 -> ~20-21 kept
  [P x1.23-1.29], halved +12-15 % tok/s at the same cycle, plus 14 % fewer tokens per task (a second +14 % per task, not
  per token).
- Tag: on record as UNTESTED (TRIED L1042 COMMENT-FREE, "test A-P-A-P prompt arm", 11-05); NEW as a certification
  object (a drafter-independent policy, see idea 4). Published: none needed; a prompt.
- Exact: YES (bitwise to the policy's own greedy output).
- Kill: paired pass@1 under the instruction drops > 3 points pooled or McNemar p < 0.05 vs the default prompt on HE+ 164 +
  MBPP 100 (tier 1's rule); or the model keeps commenting (comment-row share stays > 3 % of answer rows); or the
  tests' own harness needs the docstring (HE+ gives the docstring in the prompt, so no).
- <= 15-minute test (today, one GPU slot, not a desk test): 20 HE+ tasks plain-decoded under the instruction, 2 x 10 in
  both prompt orders; count comment / docstring rows (the engine's causal comment tagger, M-41) and pass/fail paired
  with the stored exact answers. GO if comment rows < 3 % and pass count within 2 of the default. The full certification
  is the ~40-80 min plain run (idea 4).

## Idea 2: INERT acceptance for prompts that want comments: accept every drafted token inside a comment or docstring, exact at the span ends

- Mechanism. The engine's causal comment tagger marks in-span rows; inside a span the drafter's token is accepted without
  comparing it to the 125B (both tie and drift misses vanish there); the span-closing token and everything outside stay
  exact. The guard (G32): the first code row after a span is verified against the 125B given the relaxed comment, which
  the walk does anyway.
- Numbers. [K] replay (11-05): tok/s x1.27 / x1.47 HE+ (fresh windows x1.17 / x1.30), x1.63 / x1.79 MBPP at 16 / 32 rows;
  guarded G32 x1.26 HE+ / x1.55 MBPP. Risk on record: [K] on the 2B a relaxed comment changes the NEXT CODE in 49 / 102
  cases (WEEK98-H, J(A)); the code may still pass (that is what the gate measures), and SPANRELAX-C's milder top-3 form
  kept 0 code-token deviations in M-41's replay but only x1.074 HE+ fresh. [P] per band: the same 0.55x miss-rate effect
  as idea 1 wherever the walk is inside a span, plus the span rows themselves kept at 100 %. Budget: kept x1.17-1.30
  HE+ at 32 rows, halved +8-15 %; MBPP x1.55-1.79, halved +27-40 %.
- Tag: TRIED, UNTESTED in the engine for quality (TRIED L1047 INERT; M-41 SPANRELAX-C BETWEEN / PASS MBPP; WEEK98-H
  BETWEEN; the Rule C engine arm KILL-IN-ENGINE was host cost, not the rule).
- Exact: NO (comment text changes; code tokens exact by construction).
- Kill: tier 1 FAIL (pooled delta < -3 or McNemar p < 0.05 on HE+ + MBPP), or the drift bar on long rows, or the 2B's
  48 % next-code change turning into > 5 % of tasks whose code differs AND fails where the exact one passed.
- <= 15-minute test (tomorrow, CPU): a replay on the fresh 12 HE+ windows with the 125B's top-8 (M-41's harness): with
  INERT on, count (a) rows accepted inside spans, (b) code rows whose INPUT token differs from the exact walk after a
  relaxed span (the "next-code" exposure), (c) kept/32. GO if (b) <= 10 % of post-span code rows; the quality itself needs
  the 45-min tier-1 gate (idea 4).

## Idea 3: span-conditional close-call: delta 1 nat on code rows (passed), delta 2-3 inside comments only

- Mechanism. The close-call rule accepts a draft within delta nats of the 125B's top-1 (the exact margin from the check).
  JCC says delta 1 is safe and delta 2 is not, on the 2B. Make delta a function of the row class: 1 nat on code rows,
  2-3 nats on comment / docstring rows (where 71 % of the ties are and where a flip cannot change the code except through
  the next-code exposure of idea 2, here bounded by the margin). This captures most of delta 2's kept at delta 1's code
  risk.
- Numbers. [K] HE+ kept/32: chain-32 13.77, + CC(1) 16.78 (+22 %); tree-32 16.26, + CC(1) 16.91 (overlap 0.54-0.62: trees
  and close-call rescue the same first misses). [K] JCC delta 1: pass@1 +0.76 pooled (p 0.88), kept +21 %; delta 2:
  -6.06 pooled. [G] delta 2-3 in comments only: + a third of delta 2's extra kept over delta 1 (the comment ties), at the
  code risk of delta 1; halved, +3-5 % kept on top of CC(1). Small; its value is that it is free and already built
  (R57_RELAX_DELTA, rows gated by class).
- Tag: TRIED (WEEK98-JCC delta 1 PASS / delta 2 KILL; TREES-x-CLOSECALL OVERLAPPING; close-call delta 1 / 2 is on
  tonight's EvalSuite gate); NEW: the per-class delta.
- Exact: NO.
- Kill: code-row deviations at delta 1 above JCC's rate, or the comment-row delta 2-3 shows up as > 1 pt pooled loss in
  tier 1; or the gain over CC(1) is < +2 % kept on the replay.
- <= 15-minute test (today, CPU): M-41's replay on the fresh 12 HE+ windows at (code delta, comment delta) in {(1, 1),
  (1, 2), (1, 3), (0.5, 3)}: kept/32, code-row deviations (must equal (1, 1)'s), comment-row flips.

## Idea 4: the proof by Saturday: two kinds of rule, two kinds of certificate

- Mechanism. Split every lossy lever into one of two classes, because they need different proofs:
  (a) DRAFTER-INDEPENDENT policies: the output distribution does not depend on the drafter at all (idea 1's no-comment
      prompt; any "exact greedy of a modified prompt"; distribution-exact sampling would be one too). Certify ONCE by a
      plain decode (no drafter) paired against the default on HE+ 164 + MBPP 100 (tier 1's rule: per set delta > -5 and
      McNemar p >= 0.05; pooled delta > -3). After that, every drafter, tree shape, cap or check variant that is exact
      to the policy inherits the certificate: no re-gating all week while the engine changes.
  (b) DRAFTER-DEPENDENT rules (INERT, close-call, any rank-k rule): the output depends on what the drafter proposed, so
      the certificate is per (rule, drafter, tree) and must be re-run when any of them changes. Freeze the rule and the
      drafter by Wednesday evening; run tier 1 (45 min at 45 GB, engagement counter required) Thursday; the release gate
      (5-7 h) Friday night on the one shipping stack. Tier 0's 15-minute replay pre-screen kills bad settings first, with
      F_hi (flips at margin >= 2 nats) counted on CODE rows only, since INERT flips confident comment tokens by design.
- The 97.8 % claim itself. Under a lossy rule the gated quantity must be ACCEPTANCE per guess (accepted / guessed,
  walk-strict per band on the fresh CERTIFY98 sets), reported beside the exact accuracy, never in its place. Section 0
  shows exact accuracy cannot reach 97.8 at bands 1-10 on code with any acceptance rule, so a plan that gates a lossy
  rule by exact accuracy fails by construction. NewIdeas' 14:00 plan should state the quantity.
- Numbers. [K] the tier-1 gate costs ~45 min at 45 GB (364 items; 2,045 tokens/min with fans up, half that when hot); the
  release gate 5-7 h; a plain-decode HE+ 164 + MBPP 100 run is one tier-1-sized slot. [K] power: 76 % at the measured 3.4 %
  code discordance for a -5 pt effect; a drafter-independent policy run has the same power with NO engine risk.
- Tag: NEW as a procedure; the gates themselves are EvalSuite's (LOSSY_GATE, PREREG_lossygate, CERTIFY98).
- Exact: n/a.
> (a quote from the project owner, kept private)
  the per-stack gate); or EvalSuite cannot free two 45-min slots Thu-Fri.
- <= 15-minute test (today, CPU): EvalSuite's certify98_power.py with the acceptance-per-band quantity instead of exact
  accuracy: how many fresh windows certify 97.8 with LB 97.0 under a rule whose acceptance is ~98 %; if the answer is
  "the same 56 %", the plan holds without a download.

## Killed on the desk (so nobody re-opens them this week)

- Distribution-exact speculative sampling at T (Leviathan / Chen; SpecInfer / SpecTr for trees): TRIED A1 KILL (Leap,
  10-23) on IFEval and confirmed on code tonight (+0.6 .. +2.4 points per band at 0-6, ~0 deeper). The principle survives
  only as a certification route (idea 4a); the kept gain does not exist with this head.
- Typical acceptance (Medusa's entropy rule): a deterministic cousin of close-call with an entropy term; nothing beyond
  delta 1 on code where the ties are, and no certificate of its own. Not worth a slot.
- Any rank-k or margin rule on CODE rows beyond delta 1: JCC's delta 2 KILL (-6.06 pooled) is the line.
- "Accept equivalent whitespace / formatting tokens": [K] newline / indent rows are 10 % of HE+ rows with 15 % of the
  ties, and indentation is semantics in Python. Dead.

## Stack arithmetic for the 900 budget (HE+, 32-row check), halved gains

- tree-32 16.26 -> idea 1 (x1.23-1.29 halved to x1.12-1.15) ~18.2-18.7 -> idea 3 (+2-4 %) ~18.7-19.4 kept; with the 2B
  audit at 15-30 the record's 15.00 line does not stack cleanly (chain only). Against 23.15 that is 80-84 % of the way
  at the SAME check cost, with every answer 14 % shorter (another +14 % per task). The rest of WALL A is drift on code
  rows (section 0), which is problem #4's and the audit's, not acceptance's. A K64 tree (20.77 exact) under idea 1
  reaches ~23-24 kept [P x1.12-1.15], at problem #2's check price.

Scripts and outputs: `accept_T_1116.py` -> `accept_T_1116.out` (this folder; 20 s to re-run).
