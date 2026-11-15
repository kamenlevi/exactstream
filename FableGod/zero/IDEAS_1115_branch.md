# IDEAS_1115 #1: WHERE TO BRANCH (fable-zero, Sun 15 Nov 22:0x-22:45, desk only)

The question: with 32-128 exact tree nodes, where do branches go to cover the 125B's near-ties in bands 1-14, and
how is the budget split. Numbers below are from the records read tonight (WEEK98.md, PLAN4D_REAIM978.md,
tree_k98.txt, checkstart_margins.txt, TRIED.md). Nothing was run.

**What the records fix.** The misses that matter sit on the 125B's own near-ties: at the walk's check starts, rows with a
125B margin under 1 nat are 10.2 % on HE+ and 12.8 % on MBPP and are right only 17-61 % of the time (checkstart_margins);
92 % of check-start misses have a margin under 2 nats. A #2 branch at a 50/50 position recovers about half of its
misses, so knowing WHERE m < 1 is worth ~5 pt at depth 0 on HE+ (92.0 -> ~97; the tree-16 gets 98.6 there by spending
broadly). EAGLE-2 best-first by path probability already branches where the HEAD is unsure; its calibration and shape
tweaks are KILLed (CALIB-TREE, TREE-SHAPE). So the open part is the 125B's near-ties the head does NOT see (its
confident misses), and the budget rule once a branch is placed (K98: a non-#1 step gets only #1 continuations and
ranks below the main paths; fixed-width beams are worse than best-first at equal K).

## Idea 1 (best): TIE-RUN -- the last check's own margins say where the next block's ties are
- **Mechanism.** The check computes the 125B's full distribution at every verified row, so at the moment the next
  block is drafted the 125B's margins at the last 1-32 positions are known exactly and cost nothing. Near-ties cluster:
  after a miss the 125B's median margin is 3.87 nats vs 11.87 after a hit, near-ties (m < 0.5) 12.2 vs 1.6 %, and
  "hard rows run in a row" (S6). Rule: the recent margin run (e.g. the min margin over the last 3 verified rows, or
  "the check ended in a miss") sets the root band's width: run of low margins -> top-3 at depths 0-2 plus idea 3's
  stubs; high margins -> spine-heavy (the K128 shape's wide band moved to where the ties are, not to depths 2-8 by
  default).
- **Numbers.** Check starts land on m < 2 at 16.9 vs 5.4 % (HE+) and 23.8 vs 16.9 % (MBPP) of other anchors; margin
  mix alone explains 54 % / 40 % of the check-start dip; the dip is all at after-miss starts (89 %) and absent after a
  full accept (99-100 %). Those are lag-0 facts; the lag-1..4 autocorrelation of (m < 1) is NOT in any read-out yet.
- **Tag:** new as a budget signal. TRIED closes the check's rows (K-next) as drafter INFORMATION (state refresh), not
  as a predictor of where to widen. **Exact:** yes (allocation only).
- **Kills it:** lag-1..3 lift of P(m < 1 | last margins low) under 2x the base rate, or the ties sitting at lag >= 4
  where any block already has its wide band.
- **Monday test (<= 15 min, CPU):** from BigMoE's cells with the margin per row (cells_t30b_heplus24 b316365 logs it;
  the ceiling/checkstart_*.json extension has MBPP), compute P(m < 1 at t + k | min margin over t-2..t < 2) for k =
  1..6 vs the base rate, per set. Then a desk join on the 32-node tree dumps: kept/32 when the root band's width is
  set by the rule vs EAGLE-2-16 at the same K (NewIdeas' tree_shape_cc.py framework). Line: lift >= 2x at k <= 3 and
  kept +2 % or better.

## Idea 2: TIE-STUBS -- a reserved continuation for every placed branch (the budget rule K98 is missing)
- **Mechanism.** Once a branch takes the #2 token at a predicted tie, its subtree must not compete on path probability
  (it always loses: K98). Give each placed branch a fixed stub: its own #1 chain for s = 2-4 steps, charged to a
  reserved part of K (t ties x s nodes), and let EAGLE-2 fill the rest. This is a two-level budget: spine + stubs
  where ties are predicted, instead of one global best-first list or a fixed-width beam.
- **Numbers.** K98: "once the true path takes a non-#1 step it ranks below the main paths and only gets #1
  continuations"; no best-first tree to 128 nodes holds 98 % in any band; beams (b2-b8, K 63-249) keep less than
  best-first at the same K; 82-86 % of K128's nodes are off the spine already, so the budget exists, it is placed
  by path probability rather than by where the 125B ties.
- **Tag:** new (TREE-SHAPE killed depth discounts and root tempering, not reserved stubs; D-24 leaf trees added single
  rows at low-confidence depths with no continuation and lost on row cost). **Exact:** yes.
- **Kills it:** kept/32 <= EAGLE-2 + 2 % at equal K on the 127-node dumps for every (t, s), or the stubs' rows
  raising the expert union past the 32-row floor (check with fable-b's union numbers: 4 siblings at one position use
  ~21 experts vs ~10 for one).
- **Monday test:** desk join on the 127-node dumps (Wednesday's capture spec; the 32-node dumps now): oracle kept for
  stubs s = 2 / 3 / 4 at the t = 2 / 4 lowest-margin positions of the block (oracle placement first, idea 1's rule
  second) vs EAGLE-2 at K 32 / 64 / 128. Reports the ceiling (oracle placement) and the realised gain (rule
  placement) side by side.

## Idea 3: SLOT PRIOR -- where code ties live is a token-class fact, known before any check
- **Mechanism.** The 125B's near-ties in code are not uniform over positions: they sit at identifier choices, line
  starts after a statement, and inside comments and strings. A per-class prior P(m < 1 | class of the drafted prefix's
  last token / span) costs nothing and is available at draft time; the budget rule widens at high-prior slots and
  keeps the spine elsewhere. It composes with idea 1 (recent margins) as two independent features.
- **Numbers.** Lossy close-call at delta 1 accepts non-top-1 tokens per 1K kept: code 8.1, comment 12.7, string 15.9
  (ties are ~1.6-2x denser in comments / strings); MBPP has 29 % of rows under 4 nats vs HE+ 10 % (prose-like text
  ties more); the drafter's MBPP misses are mostly far from the top-1 (median gap -4.5 nats), so the prior must be
  about the 125B's margin, not the head's.
- **Tag:** new as a branch prior (WEEK98-H's comment relaxation is the lossy cousin; nothing in TRIED allocates tree
  width by token class). **Exact:** yes.
- **Kills it:** no class with lift >= 2x that covers >= 25 % of the ties (then the prior is too diffuse to move K).
- **Monday test:** join the cells' per-row margin with the token ids (tokenizer.json): P(m < 1 | class) for ~8
  classes (identifier, punctuation, newline / indent, keyword, number, inside comment, inside string, other), with
  coverage. A lift table is the whole result; the width rule follows from it.

## Idea 4: DISAGREEMENT AS A TIE FLAG -- branch where the head and the 2B differ, instead of replacing
- **Mechanism.** The frozen 2B audit today REPLACES the head's token where it disagrees (P = 1 / 2). As a branch
  proposer it would put both tokens in the tree at that position: the 125B then decides exactly, and the disagreement
  set is a direct flag of where the 125B is unsure (87 % of the 2B's MBPP misses are on unsure rows).
- **Numbers.** Ceilings "head OR 2B": bands 1-2 / 3-6 walk-strict 94.5 / 95.2 vs t30b's 91.5 / 92.4 and the audit's
  91.0 / 93.2 (P = 2): the audit as a replacer leaves 3-2 pt on the table in exactly the bands at issue. Cost is the
  wall: 2.12 GB of 2B reads per check at P = 2 (the budget for drafting is <= ~0.5 GB per block).
- **Tag:** parked by NewIdeas ("2B audit as a tree-branch proposer: only if the ANE route makes the 1.06 GB pass
  ~free"); the branch form itself is not in TRIED. **Exact:** yes.
- **Kills it:** the disagreement set covering < 30 % of the check-start ties, or the 2B pass not fitting the 2-4 ms
  per block that 400 exact leaves.
- **Monday test:** rows2b_heplus.npz x the margin per row: P(m < 2 | disagree) vs base, the share of ties covered,
  and the oracle kept/32 of "both tokens as siblings" vs "replace" at P = 1 on the existing audit join
  (tree_audit.py's self-checks reproduce chain-32 13.769 / 6.924 and the audit 14.456 / 7.427).

## What I would not do
- Another calibration of the head's edge probabilities (CALIB-TREE KILL, TREE-SHAPE KILL): the head's confidence is
  already the best-first score; the missing information is the 125B's, and ideas 1 and 4 are the two free or cheap
  carriers of it that exist before the check.
- Fixed-width beams or a root-only tree (K98 beam rows; root tree at 32 rows: HE+ only loses).
