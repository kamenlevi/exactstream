# IDEAS_1116b_audit: PUSH800 section 1 audited, then one invention (Fable C, Mon 16 Nov 2026, 17:5x-18:5x EET)

Desk + CPU only (BigMoE's venv, read-only replays on the real t30b dumps; 3 scripts, seconds each). Marks: [K] measured
here or in the records, [P] arithmetic on measured parts, [G] judgement. All numbers are HE+ / MBPP unless stated.

## 0. The verdict in four lines
1. **The kept lever is mispriced for trees.** #13's booked x1.17-1.30 HE+ / x1.63-1.79 MBPP are CHAIN-32 numbers
   (GOAL, Nov 5, DFlash2 chains). Budget-correct on the real t30b tree-32 dumps they are **x1.10 / x1.18** (with close-call
   d1; INERT alone x1.08 / x1.17) [K this file]. The tree spends its 31 nodes on siblings at exactly the rows INERT makes
   free, so after an inert rescue the walk dies of node budget, not of misses.
2. **The lever comes back on a chain:** chain-32 + CC d1 + INERT = **23.2 / 17.7 kept** (x1.68 / x2.56 vs chain-32, x1.37 /
   x1.61 vs today's booked 16.91 / 10.98) [K cells replay]. 60 % / 40 % of those checks end at the 32-row cap.
3. **Even so, 775 is out of reach on HE+ with the booked rules on this head:** the best 32-row lossy shape prices at
   ~650-680 tok/s at 85-90 % of the C6 + int8 bound (needs 26.5-27.3 kept at a 3 ms draft; 28 at the chain's union).
   Tree-32 + every rule (CC d2 + #8 + #13) = 18.1 kept = ~530-560 tok/s. MBPP is half of that in either shape.
4. **Invention: INERT-SHAPED DRAFTING** (section 3): inside a comment / docstring draft a chain, branch only in code rows.
   Lower bound x1.29 / x1.61 over today's base at 32 rows (the pure chain, measured); the hybrid should add the tree's
   code-row gain on top [G]. Quality risk = #13's, no new one. Not the killed SPAN TREE (that one copied seen spans).

## 1. Section 1, term by term

| term (PUSH800 sec 1, corrected 17:5x) | booked | audited | direction | evidence |
|---|---|---|---|---|
| #13 INERT on top of tree-32 | x1.17-1.30 HE+, x1.63-1.79 MBPP | **x1.08-1.10 / x1.17-1.18** (budget-correct, real t30b tree-32) | **too optimistic x1.1-1.5** | `inert_tree127_1116b.out`: tree-32 exact 16.30 / 9.05 (published 16.26 / 9.03, PASS), + INERT c+d 17.90 / 10.57, + CC d1 + INERT 18.01 / 10.64 |
| #13 on a chain-32 (the booked numbers' own shape) | same | **x1.57 / x2.34** INERT alone, **x1.68 / x2.56** with CC d1 (vs chain-32 13.77 / 6.92) | too pessimistic for the chain | `inert_tree_1116b.out` (t30b cells, chain budget exact: 23.18 / 17.69 kept) |
| #8 span-conditional close-call (d1 code, d3 spans) on tree-32 | ON, no number | **x1.040 / x1.042** (16.95 / 9.43); worth +0.3 kept once INERT is on | small either way | same file |
| close-call d2 (section 2's first item) | listed as a lever | **+0.05 kept over d1 once INERT is on** (18.06 vs 18.01); d2 KILLED on the 2B gate anyway (WEEK98-JCC) | no lever | same file |
| today's base kept 16.91 (tree-32 + CC d1) | 16.91 | 16.69 budget-correct (NewIdeas' tree_cc32 follows the cells after a rescue; mine follows the selected chain) | -1.3 %, noise | same file |
| sum of section 2's kept levers at 32 rows | x1.25-1.42 (x1.43-1.62 needed) | **x1.07 on tree-32** (18.06 / 16.91); **x1.37 on chain-32** (23.18 / 16.91) | the x1.43-1.62 does not exist at 32 rows on this head, in any shape | [P] from the two files |
| bigger trees under the lossy stack | "early death's bigger trees" | tree-64 / 128 + CC d1 + INERT: 23.8 / 27.0 HE+ (12.1 / 13.2 MBPP); per Leap's floor ratios R(64) 1.32, R(128) 1.69 that is 18.0 / 16.0 per 32-row-cost, i.e. **no gain** over tree-32's 18.0 | no lever as trees | `inert_tree127_1116b.out` + BIGGER_TREES R(K) |
| expert term 16.4 ms (C6 4.60 GB) for a 32-row check | one number for any 32-row check | the map's union is from CHAIN walks; a tree-32's sibling rows cost 0.84-0.91 of a chain row [K M-10], so a tree check reads ~8-12 % fewer expert bytes (bound ~24.0-24.5 for trees), a chain check the full 16.4 (my chain pricing above uses +8 %) | too pessimistic for trees by ~1 ms; right for chains | M-10 SIBLINGS (TRIED) [P] |
| dense 7.6 ms int8 (byte-bound on 2.13 GB) | per 32 rows | byte-bound means FLAT in rows until int8 compute (~4.25 ms at 32 rows [P from 8.5 bf16 / 2]) passes 7.6, i.e. ~57 rows; a 64-row check pays ~+0.9 ms dense, not +7.6 | too pessimistic for 48-64-row checks by ~6 ms | [P]; Leap's exact-path R(K) (dense + all experts from SSD) overstates the lossy R(64) |
| draft 3 ms (QAT 26 MB head) | tree-32 | levels: tree-32 24.9, tree-64 41.7, chain-32 31, tree-128 59.1 (mean selected depth, the dumps) x 26 MB / 280 GB/s = 2.3 / 3.9 / 2.9 / 5.5 ms + per-level overhead (today's 19.1 ms at 177.6 MB x 24.9 levels = 0.77 ms/level, i.e. ~0.1 ms/level above bytes) | right for 32 rows; +1-3 ms for 64-128-row trees, which the K* arithmetic must carry | [P] |
| 45 % efficiency, 280 GB/s, cores 1.2, overhead 1.5 | - | not auditable from the desk; left as booked | - | - |

Reading: the plan's kept wall is HARDER than section 1 says for trees (x1.07 reachable vs x1.43-1.62 needed) and SOFTER
for chain-shaped drafting (x1.37), and no acceptance rule on this head reaches 26.5 at 32 rows. The 775 arithmetic now
needs one more multiplier that is not in section 2: a code-row accuracy gain (32 % of tree-32's first misses are in code
rows, 6 % at span edges; the other 62 % are what INERT takes), or the row count with a shape that uses it (section 3).

Pricing lines (HE+, C6 + int8 bound 25.2 ms, overhead 1.5, draft 3 tree / 2.9 chain) [P]:
- today's base 16.91: 496 / 520 tok/s at 85 / 90 % (the plan's ~480-545: agrees);
- tree-32 + CC d2 + #8 + INERT 18.06: 530 / 556;
- chain-32 + CC d1 + INERT 23.18 at the chain's union (+8 %, 26.6 ms): 649 / 682; at the tree's 25.2 it would be 680 / 713;
- MBPP, the same chain stack 17.69: ~495 / 520.

## 2. How it was measured (so NewIdeas can re-run it in 2 minutes)
- `inert_tree127_1116b.py` (budget-correct): BigMoE's real t30b 127-node dumps (`tree_t30b_{heplus24,mbppbank}_n127k8.npz`),
  tree-K = the first K pops as treek_final does; per anchor the exact accepted depth is the selected true-path prefix; at
  that depth the head's best SELECTED child is tried under the rule (close-call within delta of the 125B's top-1 from the
  bank's stored top-32 log-probs; INERT = the true text wholly inside a comment / docstring there and the drafted token
  does not end the span: no newline in a comment, no quote in a docstring); after a rescue the walk follows the selected
  best-child chain and judges every node against the true text (equal / close-call / inert). Never more than K nodes.
  Exact reproduces the published kept (16.30 / 20.77 / 23.78 vs 16.26 / 20.77 / 23.70; MBPP 9.05 / 10.21 / 11.31).
- `inert_tree_1116b.py`: the same rules on the t30b cells (chain-32, the drafter's own chain after a miss, 31 deep); its
  tree arms are NOT budget-correct (they let a rescued walk continue on the head's top-1 even where the tree had no node)
  and are superseded by the 127 replay; its chain arms are the chain numbers quoted here. Row classes: NewIdeas'
  slot_prior lexer, fresh at the answer start, docstrings split from other strings (HE+ rows: docstring 56.3 %, comment
  13.1 %; MBPP 37.5 / 19.5 %). Tree-32's first misses: comment 48.5 / 31.0 %, docstring 13.3 / 41.6 %, span edge
  5.8 / 2.7 %, code 32.0 / 24.3 %.
- `inert_shape127_1116b.py`: the shape question on the dumps (tree / the head's chain / the span-shaped selection), used
  as a lower bound in section 3 (the dump only holds the nodes EAGLE-2 expanded, so a re-shaped tree cannot use its budget:
  the span-shaped K 32 found 29.7 nodes, the chain 26.8).
- Standing approximation (theirs and mine): after a lossy acceptance the walk goes on along the true text. GOAL's variant
  A (span ends where the draft ends them) gave the same x1.30 as the true-text ends on chains; the 125B's text after a
  drafter-written comment is the quality question, not the kept question.

## 3. Invention 1: INERT-SHAPED DRAFTING (chain inside spans, branches only in code rows)
- **Mechanism.** Under #13 a comment / docstring row cannot miss, so a sibling there is a wasted node; under today's
  EAGLE-2 order 62 % of HE+ rows (69 % incl. edges) sit in spans and the head is unsure in them (right on 0.45-0.63 of
  span tokens [K GOAL]), so that is exactly where the tree branches. Rule: inside a span expand only the best child (path
  probability untouched); outside, EAGLE-2 as today. The budget flows into depth; code rows keep the tree's rescues.
- **Numbers [K/P].** Pure chain-32 + CC d1 + INERT: **23.2 / 17.7 kept** = x1.29 / x1.61 over today's base at the same
  32 rows (the lower bound: a span-shaped tree with all nodes on one chain). Tree-32 + CC d1 + INERT 18.0 / 10.6. The
  hybrid should add part of the tree's code-row gain (tree-32 beats chain-32 by +2.5 kept exact, most of it at code
  ties) [G: +0.5-1.5 kept]. In the 900 budget's terms: kept 16.91 -> 23.2-24.5 at a draft of 31 levels (2.9 ms at the
  QAT head) and the chain's expert union (+8 % on the expert term).
- **The 64-row version.** 60 % / 40 % of the chain-32 stack's checks end at the cap. A 64-row chain in spans with the
  int8 dense flat in rows (section 1) and experts ~x1.35 prices at ~(7.6 + 17.7 x 1.35 + 1.8) / 0.85 + 5.9 draft + 1.5
  = 46.7 ms for a kept of ~30-33 [G: renewal at the same code-row hazard] -> ~650-710 tok/s: at best +5 % over 32 rows,
  because 63 draft levels cost 5.9 ms. Worth one dump, not a build, until the draft is fused.
- **Screen (<= 10 min, CPU, done here + one GPU dump).** CPU: this file's chain numbers are the lower bound [K]. The
  shape's own number needs BigMoE's dry-run dump tool with the one-line score change (siblings in span rows get -inf),
  32 and 64 nodes, HE+ 24 + MBPP 100: ~15 min GPU at the next free slot; then `inert_tree127_1116b.py` on the new dump
  (2 min CPU). PASS line: kept/32 >= 22 HE+ and >= 16 MBPP with CC d1 + INERT (>= 0.95x the chain bound); KILL < 20.
- **Quality risk.** None beyond #13's: the accepted text inside spans is the head's own chain in both shapes; code rows
  stay strictly checked. The risk that matters is #13's itself (a relaxed comment changes the next code in 48 % of cases
  on the 2B [K WEEK98-H]); EvalSuite's tier-1 gate decides it, and the shape rides on the same gate.
- **What kills it.** #13 failing the gate (then chain-32 + CC d1 = 14.8 < tree-32 + CC d1 = 16.7 and the shape loses);
  or a head whose code-row accuracy is already the wall (then the chain's extra depth meets code misses: the cap-hit share
  falls below ~40 %).
- **Not the kill list's SPAN TREE:** that bet (NewIdeas, KILL 12:3x) drafted deep runs by COPYING seen spans with <= 7
  head levels (x0.80 of tree-32). This one copies nothing: the head drafts every token; only the node allocation changes.

## 4. Invention 2 (check cost): span rows at the cheap expert tier
- **Mechanism.** Under #13 the 125B's verdict on a span row is never used; the row only feeds the KV state. Compute span
  rows with the cheapest expert mix (MSB-3 only, or top-5) and keep the code rows at C6. HE+ span rows are 62 % of rows.
- **Number [P].** Expert term 16.4 -> ~11.8 ms if span rows cost 0.5x (MSB-3 2.064 vs 4-bit 2.784 MB and fewer ranks);
  bound 25.2 -> ~20.6 ms = **x1.22 per kept token** at equal kept. Byte-side CPU screen: 10 min with the map replay
  tools on the 36 windows (`fablec/bytes10_replay.py` with a span mask instead of the cf mask).
- **Quality risk: real and unscreened.** The KV error of a cheap row reaches every later code row through attention; the
  map's "sure rows at 3-bit" flip 0.00-0.07 % [K WALL_1105] but span rows are not sure rows. It is a map arm (non-tie
  flips on the code rows after the span vs the neighbours q1u_0.9 / hyb_cold4), one GPU slot of ~20 min; NewIdeas / Breakthrough
  can add it to tonight's screen list. KILL if non-tie flips on post-span code rows exceed +0.2 over C6.

## 5. Desk kills and notes (so nobody re-runs them)
- Strings other than docstrings add nothing (24.52 vs 24.43 on the cells replay; 0.3-0.5 % of rows).
- Close-call d2 over d1 under INERT: +0.05 kept; not a lever (and KILLED on the 2B gate).
- Bigger trees under the lossy stack (section 1 table): per cost no better than tree-32; the row count pays only with a
  chain-shaped span policy, and then only ~+5 % at 64 rows at today's per-level draft cost.
- Comments-only INERT (docstrings exact) on HE+: tree-32 + INERT comments only 22.6 on the cells replay (not budget-
  correct) vs c+d 23.8: the comment half carries most of HE+'s lever; MBPP needs the docstring half (13.7 vs 12.7).
- The additivity of INERT with tree + CC is 0.85 / 1.00 on chains (near-additive): INERT takes misses the margin-based
  rescues never see (the 125B is often sure of its own wording in a comment the head gets wrong).

## 6. Files
`~/FableGod/fablec/inert_tree127_1116b.{py,out,json}` (budget-correct, the numbers of record), `inert_tree_1116b.{py,out,json}`
(cells / chain), `inert_shape127_1116b.{py,out,json}` (shapes). Inputs read-only: BigMoE ceiling dumps, the dev bank,
NewIdeas' lexer. No GPU, no locks, no installs, nothing committed.
