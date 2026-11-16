# IDEAS_1116 #2: WALL A x THE BUDGET (fable-zero, Mon 16 Nov 12:0x-12:50, desk only)

The question: the joint shape (trees, chains, lossy acceptance, early death, comment rules) that maximises
kept / (check + draft) under the real draft cost, which grows with tree depth. 900 tok/s needs kept >= 23.15 in a
25.7 ms cycle, or kept 28 in 31 ms. Numbers from THINK_1116.md, WEEK98.md, TRIED.md and one CPU calculation
(below). Nothing was run on the GPU.

## The arithmetic first: the draft is priced per LEVEL, so depth is the wall, not width
The head reads 177.6 MB per tree level at 0.66-0.92 ms per level, whatever the width (a level's nodes are one batched
pass). So a tree of depth 30 costs ~20-28 ms of drafting at any K, and 900 at a 25 ms check leaves <= 0.7 ms for ALL
drafting at kept 23.15 (25.7 - 25). With the per-step walk-strict accuracies on record (tree-32 bands 98.5 / 96.6 /
93.7 / 95.2 / 96.2 / 96.9), the expected kept of a D-level draft and tok/s at a 25 ms check (CPU calc, chain-of-levels
model; the real tree-32 at 20.8 levels keeps 16.26 vs the model's 14.4, so the model is 10 % pessimistic):

| levels D | kept (tree-32 acc) | kept (flat 97.8) | tok/s at 0.8 ms per level | at 0.3 | at 0.1 |
|---|---|---|---|---|---|
| 8 | 7.7 | 8.3 | 244 | 280 | 297 |
| 12 | 10.1 | 11.4 | 291 | 352 | 384 |
| 20 | 13.9 | 17.0 | 339 | 448 | 514 |
| 30 | 17.5 | 22.7 | 357 | 514 | 624 (flat 97.8: 462 / 666 / 809) |

- Marginal kept per extra level: +0.76 at level 6, +0.61 at 10, +0.43 at 20, +0.31 at 30. At 900 a kept token is worth
  1.11 ms, so level 20 pays only if a level costs <= 0.47 ms and level 30 only if <= 0.35 ms. **At today's 0.66-0.92
  ms per level the tok/s-optimal draft depth is 6-10 levels, and the deep bands are a loss**, however accurate.
- **Even a perfect flat-97.8 head at 30 levels gives 462 tok/s at 0.8 ms per level and 809 at 0.1.** 900 needs the
  per-level cost near 0.1 ms (~20 MB per level, cache-resident) AND kept >= 23 AND the 25 ms check, or kept tokens
  that cost no levels at all. That is the shape question: where do kept tokens come from that are not head levels?

## Idea 1 (best): SPAN TREE -- copy spans as nodes, head levels only at span ends
- **Mechanism.** A node of the draft tree is a SPAN, not a token: a run copied from the prompt / the answer so far
  (prompt lookup, the engine's copy gate) or a run the 125B already produced (the check's rows). The head runs a level
  only where a span ends or no span matches; all tokens inside spans cost zero levels. The check verifies the spans'
  tokens as ordinary rows (exact where exact, lossy rules on top). Early death (fable-b) keeps the check at ~32 rows'
  price while 64-128 span tokens are offered.
- **Numbers.** The extra kept of the 127-node trees is long runs (HE+ docstring copies; HE+ 23.70 vs MBPP 11.29 at
  128). HE+ answers re-emit the prompt's signature + docstring: 61-68 % of tokens (COPYRUN); the copy gate G2 gave kept
  x1.041 / x1.211 at 16 / 32 rows (HE+), the wide copy check x1.21 (CW8, killed only on the 12 GB pool). A span of 20
  copied tokens costs 0 levels where the head would spend 20 x 0.8 = 16 ms.
- **Tag:** published (prompt lookup decoding, n-gram / copy drafting) + tried in pieces (LEVER1-COPY-REPLAY,
  COPYRUN, copygate G2); the span-node tree with levels only at span ends is new here. **Exact** in the check; lossy
  rules can sit on top.
- **Kills it:** on MBPP (1-3 % copy) it is ~0, so the 900 claim would be HE+-shaped; and if the head must still run a
  level per span token to score siblings (no: spans are not scored, they are verified).
- **Test (<= 15 min, CPU):** replay the 127-node HE+ dumps with span nodes from an 8-gram prompt/answer lookup:
  levels used vs kept, against tree-32's 20.8 levels / 16.26 kept. Line: >= 16 kept at <= 8 head levels.
- **Gain in budget terms:** draft ms /2-3 at equal kept on HE+ (20.8 -> ~6-9 levels), kept +20-40 % on copy-heavy
  text at the same levels; MBPP unchanged.

## Idea 2: DRAFT DEATH -- stop levels where the survival probability no longer pays
- **Mechanism.** The same early-death logic fable-b applies to the check, applied to the draft: a node is extended
  to the next level only while its path confidence (the product of the head's p1 along the path, which is right
  99.5-100 % at p1 >= 0.95) is above tau, and the whole block stops drafting when no node clears tau. Hard blocks
  (after a miss, inside near-ties) stop at 4-6 levels; easy blocks (a docstring, a signature) run to 30. The level
  budget then goes where the marginal kept per level is highest, instead of a fixed depth per block.
- **Numbers.** Marginal kept per level 0.76 -> 0.31 from level 6 to 30 (the average); on the easy half of the blocks
  the survival stays ~0.9 per 10 levels, on the hard half it is under 0.5 by level 6 (after-miss depth 0 is 89 %,
  bands 1-6 93-96 %). The chain engine already has this cut (L1-14 running-confidence truncation t 0.3, ADOPTED).
- **Tag:** tried for chains (ADOPTED); new for trees as the level stop rule. **Exact.**
- **Kills it:** kept/levels no better than a fixed depth on the 127-node dumps (the head's confidence not separating
  easy and hard blocks), or the per-block variance making the check's row count unpredictable for the kernels.
- **Test:** desk join on the 127-node dumps: for tau in 0.3 / 0.5 / 0.7, levels used and kept per block vs fixed
  depth 12 / 20 / 30; the figure is kept per level.
- **Gain:** draft ms -30-50 % at equal kept (levels spent only where they pay); it composes with idea 1.

## Idea 3: COMMENTS ON A CHAIN WITH LOSSY RULES, CODE ON THE TREE
- **Mechanism.** In-comment rows hold 71 % of HE+'s near-ties, so an exact tree spends its branches there. Split the
  shape by region: comment / docstring rows take a single chain with lossy acceptance (SPANRELAX-C top-3, INERT or
  COMMENT-FREE as EvalSuite allows); code rows take the tree with early death. The width budget then serves code's
  near-ties only, where it changes the program.
- **Numbers.** SPANRELAX-C HE+ x1.074 / MBPP x1.125 with 0 code-token deviations; INERT replay x1.17-1.30 HE+ fresh,
  x1.63-1.79 MBPP (quality untested); COMMENT-FREE x1.23-1.29 [P]. A relaxed comment changes the next code 48 % of the
  time on the 2B, so the region split must keep the code rows exact or close-call-gated.
- **Tag:** tried in parts (M-39 / M-41 SPANRELAX, INERT, COMMENT-FREE); the region-split tree/chain shape is new.
  **Lossy** in comments, exact (or close-call) in code.
- **Kills it:** EvalSuite's code gate failing on the comment rules (tonight's delta 1 / 2 run tells), or the tagger
  not being causal at draft time.
- **Test:** replay the 127-node dumps with the tagger: kept per level with comment rows on the chain + top-3 vs the
  full tree; the line is +10 % kept per level.
- **Gain:** kept x1.07-1.3 at no extra levels; the width goes to code.

## Idea 4: ONE WIDE-SHALLOW BLOCK, NOT ONE DEEP ONE (the shape when levels are the price)
- **Mechanism.** At 0.66-0.92 ms per level, give up depth 30 for the lossy speed line: a tree of depth 8-12 that is
  wide (K 64-128 nodes) at the near-tie slots, checked with early death at ~32 rows' price. Width is free in draft
  (one batched pass per level) and expensive only in the check, which early death caps. The 97.8-to-30 target is then
  met by the LAB line (true state), while the speed line runs the shape the budget allows.
- **Numbers.** Draft: depth 10 = 6.6-9.2 ms (vs 13.8-19.1 for tree-32, 26-36 for K128 at depth 29); kept at depth 10
  ~9-10 (chain model) and more with width (K128's wide band sits at depths 2-8 already). tok/s at 25 ms check: 270-318
  from the model at depth 10 vs 339-357 at depth 30 with 0.8 ms levels -- the deep tree only wins if its levels are
  cheap, which is the point: no shape reaches 900 on the head alone (see the table).
- **Tag:** new as a stated shape (K98 beams were fixed-width at every depth; this is width at ties, depth capped).
  **Exact** (lossy rules optional).
- **Kills it:** the 127-node dumps showing most of the deep kept is NOT copy runs (then idea 1 cannot replace it) and
  early death not holding the check near 32 rows.
- **Test:** on the 127-node dumps, kept at depth caps 8 / 12 / 16 with all 128 nodes vs K32 full depth; levels and
  rows per kept.
- **Gain:** draft -50-65 % ms; kept -10-30 % without idea 1, ~0 with it.

## What the numbers say for the plan
- No head-only shape reaches 900: with a 25 ms check and today's 0.66-0.92 ms per level, the best is ~350 tok/s
  (tree-32-like) to ~460 (a flat 97.8 head). 900 needs levels at ~0.1 ms (fable-far's wall: ~20 MB per level,
  cache-resident, or several tokens per level) or kept tokens that cost no levels (idea 1).
- The shape that fits the budget: spans for the runs (zero levels), draft death for the levels, lossy rules in
  comments, a wide-shallow head tree for code's first 8-12 tokens, early death in the check. That is one coherent
  design; its parts test on the 127-node dumps today.
