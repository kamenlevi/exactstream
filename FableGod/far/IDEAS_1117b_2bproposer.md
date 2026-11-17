# IDEAS_1117b_2bproposer: a drafter with the 2B's knowledge at its located misses, as a PROPOSER, not an overrider (Fable FAR, Tue 17 Nov 2026 08:1x-08:2x, desk + three CPU screens)

BREAK98b, lens (c): fewer first misses at <= ~5 ms per block. Marks: [K] measured (source), [P] arithmetic on [K],
[G] judgement. Screens (CPU, HE+ / MBPP topk cells + MoonShots' rows2b / rows4b, Lunatic's walk; all under 1 min, < 1 GB):
`screens_1117/decider2b.py` -> `.out` (the auditor as an inline decider), `sib2b.py` -> `.out` (as a sibling
proposer), `endcover.py` -> `.out` (corrected block-end coverage; see the banner now on IDEAS_1117_futurestates.md).
Alignment self-check: rows2b's 125B argmax equals the cell's y on 99.4 % (HE+) / 100 % (MBPP) of cells.

> **CLOSED (NewIdeas 09:2x, b98_2b_reconcile.txt, ad06be2).** My 67 / 72 % counted the first misses of chains started at
> EVERY anchor (3,401 misses at 542 positions, the same hard position up to 30x); on the walk's own first misses the 2B is
> right 52 % (4B 58 %). Budgeted, 2B siblings keep 15.94 at 71.9 rows per check, against best-first K63's 20.77 at 64
> rows. **2B-INFORMED TREE = KILL.** Lesson for my screens: weight first misses by the walk, never by every anchor.

## 0. What the drafter's first misses are, and who knows the answer [K sib2b.out]
At the own chain's first misses (walk, teacher-forced cells; 3,401 HE+ / 13,797 MBPP):

| who holds the right token at the first miss | HE+ | MBPP |
|---|---|---|
| the head's #2 | 66.8 % | 59.1 % |
| the head's #2 or #3 | 81.7 % | 76.1 % |
| the frozen 2B's top-1 (true prefix) | 67.1 % | 51.5 % |
| the frozen 4B's top-1 | 71.9 % | 59.1 % |
| head #2 OR 2B | **85.9 %** | 76.2 % |
| head #2 / #3 OR 2B | **89.6 %** | 85.4 % |
| head #2 OR 4B | 87.3 % | 80.1 % |
| 2B right where the head's #2 is wrong | 19.1 % | 17.0 % |

- The 2B is as good as the head's own #2 at the misses and **complementary** to it (a fifth of the misses are fixed
  by the 2B alone). The head's flag catches them: 89 % of first misses sit at head p1 < 0.95 (71 % at < 0.8).
- So the KNOWLEDGE for ~90 % of first misses exists in (head top-3, 2B top-1): ~1.1 GB of auditor per block. What is
  missing is the 125B's CHOICE among those 2-4 candidates, which VERDICT-DEPTH says forms in the last ~4 layers.

## 1. How that knowledge converts to kept: proposer, not decider [K decider2b.out, sib2b.out]
**As an inline decider (override the head's token where it is flagged and the 2B disagrees):** kept 13.77 -> at best
15.1 (2B) / 15.5 (4B) on HE+, +10-13 %, at phi 0.27-0.30 and psi 0.2-0.3 % (theta 0.95, margin >= 2); MBPP 6.92 ->
7.7 / 8.1. Every looser gate loses to false overrides (psi 1-8 %). Even a PERFECT locator with the 2B's answer
(phi 0.67, psi 0) is kept ~20 at two rounds [P on Lunatic's floor_bar table]. **The decider form is capped by
location and false fixes, not by knowledge.** TREE-PLUS-AUDIT's +0.3-5.7 % [K PLAN900] is this cap.

**As a sibling PROPOSER (the 2B's token becomes a tree node beside the head's #2; no override, no psi; unbudgeted
bounds, the sibling's continuation teacher-forced as in Lunatic's 19.26 form):**

| HE+ kept per 32-row check, unbudgeted | HE+ | MBPP | siblings per check |
|---|---|---|---|
| own chain | 13.77 | 6.92 | 0 |
| + head #2 at every miss (Lunatic's form; 19.26 there) | 19.14 | 11.48 | 0.63 / 0.96 |
| + 2B's token at every miss | 19.03 | 11.16 | 0.70 / 1.09 |
| **+ head #2 AND the 2B's token** | **22.06** | 15.90 | 1.09 / 2.21 |
| + head #2, #3 and the 2B | **23.61** | 18.34 | 1.29 / 2.82 |
| + head #2 and the 4B | 23.27 | 16.84 | 1.25 / 2.48 |
| gated: #2 and the 2B (if it disagrees) only at p1 < 0.95 | 20.97 | 15.13 | 0.91 / 1.95 |
| gated: #2 only at p1 < 0.95 (what EAGLE-2 does) | 18.64 | 11.27 | 0.56 / 0.90 |

- Own information caps at ORACLE-31 18.48 (the best 31 of the head's 127 candidates, hindsight) [K lunatic]. The 2B's
  token lifts the same bound form from 19.1 to 22.1 (+15 %), and with the head's #3 to 23.6: **the first bound on
  record that reaches ~24 at the chain's cost in rows plus ~1.3 siblings per check.**
- Realistic conversion: the unbudgeted #2 form (19.14) became 16.26 on the real EAGLE-2-31 tree (x0.85), because the
  siblings' continuations cost rows and levels. Applied: a 2B-informed tree-32 ~18.5-19.5 (+14-20 % over 16.26) [G];
  the 22-23.6 bounds need the siblings' sub-chains budgeted, i.e. ~48-64 rows [G]. So lens (c) delivers the
  knowledge; kept 24 for a 32-row price still needs route (a)'s cheap rows for the continuations.

## 2. Idea 1 (best): 2B-INFORMED TREE: the frozen 2B proposes siblings and re-ranks the head's candidates at flagged rows  [new as a proposer; the 2B audit (override) is TRIED]
**Mechanism.** After the head's draft (chain or tree), ONE batched 2B pass over the drafted rows (tree mask; 1.06 GB,
3.8 ms at 280 GB/s; measured 4-9 ms until timed [K ZERO]) gives the 2B's top-k at every drafted position. At rows the
head flags (p1 < 0.95: 89 % of first misses) add the 2B's top-1 as a sibling node where it differs from the head's
#1 / #2, and re-rank the head's #2 / #3 by head p x 2B p (a learned 3-feature chooser later). Then the head drafts
short sub-chains (depth 4-8) under the new siblings (one wave of ~8 levels: ~2.4 ms at 0.3 ms per level, ~1 ms at the
QAT head), and the check runs the whole tree.
**Number [K/P].** Bound 22.06 (HE+) at 1.09 siblings per check, 23.61 with #3; gated-at-flag bound 20.97; realistic
tree-32 ~18.5-19.5 [G], ~21-22 at 48 rows [G]. At Leap's bound (29.7 ms incl. draft 3 + overhead 1.5): +4 ms serial for
the 2B gives 19 / 33.7 = 564 tok/s vs tree-32's 566: **flat unless the 2B pass is hidden or the siblings' rows are the
cheap kind**. Hidden on the Neural Engine (WHOLE-CHIP: ANE +23-25 % beside the GPU's expert kernel, PASS at the
line's edge [K TRIED 624]; "the one overlap route measured to work" [K WEEK98:2881]) it is 19 / 29.7 = 640 (+13 %) and
~21.5 / 31 = 690 at 48 rows [P/G].
**Why it beats the audit:** the audit overrides (psi ends blocks); a sibling is only a row. The information is the
same 2B pass.
**Screen (NewIdeas, CPU + one short GPU job, <= 30 min):** (1) CPU, now: on tree_t30b_heplus24_n127k8 + rows2b_heplus,
for each anchor take the first-32-pops tree, find the spine's end (deepest on-true node) and the missing true child;
score whether the 2B's top-1 at that position equals y; credit a sub-chain continuation from the topk cells of the same
anchor at the next depths (teacher-forced own chain, up to 8); report kept vs tree-32's 16.26 at +1 and +8 sub-chain
rows. PASS >= +12 % kept at <= 40 rows; KILL < +6 %. (2) GPU (~10 min, <= 5 GB): the 2B over the 127-node trees'
nodes (tree mask) so the re-ranking form (head p x 2B p for EAGLE-2's pops) gets its own ORACLE-31-style bound on
real trees. **Quality:** exact (tree nodes; the check decides). **Kills it:** the real-tree screen lands under +6 %;
or the 2B pass cannot be hidden and rows stay at 0.28 ms (then it is flat at the bound, as TREE-PLUS-AUDIT was).

## 3. Idea 2: a DISTILLED CHOOSER for the hard rows (TrainLab this week; the smarter head in the 1-layer budget)
**Mechanism.** The head's first misses are near-ties the 125B resolves in its last layers; the head holds the answer
in its top-3 at 82 % and the 2B adds 8 pt. Train a small chooser (a 2-layer MLP over [head state at the row, the
embeddings and log-probs of the head's top-3, the 2B's log-probs on those 3 and its top-1]) on the 125B's argmax at
flagged rows only (p1 < 0.95: ~3 rows per block; data = the bank's cells + kdump / kmiss rows, hundreds of thousands
of hard rows exist). It re-orders the siblings EAGLE-2 pops and, where confident, replaces the head's #1 (the decider
form, with its psi line measured). Cost: ~2-10 MB per flagged row, ~0 ms.
**Number [G].** Choosing among the head's top-3 at 82 % ceiling (90 % with the 2B): if the chooser reaches 75 % at
flagged rows vs the head's #1-then-#2 ordering, the tree's shallow bands (93.7 / 94.3 / 96.3 at 3-14 [K]) move +1-2
pt; kept +5-10 % at 32 rows. Not 24 alone.
**Screen (Sandbox98, <= 30 min):** the 2B sandbox's head + a chooser over its top-3 at p1 < 0.95, trained on the
2B's argmax (its own 125B analog), read out as the accuracy at flagged rows and `sb98/tree8.py` tree-32 kept. PASS:
tree-32 kept >= +4 %. **Kills it:** flagged-row accuracy < +5 pt over the head's ordering (the tie is not readable
from the head's features: VERDICT-DEPTH's "last 4 layers" finding applied to the head).

## 4. What a smarter head cannot do (so nobody spends the week on it)
- A bigger / ensembled head on the same inputs: capped by ORACLE-31 (18.48) in information; the sandbox says capacity
  x2, 2 layers, more steps at the same data give <= +0.1-0.8 pt [K README]. Two heads trained differently are two
  draws from the same information; their disagreement is a sibling the head's own #2 already supplies (66.8 %).
- The 2B as the drafter: 1.06 GB per sequential token = 3.8 ms per level; a 31-level block = 118 ms. Only batched.
- Fewer first misses by training on hard rows (t30c-like): the first miss is the 125B's near-tie (92 % under 2 nats
  margin [K]); the head's calibration shows it already knows it does not know. The knowledge gap is the 125B's choice,
  which a 1-layer reader of the final state cannot reconstruct and a 2B reconstructs at 67 %.

## 5. Not proposed
- The 2B's siblings for every row (unflagged rows are right 99.8 % [K calibration]).
- Any override form without a measured psi line (decider2b.out: every loose gate loses kept).

## Summary line
Best: 2B-INFORMED TREE: one batched frozen-2B pass over the draft (1.06 GB, ~4 ms, hideable on the Neural Engine)
proposes the 2B's token as a sibling at the head's flagged rows and re-ranks the head's #2 / #3; at the chain's first
misses the 2B is right 67 % and complementary to the head's #2 (together 86 %, with #3 90 %); unbudgeted bound 22.06
(23.61 with #3) vs own-information 19.14 / ORACLE-31 18.48; realistic tree-32 ~18.5-19.5 [G]; screen = NewIdeas' tree
join on the 127-node dumps + rows2b (PASS >= +12 % kept at <= 40 rows). Kept 24 at the 32-row price needs this
knowledge PLUS route (a)'s cheap rows for the siblings' continuations.
