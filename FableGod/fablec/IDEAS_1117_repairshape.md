# IDEAS_1117_repairshape: the brief's numbers audited; the lead (in-check repair) priced on the real tree shape; one design twist (Fable C, Tue 17 Nov 2026, 07:3x-08:3x EET)

Desk + one 10-second CPU script on BigMoE's t30b dumps (`~/BigMoE/data/deepdraft/ceiling/`: cells for chains, the real
127-node k8 trees). Marks: [K] measured here or in the records, [P] arithmetic on measured parts, [G] judgement. HE+ /
MBPP unless stated. Script and output: `repair_shape_1117.{py,out,json}` beside this file. Nothing ran on the GPU.

## 0. In five lines
1. The brief's "97.8 % flat, walk-strict" is the MARGINAL row (every anchor's depth-0 accuracy); walk-strict LAB reads
   93.2 / 95.3 / 96.4 / 97.8 / 98.8 / 99.2 / 99.3 by band, and t30b alone already reads 97.7 walk-strict at 23-30 [K].
2. "True-state + tie siblings 26.79" is UNBUDGETED (the sibling rows are not charged); the budget-correct way to put a
   true-state mechanism at 32 rows is below.
3. The believed limit ("a 125B row per drafted token at 0.8-1.8 ms") is the AVERAGE row; the marginal batched row costs
   ~0.11-0.18 ms at Leap's floor [P]. The wall is not bytes per row but PASSES per block: one batched pass whose rows
   are chosen before it (hindsight, ORACLE-31). The lead, ZERO's CHECK-AS-TAP-PASS = NewIdeas' IN-CHECK REPAIR, is
   exactly a second placement inside the first pass, so it is the right shape of answer.
4. **Its +38 % is a chain number; on the real tree-32 it holds [K, this file]: 16.30 -> 23.10 (x1.42) at one round,
   25.81 (x1.58) at two, perfect location + a LAB-quality guess, rows-neutral.** Unlike INERT (x1.68 chain -> x1.10
   tree), repair transfers to trees, because it acts at the first miss with a true-state refresh, which no sibling
   covers. Kept 24-26 at 32 rows is inside route 1 alone IF location x guess >= ~0.8 with two rounds; at q 0.5 it is
   20.0 / 20.6.
5. **Location is the mechanism, not the refresh**: a probe-free refresh at a fixed row buys x1.04-1.16 of the chain
   (KILL, below tree-32 itself). So the one number that decides route 1 is the layer-L probe's first-miss RECALL, and
   my design twist (section 3) takes its precision off the critical path.

## 1. Audit of the brief's numbers (my assigned paragraph)
"LAB 97.8 % flat to 30, walk-strict": wrong attribution. NewIdeas' lab_truestate.py marginal row is 98.0 97.9 97.9 97.9
97.8 97.8 97.7 (every anchor, teacher-forced, CERTIFY98 calls it "information only"); its walk-strict row is 93.2 / 95.3
/ 96.4 / 97.8 / 98.8 / 99.2 / 99.3; t30b's own walk-strict 23-30 is 97.7 (survivorship) while its marginal is 89.65. So
GOAL 2 as worded is met or missed by definition; the quantity that buys kept is the marginal per-position accuracy at
bands 3-14 (own 94.4 / 93.1 / 92.2 vs 97.9) [K THINK_1116c, drift_bits.out]. "ORACLE-31 18.48": right, and a hindsight
bound (the head is calibrated: the expectation cap at 32 rows is 16.26), as the 07:3x refinement says. "True-state +
tie siblings 26.79": LAB chain + a rank-1 sibling at every 2nd-choice miss with the sibling rows UNCHARGED (~20 extra
rows per check); "drift fixed + a tree ~24.5-26.8" is [G] (the own head's tree/chain ratio applied to LAB). The
budget-correct statement is section 2: tree-32 + one in-check repair round = 23.1 at 32 rows (perfect). "550 needs kept
~24-26 at <= 44-47 ms": the arithmetic holds, but note two things. (a) At Leap's C6 + int8 bound the cycle is ~29.5 ms
(check 25 + draft 3 + 1.5) and today's tree-32 already projects 566 (Lunatic's table), while at today's 79.5 ms cycle
even kept 26 gives 327; the check's fall is worth x2.7, the drafter channel x1.5, and 550 needs both, as the brief says.
(b) Each repair round adds its catch-up (ZERO: 2.6-3.8 ms at L 8 / 16) times rounds per check (0.64 / 1.0 at one / two
rounds, HE+ tree [K]), i.e. +1.7-3.8 ms on the cycle; kept 24 at 44 ms then needs the check at <= ~34-36 ms. "The
believed limit": a row costs 0.8-1.8 ms on AVERAGE; the marginal row in the batched pass is R(K)-priced at ~0.11-0.18
ms (Leap's 1.44 / 1.69 at 128 rows) [P], and trees already buy it; what no shape beats is one placement per pass. "92 %
persistent" is mine and stands (non-tie rows, between-window variance of the drift gap).

## 2. In-check repair on the real shape [K `repair_shape_1117.out`]
Frame = NewIdeas' b98_repair (reproduced: chain one round 18.97, two 20.90 vs their 20.83): at the first miss the head is
refreshed with the TRUE state of the last right row and re-drafts from it; its depth 0 is the correction guess (a
tree's depth-0 nodes are several guesses), its deeper nodes the tail; perfect location; rows-neutral = the re-draft
gets the rows the dead nodes release; kept = right tokens + the exact correction at layer 48 (certify98's walk). q =
P(a round succeeds) = location x guess quality (NewIdeas' q), drawn per round.

| kept per 32-row check, HE+ / MBPP | plain | 1 round | 2 rounds | unlimited | 1 round, full 31-node re-draft | 2 rounds, full |
|---|---|---|---|---|---|---|
| chain-32 | 13.82 / 6.94 | 18.97 / 10.69 | 20.90 / 13.14 | 22.37 / 15.08 | 19.50 / 10.85 (+21.5 rows) | 22.77 / 13.59 (+40) |
| **tree-32 (real 127-node dumps)** | **16.30 / 9.05** | **23.10 / 15.01** | **25.81 / 19.45** | 28.98 / 26.33 | 24.61 / 15.70 (+20 rows) | 29.93 / 21.63 (+34) |

| tree-32, HE+ / MBPP | q 1.0 | q 0.7 | q 0.5 | q 0.3 |
|---|---|---|---|---|
| 1 round | 23.10 / 15.01 | 21.39 / 13.32 | 19.98 / 11.97 | 18.42 / 10.87 |
| 2 rounds | 25.81 / 19.45 | 23.61 / 15.77 | 20.62 / 13.24 | 18.86 / 11.27 |

- Rounds per check: 0.64 / 1.01 (HE+ tree, one / two rounds), 0.94 / 1.79 (MBPP). Price per round from ZERO: 2.6-3.8 ms
  (catch-up through layers 1..L + the head's levels). At q 1.0, one round on the tree is +6.8 kept for ~1.7-2.4 ms
  per check = 0.25-0.35 ms per kept; at q 0.5, +3.7 for the same = 0.5-0.65. Both far under Lunatic's 1.8 ms bar.
- **The shape lesson, inverted:** yesterday INERT's chain x1.68 fell to x1.10 on trees (siblings already sit where INERT
  frees rows). Repair rises on trees (x1.37 -> x1.42 at one round) because the tree's own siblings cover the head's
  SECOND choice without a refresh, while repair places a refreshed multi-guess subtree at the first miss. The two
  levers do not overlap.
- **Probe-free refresh is dead [K]:** original chain to row r*, then the head refreshed with the true state at r* and a
  chain or tree re-drafted from there (no location): r* 4 / 8 / 12 / 16 give x1.11-1.16 / x1.09-1.11 / x1.04-1.05 /
  x1.05 of the chain on HE+ (MBPP x1.14-1.27 / 1.09-1.14 / 1.04-1.05 / 1.02), all below tree-32. Blocks that miss
  before r* get nothing and the restarted tail still meets the ties; the value of a round is the second chance AT the
  miss with a true state, not the reset of drift. Hence: **the deciding number for route 1 is the layer-L probe's
  first-miss recall**, not its byte saving.

## 3. The design twist (new to the records): ADDITIVE REPAIR, so the probe runs for recall, not precision
- **Mechanism.** Early death's prereg demands <= 2 % false drops because it DROPS rows; a false flag loses kept. In-check
  repair as a REPLACEMENT inherits that constraint: a false flag re-drafts a tail that was right. Run the repair as an
  ADDED sibling branch instead: at the flagged row the original tail stays (its rows are already in flight), the
  refreshed re-draft joins as a second subtree (catch-up through layers 1..L, then layers L+1..48 with everyone). A
  false flag then costs rows (the added subtree, placed at a right row = a refresh there, which section 2 says is worth
  ~x1.05-1.1, never less than 0) but never kept; a true flag gives the full repair. So the probe can be tuned for
  RECALL (flag every row where the layer-L lens prefers another token at all), and the realized q becomes ~recall x
  guess, with recall near 1 instead of the <= 2 %-false-flag operating point.
- **Number.** The full-re-draft column above IS additive repair at perfect location: tree-32 24.61 / 29.93 at one / two
  rounds (HE+), 15.70 / 21.63 (MBPP), +20 / +34 rows per check. At guess quality 0.7 (LAB's single guess at misses is
  ~0.63; the tree's 2-4 depth-0 guesses plus the layer-L lens's top-2 should reach 0.7-0.85 [G]) and recall 0.9: q ~0.65
  -> ~22 / ~23.5 kept at one / two rounds [P on the q table, halved gain over rows-neutral]. Versus rows-neutral
  repair at a 2 %-false-flag probe with recall r: q = r x 0.7; if r is 0.5 (TRIED Rnd17#7's napkin and amendment 2
  suggest a weak chain flag at that operating point) the rows-neutral form gives ~19 (q 0.35) and the additive form
  ~22.
- **Price [P].** The added subtree's deep layers cost the marginal rows of a K64-class pass: R(64)/R(32) 1.32 at the
  floor for +31 rows, i.e. +20 rows ~ +5 ms at the 25 ms bound (today's engine ~x2), plus the catch-up shared with the
  lead (2.6-3.8 ms per round). ~+6-9 ms per check at the bound for +5-8 kept: ~1.0-1.3 ms per kept, under the 1.8 bar
  but above rows-neutral repair's 0.3-0.65. A hedge in between: keep the original tail only k rows past the flag.
- **Exact.** Every emitted token is the 125B's own (both subtrees are full rows); lossy rules apply as today.
- **Screen (adds two lines to NewIdeas' Wednesday earlydeath run on the mid capture; CPU; no new tensors).**
  1. Chain version, per block: the layer-L probe's first-miss RECALL as a function of its false-flag rate (the full
     curve, not the 2 % point), L 8 / 16 / 24, held-out by file parity. Additive repair's realized q = recall(at the
     chosen rate) x guess.
  2. At the last right row of each block: P(the exact correction is among {the head's top-2 from the true state at
     that row} union {the layer-L lens top-2}) — the multi-guess quality. (The head's top-k is in the dumps' top_ids;
     the lens needs the capture.)
  Lines: PASS if recall >= 0.8 at any false-flag rate <= 20 % AND multi-guess >= 0.75 at L 16 -> additive repair projects
  >= 22 kept (x1.35) on the tree; KILL if recall < 0.5 at every rate <= 20 % (then no form of in-check repair clears
  q 0.35, and route 1 stays under +15 %).
- **What kills it.** The lens at layer 16 not seeing silent misses (recall stuck < 0.5: the 125B decides late at exactly
  the rows the head gets wrong); or the mid-pass stall (the head's re-draft levels on the critical path) at today's
  0.27 ms per level making a round 4-8 ms on the engine, which halves the margin above the bar; or the K64-class union
  growth at the bound being worse than R(64) 1.32 on real tree + subtree rows.
- **Tag.** The lead (ZERO 07:4x, NewIdeas 07:3x) is replacement repair with a precision probe; TRIED has tail
  re-drafting without taps (SD-semiAR KILL), layer-L lenses only to drop rows (STOPPRED, ED-CHAIN, Rnd17#7). Repair as
  an added branch with a recall-tuned probe is not in the records.

## 4. For tonight's capture (by 18:00)
No new tensors from me: the probe lines above read the true-text rows (every last-right row is one), and the head's
candidates are in the dumps. If BigMoE has budget, ZERO's item 3 (the layer-16 state of the first rejected row per block)
would let line 1 be read at the exact rows instead of the true-text proxy.

## 5. Files
`repair_shape_1117.{py,out,json}`; records read: BREAK98.md, WEEK98.md 1-100 / 1585-1590 / 1966-1970 / 2930-2938,
Lunatic IDEAS_1116c_drift.md + drift_*.out, ZERO IDEAS_1117_channel.md, NewIdeas b98_repair.{py,txt},
EARLYDEATH_PREREG.md, TRIED greps (GLIDE / KV / induction / early-exit / Jacobi / semi-AR / draft-ahead). No GPU, no
locks, nothing committed.
