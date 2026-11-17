# IDEAS_1117b_closure: the VERDICT-DEPTH closure audited; (b) two located decisions priced; what is left of the frame (Fable C, Tue 17 Nov 2026, 08:2x-09:1x EET)

Desk only, on NewIdeas 2b30212 (floor_lens_capture.py / _he.out), Lunatic's IDEAS_1117_floor.md, NewIdeas' b98sim /
ADAPTK, and the day capture's layout (mid_heplus24.npz: X (51000, 10240) fp16 = 8 taps x 6,375 true-text rows, 24 HE+
files). Marks: [K] measured in the records, [P] arithmetic on measured parts, [G] judgement. No GPU, nothing committed.

## 0. In five lines
1. The closure's CONCLUSION stands (in-check repair, partial depth, mid-layer early death are dead for Saturday), but
   its stated REASON ("the 125B decides in its last ~4 layers") is not established: the mid-layer readout is a
   3,200-row cross-fitted ridge in 10,240 dims with its regulariser pinned at the same value at all nine taps, and the
   4-layer map it must learn at L44 is real computation; the verdict it can show is "no cheap readout reads the
   decision", which is the only verdict the week needs.
2. What makes the conclusion robust is cost, not readout: FIX-r with the EXACT verdict at L48 is flat in tok/s at the
   bound (430-466 vs 462 plain) because the catch-up at late layers is a pass; and at L <= 24, where the catch-up is
   cheap, even a 2x better readout leaves repair at >= 4 ms per kept [P], above Lunatic's 1.8 ms bar.
3. (b) two located decisions per block for <= 1.3 passes: KILL by arithmetic. A second pass costs >= 0.66 of a pass
   (dense 0.26-0.30 + a chain-16 union 0.40-0.48 at the floor), so two decisions are 1.66-1.78 passes for 23.65 kept =
   13.3-14.2 kept per pass, below tree-32's 16.26. Break-even needs the second pass at <= 0.45, under the dense alone.
4. The one-pass form of two decisions already exists: K127 (23.70 kept) at 1.44 (floor) / 1.69 (today) passes. So the
   frame "kept 24 for a 32-row check" reduces to one number: **the marginal row at <= ~0.08 ms** (96 rows for <= 7.5
   ms), against 0.28 today and 0.11-0.18 at Leap's floor. That is (a) in full; (b) is not a separate route.
5. Nothing on the decision side beats the frame. The levers left on kept per cost are the recorded ones: ADAPTIVE K
   (+8-15 % HE+, hindsight +17-34 %, gated build), LANDMARK-ANS / future-state rows (+5-15 % in test), and the cycle's
   other terms (the check to <= 30-35 ms, the draft to <= 5 ms). 550 on HE+ is kept ~20 at a <= 36 ms cycle [P].

## 1. Audit of the closure (the assigned paragraph)
VERDICT-DEPTH reads the 125B's layer-L widened stream (10,240) through a ridge to the final mixed vector, cross-fitted
by file parity on the day capture itself, then the real mixer + lm_head on the head's top-8. The data are 24 HE+ files,
6,375 true-text rows per tap, ~3,200 per fitting parity, for a linear map of 10,240 x d_final: the ridge can only learn
the near-identity part. Three readings in the output say exactly that: lambda = 0.1 is chosen at all nine taps (a grid
edge or a flat optimum, not a per-layer fit); the validation MSE barely moves from L2 (4.26) to L44 (3.40) while the
L47 control, the final layer's own 4-stream mean through the same ridge, reaches 1.22 at once (near-identity is all
the ridge gets); and the lens-on-top8 decision rate rises 20 -> 29 -> 42 -> 59 -> 87 % over L2 -> 32 -> 44 -> 47 -> 48,
which is as consistent with "the readout cannot follow four real layers of computation" as with "the decision forms
there". The 125B's last layers DO compute (the L47 -> L48 step alone moves the decision from 59 to 87 %), so part of the
late rise is real; how much is the readout's is unknowable at 3,200 rows. A translator trained at scale would settle it:
a rank-limited map per tap fitted on ~10^6 rows needs one 125B prefill over the dev corpus (~0.5-1 h of GPU) plus a
CPU fit, which no one has run; the tuned-lens literature says such maps read far more at 80-90 % depth than the
model's own final readout does. The LOCATOR rows are cleaner: the head's own margin at AUC 0.954 (recall 97.6 % at 20 %
false flags) is the asset; "lens + head below head alone through layer 40" is an unfitted combination (a cross-fitted
logistic on both margins cannot fall below its better input), so that line is cosmetic, not evidence. None of this
rescues route 1, for a reason independent of readouts: FIX-r with the exact L48 verdict (the mixer, decision 87 %)
yields 430-466 tok/s at the bound against 462 plain, because a round at a late layer re-runs nearly a pass for the new
rows; at L8-24, where a round costs 2.6-5 ms, the decision rate is 24-29 % and would need to reach ~70-80 % for the
rows-neutral form to clear 1.8 ms per kept [P on IDEAS_1117 section 2's q table: q 0.3 -> +2.1 kept, q 0.5 -> +3.7 on
tree-32], which no translator at a third of the depth will plausibly give on rows selected for being the head's
hardest. So: closure CONFIRMED for in-check repair and partial depth; the sentence "the 125B decides in its last ~4
layers" should read "no readout available this week reads the decision before the last layers"; and the only use a
trained translator would have (a lossy check exit at L44, -8 % check bytes) is already gated out by tier-0a's F_line
(no >= 2-nat flips).

## 2. (b) Two located 125B decisions per block for <= ~1.3 passes: KILL by arithmetic
- The second decision needs rows conditioned on the first correction, which is known only at layer 48 of pass 1. So it
  is a second pass over the re-drafted tail (Lunatic's FIX-r at r = 2: 23.65 HE+, 1.10 decisions used per check).
- Price of that pass at the bound [P from Leap's parts]: the dense is read again, 2.13 GB (7.6 ms int8, 0.30 of a 25 ms
  pass; V4 1.80 GB, 0.26), plus the tail's experts: a chain of ~16 rows unions to ~25-30 experts per layer at the
  lossy mix (~2.3 MB each), 2.8-3.3 GB, 10-12 ms, 0.40-0.48. No on-chip memory holds pass 1's union (the system cache
  is ~tens of MB), so nothing is "already loaded". Second pass 0.66-0.78 of a pass; with an 8-row tail 0.51-0.59 for a
  smaller second decision.
- Two decisions per block therefore cost 1.66-1.78 passes for 23.65 kept = 13.3-14.2 kept per pass, against tree-32's
  16.26 per pass. Break-even needs the second pass at <= 0.45 of a pass, below its dense read alone. The empirical
  confirmation is in the closure itself: FIX-r at L48 (the exact decision, a full catch-up) is flat at 430-466 tok/s.
- The one-pass form of "two located decisions" is the tree whose alternatives at the head-located rows carry chains:
  that is K127 (23.70, ~2 decisions in expectation) at 1.44 (floor) / 1.69 (today) passes, and ADAPTIVE K is its
  per-check version. So (b) collapses into (a): **kept 24 at a 32-row price = K127's 96 extra rows for <= ~7.5 ms =
  <= ~0.08 ms per marginal row**, vs 0.28 today (int8 dense 0.15 + union 0.08 + cores 0.04) and 0.11-0.18 at the floor.
  Per-row compute is the whole question; bytes (the union growth, 0.08 of the 0.28) are a quarter of it.

## 3. (d) What is left, honestly, with numbers
- **Placement is already optimal in expectation.** The head's margin locates first misses at AUC 0.954, and best-first
  by path_p is the expected-on-true-maximising order for a calibrated head (NewIdeas); the hindsight cap at 32 rows is
  ORACLE-31 18.48 (+14 %). No located-tree redesign reaches 24 at 32 rows; the room is rows (a) or knowledge (c).
- **The recorded kept-per-cost levers are the frame's remaining budget:** ADAPTIVE K +8.0-10.3 % with today's kernels,
  +10.3-15.6 % with one-pass check forms to 128 rows, hindsight +17-34 % (ADAPTK.md, gated build this week); LANDMARK-ANS
  and future-state rows +5-15 % (Sandbox98 today). Stacked and halved: kept ~19-21 at ~1.1x a 32-row cost [G].
- **550 is then cycle arithmetic, not a drafter breakthrough** [P]: tok/s = kept / (check + draft + ~1.5).

| kept | check 25 (bound) + draft 3 | check 32 + draft 5 | check 32 + draft 12.6 (today's head floor) | check 56 + draft 19 (today) |
|---|---|---|---|---|
| 16.3 (tree-32) | 551 | 423 | 355 | 213 |
| 20 (ADAPTK + free rows, halved) | 678 | 519 | 436 | 261 |
| 24 (the frame's target) | 814 | 623 | 523 | 314 |

  So kept 20 clears 550 only with the check at ~25-30 ms AND the draft at ~3-5 ms (Leap's modules + the QAT head /
  draft-ahead); kept 24 clears it at a 32 + 5 cycle. The drafter-side breakthrough buys ~x1.2-1.5; the engine side
  ~x2-2.5. Both are needed, and the engine side is the larger, as yesterday's audit said.
- **A crumb on per-row compute, exact mode only [P]:** the lm_head (vocab x d_final, ~0.4 G params) is ~9 % of a row's
  dense FLOPs (0.77 of 8.6 GFLOP). In exact greedy verification only the deepest accepted row needs the full argmax;
  every other row needs logits on its <= 8 children only. Lazy lm_head (children-only dot products in the pass, one
  full lm_head on the final row after the walk) saves ~0.013 ms per row, ~1.3 ms per 96 extra rows, ~4-5 % of the
  marginal row. Not in lossy mode (the close-call rule needs the row's top-1). Not a lever on its own; part of (a)'s
  0.28 -> 0.08.
- **Not proposed:** any new lossy acceptance at misses (tier-0a's F_line forbids >= 2-nat flips and d2 failed at 1-2
  nats, so the band between CC(1) and a KILL is empty); draft death / confidence-cut variants (K* already truncates at
  a product of 0.3; a 20 %-false-flag stop would fire before the true miss in ~96 % of blocks [P: 0.8^14]); a KV-cache
  cross-attention drafter (GLIDE; the 2B's cross-attention to the 125B's final states measured +0.18 kept, flat).

## 4. For tonight / the screen queue
- Nothing new to capture for this file. If anyone wants the verdict-depth QUESTION answered as a fact (not needed for
  Saturday): one 125B prefill over the dev corpus with the L40 / L44 widened streams stored (~10^6 rows, ~10 GB fp16
  per tap at 10,240 dims, or 2.5 GB after a 2,560-dim PCA), then a rank-256 ridge per tap on CPU; report the lens-on-top8
  decision rate at the chain's first misses vs the 3,200-row ridge's 36-42 %. PASS for the question: >= 70 % at L44.
  It changes no Saturday decision either way.

## 5. Files
This file only (the arithmetic is in the text; the repair q table is in IDEAS_1117_repairshape.md / repair_shape_1117.out).
Records read: BREAK98b.md, NewIdeas floor_lens_capture.{py,_he.out} (2b30212), BREAK98_SCREEN.md, ADAPTK.md, b98sim
(lounge note), Lunatic IDEAS_1117_floor.md sections 3-4, WEEK98.md 505-513 / 544-552, the day capture's npz header.
