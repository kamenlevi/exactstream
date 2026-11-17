# BREAK98b: kept 24 for the price of a 32-row check, now that partial depth is closed (Tue 17 Nov 08:2x; desk sprint to ~09:45; orchestrator)

Kamen's goals stand (WEEK98.md, KAMEN 07:2x): by Saturday, the cheapest way to give the drafter the 125B's information,
the drafter at 97.8 % deep, and lossy >= 550 tok/s on code. Kamen's examples are options, not rules.

## Why a second round
**VERDICT-DEPTH** ran on BigMoE's day capture (NewIdeas 08:1x, 2b30212; HE+, all self-checks pass). **Route 1 by partial
depth is CLOSED on the 125B itself.**
- **Location:**
  - The head's OWN margin finds its first misses at AUC 0.954 (recall 97.6 % at 20 % false flags).
  - The 125B's lens: 0.69-0.73 at layers 2-32, 0.756 at 40, 0.839 at 44, 0.97-0.999 only at the final layer.
  - Lens + head stays below the head alone through layer 40.
- **Decision:** the lens picks the right token at only 20-29 % of first misses through layer 32, 37-42 % at 44, and
  52-87 % only at the final layer.
- **Kept:** in-check repair driven by any layer <= 44 LOWERS kept (L16 10.5, L24 11.3, L44 13.8, against the chain's
  13.77). The 125B decides in its last ~4 layers.

## What we know now (quote with headings)
- The head knows WHERE it is wrong. What it lacks is the 125B's DECISION there.
- Two located decisions per block give HE+ 23.65 / MBPP 15.39 kept. Compare tree-32 today 16.26 and true states
  everywhere 20.77 (Lunatic, ~/FableGod/lunatic/IDEAS_1117_floor.md). Decisions come only from full depth.
- **Rows:**
  - K127 (own information, 128 rows) keeps 23.70.
  - Big trees are flat at today's per-row cost: ~0.28 ms per marginal row = int8 dense 0.15 + union growth 0.08 +
    cores 0.04 (FAR's pricing); 0.11-0.18 ms at Leap's floor (Fable C).
  - Expert routes are partly context-driven, so a big tree's union sits ~15-25 % under the chain law (NewIdeas).
- **Draft:** the planned head reads 177.6 MB per level, a byte floor of 12.6 ms per tree-32 block (MoonShots,
  DRAFT_PLAN_1117.md).
- **Check:** 56.4 ms for a lossy 32-row check today; bound ~25 ms; GPU-paced.
- **Small free levers in test:** LANDMARK-ANS and FAR's future-state rows (realistically +5-15 %; Sandbox98 today).

## The question
Kept >= 24 on HE+ for about the cost of a 32-row check, by one of:
- **(a) Nearly free extra rows:** ~96 more rows for <= ~5 ms, i.e. <= ~0.05 ms per row. That is COMPUTE, not bytes.
  Where can it come from on the M5 Pro? (GPU efficiency, lower-precision activations, the ANE / CPU AMX for some rows,
  work shared between sibling rows, ...)
- **(b) Two located 125B decisions per block** for <= ~1.3 passes' bytes.
- **(c) A drafter with more knowledge** (fewer first misses) at <= ~5 ms per block.
- **(d) Anything else** that beats the frame.

## Lenses
- **Lunatic:** the limits of (a)-(c). What is the least compute an extra row can cost? How much knowledge can a head
  of <= ~60 MB per level hold?
- **ZERO:** (a) on the M5 Pro: every compute unit and every format; the cheapest extra row.
- **FAR:** (c), a smarter drafter: distillation, ensembles, the 2B inside the draft budget, ...
- **Fable B:** (a) on the dense pass: per-row dense compute (lower-precision activations, sharing between siblings and
  between tree rows with a common prefix).
- **Fable C:** audit the closure (VERDICT-DEPTH) in one paragraph, then (b) or (d).

## Each idea gives
the mechanism; the number it moves, from measured parts; a <= 10-min screen (CPU on the dumps / today's capture, or a
microbench); the quality risk; what kills it.

## Kill list (adds to BREAK98's)
- partial-width rows per token;
- in-check repair, shallow taps, partial-depth rows, mid-layer readouts (VERDICT-DEPTH);
- early death by a mid-layer lens (EARLYDEATH stage 2, NewIdeas 08:1x: the lens drops 3.0-4.6 % of wrong children at
  <= 2 % false drops; the head's own edge gap alone drops 31.4 %);
- session value prediction;
- swap-only repair;
- probe-free refresh at a fixed row.

## Output
- Write IDEAS_1117b_<topic>.md in your own folder by ~09:45.
- Then send ONE line to NewIdeas (`peer send ojox-28`) and the orchestrator: the best idea, its number, its screen.
