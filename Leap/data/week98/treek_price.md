# Bigger trees: the exact check's cost R(K) and the tree overhead c(K), first cut (Leap, Sun 15 Nov 2026 22:0x)
> (a quote from the project owner, kept private)
tests"). Desk only: scripts/week98/treek_price.py -> data/week98/treek_price.txt. Every number above 32 rows is [est].
Refinements: tonight's C7c (the slice, 32 rows), Monday night's E2 (the engine, 32 rows), and Monday-by-day microbenches
of one-pass 64 / 128-row forms (below).

## The expert union per layer, measured (84 route files, generated tokens, non-overlapping windows)

| K rows | 16 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|
| union, mean | 73.2 | 109.7 | 135.5 | 156.5 | 187.4 | 211.1 |
| the busiest layer per check, p50 / max | 119 / 152 | 181 / 226 | 223 / 290 | 256 / 317 | 298 / 357 | 332 / 380 |

The means match BigMoE's step 0 (72.5 / 108.5) and pipe_price's extrapolations (136.5 / 156.5).
**Engine limit:** the 32-row build's slot pool per gate call is 320 (MAXK). From 96 rows the busiest layer usually
needs more: p50 298, max 357 at 96; p50 332 at 128. So K >= 96 needs a bigger pool or a split check. That is BigMoE's
part, and it carries a memory cost.

## R(K) = check(K) / A16 (A16 = 62 ms on the engine)

| K | floor ms | floor / A16 | REF2-T, split as today | REF2-T, one pass (to build above 32) | + BOF (exact) | contract (track B) | c(K) |
|---|---|---|---|---|---|---|---|
| 16 | 39.1 | 0.63 | 1.00 | 1.00 | 0.96 | 0.94 | 1.025 (measured) |
| 32 | 52.4 | 0.85 | 1.62 (G32T 1.625) | 1.43-1.46 | 1.42 | 1.39 | 1.03-1.05 |
| 48 | 61.9 | 1.00 | 2.18 | 1.79-1.85 | 1.81 | 1.78 | 1.04-1.07 |
| 64 | 69.6 | 1.12 | 2.71 | 2.12-2.21 | 2.18 | 2.14 | 1.05-1.09 |
| 96 | 80.9 | 1.30 | 3.70 | 2.72-2.86 | 2.83 | 2.80 | 1.05-1.13 |
| 128 | 89.5 | 1.44 | 4.65 | 3.27-3.47 | 3.44 | 3.40 | 1.06-1.16 |

- **The floor grows slowly:** 1.44 x A16 at 128 rows, because the union saturates (x2.9 for x8 rows).
- **The realistic cost grows much faster.**
  - Per-row compute dominates beyond 32 rows. The attention and GDN cores grow from 5.7 ms at 16 rows to 35-46 ms at 128
    (GDN's recurrence is sequential over rows; attention is linear to sub-linear).
  - The weight passes add ~25 % of their 16-row time per extra 16-row tile.
  - So the excess over the floor, not the bytes, is what a big tree pays.
- **One-pass K-row forms are the first build above 32 rows.**
  - Today's code path splits every op into 16-row calls.
  - One pass (as v32one / qmm16c2a do at 32) saves 0.4-1.2 x A16 at 64-128 rows.
- **The exact BOF levers and the contract shave a flat ~2-4 ms.** That matters at 16-32 rows and is small beyond. At large
  K the cores are the target (attention by key lists is track B's lever 9, not priced here).

## c(K), the tree overhead over a K-row chain
- **Anchor:** TREE16-TIME v2, a bushy 16-node tree, +1.55 ms engine-scaled (c16 1.025).
- **Scaling:** the overhead (route (b)'s gathers, per-path tails, GDN fork copies) scaled by nodes^1 .. nodes^1.5 over the
  one-pass chain gives the c(K) column.
- **No TREE32-TIME on the slice:** tree16 / tree_att / tree_gdn pad to 16 rows, so a 32-node tree needs their 32-row
  forms first. Not cheap.
- **One build requirement for big trees:** route (b)'s per-node masks are built in Python loops per class and per
  attention layer, O(T x 16) today and O(T^2) at T rows. They must be built once per check and vectorized. At T 128 the
  loops alone would cost ~0.1 s per check.

## Assumptions
- The experts stay at 1.04 x their floor at every K. That needs the slot pool (above) and no extra SSD misses; at 45 GB,
  a bigger union also means more misses from SSD (BigMoE's numbers).
- The dense side's per-class split at 16 rows is Breakthrough's slice profile x4 (engine-scaled).
- Per-tile growth is CB1's 16 -> 32 one-pass ratios held constant. The cores grow per row (attention 0.5-1.0 per tile).
- The tree overhead is TREE16's, scaled; a tree of K nodes routes like K consecutive rows (siblings may widen the union a
  little).

## Next (Monday by day, small tickets)
- **Microbenches:** a one-pass multi-tile qmm16c (64 / 128 rows) on the 5 dense shapes, S4 at 64 / 128 rows, and the
  attention / GDN cores at 64 / 128 rows on the 4-layer slice. They replace the per-tile growth assumptions.
- **Folding in tonight's C7c and Monday's E2:** the 32-row anchors, on the slice and on the engine.

## Update Sun 23:4x: C7c measured the one-pass 32-row check; interim costs v1 (data/week98/treek_costs_v1.json)
- **C7c (the 12-layer slice, tonight): the one-pass 32-row check is 1.345 x A16.** Today's split path is 1.632 on the
  slice vs 1.625 on the engine, so the slice tracks the engine. E2 (Tue 00:30) measures the one-pass check on the engine.
- **One pass beyond 32 rows [est]:** the first cut's model with its excess over A16 scaled by C7c (x0.775), giving 48 1.64,
  64 1.90, 96 2.39, 128 2.84. The unscaled first cut stays as the upper end (1.85 / 2.21 / 2.86 / 3.47).
- **R(24) [desk]:** REF2-T pads every call to 16-row tiles, so a 24-row check pays the 32-row weight passes. Only the
  expert union (93.3 vs 109.7 per layer, measured at 23:5x: scripts/week98/treek_union.py -> data/week98/treek_union.txt;
  busiest layer p50 154, max 197) and the cores' 8 rows shrink, by -0.101 and -0.023 x A16:
  - one pass 1.22;
  - split 1.50;
  - contract 1.155;
  - floor 0.749.
- **The 32-row break-even on HE+ with NewIdeas' model** (kept 16.26 at tree-32 vs 11.26 at tree-16, c 1.04 / 1.025,
  4 ms overhead): tree-32 beats tree-16 when R(32) < 1.45.
  - At C7c's 1.345: 179 vs 167 tok/s (+7.5 %).
  - At today's 1.625: 150 vs 167.
  - So the one-pass 32-row check is what makes tree-32 pay.
