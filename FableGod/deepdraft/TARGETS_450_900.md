> **CORRECTED Mon 16 Nov 10:3x (`date`): every tok/s grid below that says "drafting hidden" is an UPPER bound that can't
> be reached.**
> - Pillar 3 closed. Draft bytes add in full on a bandwidth-bound check; no byte cut keeps kept within 1 % (M3).
> - Add the draft per block (full head, 0.66-0.92 ms per level): tree-16 8.3-11.5 ms, tree-32 13.8-19.1, K128 26.2-36.3.
> - HE+ like for like at the byte floor:
>   - tree-16 204-216 (not 257);
>   - tree-32 209-224 (not 277);
>   - K128 + early death 251-303 (not 410-455).
> - At the measured engine, HE+ tree-16 is ~144 tok/s (~150 with a hand-fused level), not 167 (research/week98/earlydeath_unhidden.txt).

> **CORRECTION Mon 16 Nov 08:2x (`date`): the check pillar re-priced on the MEASURED engine.**
> - **Lossy:** Fit12's LOSSY32 (number-checked) prices the "5.86 GB = 20.9 / 24.9 ms" plan per component on the engine
>   at **49.8-53.3 ms realistic** for a 32-row check (6.12 GB on code; its code floor is 21.9 ms, not 20.9).
>   - At K* lossy (tree-32 + close-call 16.91), that is **295-314 tok/s HE+, not 583-679.**
>   - Best kernels with no misses: ~29.8 ms GPU, about 500 tok/s.
>   - **900 needs a ~15 ms check** at kept 16.91. Nothing priced gets there.
> - **Exact:** the engine measures 90.5 ms per 32-row check (Breakthrough a082f8c C32one; R32 1.480).
>   - K* at the engine is tree-16, ~167 tok/s HE+ (tree-32 ~164).
>   - The exact 32-row BYTE floor on code is 47.3 ms, not 52 (REF2-T reads the LSB plane for 31 of 114 union experts).
>     Tree-32 at that floor projects to ~306 tok/s.
> - **This month's lossy check line** (Fit12 LOSSY32 ADDENDUM, received Mon 08:3x; levers vs TRIED), C5 per 32-row check:
>   - 60.9-63.5 ms today (two 16-row calls);
>   - 49.8-53.3 with a one-pass > 16-row FAST form (not built; the biggest item);
>   - ~45-48 with Leap's 1b list (-4.3..-5.7 [P], ~3 days);
>   - ~41-45 with a resident-only MSB-3 rule (-3.7 [P], quality unpriced).
>   - That is ~325-375 tok/s HE+ at K* lossy 16.91. 900 needs ~15 ms.
>   - **v2 (Fit12, Mon 09:2x, + Leap's 1c glue: 34 ops x 48 layers x 6.5 us = +10.6 ms CPU):**
>     - C5: today's route 62.4-65.9 -> one pass 51.3-55.7 -> + FAST 1b 47.5-50.6 -> + resident-only 43.9-46.9, i.e.
>       **332-353 tok/s HE+**; 42.1-43.5 with the glue fused.
>     - The 1c family's CPU floor is 38.5 ms until the glue is fused. That is the NEXT WALL: at C5's last step the GPU
>       is 42.3 ms vs the CPU's 38.5.
>   - Not in TRIED at 32 rows: expert-kernel and CPU-build levers (Leap's dispatch-diet price is pending).
> - **So, with today's drafter:** exact ~165-180 (engine) / ~306 (floor) vs 450, and lossy ~300 (realistic) / ~500
>   (best kernels) vs 900. Both targets need the drafter (x1.4-1.55 kept) AND a check at or below its floor.

# 450 EXACT / 900 LOSSY on code: what each pillar must reach (NewIdeas, Sun 15 Nov 22:0x)

> (a quote from the project owner, kept private)
The numbers come from research/week98/targets_450_900.txt.

The model is tok/s = kept per 32-row check / (check + overhead + unhidden drafting). Overhead is 4 ms. Kept values are
dev replays (= real decoding per kept_walk) unless marked LAB (true state, not deployable). "Flat-equivalent" is the
per-guess p whose flat line 1 + sum p^k gives that kept.

## Even with every other pillar perfect, flat 97.8 % gives 413-436 exact
- 450 exact at 32 rows needs **kept >= 23.9 / 25.2** (flat 98.0 / 98.4 %), with the exact check AT its byte floor
  (49.1 / 52.0 ms) and drafting hidden.
  - That is above the true-state head: 20.77, or 97.0 % flat-equivalent at the walk's check starts.
  - At today's check (~100 ms measured G32T) or the contract (~86 ms [est]), no drafter can reach 450 at 32 rows.
- **The other exact route:** 64-row checks at their byte floor (~69 ms, Leap [est]) plus a drafter holding ~97.8 % to
  depth 63 gives kept ~34.5, about 470 tok/s.
  - This needs a marginal row that costs bytes only. Leap: the real per-row cores grow ~linearly today, giving R64
    2.14 realistic vs 1.12 floor.
- **900 lossy** needs kept >= 22.4 at the 20.9 ms lossy floor (flat 97.6 %), or 26.1 at 25 ms (98.6 %).

## Pillars: need / today / gap

| pillar | 450 exact needs | 900 lossy needs | today's best | gap |
|---|---|---|---|---|
| drafter, kept / 32-row check | >= 23.9-25.2 (98.0-98.4 % flat) | >= 22.4 at 20.9 ms (97.6 %); 26.1 at 25 ms | exact: tree-32 16.26 (95.1 %), the K* candidate (C1), cert. Thu-Fri; 2B audit P=2 15.00; lossy: + close-call ~16.8 (95.3 %); LAB true state 20.77 (97.0 %) | exact x1.47-1.55, beyond perfect information; lossy x1.33-1.6 |
| check, 32 rows | <= 49-52 ms = the byte floor | <= 20.9-25 ms | exact ~100 ms measured, ~86 contract [est]; lossy not built (5.86 GB floor 20.9 ms) | exact x1.9; lossy: a build |
| drafting | hidden (<= ~2-4 ms on the path) | hidden | tree-32 ~24 ms unhidden (20.8 levels x 1.15 ms); chain head 5.4 GB per block | ~24 ms to hide |
| overhead | ~4 ms (counted) | ~4 ms | commit 3.9-5.4 + seams 2.5-3.6; -2.5 with a batched commit | 2-5 ms |

## tok/s grid (drafting hidden, +4 ms)

| drafter (kept) | exact floor 49.1 | exact 52.0 | contract ~86 | today ~100 | lossy 20.9 | lossy 25 |
|---|---|---|---|---|---|---|
| chain-32 t30b (13.77) | 259 | 246 | 153 | 131 | 553 | 475 |
| tree-32 (16.26) | 306 | 290 | 180 | 155 | 653 | 561 |
| chain-32 + close-call, LOSSY (~16.8) | 316 | 299 | 186 | 160 | 673 | 578 |
| LAB true state (20.77) | 391 | 371 | 230 | 198 | 834 | 716 |
| flat 97.8 % (23.15) | 436 | 413 | 257 | 221 | 930 | 798 |

## K* and PLAN4D in this frame
- **K* = tree-32 is today's best exact drafter configuration:** 290-306 at the exact floor and 653 at the lossy floor,
  drafting hidden.
- **PLAN4D's levers each add ~+5-9 % kept** (audit P=2 +8.9 %, lookup UB +8.7 %, chooser +4.5 %, t30c ~+2-4 %). Each
  is about a sixth of the 450-exact drafter gap, and they overlap.
- **The drafter gap is the check-start wall,** the 125B's own near-ties right after a miss:
  - Even the true state is 93.2 / 95.3 / 96.4 walk-strict at bands 0 / 1-2 / 3-6.
  - Trees are the only cover found: tree-32 reaches 98.5 at band 0 but 93.7-96.6 at 1-14.
- **Where a breakthrough would have to land** (honest; none is in hand):
  1. **The exact check at its byte floor** (x1.9 from today). One-pass K-row kernels, with per-row cores reduced to
     bytes; that also opens 64-row checks.
  2. **Covering the 125B's near-ties after a miss at >= 98 %** at depths 1-14, not only depth 0. Neither a better state
     nor an auditor does it, since both are bounded by the true state.
  3. **Drafting off the critical path** (~24 ms today for tree-32).
- **The Saturday story, certifiable by tests:**
  - per-pillar bests on fresh data: tree-32 T / band 0 (C1 at K*), the audit's deep bands (C2), the LAB knowledge line;
  - Leap / BigMoE's measured exact check (E14 / E2);
  - tok/s stated as projections at the floor next to measured.
