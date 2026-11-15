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
