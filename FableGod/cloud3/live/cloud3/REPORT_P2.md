# REPORT P2: the anatomy of our own unions on the real data (cloud3, Fable at max effort)
Thu 5 Nov 2026, 00:26-00:5x EET (sandbox clock; ahead of the 03:00-04:45 slot). Marks: [K] known (file:line), [P] computed here from named data, [G] judgement. Scripts (numpy only, 7-22 s each): `sim_union.py`, `sim_extra.py`, `anatomy.py`, `mass_anatomy.py`, `sim_bounds.py`, `sim_reassign.py`, `cap45_check.py`, `per_layer.py` (kept in the session's scratch directory; copied to `cloud3/scripts/` at the end of this phase).

## 0. Data used (all inspected field by field; P0 section 4)
- D1: the 36 decision-map windows, MoonShots export (`~/c3mac/data/decision_map_moonshots`): teacher-forced rank-sorted top-10 ids per row and layer along the 125B's own greedy answers, with the HPD-0 walk's live-cut checks (t 0.3), the drafter's cf per draft, and p1. 381 / 1,356 / 1,833 live-cut checks (code / IFEval / multi), 373 / 1,301 / 1,750 uncut 16-row checks. No gate weights.
- D2: `Breakthrough/keepset/qom_vectors.npz`: the realized live-cut rows of one window per domain (38 / 161 / 98 blocks x 48 layers) WITH the renormalised top-10 gate weights, the drafter cf and a C147 residency flag.
- D3: `Breakthrough/data/expA/traces125/*_800_g.npz`: 800 single-stream positions x 48 layers x top-20 ids + raw gate weights (code / chat / math / bulgarian); pseudo-blocks of 16 consecutive positions.
- D4: `~/c3mac/data/band45_cap45.pkl`: 176 real engine checks (the exact REF2 loop at C 160, real DFlash2 16-row drafts, 6 prompts: 2 HumanEval, 4 math): ids per layer (16, 10) + kept.
- D5: `Leap/data/r24_routes.npz`: 20 real drafted 16-row checks, layers 0-11.
- Metric: distinct (expert, layer) pairs per kept token, "dpk" = sum over layers of the union / kept; "dpl" = distinct per layer per check. Every rule is the STRICT form (resident experts count; P1 section 1).

## 1. Where a check's distinct experts come from [P on D1, live-cut checks]
Each distinct expert of a (check, layer) is attributed to the first class that holds: A = a kept row needs it at rank 0-1; B = a kept row needs it only at rank 2-9; C = only rejected / past-cut rows need it. "Kept rows" = the accepted draft rows plus the first-miss or bonus row (the rows whose tokens are committed).

| domain | A: kept rows' rank 0-1 | B: kept rows' rank 2-9 only | C: rejected rows only | singletons (1 row) | used by 2 rows | used by 3+ rows | private to the last kept row |
|---|---|---|---|---|---|---|---|
| code (10.3 rows, 8.5 kept) | 0.206 | 0.658 | 0.136 | 0.588 | 0.206 | 0.206 | 0.057 |
| IFEval (3.7 rows, 2.4 kept) | 0.157 | 0.563 | 0.280 | 0.673 | 0.206 | 0.121 | 0.114 |
| multi (3.0 rows, 1.9 kept) | 0.158 | 0.586 | 0.256 | 0.650 | 0.221 | 0.129 | 0.089 |

By layer band (code): A 0.196 / 0.210 / 0.202, B 0.653 / 0.659 / 0.661, C 0.151 / 0.131 / 0.137 for layers 0-7 / 8-35 / 36-47; singletons 0.670 / 0.569 / 0.565. IFEval: C 0.315 / 0.268 / 0.278; singletons 0.786 / 0.646 / 0.642. The attribution is nearly flat over layers; the early band has the most singletons.

Readings:
1. **Two thirds of the union is the rank 2-9 tail of the kept rows (B).** The rank 0-1 experts of the kept rows are only 16-21 % of the union: a check in which every kept row ran at top-2 and the rejected rows did not exist would compute ~1/5 of today's distinct experts. Everything between x0.2 and x1.0 is a decision about B and C.
2. **The rejected rows (C) are 14 % of the union on code and 26-28 % on chat at the live cut** (at 16 uncut rows they are 35 % on code and 73-75 % on chat: section 2). They are the one class that is free of quality cost if it could be known in advance; the drafter's reach product is the only realizable signal (P1 table rows 5-7).
3. **The union is mostly singletons** (59-67 %): experts that one row alone needs. Sharing between rows is real (consecutive rows' top-10 overlap 27 / 38 / 47 %, random 2 %; the previous row's top-10 holds this row's rank 0-1 experts 39 / 51 / 65 % of the time) but it does not reach the tail: the shared experts are the high-weight ones (D2: shared experts' max row weight 0.165, singletons' 0.080).

## 2. Cross-checks on real engine checks and at 16 rows [P]
| source | rows | kept | union per layer, all rows | kept rows only | ratio |
|---|---|---|---|---|---|
| D1 code, uncut walk (373) | 16 | 8.77 | 73.3 | 47.7 | x0.651 |
| D1 IFEval, uncut (1,301) | 16 | 2.49 | 66.4 | 18.1 | x0.273 |
| D1 multi, uncut (1,750) | 16 | 1.94 | 57.7 | 14.5 | x0.251 |
| D4 cap45, real engine, exact loop C 160 (176 checks; 2 code + 4 math prompts) | 16 | 6.3 | 70.5 | 35.9 | x0.509 |
| D5 r24, real drafted code checks (20; layers 0-11) | 16 | 9.4 + 1 | 89.3 | 63.8 | x0.714 |
| D1 code, live cut (381) | 10.3 | 8.5 | 54.0 | 46.6 | x0.864 |
At 16 rows the union per layer is 70-75 on code and math and ~58-66 on chat, matching the mission's 78-80 (KERNEL90's traces) within the trace mix; r24's 89 is layers 0-11, which run 20-40 % larger unions than the rest (section 5). The live cut (t 0.3) removes most of the rejected-row cost on code (x0.651 -> x0.864 of what is left) but little on chat, where the cut leaves 3-4 rows and 1.1-1.4 of them are still wasted. **Per kept token at 16 rows: code 401, IFEval 1,283, multi 1,427 distinct (expert, layer) pairs; at the live cut 305 / 492 / 490.** The cut is worth x0.76 on code and x0.35-0.38 on chat in distinct experts per kept token at no quality cost (exact): it is the largest lever in the live default, and it is why the 16-row union is the wrong baseline for both quality (the mission's trap) and cost.

## 3. Union growth inside a check (D1, live-cut checks, mean distinct per layer after r rows) [P]
code: 10, 17, 23, 28, 34, 39, 44, 49, 53, 58, 62, 66, 69, 73, 76, 79; IFEval: 10, 16, 22, 27, 32, 36, 41, 45, 50, 53, 56, 61, 61, 64, 67, 70; multi: 10, 15, 20, 24, 29, 34, 39, 44, 48, 51, 54, 60, 62 (no longer checks). Each added row brings 3.5-4 new experts per layer: the k^0.67 law of M-14 (10 x 16^0.67 = 64; measured 79 on code because early rows add more). The marginal row's new experts are its singleton tail (ranks 3-9 mostly: section 4), so the growth is set by the tail, not by the rank 0-1 experts, which the previous rows already hold 39-65 % of the time.

## 4. Gate-weight anatomy (D2, one window per domain with scores; 38 / 161 / 98 live-cut blocks) [P]
- Renormalised weight by rank (identical across domains): 0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010. Top-2 = 48.7 % of a row's mass, top-5 = 82.3 %, top-8 = 96.9 %. (Raw 512-way softmax: the top-10 holds 0.13-0.15 of it; D3.)
- Mass coverage per (block, layer): the experts covering 80 / 90 / 95 % of the block's summed mass are x0.39 / x0.55 / x0.67 of the union on code, x0.42 / x0.57 / x0.69 IFEval, x0.44 / x0.59 / x0.71 multi.
- A MoE-Spec budget of B = 0.5 x union by summed mass leaves every row's rank 0 inside the budget (0.000 outside) and rank 1 almost always (0.000-0.002 outside): at that budget the mass rule drops only ranks 2-9, like a per-row top-k with a floating k. The b16 family's BETWEEN verdict (TRIED) was at 4-row checks with 16 experts per layer (a much tighter budget); the half-union budget is gentler than I assumed in P1 row 4 and deserves its 5-minute map arm.
- Singletons vs shared experts: a singleton's weight in its one row averages 0.080 (rank ~4-5); a shared expert's max weight 0.165 (rank ~2). 56-60 % of the union's experts have max row weight < 0.1 (rank >= 4 for every row that uses them); 15-16 % < 0.02 (rank >= 8).
- Residency at C147 (the 24 GB pool) after the block's loads: 79-82 % of (row, rank) pairs resident: the QOM resident keep computes 4 of 5 pairs at 24 GB and more at 45 GB, which is the P1 section 1 point (bytes, not tiles).

## 5. Layers [P on D1]
Uncut 16-row union per layer on code (layers 0..47): 118 104 94 79 85 88 90 92 71 70 79 70 83 76 74 54 58 71 84 66 66 75 92 66 68 64 73 62 73 72 69 47 53 67 75 64 61 71 87 58 60 69 75 65 64 64 68 79. Layers 0-2 route 1.5x more distinct experts than the mean (118 / 104 / 94 vs 73); layers 15-16 and 31-32 route the fewest (47-58). The same shape holds on IFEval and multi (112 / 99 / 94 ... 35 / 40 at layers 31-32). So the early layers are the most expensive AND, by the map, the most sensitive (reuse 0-7: 19-23 % flips; the 125B's early layers are token-driven: FAR.md 1a, token recall 0.75 at layer 0). Any per-layer cut must spend its budget where the union is large and the tolerance is high: the band 8-35 holds 28 of 48 layers, ~58 % of the union, and is the P-frame-tolerant band on sure rows. Live-cut union per layer on code, exact: 80 70 66 57 61 62 63 65 53 54 58 54 59 56 56 45 48 54 60 51 50 54 64 51 50 49 54 49 52 54 53 40 43 51 53 49 46 52 60 47 47 53 54 49 47 47 48 56 (band means 65.4 / 52.3 / 50.5 for 0-7 / 8-35 / 36-47); under Q1 0.5 strict: 27 25 24 21 22 22 23 24 20 21 23 20 22 21 21 18 19 21 22 20 20 21 24 19 19 19 21 19 20 21 20 16 18 21 21 19 19 20 22 18 18 20 21 19 18 19 19 21 (23.6 / 20.2 / 19.5). Singleton share per layer runs from 0.78 at layer 0 to ~0.50 at layers 31-32. IFEval / multi have the same shape at 24-34 / 15-25 per layer. So the Q1 cut flattens the profile (the early layers' excess is mostly sure rows' tails), and after it the band 8-35 is 28 of 48 layers and 58 % of what remains: the per-layer lever's home.

## 6. Upper bounds per policy kind (D1, live-cut, strict; distinct per kept token and the ratio to exact) [P]
| policy kind | bound | code | IFEval | multi |
|---|---|---|---|---|
| nothing | exact | 304.5 x1.000 | 492.3 x1.000 | 490.0 x1.000 |
| rejected-row cuts | the kept rows only (a perfect stop) | 263.0 x0.864 | 354.6 x0.720 | 364.4 x0.744 |
| sure-row cuts, realizable | Q1 cf >= 0.5 top-2 | 116.1 x0.381 | 295.3 x0.600 | 334.1 x0.682 |
| sure-row cuts, oracle | the 125B's own sure rows (p1 >= 0.99) top-2 | 119.5 x0.392 | 401.1 x0.815 | 360.2 x0.735 |
| both sureness signals | Q1 0.5 + 125B-sure top-2 | 90.7 x0.298 | 268.3 x0.545 | 263.8 x0.538 |
| sure-row cut + rejected oracle | Q1 0.5 + kept rows only | 85.6 x0.281 | 175.2 x0.356 | 204.4 x0.417 |
| both oracles | 125B-sure top-2 + kept rows only | 98.3 x0.323 | 289.5 x0.588 | 266.7 x0.544 |
| reuse across positions (P-frame), sure rows in 8-35 post nothing | on Q1 0.5 | 77.7 x0.255 | 244.2 x0.496 | 295.4 x0.603 |
| reuse on every other row in 8-35 (alternating), no other cut | on exact | 251.1 x0.825 | 415.3 x0.844 | 419.5 x0.856 |
| per-layer k on unsure rows in 8-35 | Q1 0.5 + k 5 / 6 | 100.2 x0.329 / 103.3 x0.339 | 228.6 x0.464 / 241.6 x0.491 | 249.9 x0.510 / 266.4 x0.544 |
| reassignment within a gate tolerance (D3, 16-row pseudo-blocks, union ratio) | swap singletons into union members at delta 0.002 / 0.005 raw | x0.745 / x0.638 (code) | x0.794 / x0.681 (chat) | math x0.716 / x0.609 |
| dropping light singletons, no substitution (D3) | wmax 0.01 / 0.02 raw | x0.702 / x0.539 | x0.766 / x0.571 | math x0.661 / x0.518 |
| the realizable stack | Q1 0.5 + gated bonus + REACH + unsure k 6 in 8-35 (each PASS or pending) | ~x0.30 [G, products] | ~x0.42 | ~x0.47 |

Readings:
1. On code the drafter's cf is already an oracle-grade sureness signal for the union (x0.381 vs the 125B's own x0.392); what is left on code is the tail of the unsure rows (B on unsure rows) and the rejected rows: the per-layer k and the reach cut, together ~x0.75 more.
2. On chat the sureness gate is weak because 52-60 % of rows are mid rows, and the two oracles together only reach x0.54-0.59: the chat union is a mid-row problem that no sureness signal solves; the levers are the bonus row (P1 section 2: x0.60 -> x0.23 at cf 0 is the gated-bonus effect, which on chat is BETWEEN) and the per-layer band on unsure rows (x0.60 -> x0.46-0.49).
3. P-frame reuse on sure rows in the band is the largest single bound after Q1 (x0.255 on code) but the map killed the chained form on code (8.11 % sure flips) and the 24-layer one-row-old form is BETWEEN (1.03 %); the one 12-layer group at a time passes (0.10-0.14 %), which is half the bound (~x0.32 [G]); it is a tile lever at 45 GB (resident experts count), not a bytes lever, which is why it was KILLED on bytes (x0.91) and should be re-priced for tiles.
4. Reassignment within a tolerance looks large by weights and is a KILL by function (orthogonal experts); dropping without substitution reaches the same union only by dropping 6-13 % of every row's mass, i.e. on mid rows too: the row-keyed cuts are strictly better for the same union.

## 7. What is missing (exactly)
- Gate weights for all 36 windows (D1 has ids only; D2 has one window per domain): needed to price mass budgets and ACE-style slot rules on the full map. The harness has them live (dm_common `_pert_scores`); a 13 MB export of `scores` beside `ids` would do.
- The map's per-row flip results for the strict (no resident keep) Q1 forms at cf 0.5 / 0 and for the per-layer band arms: every number above is a union count; the quality of the untested arms is a REQUESTS.md item (section 8 of MISSION.md; written to `REQUESTS.md`).
- Real engine checks with acceptance on chat (D4 has code and math only; D1's chat checks are the walk's teacher-forced checks with the cut applied, which the export note says starts the next check earlier on 4-7 % of checks).
- The engine's expert-GROUP count per call under QOM at 45 GB (to confirm that QOM computes ~the full union; a one-line log in BigMoE's r57 screen).
- The drafter's top-2 candidates per depth with their routes (the HPD-0 walk arrays on the Mac) for the EcoSpec row.

PHASE 2 COMPLETE 00:33 EET
