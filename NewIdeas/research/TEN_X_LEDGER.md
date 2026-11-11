# TEN-X LEDGER (BYTES10): SSD MB per kept token, lever by lever, at 45 / 24 / 12 GB

Wed 4 Nov 2026, 03:2x-04:0x EET. NewIdeas, the BYTES10 lead. CPU only (at most 4 cores, share locks, beside
Breakthrough's FLOOR night); no GPU, no downloads.
Marks: [K] measured (source named), [P] replay or arithmetic on recorded routing, [G] judgement. Decide on halved gains.

Inputs:
- the decision map's 36 windows: the 125B's own teacher-forced, rank-sorted routing, 12 each on code (HumanEval),
  IFEval and multi; ~280 answer tokens per window;
- MTP Head's HPD-0 walks, cut at the live t 0.3 as in every live map arm;
- three pool lines: 45 GB = laysz45_256's per-layer C_m / C_l (mean 295 / 89); 24 GB = 147 / 46; 12 GB = 51 / 32;
- records3: the 3-bit MSB part is 2,162,688 B per miss. The 1-bit LSB sidecar is 770,048 B and is read for ranks 0-1
  only (REF2-T, the exact 45 GB reference).

Scripts, with outputs in data/ledger/:
- scripts/ten_x_ledger.py: my replay, parts A / B / C;
- scripts/bytes10_replay_fix.py: Fable C's replay with one fix;
- scripts/ledger_lastrow.py: the last-row reconciliation.

## 0. The answer (read this first)

**1. Fable C's BYTES10 numbers (~/FableGod/fablec/BYTES10.md, 00:35) were 2-4x too low, from a one-line indexing bug.**

- Where: `bytes10_replay.py` line 135 builds the keep mask as `keep[sure, :ks]` / `keep[~sure, :ku]` on a
  `[rows, 48 layers, 10 ranks]` array. That slices layers, not ranks.
- Effect, verified on one code block:
  - EXACT read 3-bit base records on layers 0-9 only. The unions were 120 / 79 experts on layers 0 / 9 and empty on
    layers 10 / 47, where the true unions are 80 / 89.
  - Sure rows read all 10 ranks, but on layers 0-1 only.
  - The 1-bit sidecar part was right.
- The fix: `keep[sure, :, :ks]` / `keep[~sure, :, :ku]`. The fixed copy's rerun (data/ledger/bytes10_replay_fix.out)
  matches my independent replay up to its miss-counting convention (section 3). Both now match the engine:
  EXACT at 45 GB on code is 60.4 [P] against W1B's 62.4 [K] (Fable C had 18.7).
- Scope: no other script has the bug. I searched FableGod, Breakthrough, MoonShots and NewIdeas. The three similar
  lines (`dm_cold_lru`, `dm_cold_masks`, `dm_u3c_replay`) index a one-layer `[rows, 10]` array, which is correct.
  Fable C's screen-1 masks (`bytes10_bonus_masks.py`) index rows and layers explicitly, also correct: tonight's and
  13:45's map tests are unaffected. Only byte prices are.

**2. Corrected levels**, in SSD MB per kept token (code / IFEval / multi). Plain LRU, today's live checks; Fable C's
replay with the fix.

| line | EXACT | FAST today: QOM cf 0 + 3-bit sure rows | + gated bonus row | + top-1 on sure rows | + rank-class pools | + 11-bit scales |
|---|---|---|---|---|---|---|
| 45 GB | 60.4 / 38.1 / 33.2 | 13.5 / 19.3 / 20.2 | 5.7 / 4.5 / 4.4 | 2.8 / 2.8 / 2.9 | - | 2.3 / 2.3 / 2.4 |
| 24 GB | 245.2 / 210.2 / 182.6 | 61.7 / 86.1 / 83.1 | 30.8 / 20.0 / 14.2 | 16.8 / 11.8 / 7.9 | **8.8 / 8.7 / 6.9** | **7.3 / 7.2 / 5.7** |
| 12 GB | 636.8 / 621.6 / 486.6 | 135.0 / 250.5 / 225.8 | 69.6 / 75.7 / 55.5 | 32.1 / 33.5 / 24.4 | 29.5 / 31.7 / 22.6 | 24.3 / 26.1 / 18.6 |

For comparison, Fable C's pre-fix code rows:
- EXACT: 18.7 / 96.5 / 185.6 at 45 / 24 / 12 GB;
- FAST start: 3.3 / 32.6 / 55.9.

Notes on the table:
- "+ gated bonus row" = every bonus row gated ("all"). Fable C's next-draft proxy gives 31.4 / 21.6 / 15.0 at 24 GB.
- 11-bit (CORRECTED Wed 13:4x, QUEUE erratum): the packed parts are 2,064,384 / 720,896 B, x0.955 / x0.936 per part. The
  x0.824 used here at 03:4x is MoonShots' NET replay result at the 45 GB C 256 pools (it already counts ~4.6 % more parts
  per GB), not a per-byte factor. The 45 GB 11-bit cell applies that net factor to the top-1 column.

**3. Under 10 at 24 GB exists on paper, but only in FAST, and only with five levers stacked.** Two of the five still
need a quality result (the gated bonus row, top-1 on sure rows). The pool policy is exact for any rule (it changes only
what stays resident) but exists only in the replay.

| lever | quality status |
|---|---|
| QOM cf 0 | map PASS, 03:26 |
| 3-bit sure rows | map PASS at cf 0.5 on all three, at cf 0 on code / IFEval; multi BETWEEN at cf 0 (4.14 %), 03:54 |
| the gated bonus row | screen 1, 13:45 |
| top-1 on sure rows | screen 2, 13:45 |
| a frequency-aware pool (rank-class pools, an SLRU-like split) | no quality question (a cache policy); replay only, needs an engine build. FlashMoE-style learned replacement is the published form |

- 11-bit scales are exact and come on top.
- Halving each lever after the QOM + 3-bit point: 24 GB lands at ~25 / 33 / 32. Halving every factor of a five-lever
  product is harsh; the truth sits between 6-7 and ~30 until the screens and the engine calibration (screen 3) run.
- **EXACT at 24 GB has no path to 10.** Its OPT bound (Belady, a perfect future-knowing cache) is 116.9 / 93.9 / 83.5.
  The answer's working set at all 10 ranks (292 / 207 / 219 experts per layer) does not fit 147 slots.

**4. 12 GB:** the full FAST stack lands at 24-26 / 18.6 [P] unhalved. Under 10 there needs the resident set to hold the
sure rows' rank-0 working set:
- 2-bit resident records (MoonShots' records4);
- or a 2-pass check whose first pass fits the pool.

EXACT at 12 GB is ~300-390 even at OPT.

**5. A correction to my own decision-2 price.**
- My capture replay (data/band45_oea/band45_capture_qom.txt) gave the numbers Kamen saw:
  - QOM cf 0.5: -60.4 / -36.9 / -33.4 %;
  - QOM cf 0: -64.3 / -58.5 / -62.4 %.
- That replay gated the last row of every live check by its own draft ("cf per row = the draft that row verifies":
  the engine's convention).
- The map's QOM arms keep that row exact (cf -1): the bonus row.
- Part C (section 2) prices QOM exactly as the map tested it; section 4 reconciles the two.
- The gap between the two prices is Fable C's gated bonus row, which screen 1 tests at 13:45. Reconciled in section 4:
  the engine convention reproduces the capture price within 1-4 points at cf 0.
- Decision 2 itself (FAST may depend on the cache state) is untouched.

**6. Fable C's findings, sorted by whether the fix changes them.**

Survive:
- The mechanism ranking: the gated bonus row first, top-1 on sure rows second, 11-bit third.
- Its novelty claim for the gated bonus row.
- Kept per check is flat for EXACT. My part A: x0.91 from k 1 to k 16 at 24 GB on code.
- Wasted rows cost: k 14 + 2 wasted is +24 % at 24 GB on code.
- 24 / 12 GB bytes are capacity misses: OPT is ~x0.45 of LRU on EXACT.

Change:
- Every level in the BYTES10 tables (section 3).
- "45 GB is not a cache problem": on code the per-request compulsory set (decode, prefill reuse) is 40.3 of 60.4, so
  ~2/3 footprint and ~1/3 capacity.
- The gated bonus row is worth more: x0.50 / x0.23 / x0.17 at 24 GB, where Fable C had x0.66 / x0.48 / x0.47.
- Popularity pinning and rank-class pools were killed on bugged numbers. On the FAST stack the policy gap is large
  (OPT x0.29 / x0.37 / x0.51 of LRU at 24 GB), and rank-class pools take x0.52 / x0.74 / x0.87 of LRU with top-1 + the
  gated bonus row. A policy lever is back on the board.

**7. Map status of the cf 0 FAST stack (Wed 05:13).** Quality is from the map [K]; MB per kept token at 24 GB comes from
my replay [P], counting each block's misses before the block.

| rule | code | IFEval | multi | 24 GB MB per kept token |
|---|---|---|---|---|
| EXACT | - | - | - | 212.8 / 200.4 / 175.8 |
| QOM cf 0 (last row exact) | PASS | PASS | PASS | 74.8 / 102.7 / 93.7 |
| + 3-bit sure rows | PASS | PASS | BETWEEN | 54.5 / 84.4 / 81.7 |
| + the gated last row (QOM form), 4-bit | PASS (free) | BETWEEN, under Q1's level | BETWEEN, under Q1's level | 58.3 / 47.3 / 34.4 |
| + the gated last row + 3-bit | PASS | BETWEEN | BETWEEN | 35.7 / 19.4 / 13.9 |
| top-1 on sure rows (cf 0.9, union-only; screen 2) | BETWEEN | KILL | BETWEEN | its QOM form (2b) below |
| QOM cf 0, top-1 on sure rows, 3-bit (2b, Wed 07:27; a kept rank 1 at 4 bits) | PASS 0.59 % | PASS 4.65 % (information) | PASS 3.24 % | 52.0 / 87.7 / 83.6 (rank 1's LSB read on a miss); the cheap form 44.0 / 79.3 / 78.1 is untested |
| + the gated last row + 3-bit + 2-bit sure rows (lever 6, 07:27) | BETWEEN 1.49 % | KILL 10.13 % | KILL 9.77 % | off the path |
| **24 GB POOL (screen 4, 08:48):** QOM cf 0 + the gated last row + 3-bit (C) | **PASS 1.07 %** | BETWEEN 8.07 % | KILL 5.99 % | 35.7 / 19.4 / 13.9 |
| 24 GB pool: top-1, cheap form (t3keep, every kept expert on a sure row at 3 bits; B') | BETWEEN 1.69 % | KILL 11.25 % | KILL 8.61 % | 24.0 / 11.5 / 7.7: off the path |

**Screen 4 answered the caveat below for the code stack (08:48): PASS at the 24 GB pool (1.07 %), so code's 35.7 MB per
kept token at 24 GB stands, with the replay calibrated against W1B 24 GB (screen 3). On chat the same stack is BETWEEN /
KILL at 24 GB; the pool costs +0.30 / +1.98 / +0.70 points against the 45 GB pool.**

**Wed 07:4x caveat: every quality cell above is at the 45 GB pool's residents** (the two 24 GB POOL rows excepted). The QOM masks keep the experts resident
in laysz45_256's base pools (C_m mean 295 of 512 per layer); the MB column replays 147 slots per layer. At C 164 the
smaller pool cost +0.23 / +0.79 / +1.02 points (cf 0.5, 4 bits; QUEUE 03:58), and top-1 leans on residency more than
top-2 (2b passes because a resident rank 1 stays at 4 bits). Screen 4 (QUEUE 07:4x) re-tests the code PASS stack and the
top-1 stack at a 147-slot pool. Until then, "map-PASS at 24 GB" reads "map-PASS at the 45 GB pool, priced at 24 GB".

**Wed 07:5x: 2b tested the dear form of top-1** (scripts/ledger_top1.py; QUEUE 07:5x). The map served a kept rank 1 at 4
bits, so an engine that matches it reads rank 1's LSB part on a miss. With the gated last row at 24 GB that is 32.4 /
21.9 / 15.3, more than top-2 + 3-bit on chat (35.7 / 19.4 / 13.9). The cheap form (a kept rank 1 from its 3-bit base
record, no LSB on sure rows, the ledger's lever 2 as priced) is 24.0 / 11.5 / 7.7. Its quality is untested. Screen 4's
B' tests it at the 24 GB pool with the harness's t3keep gate: every kept expert on a sure row at 3 bits, a conservative
form.

Reading:
- Code has a map-PASS FAST stack at 35.7 MB per kept token at 24 GB, x0.17 of EXACT.
- Chat's map-PASS stacks sit at 84-100 MB.
- The BETWEEN stacks reach 13.9-47.3 MB on chat; EvalSuite's same-scores test decides those.
- The remaining distance to 10 at 24 GB: x0.28 on code, x0.5-0.7 on chat from the BETWEEN stacks. The candidates:
  - top-1 on sure rows (2b);
  - a frequency-aware pool;
  - 11-bit scales (x0.82).

## 1. The bug and the fix (the audit of BYTES10.md)

```
bytes10_replay.py:135   keep = np.zeros(E.shape, bool); keep[sure, :ks] = True; keep[~sure, :ku] = True      # E = ids[rows]: [r, 48, 10]
fixed                   keep = np.zeros(E.shape, bool); keep[sure, :, :ks] = True; keep[~sure, :, :ku] = True
```

Check (scripts/bytes10_bug_check.py): one code block of 16 rows.
- Fable C's EXACT Ub had experts on layers 0-9 only (120 on layer 0, 79 on layer 9, 0 on layers 10 and 47).
- The true unions on layers 10 / 47 are 80 / 89.
- Its "Q1 0 t3" Ub: layers 0-9 only as well. The bonus row covers layers 0-9; the sure rows cover layers 0-1 at all
  10 ranks.
- The LSB mask (`lsb[:, :, :2]`, `lsb[sure] = False`) indexes ranks correctly, so its sidecar numbers match mine
  exactly (code 45 GB EXACT: 60,266 sidecar misses in both).

Everything downstream of the MSB part moves:
- every level;
- the OPT / PIN / RCP verdicts;
- the compulsory floors (its 14.7 / 43.8 on code are 40.3 / 129.5);
- the per-lever ratios.

Things that did not use the keep mask are right:
- the working-set counts in section D (prompt 253, answer 292, new 90 on code);
- the bonus row's share of pairs (34 / 65 / 71 %);
- the union growth with rows.

## 2. My replay (scripts/ten_x_ledger.py; independent code, the same conventions)

**Calibration against the engine.**
- At 45 GB, code: my replay gives 55.4 for EXACT and the fixed Fable C replay 60.4. W1B measures 62.4 [K].
- W1B is at C 256 with a live kept of 5.9 per check (the map's live cut keeps 7.91) and counts prefill reads. The
  replay counts decode bytes only, with plain LRU.
- At C 160, BigMoE's W1B night measured 169.4 [K]. The 24 GB replay line (147 base slots) gives 212.8-245.2 on code.
  That is 1.25-1.45x above, consistent with the engine's LRU + 30 x popularity and its ~12 % larger base pool (164 vs 147 slots).
- Screen 3 (13:45, a CPU read-out of the W1B 24 GB arm) closes this at the 24 GB line.

**Part A, kept per check (EXACT, MB per kept token; data/ledger/ledger_A.txt).** "k" = a perfect drafter keeping k rows.

| line | domain | today (live) | k 1 | k 4 | k 8 | k 16 | k 14 + 2 wasted |
|---|---|---|---|---|---|---|---|
| 45 GB | code | 55.4 | 48.3 | 47.9 | 47.4 | 46.6 | 52.1 |
| 24 GB | code | 212.8 | 175.4 | 171.6 | 167.4 | 160.1 | 200.7 |
| 24 GB | IFEval | 200.4 | 125.0 | 121.9 | 118.4 | 112.5 | 140.3 |
| 24 GB | multi | 175.8 | 110.2 | 107.9 | 105.3 | 100.9 | 122.5 |
| 12 GB | code | 461.3 | 440.4 | 413.6 | 384.6 | 349.5 | 433.3 |

- The drafter acts on EXACT bytes through wasted rows, not through kept: k 1 -> k 16 is x0.91 at 24 GB.
- Today's live waste costs +27 % on code and +61-62 % on chat at 24 GB, against a perfect drafter at the same kept.
  On chat the live checks verify 3.68 / 2.96 rows for 2.32 / 1.86 kept.
- So D98's drafter quality buys bytes on chat (x0.6), not on code (x0.79). [P]

**Part B, per-request locality (ledger_B.txt).** Answers of 280 / 283 / 296 tokens.

| | code | IFEval | multi |
|---|---|---|---|
| distinct experts per layer in the answer | 292 | 207 | 219 |
| experts per layer covering 90 % of the router mass | 146 | 95 | 92 |
| experts per layer covering 95 % | 188 | 123 | 124 |
| of the answer's mass, on experts the prompt's prefill already routed | 86 % | 66 % | 63 % |

Compulsory floor without prefill reuse (every distinct expert read once): 123 / 87 / 87 MB per answer token at
~280 tokens. A power-law fit (distinct ~ t^0.5) gives ~55-59 at 1K tokens and ~15 at 4K [P, extrapolated]. With
prefill reuse, the fixed replay's per-request floor is 40.3 / 47.3 / 60.7.

**Part C, FAST per pool line (ledger_C.txt; the map's convention: the bonus row exact).**

| line | domain | EXACT | QOM cf 0.5 | QOM cf 0.5 + 3b | QOM cf 0 | QOM cf 0 + 3b |
|---|---|---|---|---|---|---|
| 45 GB | code | 55.4 | 24.0 (-57 %) | 13.4 (-76 %) | 23.0 (-59 %) | 12.0 (-78 %) |
| 45 GB | IFEval | 37.7 | 27.4 (-27 %) | 22.0 (-42 %) | 25.5 (-32 %) | 19.2 (-49 %) |
| 45 GB | multi | 33.0 | 26.0 (-21 %) | 22.7 (-31 %) | 24.2 (-27 %) | 20.2 (-39 %) |
| 24 GB | code | 212.8 | 79.1 (-63 %) | 59.4 (-72 %) | 74.8 (-65 %) | 54.5 (-74 %) |
| 24 GB | IFEval | 200.4 | 116.8 (-42 %) | 101.5 (-49 %) | 102.7 (-49 %) | 84.4 (-58 %) |
| 24 GB | multi | 175.8 | 109.2 (-38 %) | 99.5 (-43 %) | 93.7 (-47 %) | 81.7 (-54 %) |
| 12 GB | code | 461.3 | 158.4 (-66 %) | 134.3 (-71 %) | 148.5 (-68 %) | 123.6 (-73 %) |
| 12 GB | IFEval | 546.4 | 308.1 (-44 %) | 287.9 (-47 %) | 262.6 (-52 %) | 238.7 (-56 %) |
| 12 GB | multi | 444.6 | 275.2 (-38 %) | 262.7 (-41 %) | 233.1 (-48 %) | 217.5 (-51 %) |

QOM's cut does not shrink at smaller pools. Fable C reported x0.18 at 45 GB vs x0.34 at 24 GB, but that was the bug.
In my replay QOM cf 0 + 3b is:
- code: -78 / -74 / -73 %;
- chat: -39..-49 % at 45 GB growing to -51..-58 % at 24 / 12 GB.
A QOM ratio still belongs to its own pool line: never multiply one line's ratio onto another.

## 3. The two replays agree once they count misses the same way

My replay counts a block's misses against the pool as it stood before the block, then touches the union. Fable C's
touches the union in expert-ID order and counts in the loop. When a block's union is larger than the free pool,
in-loop counting evicts residents of the same union before they are used.

| | 45 GB | 24 GB | 12 GB |
|---|---|---|---|
| in-loop counting / my "before" counting | x1.09 | x1.15 | x1.38 |

- With my loop changed to Fable C's warm-up and counting, my replay reproduces the fixed replay exactly: 60.4 at 45 GB,
  sidecar misses 60,266 in both.
- The "before" count is what an engine can reach: use the resident experts first, then stream the misses. The in-loop
  count is the pessimistic order.
- At 12 GB a check's union (75 experts per layer for 16 rows) exceeds the 51 slots, so the 12 GB numbers depend on the
  check design. Read them as a range: code EXACT 461-637.

## 4. The last-row reconciliation (decision 2's price): scripts/ledger_lastrow.py

Same replay as part C (plain LRU, my miss count; data/ledger/ledger_lastrow.txt). Four conventions for the last row
of each live segment:

| convention | what it does | status |
|---|---|---|
| **map** | the last row stays exact (cf -1), both when it is a realized bonus row and when it is the discarded last row of a segment that ended in a miss | what the map's QOM arms tested |
| **engine** | the last row is gated by the cf of the draft it verifies: the walk's next draft after the cut. It stays exact only when no draft is left. | the convention of my capture replay |
| **disc** | the engine's gate on discarded last rows only | a decomposition, not an option |
| **all** | every last row gated | Fable C's bound |

- **engine:** the engine already has that draft for 292 / 1,448 / 1,907 of 428 / 1,467 / 1,921 segments (code / IFEval
  / multi). DFlash2 drafts all 15 tokens in one pass and the live cut drops only inputs.
- **disc:** a discarded row cannot change a kept token (causal attention), so the map's PASS already covers it. But the
  engine cannot tell those rows apart before the forward pass.
- **all:** it needs a 16th draft on segments that used every draft.

At 45 GB, % vs EXACT:

| rule | map (as tested) | disc | engine | all | my capture price (decision 2) |
|---|---|---|---|---|---|
| QOM cf 0.5 | -57 / -27 / -21 % | -58 / -32 / -24 % | -59 / -32 / -25 % | -66 / -45 / -41 % | -60.4 / -36.9 / -33.4 % |
| QOM cf 0 | -59 / -32 / -27 % | -62 / -46 / -42 % | -64 / -58 / -61 % | -68 / -58 / -61 % | -64.3 / -58.5 / -62.4 % |
| QOM cf 0 + 3b | -78 / -49 / -39 % | -83 / -70 / -61 % | -86 / -88 / -87 % | -92 / -88 / -87 % | 6.5 / 4.6 / 4.6 MB = -86 / -85 / -85 % |

At 24 GB, QOM cf 0 + 3b in MB per kept token:

| map | disc | engine | all |
|---|---|---|---|
| 54.5 / 84.4 / 81.7 | 42.0 / 41.4 / 40.3 | 35.7 / 19.4 / 13.9 | 25.2 / 19.3 / 13.9 |

Reading:
- **The engine convention reproduces my capture price.** At cf 0 it is within 1-4 points. At cf 0.5 it is within 2-9
  points, where the capture's own cf vectors differ. So decision 2's numbers were QOM plus the gated last row.
- **The map tested QOM with the last row exact.** On chat, the last row is most of QOM's saving: at 45 GB cf 0, IFEval
  is -32 % as tested and -58 % with the row gated.
- **Screen 1 (a) is the test of the engine convention at cf 0.** `qom_0_live_bonus` gates every realized bonus row;
  discarded rows cannot change flips.
  - Pass: the engine numbers stand: decision 2's price, and the ledger's 24 GB FAST start of 35.7 / 19.4 / 13.9 instead
    of 54.5 / 84.4 / 81.7.
  - Fail: the map numbers stand.
- **Building it needs no extra draft on 68 % / 99 % / 99 % of segments.** Fable C's 16th draft only adds the segments
  that used every draft (mostly code).

**Screen 1 (a) result (Wed 04:41, QUEUE 04:42): PASS on code, BETWEEN on IFEval, KILL on multi.**
- Flips: 0.65 / 6.12 / 6.50 % against the ungated 0.42 / 2.53 / 1.77 %.
- The gate's cost: +0.24 points on code (z +1.79), +3.59 on IFEval (z +8.76), +4.73 on multi (z +11.38).
- What it tested: the gated row kept ranks 0-1 plus the block's union members only. QOM's resident keep, which costs no
  SSD reads, was not on that row.
- So for now:
  - code takes the engine convention's bytes: 24 GB FAST start 35.7;
  - chat keeps the map convention's (84.4 / 81.7).
- qom_0_live_bq (QUEUE 04:4x, pre-registered) tests the gate in QOM's own form: the same reads, with the resident keep
  on that row.

**qom_0_live_bq result (Wed 05:06): PASS on code, BETWEEN on IFEval and multi.** Flips 0.39 / 4.09 / 3.26 %.
- The QOM-form gate costs nothing on code (z -0.28).
- On chat it costs +1.56 / +1.49 points: over screen 1's +0.3 condition, but under Q1 at 0.99's level on all three.
- The resident keep removed 57-68 % of the union-only gate's chat cost (z -4.62 / -7.67 vs qom_0_live_bonus).
- So the engine convention's prices now have a map result: decision 2's -64.3 / -58.5 / -62.4 % at 45 GB, and 35.7 /
  19.4 / 13.9 MB per kept token at 24 GB with 3-bit sure rows.
  - Code: PASS.
  - Chat: BETWEEN by the screen's line, inside the union arms' line.
- qom_0_live_bq_t3, the 3-bit form, is running; EvalSuite's same-scores test decides adoption.

## 5. The ranked path to <= 10 at 24 GB, re-priced

**Wed 08:59 re-rank at 24 GB, after screens 3-4 (QUEUE 08:3x-08:59).** The table below is the 03:4x pricing; read these
first:
- Lever 1 (the gated bonus row, QOM form + 3-bit) is in the map-PASS code stack at the 24 GB pool: 35.7 / 19.4 / 13.9
  (chat BETWEEN / KILL there).
- Lever 2 (top-1 on sure rows) is off the 24 GB path: its cheap form is BETWEEN / KILL / KILL at the 24 GB pool, and
  the form that passed at the 45 GB pool (2b) reads more than top-2 on chat.
- Lever 3 (the pool policy) re-priced on the C stack: SLRU x0.94 / x0.91 / x0.96 of LRU's load misses; decayed LFU is
  worse than LRU (x1.15-2.9). No policy recovers the keep lost to the smaller pool (QUALITY POOL: KILL).
- Lever 4 (11-bit), CORRECTED 13:4x: x0.898 / x0.888 / x0.894 on the C stack at 24 GB (154 / 49 slots at equal RAM with
  the packed parts; 12 GB x0.92-0.93). The 08:59 x0.633 / x0.595 / x0.615 double-counted capacity (x0.824 per byte, which
  already held the extra parts, plus 1 / 0.824 more slots): WITHDRAWN.
- Code at 24 GB, map-PASS stack, priced (CORRECTED 13:4x): 35.7 -> 32.1 with 11-bit [P]; chat 17.3 / 12.4. The 08:59
  '22.6 / 11.6 / 8.5' is withdrawn.
- **Wed 13:28, POOLSPLIT (new lever 5, QUEUE 13:25 / 13:28; 11-bit rows CORRECTED 13:4x):** under the C stack only
  unsure rows read LSB parts (chat has none left: 0.0 MB per kept with no LSB pool), so the LSB pool's RAM goes to base
  slots at equal GB. 24 GB 163 / 0: x0.917 / x0.849 / x0.861; with 11-bit 171 / 0: **code 29.3, IFEval 14.5, multi 10.6**
  [P] (x0.914 / x0.839 / x0.852 of 11-bit alone); keep +21-25 % of the gap to the 45 GB pool. 12 GB 11-bit 65 / 0: 70.1 /
  54.6 / 39.4. Chat at 24 GB stays above 10 MB; map screen 5 (198 slots) was withdrawn with the 11-bit error. Under 10 at
  24 GB needs ~x0.34 more on code (~27.5 with SLRU [P]) and ~x0.7 / ~x0.95 on chat, plus chat's quality.
- **Wed 13:44, TRELLIS FAST STORE (lever 6, QUEUE 13:4x / 13:44): the only path found to <= 10 on code at 24 GB.**
  Spend the 24 GB pool on QTIP-style trellis records for the sure rows (B10's codes, killed Sun on an EXACT bar).
  - TRL-2 (276 slots): code 5.8, IFEval 3.0, multi 2.7 [P]. The 2.40 b mix (239 slots): 10.1 / 5.0 / 4.1.
  - Code's unsure rows add +39 MB if read uncached from v3, so code needs a 17th draft or trellis-served unsure rows.
  - Quality untested [G]: code PASS-ish at every form; chat BETWEEN / KILL except TRL-3 (185 slots, IFEval 11.4 /
    multi 8.5).
  - Costs: encoder, disk, decode kernel.
- **Wed 14:04, DRAFT17 (lever 7, QUEUE 13:5x / 14:04): a drafted bonus row gates code's last rows that have no next
  draft today, and those were a third of code's bytes.**
  - Code at 24 GB: today's stack 35.7 -> 25.2 (x0.706); with 11-bit + POOLSPLIT 29.3 -> 19.5 (x0.665). 12 GB x0.742.
  - With a trellis store: TRL-2 code 4.0, the 2.40 b mix 7.2 MB per kept token [P].
  - Cost ~0 ms after Leap's one-line _rows16 fix [K MTP Head 14:03]; quality = map screen 6 (masks READY).
  - Without a trellis store, code at 24 GB still needs ~x0.51 for 10.

Base: QOM cf 0 + 3-bit sure rows: 61.7 / 86.1 / 83.1. The cf 0 quality is a map PASS. The 3-bit quality is a map PASS
at cf 0 on code and IFEval; multi needs cf 0.5 for its 3-bit rows (cf 0 is BETWEEN there).

| # | lever | x at 24 GB (code / IFEval / multi) | halved | quality status | owner |
|---|---|---|---|---|---|
| 1 | gated bonus row | x0.50 / x0.23 / x0.17 | x0.75 / x0.62 / x0.59 | screen 1 (a) at cf 0, 13:45 | Breakthrough (map); MTP Head (does DFlash2 give the 16th draft a cf?) |
| 2 | top-1 on sure rows | x0.55 / x0.59 / x0.56 | x0.77 / x0.80 / x0.78 | screen 2, 13:45 (P1-OEA passed on code at x0.465 pass-1 reads [K]) | Breakthrough |
| 3 | frequency-aware pool (rank-class pools / SLRU-like; OPT shows the room) | x0.52 / x0.74 / x0.87 (OPT room x0.29 / x0.37 / x0.51 on stack 1) | x0.76 / x0.87 / x0.94 | exact for any FAST rule (policy only); replay | BigMoE (pool policy); screen 3 first |
| 4 | 11-bit scales + native layout (M-27b) | x0.824 at 03:4x = MoonShots' 45 GB NET figure, not per byte; on the C stack at 24 GB x0.898 / x0.888 / x0.894 (CORRECTED 13:4x) | - | bitwise | MoonShots / Leap / BigMoE (Thu conversion) |
| 5 | fewer wasted rows (D98) | EXACT x0.79 on code, x0.6 on chat | - | MTP Head's program | MTP Head |
| 6 | 2-bit sure-row records (MoonShots M-34: a 2-bit read is a 1.54 MB prefix of the 3-bit part) | alone x0.92 / x0.98 / x0.985 (KILL by its line); after lever 1, engine convention (M-34 rerun 05:5x): 24 GB code 35.7 -> 29.7 (x0.833 fixed, x0.777 byte-budget), IFEval 19.4 -> 13.9 (x0.715 / x0.639), multi 13.9 -> 9.9 (x0.712 / x0.627); 12 GB code x0.813 / x0.741, chat x0.71 / x0.54-0.55 [P] | - | quality screen qom_0_live_bq_t2 pre-registered (QUEUE 05:5x); MoonShots' helper to2bit.py (self-tested) handed to Breakthrough as a one-line dm_t3 install; 13:45+ | MoonShots (formula) / Breakthrough (harness) |

Products at 24 GB:
- unhalved, levers 1-4: 7.3 / 7.2 / 5.7;
- halved, levers 1-4: ~25 / 33 / 32.

The decision-grade reading: levers 1-2 decide it. If screens 1 and 2 pass, 24 GB is 7.9-16.8 before any cache
policy, and a policy lever plus 11-bit puts it under 10.

## 6. Screens for Wed 13:45 (Fable C's section 5, unchanged lines), with two notes

- **Screen 1, the gated bonus row.** Arm (a) `qom_0_live_bonus` gates every bonus row at cf 0, so the proxy does not
  matter there; its masks index correctly.
  - Note for (b) / (c): Fable C's proxy cf for the bonus row is the next segment's first draft. In the walk's split
    that is the draft after the one the bonus row emits: dd[L + 1], where the row's own draft is dd[L].
  - The gate file's g[p] for the bonus row p is the drafter's cf for the token that row emits. It is what the engine
    would use, and the cf my capture replay used.
  - Suggest building (b) / (c) with g[p], or reading them as approximations.
- **Screen 2, top-1 on sure rows.** Unchanged.
- **Screen 3, calibrating the engine against the replay.** Unchanged, with the corrected reference:
  - code 24 GB EXACT LRU 245.2 (in-loop count) / 212.8 (before count);
  - scale by kept as Fable C wrote (x 7.91 / kept_live).

## 7. Double counting: what must be priced jointly

- QOM / Q1 / t3 / COLD / u8 / the bonus gate / top-1 act on the same rows. Price them only as joint replays: every
  number in sections 0 and 5 is a joint replay, not a product of separate factors, except 11-bit, which multiplies.
- The pool policy interacts with the rule: rank-class pools are x1.04 on EXACT at 24 GB on code (worse) and x0.52 on
  the top-1 stack. Never carry a policy factor from one rule to another.
- A QOM ratio belongs to one pool line (section 2, part C).
- The drafter's waste (part A) overlaps with the bonus gate: both act on how many unsure / exact rows a check holds.
- Exposure levers (JOINT, keepalive, zero-copy, pickup, forecasts) move 0 bytes. They belong to BROKEN line 1.

## 8. Novelty (TRIED/NOVELTY.md tags)

| item | tag | basis |
|---|---|---|
| the gated bonus row | new (Fable C's web check) | the nearest art weights positions by reach (PSR s7, AcceptMoE), never gating the last position by a draft of its token |
| the cf the engine already has | (part of the same mechanism) | the gate file's cf for the token the row emits, which the engine computes anyway; for cut checks no extra draft is needed |
| rank-class pools on the FAST stack | adapted | SLRU / 2Q segmented caching, SMoE score eviction 2508.18983 |
| the ledger method | engineering | joint replays per pool line, an OPT bound per rule, and calibration against the engine's [K] rows |

The second row matters for the claim: the drafter drafts all 15 tokens in one pass, and the live cut drops only the
inputs.

## 9. Caveats

- One capture: 36 windows of ~280-token answers. Code is one task family. The per-request compulsory share falls with
  longer answers (part B).
- Plain LRU throughout, unless OPT / PIN / RCP is named. The engine's LRU + popularity reads fewer misses (screen 3).
- No timing here; bytes only. At 10 MB per kept token the SSD (~12.7 GB/s) leaves ~1,270 tok/s of headroom.
- Calibration at 24 GB (screen 3, Wed 08:3x; QUEUE): W1B 24 GB at the replay's own 147 / 46 slots misses 226.0 / 247.4
  MB per kept token on two code prompts against this replay's 212.8 (x1.06 / x1.16; 0.92-1.16 on all six prompts). The
  replay counts demand misses only: the engine also reads 18-26 % in unused prefetches, and prefill comes on top.
- The 12 GB line depends on the check design (section 3).
- Who wrote what:
  - Fable C's replay is Fable C's code; I changed one line. The corrected tables are in
    data/ledger/bytes10_replay_fix.out.
  - My replay was written before I read Fable C's code (Tue 23:5x, beb18d3) and ran at 03:28.
