# AUDIT 1105: the overnight answers, checked before anything is built (Fable C, Thu 5 Nov 2026, 07:0x-08:xx EET)

> (a quote from the project owner, kept private)
Inputs: `~/FableGod/cloud3/live/cloud3/REPORT_P0..P3.md` + `REQUESTS.md` (Cloud Fable 3), `~/FableGod/far/FAR.md`,
`~/FableGod/zero/ZERO.md`, `~/FableGod/goal/GOAL.md`, `~/FableGod/full/FULL.md` (+ Round 2). Measured sources for the
re-derivations: `~/FableGod/KERNEL90.md` (lines 97-114), `~/FableGod/DAY9_2026-11-02.md` (line 93, the C0 timeline; line 875,
BigMoE 5b), `~/FableGod/BUILD_DAY_1104.md`, `~/FableGod/far/FAR.md` section 7b (the SSD wait law), `~/TRIED/TRIED.md` (lines
36-37, 40, 48, 69, 101, 108, 121, 349, 364, 425, 446, 724, 848, 886, 941, 1226, 1256), `~/TRIED/NOVELTY.md`, `~/NewIdeas/QUEUE.md`
line 588, `~/Breakthrough/SCOREBOARD.md` lines 2404-2419 (RESULTS FLOOR NIGHT 1), `~/Breakthrough/data/floor1/` (the measured
arms oea_k8, gate_bands, gate_top2_cf0999, gate_top5_cf099, hyb_0.9_u8_live_t3 and the batch's baseline),
`~/Breakthrough/keepset/qom_keep.py` (the engine's QOM rule), `~/TrainLab/notes/DRAFTER_PLAN_1104.md`,
`~/MoonShots/research/FAR_NOVELTY_CHECK.md`.
Two CPU checks of my own (numpy, seconds, no GPU, data on disk; scripts and outputs beside this file):
`audit_k8_bands.py` -> `audit_k8_bands.out` (the measured arms' flips by the draft's cf band, 36 windows) and
`audit_split_replay.py` -> `audit_split_replay.out` / `.json` (bytes AND tiles per kept token for six rules at the 45 / 24 GB
pools, BYTES10's conventions). Marks: [K] measured (source named), [P] arithmetic or replay on [K], [G] judgement. Gains are
HALVED where the table says so (Kamen's rule). Nothing here is a verdict; verdicts come from the tests in section 2.

Lines: code FAST at 45 GB = tok/s (today 110-117 [K DAY9 875, brief 3.4]); code EXACT at 45 GB = tok/s (83); 24 GB = SSD MB
per kept token (code 19.5, chat ~88 [K brief 3.4]). Map lines: all-row flips <= 1.34 / 5.42 / 3.69 % (code / IFEval / multi) =
Q1 at 0.99's level [K SCOREBOARD 2363].

## 0. The four findings that change the ranking (read these first)

**F1. "Same bytes" is true against the wrong baseline [K cloud3's replay; P my replay].** The strict band table reads the
same bytes as its resident-keep TWIN (REPORT_P3 section 20: within 0.5 MB), but the twin is not what runs. Today's 45 GB FAST
preset is QOM at cf 0 + the gated bonus row (`qom_keep.py`: THR 0.0, KS 2, "a sure row keeps ranks 0-1 and each rank 2-9
expert that is in the block's loads or resident"; FULL.md 2.1; DAY9 875). Against that preset:

| rule | cloud3 (P3 sections 8 / 20): 45 GB MB / tiles | my replay (`audit_split_replay.out`): 45 GB MB / tiles, bytes vs QOM | 24 GB MB vs QOM (mine) |
|---|---|---|---|
| EXACT | 42.8 / 27.0 / 25.3 ; 315 / 530 / 532 | 44.2 / 26.6 / 24.7 ; 320 / 538 / 534 | |
| QOM cf 0 + BQ (today) | 6.9 / 4.5 / 4.5 ; 290 / 482 / 474 | 4.8 / 4.6 / 4.6 ; 291 / 493 / 479 (x0.91 / 0.92 / 0.90 of EXACT's tiles) | x1.00 |
| strict c3_b2d | 14.7 / 16.8 / 17.6 ; 155 / 382 / 403 | 12.0 / 14.7 / 15.0 ; 141 / 353 / 364 ; **bytes x2.5 / 3.2 / 3.3** | **x2.4 / 4.4 / 5.5** |
| c3_b2d's resident-keep twin | 14.2 / 16.5 / 17.2 ; 298 / 520 / 524 | 11.6 / 14.4 / 14.7 ; 300 / 526 / 522 | x2.4 / 4.3 / 5.4 |

Both replays agree (the tile ratios within 0.03; cloud3's QOM bytes are higher than mine by its bonus-row convention). So
c3_b2d reads 2.5-3.3x the bytes of today's preset at 45 GB and 2.4-5.5x at 24 GB [P]; "x0.41 / 0.52 / 0.58 of today's tiles"
are ratios to EXACT, and against today's QOM tiles they are x0.44 / 0.57 / 0.65 [P]. The extra bytes are the ranks 2-4 of the
0.9-0.99 rows and 2-7 of the 0.3-0.9 rows that QOM never loads. Priced with FAR's wait law (7b: ~12 ms floor + 0.155 ms per
miss, bandwidth-bound at the margin; these are demand reads, not QOM's optional ones): code +7-8 MB x 8.5 kept = +60-66 MB =
+28-31 misses = **+4.5-4.8 ms per check** (+5-9 ms at the by-day 0.15-0.29 ms per miss); chat +1.8-2.1 ms [P]. Cloud3's cost
model (section 13: t(r) = 90 - (1 - r) E - (1 - r) 12) CREDITS an SSD saving of (1 - r) x 12 ms: the sign is wrong.

**F2. The expert term being cut is 22-27 ms, not 38-43 [K KERNEL90 113-114; DAY9 93; TRIED 941].** The brief's "experts
~38-43 ms = 3,700-3,840 tiles" is the 16-row EXACT figure (KERNEL90: ~43 ms on NAX6 per 16-row check at C2, 7.30 GB; NAX6-NH
30.9-33.2). The live default verifies 9.7 rows on code with ~54 distinct experts per layer instead of 73-79 (cloud3 P2
section 2: x0.69-0.74 of the groups), and QOM keeps ~91 % of those tiles (my replay; cloud3 92 %): 43 x 0.72 x 0.91 = ~28 ms on
NAX6, ~22 ms on NAX6-NH; DAY9's per-pair C0 timeline (2.9-3.2 ms per row x ~10 live rows x QOM's kept pairs) and FULL.md's
table (experts 22-23 ms) land in the same 22-27 ms. Cloud3 saw the ambiguity (section 13: "24-43 ms") and kept both ends; the
high end is not the live check. Re-priced on code at 45 GB FAST [P]: c3_b2d_reach07 tiles ~x0.40 of today's -> saves 13-16 ms
of 22-27; minus F1's +5-9 ms -> **net -4 to -11 ms per check = +4 to +13 % tok/s (110-117 -> 115-132), halved +2 to +6 %**, not
"117 -> 140-165"; c3_b2d alone (x0.48): saves 11-14, net -2 to -9 ms = +2 to +11 %, halved +1 to +5 %. C3-R8's log (the group
count per call) is what settles the bracket.

**F3. The quality pricing is contradicted by four measured arms of its own family [K floor1; P `audit_k8_bands.out`].**
Cloud3 prices a table's flips as the sum over cf bands of (band share) x (the band's flip rate under the uniform topk2 /
topk5 arms), with k = 8 rows at 0.00 % ("OEA k8 is a map PASS"), times a context factor (x1.6 code / x2.5 IFEval / x1.3 multi)
calibrated on q1u_0.9_live. Wednesday's FLOOR night (Breakthrough, 03:10-07:06, 78 windows, live-cut; SCOREBOARD 2404-2419,
`data/floor1`) measured strict band tables directly, and I joined the arms' per-row argmax with the batch's baseline and the
draft cf bands on the 36 map windows (the join reproduces oea_k8's 0.62 / 3.83 / 3.07 % exactly):

| measured arm (strict, + union members, live unless said) | all-row flips code / IFEval / multi [K] | flips on the cf < 0.5 rows / 0.5-0.9 rows [P join] | cloud3's prediction for this table |
|---|---|---|---|
| gate_bands = NEW-1 (>= 0.999: 2, 0.99-0.999: 4, 0.9-0.99: 8, < 0.9 and no draft EXACT) | 1.28 / 5.15 / 3.60 on the 36 (1.12 / 5.08 / 3.64 on the 78; verdict BETWEEN, dm_floor_read) | code 9.2 / 5.1 %; IFEval 9.4 / 6.3; multi 4.9 / 4.6 (these rows are EXACT in this arm) | "NEW-1 live: 0.00 / 0.03 / 0.06 % direct", tiles x0.615 |
| gate_top2_cf0999 (only the >= 0.999 band at top-2) | 1.25 / 3.97 / 2.50 (0.0 % inside the band) | code 9.2 / 4.9; IFEval 7.2 / 4.8; multi 3.3 / 3.3 | 0.00 % direct |
| gate_top5_cf099 (only the >= 0.99 rows at top-5) | 0.71 / 3.86 / 2.11 | code 5.9 / 3.1; IFEval 8.6 / 3.6; multi 3.1 / 2.4 | ~0.0 % direct |
| oea_k8 (top-8 + members on EVERY row, 16-row unions) | 0.62 / 3.83 / 3.07 | code 6.6 / 1.8; **IFEval 7.1 / 4.8; multi 4.8 / 3.3** (0.9-0.99 band 0.6 / 1.4 / 1.9; >= 0.99 0.0) | k 8 = 0.00 % direct |
| hyb_0.9_u8_live_t3 (>= 0.9 rows top-2 at 3 bits, < 0.9 rows top-8, live) | 1.37 / 7.74 / 5.54 | code 7.9 / 5.5; **IFEval 13.2 / 8.8; multi 7.8 / 6.6** | (not priced) |

Two things follow. (a) The cost of cutting the sure rows lands on the UNSURE rows downstream (the near-ties): cutting only the
>= 0.999 band to top-2, with 0.0 % flips inside that band, costs 1.25 / 3.97 / 2.50 % all-row, and in the NEW-1 arm the
EXACT < 0.5 rows flip 9.2 % on code. A "direct flips of the cut rows" bound cannot see this, and the context effect is an
additive floor of ~1 / 4 / 2.5 points for any live strict cut of the sure rows, not a multiplier on the direct sum. (b) Top-8
on the unsure bands is not free: 5-7 % of those rows on IFEval and 3-5 % on multi at 16-row unions, 9-13 % / 7-8 % at the
live cut. c3_b2d is strictly more aggressive than NEW-1 in every band (top-2 from 0.99 instead of 0.999, top-5 instead of
top-8 on 0.9-0.99, top-8 instead of exact below 0.9), so its all-row flips sit above NEW-1's 1.12-1.28 / 5.1 / 3.6 and near
hyb_0.9_u8_live's 1.37 / 7.74 / 5.54. Corrected expectation [P/G]: **c3_b2d code 1.3-1.5 % (at the 1.34 line: BETWEEN or a
thin PASS), IFEval 6-8 % (BETWEEN / KILL), multi 4.5-6 % (BETWEEN / KILL)**; c3_b2c (top-2 only from 0.999) code 1.2-1.4,
chat the same story; c3_b2d_reach07 adds +0.03-0.07 / +0.35-0.85 / +0.54-1.3 on top. The predictions "PASS x3" and
"PASS / PASS / BETWEEN-PASS" are unsupported; the lever is at best code-only, like Q1-UNION 0.9 and REACH-CUT (TRIED 108, 941).
Cloud3 could have seen this: NEW-1's measured BETWEEN is in SCOREBOARD's RESULTS FLOOR NIGHT 1 and TRIED line 40 (M-18 BAND45
"Q2 = NEW-1's confidence bands"), and oea_k8's numbers are in QUEUE.md line 588.

**F4. The tiles can be had at today's bytes: the split form [P `audit_split_replay.out`].** The strict table's bytes come
from loading the unsure rows' tails; its tiles come from NOT computing the sure rows' resident tails. The two are separable:
keep QOM's loads (every drafted row's ranks 0-1) and QOM's resident keep on the unsure rows, and compute the drafter-sure
rows strictly (ranks 0-1 + union members, no residents). Replayed at the 45 GB pools (tiles vs today's QOM; bytes vs today's):

| rule | code tiles (vs QOM) / bytes | IFEval | multi | 24 GB code tiles / bytes |
|---|---|---|---|---|
| strict c3_b2d | 141 (x0.48) / **x2.5** | 353 (x0.72) / x3.2 | 364 (x0.76) / x3.3 | x0.63 / x2.4 |
| SPLIT99 (strict on cf >= 0.99 rows) | 175 (x0.60) / x1.01 | 434 (x0.88) / x1.00 | 437 (x0.91) / x1.00 | x0.65 / x1.00 |
| **SPLIT90 (strict on cf >= 0.9 rows)** | **137 (x0.47) / x1.01** | 367 (x0.74) / x1.00 | 380 (x0.79) / x1.00 | x0.53 / x1.00 |

SPLIT90 matches c3_b2d's tiles on code (137 vs 141) at 0.4x its bytes, i.e. at today's bytes; its quality is NOT a new
unknown: it is q1u_0.9_live's measured cut on the >= 0.9 rows (1.04 % on the 78 / 1.25 % on the 36 on code = PASS; 6.4-6.5 /
3.8-4.1 chat = BETWEEN [K TRIED 941, SCOREBOARD 2417]) plus QOM cf 0's measured gating of the < 0.9 rows (qom_0_live_bq 0.39 /
4.09 / 3.26 [K TRIED 101]); SPLIT99 is Q1-UNION at 0.99's cut (0.74 % at 16 rows [K SCOREBOARD 2347]; live ~0.9-1.1 [G]) plus
QOM's. Engine form: `qom_keep.py`'s keep with the resident clause disabled on rows whose cf >= the split threshold (one
condition); mask form: dm_oea_masks' qom arm with a strict band (one line). Priced on code at 45 GB FAST [P]: SPLIT90 saves
53 % of 22-27 ms = 12-14 ms at no SSD cost -> 90-95 -> 76-83 ms = **+15-19 % raw, halved +7-9 %**; SPLIT99 saves 9-11 ms =
+11-14 % raw, halved +5-7 %. This is the form to test, and if it passes, the form to build; c3_b2d's extra bytes buy nothing
it does not already have.

Consequence for Friday night: run the split arms beside c3_b2d / c3_b2c (masks ready for the latter), code decisive, chat as
the control, with C3-R8's group-count log in the same window; do not build an engine form on "same bytes".

## 1. The ranked table (every idea; gain halved; tried / untried; the number errors found)

Tried-status key: T-K tried and killed; T-A tried and alive (or adopted); U untried. Prior art: the nearest published work I
know (MoonShots does the web check; section 4). Cost: the cheapest decisive test.

| # | idea (session) | gain on our lines, HALVED [P] | odds the test passes [G] | cost | tried? | number errors found |
|---|---|---|---|---|---|---|
| 1 | **The split form at 45 GB FAST** (this audit, from cloud3's tile anatomy): QOM's loads + strict compute on the drafter-sure band; SPLIT90 on code, SPLIT99 on chat | code tok/s +7-9 % (SPLIT90; raw +15-19 %) or +5-7 % (SPLIT99); chat +1-2 %; 24 GB bytes unchanged (tiles x0.53 there too, useful once the 24 GB check is not SSD-bound) | map PASS code: SPLIT90 ~45 % (q1u_0.9_live's 1.04-1.25 % + QOM's < 0.9 cost, at the 1.34 line), SPLIT99 ~60 %; chat: SPLIT99 ~50 %, SPLIT90 ~25 % | 5 min map arm each (one-line mask change) + C3-R8's free log + an engine A/B 10 min (one condition in qom_keep) | U as a form; every piece measured (q1u_0.9_live PASS code, Q1-UNION 0.99, QOM-BQ) | none (my numbers; the quality estimate is [G] from the two measured parents and must be read from the arm) |
| 2 | **Rule C comment relax** (FULL R2.1; measured by MoonShots M-41) | kept x1.04 HumanEval+ (x1.074 profile, halved) / x1.06 MBPP (x1.125); tok/s by the same factor at ANY engine, both tracks' FAST side; 24 GB code bytes x0.96 | engine PASS (median S <= 0.92, T <= 1.05, C <= 1.15) ~50 %; the kept gain is measured | 10 min GPU (BigMoE 12 HumanEval C 224, A-B-A-B) | T-A: M-41 BETWEEN HumanEval+ (x1.074 < its PASS 1.10), PASS MBPP; R1 killed by M-40 | FULL's "12.0 halved" multiplied in COPYGATE G2 (x1.041) and the outside-span R1 (x1.016), separate levers; the rule's own halved gain was x1.08, measured x1.074 profile (~x1.04 real by Mission 25's halving); its PASS line 11.6 was missed |
| 3 | **c3_b2d / c3_b2c, strict three-band table at 45 GB** (cloud3 P3 16-19) | code tok/s +1 to +5 % (raw +2 to +11 % net of F1's bytes); chat ~+0.5 %; 24 GB: negative (bytes x2.4-5.5) | map PASS code ~35 % (F3: 1.3-1.5 % expected), chat ~10 %; the engine cash-in needs the per-GROUP kernel (NAX6-NH, Thursday) and R8 | map arm 5-10 min (masks ready) | U as a table; its pieces T-A | F1, F2, F3; "of today's tiles" = of EXACT's (x0.44 / 0.57 / 0.65 of QOM's); 140-165 tok/s -> 115-130; NEW-1 priced at 0.00-0.06 % direct vs measured 1.12-1.28 / 5.1 / 3.6 |
| 4 | **c3_b2d_reach07** (cloud3; the arm named for Friday) | code +2 to +6 % (raw +4 to +13 %); chat ~+1 % | code ~30 %; chat ~5 % | as #3 (masks ready) | U as a stack; REACH-CUT T-A (PASS code only, TRIED 108) | as #3; the headline x0.52 / 0.58 on chat is the tau-0.7 form cloud3 itself says not to use on chat (tau 0.5: x0.60 / 0.65); the X rule's kept loss (x0.997 IFEval) unpriced |
| 5 | **Chat bytes lever at 24 / 12 GB** (cloud3 sections 21 / 25: the QOM 0.5 chat route + top-8 on its unsure rows, then + REACH-CUT 0.5, resident keep) | 24 GB chat SSD MB per kept token x0.91 (u8, raw x0.82) / x0.82 (both, raw x0.64); chat tok/s at 24 GB +2-4 % [P: ~113 misses per chat check x 0.155 ms]; code untouched | u8 PASS ~65 % (expect +0.3-0.6 points, not "+0 to +0.1": at 147 slots the cut drops the non-resident ranks 8-9 of 26-34 % of rows, ~1/5 of OEA k8's cut, whose < 0.5 band flips 7.1 / 4.8 %); + reach 0.5 ~40 % (multi 2.73 + 0.54 + 0.4 = ~3.7 vs 3.69) | 5 + 5 min map arms (C3-R15, R15b; masks = dm_oea_masks' qom form with `_u8`) | U as a combination; pieces T-A | the u8 price rests on "k 8 is free" (F3), softened by the resident keep; R15b's multi margin is ~0, not "within the route's lines" |
| 6 | **FAR-HYBRID near / far tiered eviction** (FAR section 8; Kamen's idea) | 45 GB EXACT tok/s +2-3 % (raw -20 % of 100-200 live misses = -3 to -6 ms of 85-126); 24 GB code bytes x0.89 (raw x0.78); FAST ~0 (QOM's optional misses) | FAR-G1 PASS (>= 20 % fewer misses at K 266, set recall >= 0.80) ~30 % (FAR's own 25 %): the near tier is a noise model at V1's recall, the far map an oracle-state stand-in | FAR-G1 capture 6-7 min GPU (BigMoE builds it Thursday) + CPU | T-A on CPU stand-ins (TRIED 36-37); the one-window form Rnd17#2 / R-57b BETWEEN / KILL | FAR section 4 inherits GOAL's "EXACT SSD wait 55-60 ms": by day at C 224-256 the exposed wait is ~27-45 ms (FULL's table; BUILD_DAY; FAR 7b's own io 29-53 ms), so its +7-18 % column is x0.5-0.75 of that |
| 7 | **GOAL lever 1: the SSD off the critical path** (draft-state prefetch A, stride B, one command buffer C; "fit" at <= 2.8 bpw) | beyond owned work ~0: A/B = lever 2 (look-ahead prefetch into the volatile tier, IN FLIGHT, TRIED 39: EXACT +9 % at recall 0.823, FAST -5 ms [K BUILD_DAY]); ROUTEPRED / Rnd23 (draft-state route prediction) KILL; C = the megakernel KILL (grid barrier 2.0-2.5 us vs dispatch 1.3-1.6 [K FULL]); "fit" = the trellis store (chat quality FAIL [K brief 3.7]; a code bytes lever) | n/a | the oracle-replay test (all of a logged check's reads at t = 0, QD 16) is still worth 10 min: it measures the ~12 ms floor directly | T-A / T-K | dense ~6 ms vs measured ~32-36 ms of GPU per check (DAY9 93: GDN 12.4 + HC reads 11.4 + attention 7.3 + router / shared / tail 5); "EXACT SSD wait 55-60 ms = 45 %" is that error's residual (measured remainder ~27 ms at C 224); "latency problem, not bandwidth" is half right (FAR 7b: bandwidth above a ~12 ms floor that moves only with stalled layers, -5 % under perfect foresight); so FAST +36-45 % -> ~+5 %, EXACT +60-85 % -> +9-15 % [P]; "fit" re-bases EXACT on a lossy model, against the two-track rule |
| 8 | **GOAL lever 2: confidence-tiered k** (sure rows k 4) | subsumed by #1 / #3 (the Q1 / QOM family) | as #1 | as #1 | T-A (QOM, Q1-UNION); U (k 4 strict) | "tiles -35-45 %" agrees with the replays; its quality argument (Expert-Sample, HookMoE) ignores the map's band rates and F3's context floor |
| 9 | **GOAL lever 3: adaptive draft length / drafter parallel commit / one buffer / FR-Spec** | beyond owned ~0 (the live cut at t 0.3 is the adopted form; CALSTOP / STOPCLS / BWIDTH KILL; Mission 29 design D owned, -2.5 ms; megakernel KILL; FR-Spec ADOPTED, SC1 / BP-1c) | n/a | none | T-A / T-K | "chat x1.6-2 from adaptive length" is already in the default (3-4 rows per chat check); "24 GB 19.5 -> 14.6" credits the live cut twice |
| 10 | **ZERO Design B** (3-bit experts, gate-threshold k, in-expert activation sparsity, cache-aware selection bias) | 3-bit: adopted as bytes (bits do not buy ms at 45 GB [K KERNEL90; L1-30 x1.22 where bytes allow 1.80]); gate-threshold k: T-K (R-26, L1-17b/c); selection bias to residents: T-K by orthogonality (REROUTE b25, I6, SERE fallback, MoEPrefetch substitution; cos 0.024); **in-expert activation sparsity on the experts: U** (TEAL on GDN PARKED, TRIED 364): expert ms x0.7 if it held = +8 %, halved +4 %, x odds 0.15 = ~+0.6 % expected | ~15 % (2-bit sure rows went BETWEEN / KILL / KILL; the kernel needs a two-phase gather inside packed 3-4-bit blocks) | 10 min CPU on D2's recorded MoE inputs x a few experts' weights: output error at 50 % sparsity vs the 3-bit tier's (L1-17c's bar) | one U, the rest T-K / T-A | expert 1.84-2.76 MB (records: 2.16 + 0.77); EXACT ceiling "130-170" assumes an MTP chain at k 4 (A 3.9): the drafter keeps 10.5 per 16 rows and the floor engine is 210-280 at that kept [K FULL]; time = bytes / (0.8 x 286 GB/s) ignores the per-group knee; "10 MB needs 99 % hits" ignores the gate; Design C (REAP) dead by Kamen's rule (TRIED 425, PRUNE26) |
| 11 | **c3_b2b_band5** (R12) | code +1 % | ~25 % | 5-10 min, after #1 / #3 | U | the band's "+0.66 % direct" sits on F3's floor |
| 12 | **c3_b2b_top1** (R13) / **c3_b2d_reach07_u7** (R17) | +0.5-1 % / chat only | 35 % / ~5 % (k 7 drops 3 ranks of rows that flip 5-13 % at k 8) | 5 min each | U | u7's "quality unbounded" should read KILL-likely |
| 13 | **cloud3's strict cf 0.5 / 0 forms** (R1, R2, R4, R5, T1-T5) | n/a | KILL predicted by cloud3 itself (2.8-3.5 / 14-17 / 15-18 % direct); F3 adds the floor | do not run | predicted T-K | the P1 / early-P3 tok/s (220-244) are the superseded pricing |
| 14 | **FULL R1: parts of one answer in one pass** | n/a | KILL (M-40: 0 % of HumanEval+ / MBPP answers have >= 2 independent parts; 1.2 % Magicoder) | done | T-K | FULL's x1.4-2.1 was conditional on a workload absent from the lines |
| 15 | **FULL R2: the capacity drafter (DFlash3)** | kept 12-12.5 if it works (+15-20 %), halved +8 %, both tracks; earliest live Nov 20-25 [K DRAFTER_PLAN corrected] | P-CAP >= +0.3 kept ~30 % | one night (armed by MTP Head) | T-A (armed) | DRAFTER_PLAN's corrections (prefill 56-230 tok/s; 35-45 h per epoch) postdate FULL's "Nov 13-18" |
| 16 | **GOAL / ZERO's own "would not spend" lists** (tree drafts, a bigger standalone drafter, kernel redesign #13, 2-bit for speed) | agree: T-K (TREES; KERNEL90's twelve KILLs; L1-30) | | | T-K | none |

Reading the table: nothing here moves a line by >= 1.3x. The two largest halved gains are a measured lever (Rule C, +4-6 %
kept) and the strict-compute family in its byte-neutral form (+5-9 % on code tok/s), and they multiply; cloud3's own form of
the tile lever ranks below both because it pays 2.5x the bytes for the same tiles and its quality prediction is contradicted
by Wednesday's arms. The one untried piece with a sizeable upside (in-expert sparsity) has low odds. GOAL's bet is already
owned (lever 2 in flight); ZERO's are adopted or killed.

## 2. Top 3 tests (each <= 10 min; the PASS / KILL lines written now)

**Test 1 (first): the split arms beside cloud3's, on the decision map, code decisive, with C3-R8's log in the same window.**
Arms: `split_0.9` and `split_0.99` (dm_oea_masks' qom_0_live_bq form with the resident clause off on rows whose draft cf >=
0.9 / 0.99; one line) and `c3_b2d`, `c3_b2c` (cloud3/masks, as built), PERT renorm off, the live read-out; the chat windows
run as the control. In parallel, free: BigMoE's r57 screen under the 45 GB FAST preset logs the expert GROUP count per layer
call beside the miss count for 12 HumanEval prompts, exact vs QOM (C3-R8).
- PASS (code): all-row flips <= 1.34 % for split_0.9 (prediction [G] 1.2-1.5 %) or split_0.99 (0.9-1.2 %), AND R8 shows QOM's
  group count >= 0.85x exact's (then the term is 22-27 ms x (1 - r) and SPLIT90's -12-14 ms is real). BETWEEN: 1.34-2.0 %,
  or R8 at 0.6-0.85x (the gain halves again). KILL: both split arms > 2.0 % on code, or R8 <= 0.6x (QOM already computes few
  tiles; nothing to cut).
- The c3 arms: prediction code 1.3-1.5 / 1.2-1.4 %, IFEval 6-8 %, multi 4.5-6 % (BETWEEN / KILL on chat). If c3_b2d lands
  under 1.0 % on code and under 4 / 3 % on chat, F3's reading is wrong and cloud3's pricing stands.
- Then, only if a code arm PASSes: c3_b2d_reach07 on code only (prediction +0.05 on its parent; PASS <= 1.34, KILL > 2.0),
  and the engine A/B of the passing split form (12 HumanEval, C 224, qall vs qall + split, A-B-A-B; PASS tok/s >= +8 % at
  kept within 1 %, then EvalSuite's same-scores).

**Test 2: Rule C's engine arm (FULL R2.1's lines).** BigMoE, 12 HumanEval at C 224, qall vs qall + SPAN (rule C, the causal
tagger, the line-end guard), A-B-A-B, same prompts; per task T = generated tokens to EOS, S = wall seconds, K = kept per check,
C = the share of generated tokens inside '#' comments.
- PASS: median S ratio <= 0.92 AND median T ratio <= 1.05 AND C ratio <= 1.15 (prediction [G]: S 0.93-0.96, T 1.00-1.03, K
  1.04-1.07: BETWEEN is the likelier read). BETWEEN: S <= 0.97. KILL: S >= 0.97 or T >= 1.15 (the 2610.23818 regime).
- Second, not first, because its kept gain is already measured (M-41); the engine arm decides time vs length. It multiplies
  with test 1.

**Test 3: C3-R15, the chat route + u8 at 147 slots** (the 24 GB chat bytes line; 5 min; masks = dm_oea_masks' qom form with
`_u8` on rows with cf < 0.5 and the bonus row below 0.5).
- PASS: all-row flips within +0.6 of the route's 0.71 / 4.30 / 2.73 % AND under 1.34 / 5.42 / 3.69 (prediction +0.3-0.6 on
  chat); bytes x0.82 on chat at 24 GB [P cloud3 section 21]. KILL: IFEval or multi over its line. R15b (+ reach 0.5) only if
  R15 lands with >= 0.5 of margin on multi.

Not proposed as a top test: TEAL on experts (odds 15 %; a CPU pre-test is in table row 10 if a free hour exists), the oracle
replay of GOAL lever 1 (worth 10 min, but lever 2 is already building what it would motivate), FAR-G1 (already scheduled).

## 3. Cloud Fable 3's request list, audited (REQUESTS.md; run = as filed, fix = with the change named, no = do not spend GPU)

| request | verdict | why |
|---|---|---|
| C3-R11 := c3_b2d (map) | run beside split_0.9 / split_0.99 | test 1; re-register chat as BETWEEN / KILL-expected (F3) |
| C3-R14 := c3_b2c (map) | run with R11 | the safe form; if b2d BETWEENs on code, b2c and split_0.99 are the fallbacks |
| the reach arm c3_b2d_reach07 | fix: code only, after a code PASS | F3; the tau-0.5 chat form has no masks and would not change the chat verdict |
| C3-R8 (engine log, 1 line) | run first, it is free | decides F2's bracket; without it no tok/s claim for any tile form is priced |
| C3-R10 REACH-CUT-X engine rule (Friday) | hold until a code arm PASSes | its +5 % line is on the 22-27 ms term; log the X rule's kept loss and false accepts |
| C3-R15 / R15b | run R15; R15b after it | #5; lines in test 3 |
| C3-R12 c3_b2b_band5 | after test 1, code only | BETWEEN-likely |
| C3-R13 c3_b2b_top1 | after test 1 | probe; k 1 unmeasured in the strict form (TOP1-QOM 2b passed with residents) |
| C3-R17 c3_b2d_reach07_u7 | no | KILL-likely on chat (F3) |
| C3-R1, R2, R4, R5 (strict cf 0.5 / 0) | no (cloud3 agrees) | predicted KILL by its own section 16 |
| C3-R6 massB_0.5 | low priority; (a) form is fine | a 5-min arm; the b16 family was BETWEEN at tighter budgets |
| C3-R7 scores export (13 MB) | yes if cheap (CPU) | mass-budget pricing only; not on tests 1-3's path |
| C3-R9 | withdrawn by cloud3 (correct) | the live masks already count realized rows only |
| paper requests (29 PDFs) | delivered (P1b) | no action |

## 4. Prior art, in one place (for MoonShots' full check)

- The three-band k table keyed on the DRAFTER's confidence, plus union members: adapted. PSR section 7 (arXiv 2610.34721:
  expert demand weighted by drafter confidence; the table is its discretization), AcceptMoE (2608.02989), OEA (2511.02237:
  the union step), LExI (2509.02753) / Yang et al. ACL Findings 2026 / HookMoE for per-layer k. Cloud3's tag ("adapted, new as
  a table") is right; the project's own NEW-1 bands (DAY9 Q2, Wed) are the same family, measured first.
- The split form (#1): a per-band resident clause on QOM; engineering on QOM (itself adapted from cache-conditional routing,
  NOVELTY line 55) and Q1-UNION (OEA).
- REACH-CUT-X (end the check at the previous row when the first miss lands on a cheap row): TriSpec-style escalation
  (NOVELTY line 47) inside one check; adapted.
- The chat bytes lever: OEA k8 + REACH-CUT on QOM; engineering.
- Rule C: NEW as a span rule (NOVELTY: no paper relaxes acceptance only inside code comments); the inflation risk is published
  (arXiv 2610.23818, 2607.26627).
- FAR-HYBRID / the ~100-token horizon: adapted (MoonShots FAR_NOVELTY_CHECK: fMoE, MoE-SpAc, MoE-SpeQ, SeqMoE; tournament
  predictors); FAR's four measurements tagged new (measurement).
- GOAL lever 1: published (SP-MoE, MoE-SpeQ, DraftExpert, ProMoE, Pre-gated MoE, HOBBIT); lever 2: published family (AdapMoE,
  Lu et al. 2402.14800, CALM, Expert-Sample, HookMoE, LExI, PCoMoE); lever 3: published (SpecDec++, AdaEDL, FR-Spec, EAGLE-3).
- ZERO: TEAL (2408.14690) / CATS for in-expert sparsity (untried here on experts); loss-free balancing bias (2408.15664) as a
  cache-aware selection bias (the substitution KILLs apply); REAP (2510.13999, dead by rule).

## 5. The three context levels, compared (one paragraph)

Knowing nothing (ZERO) reproduced the project's measurement programme (its T1-T6 are the decision map, cap45, KERNEL90's
bandwidth probes, the DFlash2 acceptance walks and FAR-WAIT) and most of the adopted stack (3-bit experts as a bytes lever,
gate-mass LFU, next-layer router prefetch, FR-Spec, GPU-driven stepping), but it priced the machine by DRAM bytes and so missed
the two facts that set this project's walls, that expert time is per group not per byte and that the drafter already keeps
10.5 of 16, which put its EXACT ceiling at 130-170 against the measured floor engine's 210-280; its three untried knobs are two
kills (selection bias, pruning) and one long shot (in-expert sparsity). Knowing the goals and the facts (GOAL) found the right
shape in one page (at 45 GB the wall is tiles and bits buy nothing; wasted rows are 40-85 % of rows; kept is capped) and then
mispriced the pieces it could not see (dense 6 ms for a measured 32-36, an SSD wait twice its size, latency where the law says
bandwidth) and proposed as its bet a lever the project already has in flight. Knowing everything (FULL) produced the one lever
in this audit that is new, cheap and now measured (rule C: PASS MBPP, BETWEEN HumanEval+), killed its own runner-up with a
five-minute census, and priced the engine within the measured brackets; its error was multiplicative optimism in its own
halving. Knowing everything plus the data and compute (cloud3) did the most precise anatomy of the night (where the tiles are,
which is what made the split form visible) and the largest unverified claim, and its two load-bearing assumptions (same bytes;
top-8 free) sat beside measured numbers it had cited or could have read (its own replay table; Wednesday's FLOOR arms): more
context raised the quality of the questions and the precision of the arithmetic, not the probability of checking the one
number a claim rests on. The value of the fresh levels was calibration (ZERO) and framing (GOAL's tiles, which cloud3 then
quantified); every priced lever in the three fresh answers was either owned, killed, or cloud3's; the audit's own candidate
(#1) is cloud3's anatomy plus the project's measured arms, which only the full-context reading could combine.

## 6. What this audit did and did not do; next step

Done on CPU (seconds, data on disk): the join of five measured floor1 arms with the draft cf bands (`audit_k8_bands.out`,
reproduces oea_k8's published all-row rates exactly) and a bytes-and-tiles replay of six rules at the 45 / 24 GB pools
(`audit_split_replay.out`, reproduces cloud3's EXACT and QOM tile counts within 1-2 %). Not done: a per-row re-derivation of
cloud3's band flip bound (unnecessary now: the measured arms replace the bound); any GPU. Open [G] numbers: the split arms'
quality (table row 1; read from test 1) and F2's bracket (C3-R8). Next step for the orchestrator: order tests 1-3 (test 1's
split masks are a one-line change to dm_oea_masks' qom arm; cloud3's masks are ready); the Friday-night arm set becomes
split_0.9 + split_0.99 + c3_b2d + c3_b2c with R8's log, not c3_b2d_reach07 alone; nothing in the engine should be built on
"same bytes" or on the 38-43 ms term. TRIED lines (for the recorder; I write only here): AUDIT-SPLIT (the split form, CPU
replay: tiles x0.47 / 0.74 / 0.79 of QOM's at x1.01 bytes, 45 GB; U, map arm proposed) and AUDIT-B2D (cloud3's strict table:
bytes x2.5 / 3.2 / 3.3 of today's preset at 45 GB, x2.4-5.5 at 24 GB; quality prediction contradicted by NEW-1 / oea_k8 /
hyb_0.9_u8_live; map arm still worth its 10 min on code).

## 7. Rulings for Breakthrough (the orchestrator 07:0x-07:1x; answered 07:2x)

Fact received: Breakthrough rebuilt cloud3's band masks independently from the spec text; bit-equal on 36 / 36 windows for
c3_b2d, b2c, reach07, u7, b2b, b2b_top1, new1 and band5; nesting holds; rank 0 kept everywhere [K Breakthrough]. So the
numbers in sections 0-3 refer to masks two builders agree on.

1. **C3-R12 (band5), "rows with cf < 0.9 keep ranks 0-4 in layers 8-35": drafted rows only.** A no-draft bonus row stays
   exact, as in c3_b2d ("the bonus row by its next draft's band, exact if none") and as cloud3's own builder did (the masks are
   bit-equal under this reading; the other reading changes 14 / 36 windows). The rows at stake are the 83 no-draft bonus rows
   on code (2.5 % of its 3,366 realized rows; 1 on IFEval, 0 on multi) [K `audit_k8_bands.out`]. Note for the read-out [P]:
   in the engine under DRAFT17 every bonus row has a cf, and 95 % of code's next-draft cfs are < 0.9 (cloud3 P3 section 12),
   so the engine's band5 would cut most of those rows too: the map arm understates band5's tile cut by ~1.5-2 % on code
   (83 rows x ~100 tiles each over 3,378 kept tokens) and its code flips by ~+0.1-0.15 points (83 rows at ~half the < 0.9
   band's topk5 rate, 12 %). Small, bounded, and the conservative side for the map's PASS line; no rebuild.
2. **C3-R6 (massB_0.5), stand-in rows without router scores: give them the rank profile, keep their ids.** My count on the
   36 windows [P, the same block convention as every arm]: stand-in rows are 18.0 / 36.9 / 37.3 % of block rows (code /
   IFEval / multi), and of those 51 / 18 / 10 % wrap onto prompt rows (p + 97 >= N), i.e. 9.2 / 6.6 / 3.6 % of all block
   rows: not rare on code. Convention: a row without scores takes the domain-independent mean renormalised weight by rank
   (0.294, 0.193, 0.143, 0.109, 0.084, 0.064, 0.048, 0.033, 0.021, 0.010; cloud3 P2 section 4, "identical across domains") as
   its mass vector; its ids stay (p + 97) mod N as in every other arm, so unions and members remain comparable. Why not the
   alternatives: dropping their mass (ids in the union count, zero mass) deletes the stand-ins' experts from the kept set and
   flatters the tile ratio against the engine, where wrong-draft rows have real scores; remapping the stand-ins to answer
   rows changes the union convention of this one arm. Stand-ins are not masked or counted for flips (dm_oea_masks: realized
   rows masked, stand-ins only in the union), so the convention moves the arm's tiles and members, not its measured quality.
3. **C3-R6, the rule as written drops rank 0 on 1.05 / 0.43 / 0.40 % of row-layers and rank 1 on 4.8 / 2.2 / 1.9 % (kept share
   0.73 / 0.65 / 0.64) [K Breakthrough, afe506e], against cloud3's "never" (P2 section 4, one window per domain): always keep
   ranks 0-1 per row, then the budget.** Reasons: (i) a row that loses its rank-0 expert in one of 48 layers is the case the map
   already condemns (topk2 on the < 0.9 bands flips 15-46 %; rank 0 is 29 % of a row's mass); at 1.05 % per row-layer, 1 -
   0.99^48 = 40 % of code rows would lose rank 0 somewhere, so the as-written arm would KILL for a reason that teaches nothing
   about the mass rule; (ii) cloud3's own choice (a), masks precomputed from the baseline's scores, was justified by "the
   budget keeps every row's rank 0 and 99.8 % of rank 1, so perturbed scores move the cut only among ranks 2-9"; the measured
   drops void that argument unless the floor is enforced; (iii) the tile cost of the floor is small (adds back at most 1-6 %
   of row-layer pairs; the kept share rises from 0.73 to ~0.76-0.78 on code, less on chat). With the floor the arm is a fair
   tail-cutting test, comparable with the band tables; expected [G]: x0.75-0.80 of the union, BETWEEN on chat (it cuts
   unsure rows' tails by mass, where F3 says the cost is), PASS-to-BETWEEN on code. If an as-written form is wanted for the
   record, it is a second arm later, not Friday's.
