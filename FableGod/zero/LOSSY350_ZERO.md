# LOSSY350_ZERO — the whole design for 350 tok/s lossy at 45 GB (v2.1, Thu 5 Nov 2026, 16:5x EET)

Mission: orchestrator 15:55 (Kamen: FAST is the main mode; 350 tok/s lossy at 45 GB, all 512 experts, top-10,
48 layers). Read: `~/TRIED/TRIED_SHORT.md`, `~/FableGod/fablec/AUDIT_1105.md`, `~/FableGod/full/FULL.md` section 1,
`KERNEL90.md` 90-116, `DAY9` 93 / 875, `far/FAR.md` 7b, `far/LOSSY350_FAR.md`, greps of `~/TRIED/TRIED.md`.
Marks: [K] measured (source), [P] arithmetic or replay on [K], [G] judgement. No GPU. Three CPU replays of my own
(numpy, one process, 3-12 s each, swap unchanged): `late_count.py` -> `late_count.out` / `.json`,
`late_count_pools.py` -> `late_count_pools.out`, `late_terms.py` -> `late_terms.out`, `late_terms_base.py` ->
`late_terms_base.out`, beside this file.

**What changed since v1.** (a) v1 called LATE-INJECT "never tried": wrong. It is the project's deferral family (TRIED
29, 33, 76, 183, 608, 631, 959, 970; my first grep missed "half a layer late"), and FAR reached the same lever
independently as NEVER-WAIT with measured engine numbers (`far/LOSSY350_FAR.md`). FAR owns it; my arm names are
withdrawn for FAR's. (b) v2 priced the SSD wait from a replay at the 45 GB mask pool (5-6 ms); FAR's logs measure
18.8-20.6 ms on the engine's "C 208" run. The gap is not only pool size: that run's reads are ~3.0 base + ~3.6
sidecar per kept token, and my replays model base misses only (section 5d; my first reading, "208 slots = the
engine", was wrong and FAR corrected it from the log). (c) Nothing here is a new idea: it is a constraint sheet, a
count that brackets the joint test's exposure, and one decisive measurement.

## 0. Verdict in seven lines

1. 350 tok/s = **2.857 ms per kept token**. Measured today at C 208: 88.4 ms check + 10.1 ms draft = 98.5 ms per
   cycle at kept 9.38 = 95 tok/s, i.e. 10.5 ms per kept [K FAR 1a, BigMoE's qallt_208 logs]; the 110-117 headline
   is the same engine at C 224 and kept 10.5 (8.7 ms per kept [K brief]).
2. The engine is one inequality: **flat + e x K <= 2.857 x K** (flat = ms per cycle that do not scale with kept:
   dense + SSD wait + draft + commit; e = expert ms per kept token; K = kept per check).
3. **Once the split form is in, the experts stop being the wall.** The split form is ~1,090 tiles per check on
   the map's code blocks [P AUDIT F4] x 8.4 us [K NAX6-NH] = 9.2 ms, so e = 0.77-0.9 ms per kept at K 10.5-12.
   Then **flat <= 23.5-25 ms at K 12, 25-27 at K 12.85, ~20 at K 10.5.**
4. Flat today, measured, is **~76-78 ms**: compute 67.8 less ~20.7 of rows (1.94 ms per verified row) = ~47, plus
   18.8-20.6 of wait, plus 10.1-10.6 of draft [P on K FAR 1a]. FULL's projections: ~51 with tonight's levers, 32.5
   by Nov 14 at night, floor 23. So **350 = the flat term at its floor plus K >= 12**: a 3.1x cut of flat, and
   nothing that touches rows, tiles or bits can deliver it. Of the walls in flight, dense + fixed and the SSD
   wait decide 350; groups and cost per group stop mattering at e ~0.9.
5. Three flat terms can LEAVE the critical path instead of being ground down: the SSD wait (FAR's NEVER-WAIT:
   18.8-20.6 ms today), the drafter (10.1 ms today, 3.5 at its floor; PARKED Mirror-SD), the commit (design D).
   With all three off the path, dense may sit at 19-21 ms and 350 closes at K 12 [P].
6. For the joint test [P my replays; K FAR on BigMoE's log]: QOM's base-pool demand misses per kept token on the
   map's code windows are 2.2 at the pool QOM's masks were built on, 2.7 at 290 uniform slots, 3.5 at 266, 6.6 at
   208. The engine's "C 208" run is C_m 226 + a 64-slot volatile tier, and its 6.6 reads per kept are **~3.0 base
   + ~3.6 sidecar** by its own bytes (9.3 MB over 6.6 reads). So its late-term exposure is the 266-290 rows: 4-5
   late terms per realized row, 1.4-1.9 % of the MoE mass, on 82-87 % of rows (section 5d). The 208-slot arm is
   about twice that (the upper end); the mask-pool arm is ~70 % of it, not a third. My earlier "208 slots = the
   engine" was a coincidence of totals (corrected by FAR, 17:0x). Over half of the reads the wait pays for are
   sidecars, which need no late term at all: the expert can run on time from its resident 3-bit record.
7. **The measurement that decides whether 350 is possible at all: FLAT-2**, the steady cycle
   [draft + 2-row warm all-resident check + commit] with every owned lever on. 350 needs flat <= 25 ms.
   Today ~57 with no waits [P on K FAR 1a]. Lines in section 4.

## 1. The budget: what each term must cost

Per cycle on code at 45 GB FAST, ms. Column 1 is today's logs [K FAR 1a; the dense / expert split of the 67.8 ms
of compute is [P] from FAR's 1.94 ms per row]; columns 2-3 are FULL's table [G / P]; column 4 is this design.

| term | today, C 208 [K / P] | realistic ~Nov 14 [G FULL] | floor of today's structure [P FULL] | **350 engine at K 12 (34.3 ms)** | per kept at K 12 | today per kept (K 9.38) | ratio needed |
|---|---|---|---|---|---|---|---|
| dense pass + head + commit (flat 1-16 rows) | ~47 | 22 + 3 | 17 + 2.5 | **16-17 + 2.5-3** (19-21 + 2.5 if draft and wait leave the path) | 1.5-1.7 | 5.0 | / 2.4-2.9 |
| experts | ~20.7 (QOM, ~10.7 rows x 1.94) | 19 | 14.5 | **9.2-10** (split form on NAX6-NH) | 0.77-0.83 | 2.2 | / 2.1-2.3 |
| SSD wait | 18.8-20.6 | 3 by night (C 320 / lever 2) | 0 | **<= 1** | 0.08 | 2.0-2.2 | / 19-21 |
| draft | 10.1-10.6 | 4.5 | 3.5 | **3.5-4.5** (or ~1 exposed if drafted off the GPU) | 0.3-0.38 | 1.1 | / 2.3-3 |
| **total** | **98.5** | 51.5 | 37.5-40 | **33-34** | **2.8** | 10.5 | / 2.9-3 |
| kept per check | 9.38 (10.5 at C 224) | 10.5 | - | **12** (RELAX 0.5 nat 11.7 [K TRIED 1274] x rule C; 12.85 = FULL's profile) | | | x 1.14-1.28 |

Closures at K 12 [P]:

| closure | dense | experts | SSD | draft exposed | commit | cycle | tok/s |
|---|---|---|---|---|---|---|---|
| C1 three terms off the critical path, dense between realistic and floor | 20 | 9.2 | 1 | 1 | 2.5 | 33.7 | **356** |
| C2 nothing leaves the path: every flat term at its floor | 17 | 9.2 | 0-1 | 3.5 | 2.5 | 32.2-33.2 | **361-373** |
| for scale: realistic Nov 14 + split form + RELAX (night) | 22 | 10 | 3 | 4.5 | 3 | 42.5 | 282 |
| for scale: today's structure at its floor, no split form | 17 | 14.5 | 0 | 3.5 | 2.5 | 37.5-40 | 300-320 |
| for scale: today + NEVER-WAIT + U3C's strict compute, kept 9.5 [P FAR 5] | | | | | | ~60 | 158 |

Reading: the split form and RELAX are x1.2 each and are owned; NEVER-WAIT is x1.2 on QOM and up to x1.6 under a
strict form [P FAR]. After all three, ~10 ms of flat are still to be found: dense stages (5), draft (1-3.5 on the
GPU, or the whole 3.5-4.5 if it leaves it), commit (0.5). Bus check [P]: at C1 the cycle moves 2.7 GB dense + 1,090
x 1.98 MB = 4.86 GB in 33.7 ms = 144 GB/s, half the bus; the rest is ~1,000 dependent launches (dense ~400, experts
~100-150, draft ~500 [K FULL]). **The 350 engine is latency-bound, not byte-bound.**

Necessary conditions (one failing = no 350) and who holds each:
- K >= ~12 at the quality line: RELAX + rule C (EvalSuite's RELAX; GOAL).
- e <= ~0.9 ms per kept: the split form's engine A/B + R8, NAX6-NH (Fable C, Lunatic, A). Map PASS on code only
  [K TRIED 1281], so this is a code-only design; chat keeps 2-4 rows per check and would need flat <= 6-10 ms.
- SSD wait <= ~1-2 ms by day: NEVER-WAIT (FAR), or night C 320 + lever 2.
- dense + head + commit + exposed draft <= ~22-24 ms (B, Leap, the drafter's owners). **This is the wall that has
  no single lever.**

## 2. The engine, in seven lines

1. Draft 16-17 rows (DFlash2 l120 or the capacity drafter), live cut t 0.3, RELAX 0.5 nat + rule C.
2. One GPU-driven verify pass, one-call OOO; experts by the split form (cf >= 0.9: ranks 0-1 + union members;
   below: QOM's keep) on NAX6-NH.
3. No layer waits on the SSD (FAR's NEVER-WAIT: a cold top-2 term joins the expert call that is running when
   its read lands; rank 0 may still wait as the fallback).
4. The drafter's ~500 launches run off the GPU (ANE / SME) so only its hand-off stays on the path.
5. Commit pipelined under the next check (Mission 29 design D).
6. Dense at 19-21 ms: Leap's owned levers (HC8-1P, DENSE-N-RAM, hc_b, SVH, the GDN one-kernel body).
7. One flip budget for {split + never-wait} on the map (<= 1.34 / 5.42 / 3.69 %), EvalSuite's RELAX for kept.

## 3. Today's assumptions, ranked by what breaking them is worth (with the true TRIED status)

| # | assumption today | the break | ms per cycle, code 45 GB | TRIED status (lines) | what is actually open |
|---|---|---|---|---|---|
| 1 | "The check may not run ahead of a read" (FAR's wording) | NEVER-WAIT: cold top-2 terms late instead of waited for | **-14 to -20** of 98.5 [K / P FAR: 113-120 tok/s on QOM]; -33 to -35 under a strict form | deferral family TRIED at 12 / 24 GB (29, 33, 183, 608, 631, 970): quality PASS for tails, no speed because the late phase spun; never run at 45 GB FAST without the spin | quality of a late RANK 0, at the engine's exposure (section 5); owner FAR, spec `far/NEVERWAIT_TEST.md` |
| 2 | "Experts are the term to cut" (three walls on groups and cost per group) | after the split form the inequality is flat-bound: the gap is ~10+ ms of flat, not tiles | - | a framing, from FULL's table + AUDIT F4 | move effort to flat once the split A/B lands |
| 3 | "Drafting is 6-9 % of the check, hiding it is low priority" | measured today it is 10.1-10.6 ms = 10 % of the cycle; at the 350 point 3.5-4.5 of 34 = 10-13 %, and C1 needs it off the GPU | **-5 to -6** today (to its 4.5 realistic), then -3 to -4 by leaving the GPU | PARKED (796, Mirror-SD on ANE / SME, napkin <= 6-9 %) | a re-pricing; the hand-off (correction token + taps) still waits for the head |
| 4 | "The CPU cannot help: every hand-off is a sync" (601 KILL: 70-150 us per sync) | late terms need no sync (two shared buffers); the CPU could compute them on SME while the GPU runs on | ~-1 to -3 on top of #1 [G] | published (KTransformers Expert Deferral, SOSP '25 = CPU experts deferred); 601 killed the synced form | small; only after #1's engine form exists |
| 5 | "Dropped is the only alternative to waited" (QOM drops a sure row's non-resident tails: 227 experts per check, 491 MB [P my replay]) | late-load the heaviest instead: a quality dial paid in tiles and SSD traffic | 0 | B12LL is this at 12 GB (ADOPTED there); FAR row 6: no speed form at 45 GB (bandwidth) | only where a split arm is short of its line (chat) |

Considered and NOT proposed:
- Re-using the rows after the first miss when a fresh redraft agrees: M-06 RECYCLE [K TRIED 869], 24 / 18 % hits.
- A two-deep layer pipeline (row groups 24 layers apart sharing each launch): shares only stage latency and
  needs the next block before this one's verdict = the 32-row KILL (M-18c, U-32).
- Trusting sure rows without a verdict: rows must still be ingested, the pass is flat in rows, and passes are
  counted by the drafter's misses. No gain.
- The whole MoE one sublayer late: I4 [K TRIED 631], KLD 0.400.

## 4. The decisive measurement for the whole design: FLAT-2

**FLAT-2 (<= 10 min GPU, BigMoE or Leap's C0 harness, after Friday's native conversion, every owned lever on).**
Steady loop of 200 cycles on one warmed code prompt at the 45 GB FAST preset with zero misses (the prompt's
experts resident, or FAR 4b's oracle-prewarmed form): [draft + 2-row check + commit], wall ms per cycle, plus the
C0-style split (GDN, HC reads, attention, router / shared / tail, head, draft, commit).
- Why this one: with the split form and never-wait the inequality has one unknown left, flat without waits, and
  a 2-row check is that plus ~4-6 ms. 350 needs it <= 23.5-25 at K 12. Today ~57 (compute ~47 + draft 10.1-10.6
  [P on K FAR 1a]); FULL's "tonight" ~39 (28 + 5.8 + 5); realistic ~29.5; floor 23.
- **PASS: flat <= 30 ms** (cycle <= 36): the last 5-6.5 ms have named owners (drafter off the GPU -3 to -4,
  dense stages -2 to -3) and 350 is live on code at K >= 12.
- **BETWEEN: flat 30-36.** 350 then needs K >= 15-18: only with the capacity drafter AND RELAX at 1.39 nat.
- **KILL: flat > 36 ms with every owned lever on.** No kept the chain allows closes it (K >= 18.4 at e 0.9);
  state the code line's ceiling as ~300 and stop spending on tiles for speed.
- It also sizes each flat term for the next cut, which no map arm does.
- **Built Thu 19:0x, not yet run** (`build_1105/`, INSTALL.md there; owner Leap or BigMoE, ~2-3 min GPU per arm, two arms,
  MEM 32-40 GB by the C ladder): BigMoE's own `run()` at its HEAD with every check held to 2 rows, Friday's all-in
  levers + the split 0.9, pools put back before each pass so the checks miss nothing, Leap's C0 probe for the split and
  the row slope. FLAT = median start-to-start cycle - 2 x the warm row slope. The CPU dry-run (43 checks) runs the real
  patched loop on a stub engine. Two things the build turned up [K by reading the code]: (1) the prefill adopts the
  prompt's experts over the pools (`ext.adopt`), so a repeated prompt starts as cold as the first: any "warm repeat"
  measurement needs the pools put back; (2) BigMoE's Thu 19:0x per-check QOM counters are read from the GPU right after
  the commit submit, in QOM arms only: that read waits for the commit replay.

The SSD term's own test is FAR's (`qom_nw1` on the map + the oracle-prewarmed A/B). My inputs to that spec, sent to
FAR 16:5x: (1) the decisive arm's cold set from 208 uniform slots (`qom_nw1_c208`), the mask-pool arm beside it;
(2) one combined arm with the split form, because SPLIT90 is at 1.31 of 1.34 and the two share one flip budget;
(3) the term added where the engine will add it (the NEXT call's MoE output, i.e. through layer L+1's inject
weights); (4) nw1 licenses today's pace only: at the 350 pace 41 % of the late mass lands 2+ layers late
[K FAR 1c], so the 350 claim needs `qom_nw2` or a mixed-lateness arm; (5) the engine line "no spin" (608, 183).

## 5. The count, and why three sources disagreed (`late_count_pools.out`)

QOM0BQ's loads (= the split forms' loads) per live block of the 36 map windows, Fable C's replay conventions
(plain LRU per layer, prompt rows warm the pool, one session per domain, bonus row gated by the next draft).
Code blocks: 7.9 kept / 9.6 rows.

| pool (slots per layer) | code: demand misses per block (per kept) | rank 0 / rank 1 / undrafted tails | stalled layers (all / if only rank 0 waits) | late top-2 terms per row per pass | MoE mass late per row-layer (all / rank 0 waits) | IFEval: misses per kept; late terms per row | multi: the same |
|---|---|---|---|---|---|---|---|
| laysz45_256 (mean 295, 160-448) = the map masks' pool | 17.5 (2.21) | 6.9 / 10.3 / 0.4 | 11.3 / 5.6 | 1.89 | 0.98 % / 0.49 % | 2.12; 1.36 | 2.13; 1.29 |
| 266 uniform (FAR's replays) | 28.3 (3.58) | 11.5 / 16.2 / 0.5 | 15.4 / 8.5 | 3.02 | 1.62 % / 0.79 % | 2.61; 1.66 | 2.35; 1.44 |
| 224 uniform | 45.4 (5.73) | 19.1 / 25.5 / 0.7 | 20.8 / 12.5 | 4.77 | 2.63 % / 1.26 % | 3.95; 2.44 | 3.16; 1.94 |
| **208 uniform (equals the engine's TOTAL reads per kept, base + sidecar: see 5d)** | **53.3 (6.74)** | 22.6 / 29.9 / 0.8 | 22.8 / 14.1 | **5.56** | **3.11 % / 1.48 %** | 4.72; 2.87 | 3.65; 2.24 |
| 160 uniform | 83.4 (10.5) | 35.8 / 46.2 / 1.3 | 28.6 / 19.2 | 8.47 | 4.90 % / 2.32 % | 8.03; 4.69 | 5.79; 3.50 |
| 147 uniform (24 GB) | 93.6 (11.8) | 40.5 / 51.6 / 1.5 | 30.1 / 20.8 | 9.48 | 5.53 % / 2.60 % | 9.24; 5.35 | 6.61; 3.97 |

Readings [P]:
- **One curve, two points and a coincidence.** Fable C's 4.8 MB per kept (2.2 misses) and FAR's 3.7-3.8 are this
  replay at 295 and 266 slots. The engine's 6.6 is NOT the 208-slot point: it counts base and sidecar reads
  together, and its base pool is 226 + 64 volatile slots (5d). I first read the match as the engine's exposure;
  FAR found the pool line in the log.
- **The wait.** FAR's law on the 208-slot count gives 22.8 stalled layers x 0.27 + 53.3 x 0.155 = 14.4 ms per
  9.6-row block against 18.8 measured at 62 reads; it lands near because sidecar reads stall layers too, not
  because the base misses are that many. At the mask pool the law gives 5-6 ms (v2's number).
- **The quality exposure.** The engine's base misses put 1.4-1.9 % of the MoE mass late on code (5d), not 3.1 %;
  FAR's scaling from LATE RANK 1 (2.05 + 1.31 late terms per kept row, taken at 266 slots) was already at the
  right point. The 208-slot arm over-tests by about 2x, which is the safe side for a PASS.
- **Rank 1 is 58 % of the misses at every pool size**; waiting for rank 0 only halves the late mass and removes
  ~half the wait (14.1 x 0.27 + 22.6 x 0.155 = 7.3 of 14.4 ms at 208 slots).
- At 147 slots the late mass on code is 5.5 %: the 24 GB line is a different quality regime from the 45 GB one,
  not the same check with more reads.

### 5b. The same count the way the joint test's driver will print it (`late_terms.out`, 12 s)

`far/NEVERWAIT_TEST.md` 2.3 defines late[p, l, r] = keep AND the expert is a demand miss of the check, on realized
rows, EVERY kept rank: a row's own top-2, the members it keeps (another row's missed load at its ranks 2-9), an
undrafted row's tail. The table above counted own top-2 only. With members:

| domain | pool, bonus form | demand misses per kept | late terms per realized row: nw1 (own top-2 / members / undrafted tails) | nw1r1 (rank 0 waits) | late MoE mass per realized row-layer: nw1 / nw1r1 | realized rows with a late term: nw1 / nw1r1 |
|---|---|---|---|---|---|---|
| code | 208, `_ball` (the spec's decisive arm, the upper end: see 5d) | 6.64 | **9.98** (6.05 / 3.94 / 0) | 5.39 | **3.55 %** / 1.59 % | 93 % / 90 % |
| IFEval | 208, `_ball` | 4.40 | 3.79 (2.67 / 1.12 / 0) | 2.17 | 1.48 % / 0.70 % | 64 % / 58 % |
| multi | 208, `_ball` | 3.44 | 2.81 (2.07 / 0.73 / 0) | 1.69 | 1.12 % / 0.56 % | 61 % / 55 % |
| code | 224, `_ball` | 5.64 | 8.53 (5.14 / 3.39 / 0) | 4.65 | 3.02 % / 1.37 % | 92 % / 89 % |
| code | laysz45_256, `_ball` (the published masks' pool) | 2.16 | 3.24 (1.97 / 1.27 / 0) | 1.85 | 1.14 % / 0.54 % | 77 % / 69 % |
| code | 147, `_ball` (24 GB) | 11.66 | 17.52 (10.57 / 6.96 / 0) | 9.33 | 6.23 % / 2.76 % | 97 % / 96 % |

So at the 208-slot upper end nearly every code row has a late term somewhere in its 48 layers (10 per row, 4 of
them small member terms); at the engine's own base exposure (5d) it is 4-5 per row on 82-87 % of rows. Either way
the lever is an all-row perturbation, not a rare event. If the driver prints ~5.6 late terms per row on code at
`_c208`, the mask missed the members.

### 5d. Where the engine sits between the two pools (`late_terms_base.out`, 6 s; K: `BigMoE/data/c224t/qallt_208.txt`)

The log of the run FAR priced the wait on: "tier 4433333333: C_m=226 C_l=112 per layer" plus "R57_VTIER: 64 / 32
slots per base / sidecar pool" (uniform; found by FAR). Pooled per kept token: SSD 9.3 MB, 6.6 misses. With 2.163
MB per base record and 0.770 MB per sidecar: 2.163 x + 0.770 (6.6 - x) = 9.3 gives **x = 3.0 base reads and 3.6
sidecar reads per kept** (46 / 54 % of the reads, 70 / 30 % of the bytes) [P on K].

| base slots (uniform), `_ball`, code | demand misses per kept | late terms per realized row: nw1 (own top-2 / members) | nw1r1 | late MoE mass per realized row-layer: nw1 / nw1r1 | realized rows with a late term: nw1 / nw1r1 |
|---|---|---|---|---|---|
| 226 (the base pool alone, volatile tier gone) | 5.52 | 8.35 (5.03 / 3.32) | 4.55 | 2.95 % / 1.34 % | 92 % / 88 % |
| 266 | 3.51 | 5.33 (3.20 / 2.13) | 2.96 | 1.87 % / 0.87 % | 87 % / 80 % |
| 290 (226 + the 64 volatile slots intact) | 2.66 | 3.99 (2.40 / 1.59) | 2.26 | 1.39 % / 0.66 % | 82 % / 74 % |
| for scale: laysz45_256 (the published masks' pool) | 2.16 | 3.24 (1.97 / 1.27) | 1.85 | 1.14 % / 0.54 % | 77 % / 69 % |
| for scale: 208 (the spec's upper-end arm) | 6.64 | 9.98 (6.05 / 3.94) | 5.39 | 3.55 % / 1.59 % | 93 % / 90 % |

Readings [P]: the engine's 3.0 base misses per kept fall between the 266 and 290 rows, so never-wait at today's
engine means ~4.5 late terms per realized row and ~1.6 % of the MoE mass a layer late. The other 3.6 reads per
kept are sidecars of experts whose 3-bit record is resident: under never-wait they need no late term (run the
expert from the resident planes now, read the sidecar in the background), i.e. a 3-bit top-2 on ~4 % of the top-2
picks, far below the all-sure-rows 3-bit form that already passed the map at the 24 GB pool (1.07 % [K DAY9 875]).
So more than half of the reads behind the 18.8 ms wait leave the path with no lateness at all [P/G: BigMoE's
answer to the spec's 8.1 confirms the split by rank].

### 5c. My one review of the joint spec (sent to FAR 16:5x), for the record

Confirmed in the code: the write is linear and the hyper-connection read returns the streams untouched
(`BigMoE/ref/qwen4_exp.py` 582-606, 1243-1290), so a term written later through layer L's OWN inject gates is the
identical vector: FAR's resolution is right. The builder's loop matches the spec's late set (`dm_oea_masks.py`
168-180). Six fixes asked for: the late flags must be read at the same row offset as the keep flags (`_row_rm`
advances `CUR['kpos']`); the write belongs in a post-wrap on `DecoderLayer.__call__` (not an hc pre-hook: no layer
48, and PLE layers read the stream first); the inject must come from the MLP-side read, which the hook cannot tell
from the attention-side one without an identity test; the stash must be per forward chunk; `_t3` arms drop the
top-2 from `keep` and would hide every late top-2; the new suffix must be stripped before `_t3` and `_c`. Two open
points: whether the engine's C 208 is uniform or per-layer, and a third speed pass with the oracle reads issued AT
the check's start, since pre-landed reads cannot show the cost of SSD traffic during compute.

## 6. What this file does not know [G]

- Whether "~400 true stages x 20 us" is the dense floor or only today's kernel shapes (FLAT-2's split shows it).
- The drafter's cost on ANE / SME, and how much of its 500 launches is the hand-off that cannot move.
- The rows per check in today's logs (I used kept / 0.88 to split compute into rows and flat).
- The engine's own tiles per check under the split form (1,090 is the map replay's 9.6-row block).

## 7. TRIED lines (for the recorder; I write only in this folder)

- ZERO-FLAT (Fable ZERO, LOSSY350_ZERO v2.1): the 350 inequality flat + e x K <= 2.857 x K; with the split form on NAX6-NH (e 0.77-0.9 ms per kept) 350 on code needs flat (dense + SSD wait + draft + commit) <= 23.5-25 ms at K 12 | wall: fixed-cost | line: 45 GB FAST | exact: no | verdict: FRAMING (arithmetic on FULL's table, AUDIT F4, FAR 1a); test proposed = FLAT-2 (draft + 2-row warm all-resident check + commit: PASS flat <= 30, KILL > 36 ms) | flat today ~76-78 measured (47 compute + 18.8-20.6 wait + 10.1-10.6 draft), realistic 32.5 night, floor 23 | 2026-11-05 | ~/FableGod/zero/LOSSY350_ZERO.md
- LATE-COUNT (Fable ZERO, CPU replays, 3-12 s): QOM0's base-pool demand misses and late terms per live block on the 36 map windows by pool size | wall: SSD | line: 45 / 24 GB FAST | exact: n/a (measurement) | verdict: MEASURED | code misses per kept 2.16-2.21 (laysz45_256, mean 295 slots) / 2.66 (290) / 3.51-3.58 (266) / 5.52 (226) / 5.64-5.73 (224) / 6.64-6.74 (208) / 11.7-11.8 (147); late terms per realized row with members 3.24 / 3.99 / 5.33 / 8.35 / 8.53 / 9.98 / 17.5; late MoE mass per realized row-layer 1.14 / 1.39 / 1.87 / 2.95 / 3.02 / 3.55 / 6.23 %; rank 0 : rank 1 = 42 : 58 at every size | 2026-11-05 | ~/FableGod/zero/late_count_pools.out, late_terms.out, late_terms_base.out
- NW-EXPOSURE (Fable ZERO + Fable FAR, for FAR's NEVER-WAIT test): BigMoE's qallt_208 run is C_m 226 + 64 volatile base slots and its 6.6 reads per kept are ~3.0 base + ~3.6 sidecar (9.3 MB over 6.6 reads); the engine's late-term exposure is the replay's 266-290-slot rows (4-5 late terms per realized row, 1.4-1.9 % of the MoE mass), the 208-slot arm is the upper end (x2), the published-pool arm ~70 % | wall: SSD / quality | line: 45 GB FAST | exact: no | verdict: MEASURED / in the joint spec (far/NEVERWAIT_TEST.md 2.2, 8.1) | sidecar misses need no late term (run from the resident 3-bit record) | 2026-11-05 | ~/FableGod/zero/LOSSY350_ZERO.md section 5d
