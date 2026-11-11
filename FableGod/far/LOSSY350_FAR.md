# LOSSY350-FAR: the SSD-wait wall in FAST mode (Fable FAR, Thu 5 Nov 2026, 16:00-16:3x EET; first version 16:17, revised 16:30)

Brief: the orchestrator's MISSION LOSSY350-FAR (Kamen ~15:55): 350 tok/s LOSSY (the FAST track) on the 125B at 45 GB; my wall =
the SSD wait; lossy breaks the rule "wait for the right expert". CPU only (numpy, one process, < 1.5 GB, seconds per run, under a
share lock, swap unchanged at 9.0 GB). Marks: [K] measured (source named), [P] arithmetic or replay on [K], [G] judgement.
Scripts and raw outputs beside this file: `far_defer.py` -> `far_defer.txt` (the never-wait check on the cap45 frame),
`far_defer2.py` -> `far_defer2.txt` (where FAST's remaining misses come from), `far_defer3.py` -> `far_defer3.txt` (the same on the
decision map's 36 windows, by domain and drafter confidence), `far_ckfit.py` -> `far_ckfit.txt` (how much of the service time is
on the check's critical path). Engine numbers: BigMoE's per-check logs
`~/BigMoE/data/r57/{qallt,qallft,u3ct}_208_checks.json` (today 07:39-07:43, C 208, 12 HumanEval, apps closed, checks >= 3)
and `qalq_160` (Wed 15:46). TRIED line numbers are `~/TRIED/TRIED.md`'s.

**Revision 16:4x (after the orchestrator's and ZERO's reviews; the test spec is now `NEVERWAIT_TEST.md`, which supersedes
sections 4a-4b here).** (1) Tag: **adapted, RE-OPENED, not new.** NEVER-WAIT is KTransformers' Expert Deferral and the
project's own deferral family (TRIED 29, 33, 76, 183, 608, 631, 959, 970); what is open is the 45 GB FAST point with no
spin. ZERO reached the same lever independently (LATE-INJECT). (2) Exposure: my replays ran at 266 slots (3.7-3.8 demand
misses per kept token); the engine's logs are at C 208 with 6.6, and ZERO's count on the map windows gives 6.74 at 208
slots: with members counted, 10 late terms per realized row (6 of its own top-2 + 4 members) and 3.55 % of the MoE mass per
row-layer on code, not 3.4 and 2.1 % (ZERO's `late_terms.out`). The engine's 6.6 mixes base and sidecar reads (1.38 MB per read),
so its base-pool exposure sits between the published pool (2.2 per kept) and 208 slots; the spec runs both. The decisive map arm
therefore takes its cold set from 208 slots. (3) The engine form: a late term must be written through its OWN layer's
inject gates (as bt_b12.py's existing late path does), not joined to the next layer's MoE output; and because the late
group rides in the next layer's expert call, the no-spin form is ONE layer late, not B12LL's half layer. Section 4's
step 3 is corrected below. (4) Speed: four designs on the logs now bracket the share of the gate service that is on the
critical path at 0.7-1.0 (within-run fits, a knee model, same-token pairs at two cache sizes, the engine's own
cold-vs-warm samples; `far_exposed.txt`, `far_pairs.txt`, `far_warmsplit.txt`): QOM 95 -> 112-120 tok/s, U3C 100 ->
128-150. (5) Odds for the decisive arm at the engine's exposure: code PASS 60 % at 208 slots, 80 % at the published pool (was 75 % at the 266-slot dose).

## Verdict in seven lines

1. **The wall is bigger than the audit's table says, and it is what keeps the cheap forms from winning [K].** Today's FAST
   (QOM all-in) waits 18.8-20.6 ms of an 88.4 ms check (21-23 %). The strict code form U3C would need ~59 ms without its wait (fitted)
   but waits 33-35 ms (42 %); QOM at a smaller cache (Wed, C 160): ~51 ms without the wait, 53-56 ms of wait. Every form lands at ~95-100 tok/s
   because each trades expert groups against misses; the wait is the hidden price of every strict (low-tile) form.
2. **With the wait gone the logs give ~114 tok/s for QOM (112-116) and ~137 for U3C (133-137) at today's kept (9.4-9.5), on the
   same prompts [P]**: the fitted share of the service time that is on the check's critical path is 0.77-0.85 (QOM) and
   0.68-0.73 (U3C) today, 0.93-0.96 in Wednesday's C 160 runs (`far_ckfit.txt`). If it were fully serial: 120 and 159.
3. **Every lossy form tried so far still waits.** QOM and B12LL wait for cold ranks 0-1; the "late" forms (B12LL, DH_W4, LATE
   RANK 1) wait for the read half a layer later, which is why two of them passed quality and gained no speed (TRIED 183, 608).
   Substitution, nearest / mean expert and low-rank sketches are dead by measurement (TRIED 265, 970, 979; sketch error
   0.92-0.99 of a drop). The unbroken rule is not "use the right expert", it is **"the check may not run ahead of a read"**.
4. **Best bet: NEVER-WAIT.** A cold expert's read is issued and the check moves on; its term is computed on the saved layer
   input and added to the residual stream at whatever layer is running when the read lands. On today's FAST loads that is
   1.3 % of the expert terms (2.1 % of the router mass): per kept row 1.3 cold rank-0 and 2.1 cold rank-1 terms out of 96, and
   98 % of that mass lands one layer late at today's layer time (89 % at 1.0 ms per layer, 58 % at the 350 pace) [P].
5. **Price [P, then halved]:** QOM all-in 95 -> 112-116 tok/s (+17-22 %, halved +9-11 %); on the strict forms 100 -> 133-137
   (+33-37 %, halved +17-18 %), which would be the first single lever on today's table above the 1.3x line raw, if the quality arm passes. It also makes the
   24 GB line run at the 45 GB line's speed (the misses there are 2.8x, the lateness the same) [P].
6. **Quality is the open number, but the 125B's own late experiments calibrate it (section 4c) [K -> P]:** at 12 GB, making
   the cold ranks 2-9 half a layer late instead of dropped took hard-text KLD from 0.1117 to 0.0178 (-84 %); a cold rank 1
   half a layer late cost +0.0056 KLD on hard text (+5.5 %, top-1 0.886 -> 0.878) and nothing on code / math, at 13.9-15.2
   late terms per token. Today's FAST would have 2.05 late rank-1 and 1.31 late rank-0 terms per kept row, 80 % of them inside that
   same half layer: scaled, ~+0.002-0.003 KLD on hard text = ~+0.3-0.45 points of flips at code's exposure, ~0 on code itself. The unknown is
   rank 0 (I4: every expert a full layer late is a KILL). On the map's own windows chat rows carry under half of code's late terms (0.6 + 0.95 per kept row: ~+0.15-0.2 points there), and on code 79 % of them
   sit on drafter-sure rows (section 1d). My odds: code PASS 75 %, IFEval 65 %, multi 50 %; the sure-rows-only form 85 % on code.
7. **Test (<= 10 min GPU, Fri night): one map arm, `qom_nw1`** = QOM cf 0 live BQ's masks with every cold rank 0-1 term moved
   one layer late instead of waited for; PASS / KILL lines in section 4, written now. Speed side needs no new engine code
   for its ceiling: an oracle-prewarmed re-run of the same 12 prompts (section 4b).

## 1. The wall today, measured

### 1a. The engine's own logs [K]

| arm (C, date) | check ms | SSD busy / gate service ms | check minus service (a lower bound on the no-wait check) | misses per check (per kept) | kept | tok/s now | tok/s with the wait removed (at the fitted slope b) |
|---|---|---|---|---|---|---|---|
| qallt_208 = today's FAST, QOM all-in | 88.4 | 18.8 / 20.6 | 67.8 | 62 (6.6) | 9.38 | 95.2 | 112-116 (b 0.77-0.85); 120 if serial |
| qallft_208 = + lever 2 prefetch | 93.6 | 17.3 / 19.7 | 74.0 | 62 (6.6) | 9.38 | 90.5 | 108-110 (b 0.90-0.93) |
| u3ct_208 = U3C, strict code form | 84.7 | 33.2 / 35.2 | **49.5** | 162 (17.0) | 9.51 | 100.0 | **133-137** (b 0.68-0.73); 159 if serial |
| qalq_160 = QOM all-in at C 160 (Wed) | 103.9 | 52.8 / 55.7 | **48.2** | 184 (19.3) | 9.54 | 83.3 | 152-155 (b 0.93-0.95) |

All four: draft 10.1-10.6 ms per check on top (tok/s = kept / (check + draft), checks >= 3, pooled). "b" = the coefficient
of service ms in check ms ~ prompt + b x service + c x rows + d x position (my fit, `far_ckfit.py`, 234-250 checks per arm;
position = tokens generated before the check, needed because early checks have both more misses and a shorter context):
0.7-0.95 of every service ms is on the check's critical path; OOO and today's joint reads hide the rest. "Check minus
service" is therefore a lower bound on the no-wait check; the fitted no-wait checks are 72 ms (QOM) and 59 ms (U3C). The same fit prices a verified row at 1.94 ms for QOM and 1.23 ms for U3C
(c): the strict form's rows are cheaper, which is where its ~13 ms of compute go. Today's preflight (14:10, cold, 2-3 checks, C3-R8's new fields) agrees on the
mechanism: expert groups per check U3C 2,256-2,698, QOM 3,405-3,977, exact 4,212-4,905; misses 1,076-1,254 / 254-405 /
1,933-2,278.

Reading [P; the cause of the 67.8 vs 48-50 ms gap is [G] until C3-R8's group log runs on a full window]: QOM keeps the wait
low by computing whatever is resident (more groups: a 72 ms check without the wait, fitted); the strict form computes a
third fewer groups (59 ms) and pays for it in demand reads. The audit's F1 found the same sign from the bytes
side (strict c3_b2d reads 2.5-3.3x QOM's bytes); here it is in milliseconds, and larger: 33-35 ms, not 4.5-9. A smaller
cache does the same thing to QOM itself (Wed, C 160: ~51 ms without the wait, 53 ms of wait). So the forms tie at 95-100 tok/s, and the
no-wait check that is already running in the engine is ~59 ms (U3C, fitted; 51 ms for Wednesday's QOM at C 160), i.e. ~70 ms
per cycle = **~137 tok/s at kept 9.5 if nothing waited** (159 only if the wait were fully serial). That is this wall's whole
budget: x1.2 on QOM, x1.35 on the strict forms, not more.

### 1b. Where FAST's remaining misses come from [P, `far_defer2.txt`]

cap45 frame (M-38's chained replay, prompts 2-6, per-layer LRU), QOM's rule (ranks 0-1 of every row are demand loads, ranks
2-9 computed only if resident or co-loaded), live rows = kept + 2. The replay gives 3.8 demand misses per kept token at 266
slots against the engine's 6.6 at C 208: same regime.

| | 45 GB (266 slots) | 24 GB (147 slots) |
|---|---|---|
| demand misses per check (per kept token) | 23.5 (3.8) | 65.2 (10.7) |
| by the best rank that needs the expert: rank 0 / rank 1 | 41.5 % / 58.5 % | 42.9 % / 57.1 % |
| needed by a kept row / only the first rejected row / only later rows | 73.7 / 13.8 / 12.6 % | 72.5 / 14.3 / 13.1 % |
| stalled layers per check: all / from kept rows alone | 14.9 / 11.3 | 28.3 / 22.6 |
| cold demand terms per kept row over 48 layers: rank 0 / rank 1 | 1.31 / 2.05 | 3.70 / 5.49 |
| mass QOM drops outright (ranks 2-9, per row-layer) | 0.0495 | 0.1115 |
| wait only for rank 0 (KS 1): misses per check; dropped mass | 12.5 (-47 %); 0.0714 | 34.3 (-47 %); 0.1630 |

Ranks: cap45's ids are in recomputed router-logit order in 56.5 % of rows (bf16 inputs, fp16 routers: near-ties); the counts
do not depend on it (23.6 misses per check in the captured order, 23.5 re-sorted).

### 1c. What a never-wait check would add late [P, `far_defer.txt`]

Model: layer l's reads are issued at l x TAU; one SSD stream, a batch of m experts takes 0.27 + 0.155 m ms (FAR.md 7b's law
per stalled layer; it is consistent with the engine: 62 misses x 0.155 = 9.6 ms + ~30 stalled calls [G, not logged in that run] x 0.27 = 17.7 ms vs 18.8 measured); a term lands
d = ceil(read time / TAU) layers late; a term that cannot land before layer 47 is the tail.

| loads (slots) | late terms: share of terms / of router mass (ranks 0-1 part) | within half a layer (B12LL's lateness) | lands 1 / 2 / 3+ layers late / tail | SSD busy per check |
|---|---|---|---|---|
| QOM (266), TAU 1.56 ms (today's compute per layer) | 1.3 % / 2.1 % (1.7 %) | **80.5 %** | 98.1 / 1.2 / 0.0 / 0.7 % | 7.7 ms |
| QOM (266), TAU 1.0 ms (the ~50 ms forms) | same | 39.5 % | 88.7 / 10.0 / 0.6 / 0.7 % | 7.7 |
| QOM (266), TAU 0.63 ms (the 350 pace) | same | 0 % | 58.5 / 27.4 / 13.2 / 1.0 % | 7.7 |
| QOM (147 = 24 GB), TAU 1.56 | 3.5 % / 5.9 % (4.8 %) | 54.1 % | 92.4 / 6.2 / 0.0 / 1.4 % | 17.8 |
| EXACT loads, nothing dropped (266), TAU 1.56 | 6.3 % / 4.5 % | 11.8 % | 31.8 / 14.3 / 47.2 / 6.6 % | 40.3 (overrun 1.5) |
| EXACT loads (266), TAU 0.63 | same | 0 % | 2.0 / 3.0 / 56.2 / 38.8 % | 40.3 (overrun 14) |

("Within half a layer" is part of the "1 layer late" share: the read finished before the next layer's MoE, with no wait.)
So with QOM's loads the SSD has slack (7.7-18 ms busy in a 50-75 ms check): at today's pace four fifths of the late mass
arrives inside the half layer the 12 GB line already tested, and nearly all of it within one layer. With EXACT's loads it
does not (bandwidth: 40 ms of reads cannot hide in a 30-50 ms check), so "stop dropping, defer everything" is not a speed
form (row 6 below). As the check gets faster the lateness grows (the 350 pace: one to three layers), which is what the
second map arm measures.

### 1d. The same count on the decision map's own windows, by domain and drafter confidence [P, `far_defer3.txt`]

Breakthrough's DAY9 routing capture (the 36 windows QOM's flips were measured on; live checks after the 0.3 cut; real DFlash2
cf per row; rows past the first miss = the p + 97 stand-in), QOM's rule, 266 slots, windows chained per domain.

| | code | IFEval | multi |
|---|---|---|---|
| rows / kept per check | 10.2 / 8.5 | 3.7 / 2.4 | 3.0 / 1.9 |
| demand misses per check (per kept token) | 31.6 (3.7) | 6.0 (2.5) | 4.3 (2.3) |
| wait model, ms per check (stalled layers) | 9.5 (17.0) | 2.2 (4.8) | 1.6 (3.5) |
| cold terms per kept row: rank 0 / rank 1 | 1.39 / 2.06 | 0.60 / 0.97 | 0.55 / 0.90 |
| share of those terms on rows with cf >= 0.9 (share of kept rows there) | **79.0 %** (80.6 %) | 46.6 % (39.0 %) | 32.0 % (32.5 %) |
| wait left if only kept rows with cf < 0.9 are waited for: ranks 0-1 / rank 0 only | **24 % / 11 %** | 35 % / 14 % | 44 % / 17 % |
| mass QOM drops per row-layer | 0.0455 | 0.0397 | 0.0403 |
| the same at 147 slots (24 GB): misses per check; cold rank 0 / rank 1 per kept row | 102.6; 4.76 / 6.50 | 21.6; 2.27 / 3.11 | 12.3; 1.58 / 2.29 |

Three readings. (a) The code numbers repeat cap45's (3.7 vs 3.8 misses per kept token, 1.39 / 2.06 vs 1.31 / 2.05 cold terms
per kept row): two independent captures agree. (b) On code, four fifths of the late terms would sit on rows the drafter is
sure of, where the map has seen almost no flips from far harsher cuts (0 to 2 of 2,211 rows at cf >= 0.99 across the measured arms); waiting only for the
unsure kept rows keeps 24 % of the wait (11 % if those wait for rank 0 only). That is the low-risk form, row 1b. (c) Chat has
less than half of code's exposure per row and almost no wait to remove (1.6-2.2 ms per check): never-wait is a CODE speed
lever, and its chat cost is small for the same reason.

## 2. What is already tried on this wall (do not re-run)

| idea | verdict [K TRIED line] | what it leaves open |
|---|---|---|
| substitute a resident / nearest / mean expert | KILL (265, 970, 979): routed experts are orthogonal, best abs(cos) 0.024; substitution collapses GSM8K | nothing |
| low-rank sketches of a missed expert (rank 32-64, whitened) | KILL (SCOREBOARD 2323: error 0.92-0.99 of a drop) | nothing |
| re-route near-ties to residents (REROUTE b25), selection bias | KILL (236) | nothing |
| wait by mass (m50-m85) | BETWEEN, superseded (946) | - |
| drop cold ranks 2-9 (QOM) | ADOPTED in FAST (446); map 0.39 / 4.09 / 3.26 % | the top-2 wait |
| wait for rank 0 only, rank 1 if resident (TOP1-QOM 2b) | PASS x3 at the 45 GB pool, 0.59 / 4.65 / 3.24 % (121); judged on bytes | its TIME price (row 3 below) |
| cold ranks 2-9 half a layer late (B12LL), ranks 4-9 (DH_W4) | PASS quality (29, 33); DH_W4 no speed, 34.8 -> 34.9 (608) | lateness was bounded: the late phase spins |
| cold rank 1 half a layer late (LATE RANK 1, 12 GB) | KILL (183): hard KLD only +5.5 %, but "two phase-B spins eat the overlap" | the same |
| every routed expert one layer late (I4, 35B) | KILL (631): KLD 0.400, 0.206 with rank 0 in place | late for the COLD few only |
| rows past the first miss at top-2 (REACH-CUT) | PASS code (108) | the WAIT those rows cause |
| resident experts first, missed ones after the read (OOO) | ADOPTED, exact (14-15) | still inside the layer |
| route prediction, far foresight | KILL / BETWEEN / FAR-G1 Fri night (FAR.md) | - |

The common thread [P]: all of it keeps the layer (or the half layer) as the unit the read must fit in. A 2.2 MB read takes
0.23-0.35 ms end to end at low miss counts [K far_wait.txt], and the compute a layer can put under it is 0.1-0.2 ms against
0.7-0.85 ms of reads (TRIED 21: "at most ~20 % of a stall can overlap"). So the "late" forms bought quality headroom and no time.

## 3. Ranked ideas (numbers; gain halved by Kamen's rule in the last column)

| # | idea | what it breaks | tok/s on code FAST at 45 GB [P] | quality [K / G] | new? | halved |
|---|---|---|---|---|---|---|
| 1 | **NEVER-WAIT on QOM's loads**: cold ranks 0-1 are read in the background and added at the layer running when they land (98 % one layer late) | "the check may not run ahead of a read" | 95 -> 112-116 (+17-22 %) | [P from the 12 GB late runs, section 4c] code +0-0.15 points on QOM's 0.39; chat +0.15-0.4 on 4.09 / 3.26 (lines 1.34 / 5.42 / 3.69); rank 0 is the unmeasured part | adapted, RE-OPENED (KTransformers' expert deferral; TRIED 29, 33, 183, 608, 631): never run at the 45 GB FAST point, never without the spin | +9-11 % |
| 1b | **SURE-NW**: row 1 only on rows the drafter is sure of (cf >= 0.9) and on rejected-likely rows; unsure kept rows still wait (for rank 0 only in the tighter form) | the same, behind the gate FAST already uses | 76 % of the wait on code (89 % in the tighter form): 95 -> 109-112 on QOM [P section 1d]; ~120-130 under a strict form [G: a strict form's unsure rows load more] | late terms only where the map shows 0-2 flips in 2,211 rows under harsher cuts; the downstream (context) effect on unsure rows is the open part | new | +7-9 % (QOM) |
| 2 | **NEVER-WAIT under a strict form** (U3C today; SPLIT90 when it passes): the strict forms' 33-35 ms wait is the reason they tie with QOM | the same + "bytes cost time" | 100 -> 133-137 on U3C's logs; SPLIT90's tiles at QOM's loads: ~69 ms cycle = ~135 [P: 98.5 - 13 - 16.2] | U3C's own map quality (code PASS 1.13 %; chat fails) + row 1's cost; with U3C's 162 misses per check only ~1/3 lands one layer late, so use QOM's loads + strict COMPUTE (the split form), not strict loads | new as a combination | +17-18 % |
| 3 | **Wait for rank 0 only** (cold rank 1 dropped as TOP1-QOM 2b, or late) | "top-2 waited for" | misses -47 %, wait -42 % = -7 to -9 ms: 95 -> 102-104 (+7-9 %) | measured: 0.59 / 4.65 / 3.24 % PASS x3 (TRIED 121); late instead of dropped is better | tried for bytes, never priced for time | +4-5 % |
| 4 | **REACH gate on waits**: never wait for an expert only rejected-likely rows need (reach < tau) | "every verified row is worth a stall" | oracle bound: 26 % of misses, -4 to -5 ms: +4-5 %; a real gate ~60 % of it | none on kept rows if the gate is right; rejected rows are discarded | new on waits (REACH-CUT gates tiles) | +1.5-2.5 % |
| 5 | **Deadline loads at 24 / 12 GB**: rows 1 + 3 there: speed set by the GPU, quality by what the SSD delivers | "less RAM = slower" | 24 GB: 65 misses per check, 92 % one layer late: the 24 GB check runs at the 45 GB check's time | QOM's own drop grows (mass 0.11 vs 0.05): that, not the lateness, is the 24 GB quality cost | new framing; B12LL is the bounded version | (small-RAM line) |
| 6 | **Stop dropping: defer everything** (exact routing, all ranks late) | QOM's drop | none (bandwidth: 193 misses per check, half lands 3+ layers late; 39 % cannot land at the 350 pace) | the measured upside is large: at 12 GB late-instead-of-dropped cut hard KLD by 84 % (section 4c), and QOM's drop is what spends 4.09 of IFEval's 5.42 and 3.26 of multi's 3.69. But only where the SSD keeps up: "late if it lands, else dropped" gets 12 % of the late mass inside half a layer and 32 % inside one at TAU 1.56 [P]. Worth one map arm AFTER row 1 (it would free chat's flip budget for the tile levers) | adapted (B12LL at 45 GB without the wait) | 0 tok/s (a quality lever) |
| 7 | FAR foresight (near + far tiers) | - | 0 in FAST once nothing waits; -20 % late terms if FAR-G1 passes | quality only | scheduled | 0 |

Rows 1, 3 and 4 nest: 3 and 4 are row 1 restricted to the terms least likely to matter (58.5 % of the misses are rank 1;
26 % belong only to rows that end up discarded), and they are the fallbacks if row 1's arm lands BETWEEN. Not on the list
because measured dead: any stand-in for a missed expert (section 2).

## 4. The single best bet: NEVER-WAIT (rows 1-2)

Mechanism in the engine (three changes to the gate; BigMoE judges the effort):
1. A cold demand expert does not hold the layer: the gate posts the read and releases at once; the row computes without it
   (no renormalisation, as QOM's drop).
2. The layer input of every row with a pending term is kept (16 x 2,560 values per layer; already there for OOO).
3. At each later expert call, the experts whose reads have landed are appended to that call as extra groups (their own
   weights, the saved input, their original router weight). Their sum is kept apart from that call's own output and is
   written after that layer's MoE write through LAYER L's saved inject gates (corrected 16:4x: the write is per stream,
   so the term must use its own layer's gates, as bt_b12.py's late path does). A read that has landed by the next
   layer's expert call is thus ONE layer late (B12LL's half layer needed its own call); a slower one joins a later layer. At layer 47, or at the end of the check, a still-missing term
   is dropped (0.7-1 % of the late mass) or, for rank 0 only, waited for (one stall per check at most).
Nothing else changes: the loads, the cache and the tokens of every row without a cold top-2 are today's.

### 4a. Quality test (SUPERSEDED by NEVERWAIT_TEST.md section 2: pool 208 slots, own-gates write, new lines)

Arm `qom_nw1`: qom_0_live_bq's masks, 36 windows, plus: every rank 0-1 term whose expert is not resident at the check's
start under the 45 GB residency replay QOM's own masks are built from (dm_res_masks.py / dm_oea_masks' qom arm; Breakthrough knows which) is removed from its layer's MoE
output and added to the NEXT layer's. Harness change: one pending tensor per layer in the map's MoE patch (Breakthrough;
the engine-side late path B12_LATE exists for half a layer on the v3 engine). Second arm if time: `qom_nw2` (two layers
late: the 350-pace share). Read out beside qom_0_live_bq on the same windows (McNemar).
- **PASS:** all-row flips <= QOM's + 0.30 on code (i.e. <= 0.69 %; line 1.34) -> build the engine form for code FAST.
  Chat PASS: IFEval <= 5.0 and multi <= 3.69.
- **BETWEEN:** code 0.69-1.34 %: run `qom_nw1_sure` (the same, only on rows with cf >= 0.9 and on rows past the first miss:
  row 1b, 76-89 % of the wait) and `qom_nw1_r1` (rank 1 late, rank 0 waited: 58 % of the misses), same harness; build the
  one that passes.
- **KILL:** code > 1.34 % or McNemar z > +3 against QOM on code: cold rank 0 cannot be late on every row; row 1b (if `qom_nw1_sure` passes), rows 3-4 (+5-7 %) remain.
- Prediction [P from sections 1d and 4c, rank 0 by [G]]: code 0.4-0.55 % (PASS 75 %), IFEval 4.25-4.5 (PASS 65 %), multi
  3.4-3.6 (PASS 50 %); `qom_nw1_sure` on code 0.39-0.45 % (PASS 85 %). `qom_nw1` is the conservative arm (everything a FULL layer late; the engine would have 80 % inside half a
  layer). `qom_nw2` within +0.1 / +0.4 / +0.3 of `qom_nw1` [G].

### 4b. Speed ceiling (SUPERSEDED by NEVERWAIT_TEST.md section 3)

Run the 12 HumanEval prompts twice at C 208 with QOM all-in: pass A as today; pass B with pass A's demand-read list
(R57_STALL's trace) issued just before each check starts, so every demand read of the check is resident when its layer
arrives (the oracle form of lever 2). Tokens should equal pass A's (QOM's keep rule counts the block's own loads as present
either way); if a changed eviction order makes a prompt diverge, read the check times on the matching prefix. It measures
the check with the waits gone and nothing else changed. Fallback with no new code at all: each prompt twice back to back
(the second pass finds its experts resident); the project already knows the size of that effect as an A/B hazard (a warm
second arm reads +12-21 % [K, the A/B-order rule]), which is this bet's gain seen from the other side.
- **PASS:** median check <= 76 ms (>= 12 ms off 88; the fit predicts 72) -> never-wait's time is real; build it.
- **KILL:** median check > 83 ms: the wait is not on the critical path the way the logs say; drop the bet.
- Same run for U3C: PASS <= 64 ms (the fit predicts 59), KILL > 76.

### 4c. What the project has already measured about late experts on the 125B [K, ~/Breakthrough/SCOREBOARD.md]

All teacher-forced KLD, 10 % residency (C_m 51) unless said, cold = not resident, "late" = half a layer late.

| run (SCOREBOARD section) | hard text (ifeval + multi) | easy text (gsm + code) | what it tells this bet |
|---|---|---|---|
| B12-mass: cold ranks 2-9 DROPPED ("QUALITY STEP screen") | KLD 0.1117 | | the drop is the dominant cost |
| B12 + LATE3=2,10: the same ranks LATE | 0.0178 (-84 %; PASS strict) | | late recovers nearly all of what the drop loses |
| B12 + LATE3=2,4: only ranks 2-3 late | 0.0508 (-55 %) | | the top of the tail carries most of it |
| LATE RANK 1: wait for rank 0 only, cold rank 1 late ("Late rank 1 quality"), 13.9 (code) - 15.2 (prose) late per token | +0.0056 [+0.0018, +0.0093] on 0.1009 (+5.5 %), top-1 0.886 -> 0.878 | -0.0004 [-0.0038, +0.0031], top-1 0.968 -> 0.971 | a late rank 1 is cheap on chat and free on code |
| its speed | +6.7 % code / +8.8 % prose in the harness, "45-50 % of each moved read's latency hidden"; nothing on the v3 engine ("two phase-B spins ... eat the overlap") | | the bounded form hides half a read at best |
| B8 (5 %): cold ranks 2-3 late | KLD 0.1740 -> 0.0755 | 0.0507 -> 0.0198 | and "B8's late reads do NOT hide (60 ms vs 44 ms per token)" |

Scaling to today's FAST at 45 GB [P]: 2.05 late rank-1 terms per kept row against LATE RANK 1's 13.9-15.2 per token = x0.13-0.15 ->
+0.0008 KLD on hard text; 1.31 late rank-0 terms at (0.294 / 0.193)^2 = 2.3x the energy per term [G: energy scaling] ->
+0.0012; the fifth of the mass that is a full layer late at 2-3x the cost [G] -> **+0.002-0.003 KLD on hard text, ~0 on
code**. At LATE RANK 1's measured slope (0.8 points of top-1 per 0.0056 KLD) that is +0.3-0.45 points of flips at code's exposure; chat's own exposure is under half of it (section 1d): +0.15-0.2 points.
What this cannot say: a late RANK 0 was never run alone (I4, on the 35B, made every expert a full layer late: KLD 0.400,
0.206 with rank 0 in place), so the map arm is still the judge.

Cost of the bet: one map arm (5 min) + one harness patch; one engine A/B (10 min) + the gate change. No download, no new
store, no RAM.

## 5. Arithmetic to 350 (this wall's share)

Today's cycle on these prompts: 88.4 + 10.1 = 98.5 ms at kept 9.38 = 95 tok/s (the 110-117 headline is the same engine
at C 224 on steadier prompts). 350 at this kept is 27 ms per cycle.
- Never-wait on QOM: 98.5 -> ~82 ms (114 tok/s).
- Never-wait + the strict compute that already runs in U3C: -> ~70 ms (137 tok/s by the fit; 60 ms and 159 only if the wait
  were fully serial).
- What is left, ~59 ms of check + 10 ms of draft, is the other walls' (groups, cost per group, dense, kept): this wall
  gives x1.2 alone and about x1.35 under a strict form.
- 24 GB: the same cycle as 45 GB once nothing waits [P section 1c]; today's 24 GB FAST check carries 2.8x the misses.

## 6. Prior art

Row 1: KTransformers' expert deferral (SOSP '25; TRIED 970, PRIOR_ART.md lines 15, 89: "1.45x, lossy") defers chosen experts
by a fixed step to overlap CPU and GPU; here the deferred set is the cache misses and the delay is the read's own latency
(adapted; the unbounded-lateness scheduler and the "strict forms tie with QOM because of the wait" measurement are new to
this project; MoonShots should run the web check). Row 3: TOP1-QOM (project, measured). Row 4: AcceptMoE-style reach
weights (as REACH-CUT). Row 5: B12LL generalised.

## 7. TRIED lines (for the recorder; I write only in ~/FableGod/far/)

- FAR-NW (never-wait: cold top-2 read in the background, added when it lands; adapted, RE-OPENED: the deferral family of
  TRIED 29 / 33 / 183 / 608 / 631 at the 45 GB FAST point, without the spin; same lever as ZERO's LATE-INJECT) | wall: SSD |
  line: 45 FAST | exact: no | verdict: PROPOSED, CPU-priced; joint test spec ~/FableGod/far/NEVERWAIT_TEST.md | logs: wait 18.8-20.6 of 88.4 ms (QOM), 33-35 of 84.7 (U3C); wait removed = 112-120 / 128-150 tok/s (0.7-1.0 of the gate service, four designs on the logs)
  [P]; replay: at 266 slots 1.3 % of terms late (ZERO at the engine's 208 slots: 5.6 late terms per row, 3.1 % of the mass), 98 % one layer late; 12 GB late runs scale to +0.002-0.003 KLD on hard text, ~0 on code | 2026-11-05 | ~/FableGod/far/LOSSY350_FAR.md
- FAR-NW-KS1 (wait for rank 0 only) | SSD | 45 FAST | no | PRICED: misses -47 %, wait -42 % (-8 to -9 ms) at TOP1-QOM 2b's
  measured quality | 2026-11-05 | same
- FAR-NW-SURE (never-wait only on rows with cf >= 0.9 + rejected rows) | SSD | 45 FAST | no | PRICED on the map windows: 79 % of
  code's late top-2 terms are on sure rows; the wait left is 24 % (11 % with rank 0 only) | 2026-11-05 | same
- FAR-NW-REACH (no wait for experts only rejected rows need) | SSD | 45 FAST | no | PRICED: oracle 26 % of misses, -4.9 ms |
  2026-11-05 | same
- FAR-DEFER-ALL (exact routing, every miss late) | SSD | 45 | no | KILL as a speed form on paper: bandwidth, half the mass
  lands 3+ layers late | 2026-11-05 | same
