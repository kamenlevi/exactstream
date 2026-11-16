# PLAN900: 900 tok/s lossy on code by Saturday (NewIdeas, Goal A lead; the orchestrator's 11:5x ask)

Mon 16 Nov 12:2x (`date`); revised 12:2x for Kamen's comment decision.

> **CORRECTED Mon 13:5x (`date`): expert bytes were priced 23-30 % low in every check budget (Leap; NewIdeas verified
> from the engine's records3 L01.bin: 1,426,063,360 B / 512 = 2.785 MB per 4-bit expert, vs 2.135 used; MSB-3 2.064 vs
> 1.661; Breakthrough re-checked against the checkpoint, f6bbf36).**
> - Measured times don't move (engine 56.4 ms); floors, bounds and desk byte prices rise ~27 %.
> - A 25 ms lossy 32-row check is at the physical edge: at 100 % of every peak only C6 + V4 + int8 is under 25 ms
>   (24.0); C6 + int8 25.2; C5 26.8-27.7. At ~80-85 % it needs experts <= ~3.7-3.8 GB per check (C6: 4.60).
> - So section 1's "check 25 ms" is NOT reachable with any priced lever; 900 needs a byte lever beyond every priced plan
>   AND near-zero overhead (whole checks run at 35-45 % of peak today; the best kernels 74-84 %) AND kept 26.6.
> - Exact byte floors on real sizes (Leap 13:5x, 280 GB/s): 32 rows code 47.3 -> 57.9 ms; 16 rows 38.8 -> 46.9;
>   1 row 15.9 -> 17.1. GOAL B's "~55 ms realistic exact target" is below the real floor.
> - The expert kernel runs at 1.52-1.64x its real floor (171-184 GB/s), not 2x.

> (a quote from the project owner, kept private)
until Saturday."

Inputs:
- THINK_1116 (Lunatic, ZERO, FAR, Fable B, Fable C);
- Leap's CHECK25 (79a850f);
- Fit12's LOSSY32 v2;
- MoonShots' Pillar 3 (dfe0e4a, 3170b9a);
- the real 127-node dumps;
- my dev joins.
Numbers: research/week98/budget900.txt.

## 1. The 900 budget (one screen)
**tok/s = kept per 32-row check / (check + draft + overhead), in ms.** HE+ code; drafting is NOT hidden, since a
bandwidth-bound check can't hide draft bytes.

| term | today (best measured / dev) | target set A (Kamen's 3) | what 900 actually needs | gap to 900 |
|---|---|---|---|---|
| kept / 32-row check (lossy acceptance) | 16.91: tree-32 + close-call d1, dev replay (MBPP 9.50); exact tree-32 16.26 | 23.15 (97.8 % flat) | >= 26.6 (98.8 % flat) at 25 + 3 + 1.5 ms; or 23.15 at a ~20 ms check | x1.57 |
| lossy 32-row check | none built. Desk: C5 51.3-55.7 ms realistic (Fit12 v2). Measured exact engine: C32one 90.5 | 25 ms | 25 ms (or ~20 at kept 23.15) | x2.1-2.2 vs desk; x3.6 vs measured |
| draft per block | tree-32 19.1 ms (20.8 levels x 0.92, full head 177.6 MB per level, sync-free); tree-16 11.6 | 3-5 ms | <= 3 ms | x4-6 |
| overhead | ~4 ms (commit replay + seams) | - | <= 1.5 ms | x2.7 |
| **tok/s HE+** | **~215-227 lossy** (desk check), ~144 exact (measured engine, tree-16) | **681-827** (the three targets at overhead 4..0) | **900** | **x4** |

- **The three stated targets do not compose to 900.** At 97.8 % flat, a 25 ms check and a 3-5 ms draft, the result is
  681-827 tok/s.
- **900 needs one target beaten:** kept >= 26.6 per 32-row check (98.8 % flat), or a ~20 ms check with 97.8 %.
- **Three bounds that bind:**
  - Lunatic: t30b's 177.6 MB per level puts 900 out of reach at any tree shape. The head must read <= ~47 MB per level.
  - Leap: C5's bytes bind at 23.9 ms at 100 % of every peak. The realistic Friday commitment is a MEASURED ~33-38 ms.
  - Fable C: no acceptance rule certifies 97.8 % per band at bands 1-10 on code rows. Comment-free decoding was the
    exception; Kamen dropped it.
- **Saturday's honest demo:** measured parts + a projection from certified parts + (optional, labelled, "doesn't really
  matter" per Kamen) a multi-stream aggregate.

## 1b. Wall A decomposed: TIES vs DRIFT at bands 1-10 (comments stay; the orchestrator 12:2x)
Measured on HE+ dev, walk-strict:
- **Ties cap the bands even with ZERO drift.** The true-state head as a chain (LAB) reads 93.2 / 95.3 / 96.4 / 97.8 at
  bands 0 / 1-2 / 3-6 / 7-10. Those are the 125B's own near-ties after a miss, which no drafter can predict.
- **Only trees cover ties (exact).**
  - Band 0: 98.5 at tree-32. Bands 1-2: 96.6 at K32, 98.3 at K127.
  - Bands 3-14: 93-96 even at K127, because there ties AND drift both act on the tree's head.
- **So 97.8 per band at bands 1-10 needs BOTH:** trees for the ties AND a drift fix that brings the tree's head toward
  true-state accuracy.

**Drafter-side DRIFT levers, with what each is worth (measured unless marked):**

| lever | attacks drift? | worth (HE+ dev) | status |
|---|---|---|---|
| 2B audit, P = 2 (C2) | partly: repairs at confident disagreements | walk-strict 15-22 / 23-30 98.3 / 98.3; bands 1-14 +0..0.5; kept +8.9 % | certified line (amendment 8) |
| 4B audit | partly | deep 98.3 / 98.3 at P = 2; kept +14.9 % | information |
| t30c (more budget / data) | yes | sandbox: HE 23-30 +0.98 (FAIL), MBPP +2.6 | OFF (S10 FAIL) |
| carryac / tapmlp, full budget (item 5) | yes (state supervision) | 400-step screens +0.6-0.8 pt | S12 running (MTP Head) |
| SIBLING-STATE drafting (FAR: the rejected rows' exact 125B states) | yes: true-state information for free | untested; the ceiling is the LAB true state (kept 20.77 vs 13.77, +51 %) | sandbox analog -> **Sandbox98** |
| QAT-distilled small head (26 MB per level) | NO: a cost lever; at best it keeps today's accuracy | draft / 4-6 with the fused kernel | MTP Head screen |
| bigger trees + early death | no (ties + throughput) | kept 23.70 at K128 on HE+ | probe Wed |

- **Honest read:** no drift lever in hand reaches true-state accuracy by Saturday. The true-state chain itself is below
  97.8 at bands 0-6 because of ties.
- **So the per-band 97.8 at bands 1-10 is NOT reachable this week without a breakthrough on drift** (sibling states are
  the only untested candidate).
- **What IS reachable:** band 0 (tree, 98.5); bands 15-30 (audit P = 2, 98.3); the LAB knowledge line (97.8 marginal).

## 2. Bets per wall, ranked by gain x odds by Saturday (owner; the first test TODAY)

| # | wall | bet | gain if it works | odds by Sat | owner | first test TODAY |
|---|---|---|---|---|---|---|
| 1 | A + C | **SPAN TREE / deep runs from copies** (Lunatic best + ZERO): head levels only at bands 0-6 (<= 7 per block); deep runs from seen-span copies (58 % of HE+ positions sit in >= 8-token seen spans, right 99.2 %) | draft / 3 at ~equal kept; Lunatic projects ~900 at the floor on HE+ (weaker on MBPP) | medium on HE+, low on MBPP | **NewIdeas** | CPU on the real dumps + the bank's ids: kept / 32 with <= 7 head levels + span continuation vs tree-32. Line >= 0.90x tree-32 kept on HE+ |
| 2 | B | **one-pass > 16-row FAST check** (qmm32n + FAST-NAX one-call at 32 rows; bits PASS 4663870) | -10 ms per check (C5 61-65 -> 51-56) | high | **Leap** | the <= 5-min QUIET qmm32n_bench timing in an EvalSuite step gap (slot given 12:2x) |
| 3 | B | **CPU diet:** fuse the 1c glue into router_fused (34 ops per layer = +10.6 ms CPU), then Fable B's 4-day diet (-1,500 launches) | CPU floor 38.5 -> ~17 ms; without it no check gets below ~38 ms | medium | **Leap** (Fable B's plan) | the glue-fusion design + a launch count on the scoreboard |
> (a quote from the project owner, kept private)
| 5 | B | **check BYTES:** C6 (lambda 1.8e-2) + int8 dense + V4 (GDN / attention 3-bit), a code-scoped cut <= ~5.5 GB at quality | -3..-6 ms | medium (quality gates this week) | **Breakthrough** + EvalSuite | the byte / ms price on code, and the three quality gates pre-registered with EvalSuite |
| 6 | C | **QAT-distilled t30b** (FAR: 4-bit + 8K screen, TRAINED) at ~26 MB per level | with bet 7: draft ~2-3 ms per tree-16 block | low-medium (training) | **MTP Head** (back 13:10) | the 15-min sandbox screen: does QAT recover M3's -8 % untrained 4-bit loss? |
| 7 | C | **FUSED LEVEL KERNEL** (FAR: "not optional for 900"): fixed cost 0.31 -> ~0.05 ms per level | with bet 6, draft <= 3 ms. Alone it is +4 % (exact engine gain) | low by Sat (multi-day) | **MoonShots** (free) | the kernel's dependent-step plan (<= ~35 per level vs ~144) + a 1-level prototype |
| 8 | A | **span-conditional close-call** (Fable C: delta 1 on code, 2-3 nats in comments; built: R57_RELAX_DELTA). **ON HOLD:** Kamen's answer on relaxed acceptance inside comments is pending; until then comment rows = code rows (delta 1 at most) | +4-10 % kept (lossy) | pending Kamen | **NewIdeas** (replay, desk) -> EvalSuite | CPU replay, information only, until Kamen answers |
| 9 | C | **DRAFT DEATH** (confidence-cut depth, draft -30-50 %) + **WIDE-SHALLOW** (depth 8-12, K 64-128) (ZERO) | draft -30-50 % at small kept loss | medium | **NewIdeas** (with #1, same machinery) | CPU on the real 127-node dumps: kept vs levels per block |
| 9b | A (drift) | **SIBLING-STATE drafting**, sandbox analog (FAR; the target's states at rejected rows as the head's depth inputs) | the only untested true-state-information lever; ceiling +51 % kept | low-medium | **Sandbox98** | the analog in the 2B + s30 sandbox (<= 5 GB). PASS: half of the own -> true gap closed at depths 1-8 |
| 10 | A | close-call d1 / d2 | the lossy acceptance baseline | running | EvalSuite | night1117 (tonight) |
| 11 | A + B | **EARLY DEATH** (bigger trees at a cheaper check) | +20-35 % like for like (drafting not hidden) | low-medium | BigMoE (data) / NewIdeas (probe) | Tue-night dev pass -> Wed CPU probe |
| 12 | B | misses -> 0 (a resident-only MSB-3 rule: -3.7 ms, quality unpriced) | -3.7 ms | medium | **Fit12** (back 13:10) | quality pricing + keep the LOSSY32 ladder as the scoreboard |
| 13 | A | INERT comment + docstring acceptance (replay x1.17-1.30 HE+, x1.63-1.79 MBPP; a relaxed comment changes the next code 48 % of the time). **ON HOLD** with #8 (Kamen's comment question) | large kept, high quality risk | pending Kamen | EvalSuite | none until Kamen answers |

**Scoreboard (kept per check, check ms, draft ms, overhead, launches per check), updated daily:** NewIdeas, in
TARGETS_450_900.md.

## 3. GPU plan through Saturday
Days are fully usable (Kamen 11:5x). Memory gates, swap / disk / fan guards and lid = pause stay. One > 16 GB job at a
time. The drafter certification (capture Wed, proofs Thu-Fri) is untouched.

| when | job | owner | size |
|---|---|---|---|
| Mon by day -> 22:30 | 1c-tiers gate to completion (B1 -> V1 -> B2 -> V2) | EvalSuite | ~39 GB |
| Mon by day, beside | Leap bit / accuracy tests (share); MTP Head S12 / QAT screen (<= 4 GB); MoonShots kernel work (<= 4 GB) | - | small |
| Mon, a gate gap | Leap's qmm32n QUIET timing (<= 5 min) | Leap | <= 2 GB |
| Mon night | M4 23:05 (BigMoE); E14 00:30 (Leap); night1117 close-call d1 / d2 + tier-0a (EvalSuite) | - | 45 GB, chained |
| **Tue by day, first** | Leap's quiet ENGINE 32-row checks (~40 min) after night1117 ends | Leap | ~40 GB |
| Tue by day | C6 / int8 / V4 quality gates once built and pre-registered (COMMENT-FREE dropped) | EvalSuite / Breakthrough | 45 GB |
| Tue by day, beside | BigMoE builds the early-death taps (head-test); QAT sandbox; CPU screens | - | small |
| Tue night | early-death dev pass 23:35 (~35 min), then Leap's engine slot #2 (~00:20) | BigMoE, Leap | 45 / 40 GB |
| Wed by day | Leap's int8 qmm32n + engine checks; lossy quality gates (end >= 2 h before the capture); amendment pins | Leap / EvalSuite | 40-45 GB |
| Wed night | THE CAPTURE (>= 23:30) | EvalSuite / BigMoE | 45 GB |
| Thu by day | TOPK + T8-32 dumps (~0.8 h); the 2B / 4B audit rows; T8-127 HE+ only if early death passed; Leap's engine slot #3 | BigMoE / Leap | 12-40 GB |
| Thu-Fri nights | THE PROOFS | EvalSuite | 45 GB |
| Fri by day | Leap's MEASURED lossy 32-row check (the Friday commitment); final scoreboard | Leap, NewIdeas | 40 GB |
| Sat | certificate verdicts + the 900 budget readout: measured + projected from certified parts | EvalSuite, NewIdeas | - |

## RESULTS today (my CPU bets)
- **#1 SPAN TREE: KILL** (span_tree.txt, f9c20b2). Best HE+ x0.797 / MBPP x0.757 of tree-32 kept at <= 7 levels (line
  0.90x).
  - The deep half of tree-32 on HE+ is the head's own confident drafting, not copies: a depth-6 tree alone keeps
    x0.50-0.55.
  - Levels 20.7 -> 6.9 still nets ~+9 % tok/s at a 25 ms check (-4 % at today's 53.5 ms).
- **#9 DRAFT DEATH / WIDE-SHALLOW: KILL** (depth_width.txt). No depth cap (6-24) or confidence cut (p >= 0.01-0.5) beats
  tree-32 on HE+. On MBPP, tree-16 is +11 % (its known K*). EAGLE-2's order with the right K is near-optimal.
- **Implication:** with TODAY's kept, even every draft fix (QAT 26 MB head + fused level, 0.30 ms per level) gives HE+
  ~449 tok/s at a 25 ms check. The binding wall for 900 is KEPT (the drift + ties wall) together with the check.

## UPDATES (Mon 12:3x)
- **Lossy gates: lossygate amendment 5 (EvalSuite 1e8b9a5), option A.**
  - G1 = C6 + int8 dense (code-scoped: R57_INT8D_SETS; +1.7 GB resident) Tue by day after Leap's engine slot #1:
    base ~2.5 h, then G1 ~2.5 h.
  - **G2 = G1 + V4 now runs only on a DENSE-3 map-arm PASS** (Breakthrough, Wed ~07:00-09:00, ~1 h, ~14 GB; PASS
    <= 1.34 % code flips).
  - ~~V4 was never flip-tested~~ CORRECTED 12:3x: 3-bit dense WAS flip-tested Sun 15 Nov and FAILED (bt_ga3 RTN 4.43 %,
    bt_ga3awq AWQ 3.86 % code flips vs 1.34). Only the DENSE-3 midpoint variant is untested (prior FAIL ~3.5-4 %).
  - Tonight's map screen (Breakthrough, ~25 min, ~16 GB), after BigMoE's G1 key test, which comes right after EvalSuite's
    chain ends. Arms: C6 bytes / G1 / DENSE-3 / lambda 2.0e-2. It decides G2 and pre-screens G1.
  - Leap's measured price: int8 -0.44 ms per 32-row check (-2.0 fused); C6 -2.9 ms. So G1 + G2 buy ~-3..-6 ms against
    the ~-28 ms that 25 ms needs.
- **Tree-32 + audit at 32 rows** (tree_audit32): +0.3-4.3 % (2B) / +2.9-5.7 % (4B) kept, each pass adding 1.06 /
  2.37 GB per check. Not a 900 lever.

## CORRECTED Mon 12:3x (MoonShots' measured bytes, 6bb1a0c): bet 6 + 7 do NOT reach <= 3 ms of draft
- **FAR's "QAT 4-bit + 8K screen = ~26 MB per level" is wrong.** Measured C1 (4-bit Linears + 8K screen) is 95.1 MB
  per level: dense 52.6 + head 11.8 + 10 routed experts 30.7. The 32K screen is 47.2 MB; ~37 MB is already stored at
  4 / 8 bits.
- **With the fused kernel** (f 0.056 + 95.1 MB at 292.7 GB/s = 0.381 ms per level), bets 6 + 7 draft:
  - a tree-32 block in ~7.9 ms (HE+ 20.8 levels);
  - a tree-16 block in ~4.8 ms (HE+ 12.6 levels);
  - not 2-3 ms.
- <= 3 ms per tree-32 block needs <= ~26 MB per level AND f <= 0.056. No format reaches that today: C8 (top-2 experts +
  2-bit dense + 2K head) is 39.8 MB and untrained cost 67-81 % kept.
- **Re-run projection (25 ms check model, + 4 ms):** HE+ tree-32 429 tok/s (not 449); HE+ tree-16 ~395; MBPP tree-16
  ~264. The draft wall stays at ~5-8 ms per block this week even with both bets.
- MoonShots' bet 7 line, pre-registered: PASS bitwise + f <= 0.056 at the stock head; BETWEEN f <= 0.10. The plan is 15
  dependent kernels per level vs 144 primitives today. Multi-day.
- **MEMORY:** Leap removes int8's +1.7 GB copy TODAY (the kernel reads DENSE-N-RAM's native words; native == copy
  staged bit for bit). G1's MEM stays t1cx45t1c1's. int8d bits / accuracy PASSED 12:31 (11 shapes, 1-32 rows).

- **Bet 7 M7a (MoonShots afcc3f2, quiet 12:46):** dispatches are cheap (<= 1.3 us GPU per dependent dispatch; ~2.0 us CPU
  per custom call). The stock C0 level is 0.954 ms, 0.348 ms above its bytes:
  - <= 0.19 ms of it is dispatches;
  - >= 0.16 ms is off-bandwidth work (the sort-based top-K over 32K logits, single-threadgroup reductions, GEMVs below
    bandwidth).
  - A fused 15-call level reaches f <= 0.056 only if its GEMVs run at full bandwidth and the tail (logsumexp + top-K)
    stays a few us. Those are now the design constraints.

- **G2 composition (13:0x):** G1 as built is 6.22 GB per check (no V1 / V3), not 5.77.
  - G2 = G1 + V1 (16K head list) + V3 (4-bit HC mixers / PLE) + V4 if DENSE-3 PASSes, else + lambda 2.0e-2. One stacked
    Wednesday gate.
  - V1 + V3 measured +0.12 / +0.12 flips on the map (12 windows) and save ~0.46 GB.
  - Build: BigMoE if it fits by Wed morning, else Breakthrough codes and BigMoE wires. If neither, G2 is dropped this
    week.

## UPDATES (Mon 13:4x, `date`): the measured engine check + the 1c-tiers comment fix
- **Leap, real engine, quiet (run2b; Leap data/week98/check25/README.md; every expert resident, misses 0):**
  - The best lossy 32-row check (C5 + C6 + int8, F32c6i) = **56.41 ms**, R32 1.440 of F16 38.88 (paired). 25 ms is
    **2.26x** away; the C6 + int8 byte floor is 18.2 ms (3.10x).
  - The one-pass form is real: -9.46 ms paired (F32o vs F32s).
  - **The expert kernel runs at ~2x its byte floor at 16 AND 32 rows, whatever the split** (135-145 GB/s; ~28 ms of
    experts alone at the map's bytes). That is above the whole 25 ms budget: the check pillar's main wall now.
  - int8 dense +1.13 ms and t1cf +0.49 ms SLOWER in the engine (the benches said faster; open).
  - Number check (NewIdeas): 56.41 / 25 = 2.26; 82.6 MB per call / 571-614 us = 135-145 GB/s; the "vs F16" column is
    the paired median, ~0.8 % below the ratio of medians. Consistent.
  - With 1c's misses (~60 per 32-row check, Fit12) the real lossy check is ~56-60 ms with RES. At K* lossy kept 16.9 +
    tree-16 draft 4.8 + 1.5 ms: **~260-270 tok/s HE+ today.**
  - Suggested to Leap (option): a no-math read kernel over the same expert bytes, one launch, to split the 2x into
    kernel structure (>= 250 GB/s) vs access pattern (~140 GB/s; 1.9 MB experts).
- **Fit12, 1c-tiers flips by tag (12 HumanEval windows; Fit12 PLAN "COMMENT-EXACT PRICE RESULT"):**
  - 35 of 1c's 38 flips are on "#" comment rows (6.93 % of them), all 16 non-tie flips; 0 on docstrings; 3 elsewhere
    (0.36 % = QOM's rate). Comment rows are fragile under any change: QOM 2.18 %, path noise 1.78 %.
  - Per 32-row check (1c alone 64.3-67.5; + RES 51.1-54.3): comments on QOM **+3 ms (+5 with RES)**, quality ~QOM's
    [P] ~0.42 / ~0.06; comments exact +14 ms (misses 60 -> 102); comments + docstrings exact ~170 ms.
  - Open: MBPP docstrings (authored; 2 of the 4 runaway divergences; the map has no MBPP windows).
- **Decision (NewIdeas 13:4x):** Fit12 builds ONE installer NOW (comment-QOM switch first, RES by Tue 18:00;
  docstrings a per-set option). The RES hold is lifted because both verdicts need it: PASS -> RES for G2; FAIL ->
  comment-QOM (+ RES) for G1. FAIL looks likely (4 of 45 keys vs 0; the guard fails at 6-0).
  - **Kill test (pre-registered):** free-decode the 4 runaway keys under 1c + comment-QOM, without and with MBPP
    docstrings. The fix dies if >= 3 of 4 still cap or loop; BETWEEN at 2; it survives at <= 1 (weak: any change
    moves these paths). The gate's RUNAWAY guard stays the real test.
  - **Map read (Breakthrough, Tue ~07:00-07:15, slot (c); 12-window map, Fit12's 1c masks):** bt_cq_ctrl (1c +
    hc8w_mix + int8) / bt_cq (comment rows on QOM) / bt_cq_res. Pre-registered, all-row flips: CONFIRMED if bt_cq <=
    0.5x bt_cq_ctrl -> G1's fix = comment-QOM; NOT CARRIED if >= 0.8x -> a lower lambda; BETWEEN -> the kill test
    decides. Runs under either verdict.
  - If RUNAWAY fails: G1 = 1c + comment-QOM (+ RES) by EvalSuite's dated two-branch amendment before Tuesday's gate.

## UPDATES (Mon 14:0x, `date`): amendment 5d's open items (G1's branches), fixed before data
- EvalSuite filed 5d before 1c-tiers' V1 verdict: PASS -> G1 as filed (or + RES); RUNAWAY FAIL -> by Breakthrough's
  Tue 07:15 comment-QOM map read + Fit12's 4-key kill test.
- NewIdeas fixed the four open items (14:0x):
  0. The map read is judged on NON-TIE flips (all-row beside). hc8w_mix + int8 in both arms dilute the all-row ratio
     (~0.5 expected even if comment-QOM works as measured); all 16 of 1c's non-tie flips were comment rows.
  1. Lower lambda = 0.8e-2 (s 0.98): predicted code flips 0.51 / 0.15 vs 1c's 1.08 / 0.50 (~80 % of the excess over
     QOM removed); ~+50 % expert bytes vs 1c (54.8 vs 36.4 4-bit-eq). 1.2e-2 removes only ~40 %.
  2. Between 0.5x and 0.8x: comment-QOM only if the kill test SURVIVES, else lambda 0.8e-2. At <= 0.5x: comment-QOM
     unless the kill test KILLS. No key-tested installer by 07:15 -> lambda 0.8e-2.
  3. Kill test = the 4 cap-hit keys (fixed list); still runaway = cap or loop (3a); the comments-only arm counts.
     KILL >= 3 / 4, SURVIVE <= 1, BETWEEN 2; VOID if the OFF control reproduces <= 2 / 4.
  4. IFEval identity FAIL = a scope bug: G1 not run. Pooled delta / decision-4 FAIL: the same branch rule as RUNAWAY.
  5. (14:0x, filed by EvalSuite) The MBPP-docstring arm D is DECISIVE: if the comments-only arm C fails to SURVIVE
     and D SURVIVES -> comment-QOM WITH MBPP docstrings; C KILL and D not SURVIVE -> lambda 0.8e-2. 3 of the 4 keys are
     MBPP (mbpp/122, mbpp/147, mbpp/13, HumanEval/147; Fit12 confirms).
  6. No map read by 09:15 (not 07:15) -> lambda 0.8e-2. The comment-QOM branch runs 1c at lambda 1.6e-2 (confirmed).
  7. Every branch has a with-RES and a without-RES env; RES joins only if its own 5c rule holds. g1b key-test envs (9):
     lx, g1x, g1rx, g1cx, g1crx, g1cdx, g1cdrx, g1lx, g1lrx (lambda is env R57_TIERS_LAM; ~7-8 min; waits on res1c).
  8. **CORRECTED 14:1x (Fit12 14:0x):** under 3a's rule with the gate's per-item caps (MBPP 1024, HE 768), only 2 of
     the 4 keys are 1c-only runaways (mbpp/122, mbpp/147: base eos -> 1c length); mbpp/13 and HumanEval/147 cap in the
     base too. The signal is 2 of 45 vs 0 (McNemar p 0.5), so the 13:4x "FAIL looks likely" was WRONG; the branches are
     insurance. The kill test counts DISCORDANT keys only (variant runaway + base eos): mbpp/122, mbpp/147 + any more
     from the full tier-1 rows at the runner's start (<= 6); KILL >= 75 % still runaway, SURVIVE <= 25 %, else BETWEEN;
     VOID if the OFF control reproduces < 75 %. The comment-spiral evidence for Kamen's comment question rests on 2 keys.
  9. **RE-CORRECTED 14:1x (EvalSuite, full V1 code rows, 264 keys):** 1c-only runaways 10 (mbpp/122, mbpp/147, HE/77,
     HEx/113, HEx/116, mbpp/243, mbpp/244, mbpp/304, HEx/83, HEx/99), base-only 3, both 11: exact McNemar p 0.092
     (recomputed), so V1's single-order guard should PASS; if V2 repeats, pooled ~20 vs 6, p ~0.01 = FAIL. A RUNAWAY
     FAIL is plausible again; the 5d branches are likely NEEDED. Kill test: up to 6 discordant keys (either order).
- BigMoE: g1b = g1a + res1c (switches OFF); tonight's key test per env: base / G1 / G1+RES / 1c + COMQOM + RES + int8
  (corrected 14:0x from C6 to 1c's lambda) / 1c lambda 0.8e-2 + int8 (asked). If res1c lands after ~17:00: g1a as armed.
- Breakthrough (b) per-layer lambda: experts-only band arms on 24 code windows (C6 vs lambda_hi on layers 0-15 / 16-31
  / 32-47), ~20 min, Tue after Fit12's kill test, latest start 09:00. (a) chain reach-weighting = already C6's s^d
  (s 0.98 + the s 0.935 scan): no screen. (c) 2-bit tier: PARKED (format change).
