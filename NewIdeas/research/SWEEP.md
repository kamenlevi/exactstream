# THE SWEEP: old "lossy" and error-bar kills, re-priced under today's FAST stack (v1.3 FINAL LIST, Thu 5 Nov 2026 07:0x; v1.2 Wed 19:5x; v1 14:4x)

> (a quote from the project owner, kept private)
are the kinds of things which we should be searching for." Every KILL / PARKED from before the 11-02 two-track rule
that was judged on an exact or error bar, and every published method set aside as "lossy", priced under today's FAST
stack. The candidates with sources are in research/SWEEP_CANDIDATES.md (33 rows).

Status: v1.3 (Thu 07:0x) = the FINAL RANKED LIST below; the sections after it are v1.2's working (Wed 19:5x). The quality column uses screen 7's measured terms (scripts/sweep_engine.py,
data/ledger/sweep_engine.txt). FRAME FIX: v1.1 (19:5x) added the 3-bit tail's cost (E29) to every row, judged against
the 4-bit model. But the FAST stack certifies against REF2-T (EvalSuite PREREG_f1), and REF2-T itself serves ranks 2-9
at 3 bits (4433333333). So the verdict frame is amendment 2's in-frame one: MSB-3 rows pay no E29, and a store pays
(r - 1) x E29. v1.1's numbers stay in that file as the conservative column. EvalSuite confirmed the reference (20:06):
FAST same-scores tests compare against REF2-T's rows (4433333333), not REF2b; w24t's plan is that one; after a NAX6
PASS, NAX6's rows (same tiers). MoonShots re-derived v1.1 (413d214, within +-0.2 points) and checked the frame
(cd7e1f0): sound for row 0. MSB-3 is the reference's own bits, so row 0 pays no ranks 2-9 term under any view. For
the independent quantizers (rows 1-4), two views differ enough to change verdicts:
- credit, (r - 1) x E29: the expected-score view, since a store nearer 4-bit should score at least as well as REF2-T;
- strict, (r + 1) x E29: disagreement with REF2-T, because the errors are not nested in MSB-3.
Both are shown until a same-scores run on a real store says which holds. Every number below is [P] (a replay or arithmetic on measured numbers) or [G] (a model). Nothing
goes to Kamen before MoonShots' independent re-derivation (the orchestrator's rule). DRAFT17 already has it (x0.723 vs
x0.706).

## FINAL RANKED LIST (v1.3, Thu 5 Nov 07:0x)

Ranked by what to do next. Bytes are SSD MB per kept token in decode at 24 GB (code / IFEval / multi), my replay's
convention (11-bit parts, best RAM split). Flips are measured on the decision map unless marked [G]; lines 1.34 / 5.42 /
3.69 %. "Re-derived" = an independent second computation exists (MoonShots / Breakthrough / Cloud3 / Fit12, named).

| rank | candidate | role | 24 GB bytes | flips % (verdict) | re-derived? | next |
|---|---|---|---|---|---|---|
| 1 | today's FAST stack (MSB-3 11-bit + POOLSPLIT + DRAFT17), row 0 | the 24 GB CODE route | 19.5 / 14.3 / 10.6 | 1.08 / 7.61 / 5.98 in-frame (PASS on code) | yes: bytes MoonShots M-35; screen 6 Breakthrough | EvalSuite same-scores (the FAST adoption path) |
| 2 | the chat gate rule: QOM cf 0.5 + bonus gate 0.5, 4-bit sure rows, row 10a | the 24 GB CHAT route (per request) | 61.8 / 87.9 / 87.2 | **0.71 / 4.30 / 2.73 (PASS x3)** | yes: bytes MoonShots c388167; flips Breakthrough 8a473b0 | EvalSuite same-scores on chat |
| 3 | route + unsure rows at top-8 + REACH-CUT 0.5 (Cloud3 C3-R15b) | a cheaper chat route | 49.3 / 57.1 / 54.0 (x0.65 / 0.62 on chat) | [G] ~0.74 / 4.65 / 3.27 (Cloud3) | bytes yes (Cloud3 + mine, within ~10 %) | map arm Fri 00:30 (pre-registered QUEUE Thu 07:0x) |
| 4 | route + unsure rows at top-8 (Cloud3 C3-R15) | a cheaper chat route | 54.3 / 68.2 / 64.0 (x0.78 / 0.73) | [G] within +0.1 of the route (Cloud3) | bytes yes (as above) | map arm Fri 00:30 |
| 5 | drafted 0.3 + bonus gate 0.4 | a cheaper chat route | 53.9 / 62.4 / 64.0 (x0.71 / 0.73) | [G] ~0.75 / 5.2 / 3.5, a coin flip | yes: MoonShots M-42 52.8 / 62.0 / 64.2, x0.71 / 0.73 (cd5f64b) | map arm Thu after DENSE-3, only if a share slot is free |
| 6 | FAST 3-bit prefill, row 5 | the prompt's bytes (prefill is 62-86 % of process bytes at 24 GB) | prefill x0.74 (code 117.9 -> 87.4) | unknown | bytes no | a map arm with 3-bit prompt rows (harness work) |
| 7 | GPTQ-3 g64, row 4 | same bytes as row 0, better 3-bit code | 19.5 / 14.3 / 10.6 | [G] credit 0.94 / 6.85 / 5.03 from its MSE ratio, but see below | bytes yes (M-35) | only after a KL screen (Hessians for all experts: the capture question) |
| 8 | plane-aware prefetch, row 7 | engine reads -5-9 % | - | none (prefetch only) | no | an engine arm (BigMoE) |
| 9 | gate-mass-adaptive tiers, row 9; trunk diet, row 6 | ~x0.9 / ~x0.96 | - | small, unknown | no | gate masses / a map arm; low priority |

Fable C's audit (Thu 07:05, ~/FableGod/fablec/AUDIT_1105.md item 5) on ranks 3-4:
- quality: R15 expected +0.2-0.4 points on chat (PASS ~70 %); R15b's multi margin is ~0.1 point (PASS ~50 %);
- speed: the raw byte cuts are worth only +2-4 % chat tok/s at 24 GB, since SSD is ~6 of ~70 ms per chat check.
The verdict lines were amended to the audit's before any run (QUEUE Thu 07:1x).

CLOSED (no store build, no disk):
- The trellis family (rows 1-3: TRL-2, the 2.40 b mix, TRL-3). Screen 7 KILL: 1.45 / 9.54 / 10.44 % in-frame,
  re-derived by Breakthrough. Fit12's judge on the map's own inputs shows the inputs do not explain the miss.
- The cf 0.3 chat route: 0.74 / 5.56 / 3.86 %, BETWEEN on chat, a near miss. Both split hybrids die on bytes.
- 3-bit sure rows on the chat route (10b, [G] BETWEEN on chat).
- The 2-bit warm tier (row 8): cannot reach the chat budget in the model. Plus the subsumed / dead list below.

What it adds up to:
1. **Code at 24 GB:** the cheapest measured PASS is today's stack at ~19.5 MB per kept token. No store gets code under
   10 MB with a PASS: the trellis route is closed, and GPTQ-3 only changes quality, not bytes.
2. **Chat at 24 GB:** the gate rule PASSES at ~88 MB. Cloud3's two cuts may bring it to ~54-68 MB (Fri 00:30).
3. **Data-aware codes need KL screens.** On identical reads and inputs, the trellis store cost ~1.7-3.2x more KL per unit
   of expert-output error than truncation (~1.7-3.5x across both measurements; re-derived by MoonShots M-43, cd5f64b: direct ratio 3.18 / 1.65 / 2.34x;
   code noisy, +-1.3; IFEval / multi clear 1 by > 2 SE; MSB-2's error ratio measured at 4.98-5.10x, as assumed).
   GPTQ-3's MSE advantage may not survive a KL screen.
4. **Prefill dominates the FAST stack's process bytes** for these prompts. If Kamen's 10 MB line counts the prompt, the
   next wall is prefill, and FAST 3-bit prefill (x0.74) is only a start.

**Re-derivation (MoonShots 14:4x, its own replay scripts/sweep_stores_check.py on Breakthrough's capture checks, QUEUE
M-35, d05b2ac):**
- 24 GB is CONFIRMED on all four stores (x0.99-1.04; e.g. TRL-2 4.0 / 3.1 / 2.8 vs mine 4.0 / 3.0 / 2.7). So are the
  prefill shares (120.9 / 44.3 / 17.0 vs 117.9 / 43.1 / 17.0).
- 12 GB code is CONFIRMED (x0.99-1.03).
- **12 GB chat cells are stand-in-dependent.** Both replays stand in for the rows wasted after a wrong draft (their true
  routing is not captured): mine uses another row of the window (p + 97), MoonShots' default a far row (p + R/2).
  - With the far row its cells are x1.06-1.17 of mine; with my stand-in they agree within 2 %.
  - With true-path stand-ins (the next positions' own routing) the 12 GB cells drop x0.66-0.88.
  - So the 12 GB chat prices carry a spread of roughly -34..+17 %. The 24 GB cells hold within 7 % under any stand-in.
  - What would pin it: a live capture of verify passes that routes the wrong-draft rows.

## How each candidate is priced

- **Bytes [P]:** the ledger replay (live-cut checks of the 36 decision-map windows, plain LRU per layer). The stack is
  QOM cf 0 + every last row gated (DRAFT17), so every row is sure and loads its top-2. One pool per layer holds the
  store's base records at equal RAM: 24 GB = 147 x 2,162,688 + 46 x 770,048 B per layer, 12 GB = 51 / 32. Unit: SSD MB
  per kept token in decode (code / IFEval / multi). Scripts: trellis_draft17.py, draft17.py, pool_split.py,
  prefill_bytes.py.
- **Quality [G]:** a KL model built from measured map arms (scripts/map_kl.py, data/ledger/map_kl_fit.txt).
  - Flips = a + b sqrt(KL) per domain over ten full-model arms (rms 0.07 / 0.37 / 0.15 points).
  - KL at the lines (1.34 / 5.42 / 3.69 %): 0.0082 / 0.0164 / 0.0114.
- **The KL model's terms**, measured at the 45 GB pool unless noted:
  - base (QOM cf 0 + gated last row, 4-bit sure rows) = qom_0_live_bq: 0.0013 / 0.0093 / 0.0101.
  - ranks 0-1 of sure rows at a store with error ratio r vs MSB-3: r x 0.0020 / 0.0113 / 0.0128 (the measured MSB-3
    term bq_t3 - bq; MSB-2's measured term fits r ~3.9-5.2).
  - ranks 2-9 at ratio r instead of MSB-3: (r - 1) x 0.0017 / 0.0071 / 0.0078 [G]. CORRECTED 14:5x: v1 used t4_all's KL
    here, but t4_all is the t-path's numerics floor, not this term. The map runs D-02a's 4-bit model, so in C's form the
    kept ranks 2-9 sit at 4 bits as a stand-in for the engine's MSB-3. The guess is 0.8 x (t3_all - t4_all - the ranks
    0-1 term), and screen 7's s7_ref3 - s7_ref measures it tonight.
    MEASURED Wed 18:4x (s7_ref3 - s7_ref, 239 slots): E29 = +0.0011 / +0.0065 / +0.0089 (paired z 3.4 / 9.6 / 13.6
    per row, 1.8 / 7.9 / 9.6 per window; data/ledger/s7_read.txt). It replaces the guess: a store pays
    (r - 1) x E29(c), E29(c) = E29 x keep(c) / keep(239). It pays this against REF2-T, whose ranks 2-9 are at 3 bits.
    Against the 4-bit model it would be r x E29(c) (the conservative column).
  - the pool: (1 - g) x 0.0024 / 0.0180 / 0.0084, where g = the replay's keep gap closed toward the 45 GB pool. The
    147-slot cost is measured (C at 147 minus bq_t3). v1.1: a third anchor is measured, s7_ref (C + DRAFT17 at 239
    slots): 0.0040 / 0.0232 / 0.0236. It closes 0.89 / 0.86 / 0.92 of the KL gap where the keep proxy says 0.67 /
    0.75 / 0.81, so v1's linear-in-g pool term was pessimistic for big pools. v1.1 interpolates the map-frame C form
    piecewise in g between the three measured anchors (147: screen 6; 239: s7_ref; 45 GB: bq_t3 + DRAFT17).
  - **Check of the r scale (v1.1):** r is an MSE ratio vs the 4-bit weights (Fit12's E4 units; MSB-3 = 3 E4, MSB-2 =
    15 E4, so MSB-2 is r 5). s7_cal_all (all 48 layers at MSB-2) measures 0.0180 / 0.1094 / 0.1465; the model says
    0.0177 / 0.1004 / 0.1192 (implied r 5.10 / 5.51 / 6.26). Linear in MSE holds on code; chat runs 8-19 % above it
    at this large an error, so chat cells for stores with r > 1 are optimistic.
  - DRAFT17 on code: +0.0004 MEASURED (screen 6, Wed 18:0x: C 0.0057 -> 0.0061; v1's [G] was +0.0006). Code flips
    unchanged (1.07 %, paired 12 / 12).
  - Check: the model gives C at 147 = 0.0057 / 0.0386 / 0.0313, as measured (by construction).

## Ranked (24 GB first; the BYTES10 line is <= 10 MB per kept token in decode)

The 12 GB column's IFEval / multi cells are stand-in-dependent (-34..+17 %, see the re-derivation above); its code
cells and the whole 24 GB column are re-derived.

| # | candidate (old verdict, bar) | FAST form | 24 GB MB per kept [P, re-derived] | 12 GB [P] | est. flips % [G], verdict frame v1.2 (lines 1.34 / 5.42 / 3.69; conservative: sweep_engine.txt) | what it needs |
|---|---|---|---|---|---|---|
| 0 | today's FAST stack (reference): MSB-3 11-bit + POOLSPLIT + DRAFT17 | - | 19.5 / 14.3 / 10.6 | 52.0 / 54.4 / 39.4 | 1.08 / 7.61 / 5.98 (PASS / BETWEEN / KILL; +-0.1, an uninterpolated stretch) | screen 6 PASS (18:0x) |
| 1 | B10 TRL-2 (KILL 10-25, 1.5x MSB-3 error bar) | a FAST-only store, 276 slots | **4.0 / 3.0 / 2.7** | 21.7 / 18.0 / 13.0 | credit 1.19 / 7.87 / 7.20 (PASS / BETWEEN / KILL); strict 1.40 / 9.12 / 8.66 (BETWEEN / KILL / KILL); chat +0.1..+0.5 possible | screen 7's s7_trl (the mix) first; full encode ~31 GB disk |
| 2 | the 2.40 b mix (gate / up TRL-2, down TRL-3; same B10 bar) | a FAST-only store, 239 slots | **7.2 / 5.0 / 4.1** | 28.5 / 25.3 / 18.4 | **measured (screen 7, s7_trl, the pre-registered in-frame verdict): 1.45 / 9.54 / 10.44 (BETWEEN / KILL / KILL)**; est_abs 1.54 / 10.07 / 10.98, strict 1.63 / 10.57 / 11.48. At the 4 layers the damage is 1.6-2.5x the MSE model's: r_eff 2.72 / 2.55 / 3.15 vs r 1.62-1.87 (v1.2's [G] was credit 1.13-1.21 / 7.63-8.09 / 6.79-7.28) | s7_trl landed 22:20 (data/ledger/s7_read.txt; re-derived by Breakthrough's read-out 8a473b0): KILLED as a 24 GB route (code BETWEEN, chat KILL) |
| 3 | TRL-3 / T3R (PARKED / KILL on a 12 GB bytes line; quality passed: ranks 2-9 at 0.5x MSB-3) | a FAST-only store, 185 slots | 15.8 / 11.4 / 8.5 | 44.9 / 45.2 / 32.8 | credit 0.86 / 6.32 / 4.61 (PASS / BETWEEN / BETWEEN); strict 1.11 / 7.74 / 6.54 (PASS / BETWEEN / KILL) | the best chat quality of any store; ~47 GB disk; 3INST decode at ~3c speed |
| 4 | GPTQ-3 g64 (HOBBIT-style copy; "lossy", T:368 0.61x MSB-3 at 3.5 b) | a FAST-only store, 171 slots (MSB-3's size) | 19.5 / 14.3 / 10.6 | 52.0 / 54.4 / 39.4 | credit 0.94 / 6.85 / 5.03 (PASS / BETWEEN / BETWEEN); strict 1.17 / 8.16 / 6.83 (PASS / KILL / KILL); +-0.1 | the cheapest encoder (GPTQ, not trellis); quality at no byte cost |
| 5 | FAST 3-bit prefill (KILL on paper 10-20, exactness) | prompt rows read MSB-3 only | prefill x0.74: code 117.9 -> 87.4 | - | unknown (the prompt's KV at 3 bits; not on the map) | a map arm with 3-bit prompt rows (harness) |
| 6 | dense / trunk diet, e.g. 8-bit HC + PLE (KILL on hard KLD) | -627 MB of trunk to expert slots (+3.7 %) | ~x0.96 [P, elasticity ~1] | ~x0.96 | small, unknown | a map arm (HC8 was FAST-judged already) |
| 7 | Rnd19#5 plane-aware prefetch (PARKED, never run) | prefetch only the MSB-2 plane of predicted experts | engine reads -5-9 % (prefetch waste 18-26 % at 24 GB [K]) | - | none (prefetch only) | an engine arm (BigMoE) |
| 8 | B9 / B7 2-bit warm tier for ranks 4-9 (killed by a bytes line; quality never measured) | under QOM: low-bit residents raise the rank 2-9 keep | not priced (needs a two-tier replay) | - | the chat lever in principle (the pool term is 0.0180 of IFEval's 0.0386) | a two-tier replay design |
| 9 | Rnd19#2 gate-mass-adaptive tiers / HOBBIT on-miss low precision (PASS on 35B, never built) | rank 1 at 2 bits when its gate mass is small | ~x0.9 [G] | ~x0.9 | between C and lever 6 for those rows | gate masses (not in the routing files) |
| 10a | chat's gate threshold: QOM cf 0.5 + the bonus row gated at 0.5 (Fable C's 1(c), measured Wed 19:38) | sure rows read their top-2's LSB parts (4-bit, = REF2-T's ranks 0-1); unsure rows = REF2-T's reads; split Cl 69 / Cm 147 | 61.8 / 87.9 / 87.2 (EXACT at its best split 190.6 / 176.8 / 155.1) [P, re-derived: MoonShots c388167 within 0-3 %, far-row stand-ins +2..+6 % on chat] | - | **measured 0.71 / 4.30 / 2.73 (PASS / PASS / PASS)** on the map at 147 slots (qom_0.5_live_c147_bonus5g, Wed 22:14; read scripts/chat_arm_read.py, data/ledger/chat_arm_read.txt; first read, independent re-derivation pending); [G] was 0.65 / 4.34 / 2.85. Against 1(c) at the 45 GB pool, the small pool costs +1.42 IFEval / +0.82 multi points (McNemar p 5e-6 / 0.003) | a per-request rule (code cf 0, chat cf 0.5). The cf 0.3 arm (bonus row gated at 0.3; 49.2 / 51.7 MB on chat, x0.56-0.59) measured 0.74 / 5.56 / 3.86 % (PASS / BETWEEN / BETWEEN, a near miss, Wed 22:42; QUEUE 22:4x), so the PASS route stays cf 0.5 |
| 10b | the same with 3-bit sure rows (t3) | no LSB reads on sure rows; Cl 23 / Cm 163 | 44.5 / 73.0 / 75.9 [P, re-derived c388167] | - | 0.92 / 5.56 / 4.35 (PASS / BETWEEN / BETWEEN; two perturbations on the same rows, so chat may be super-additive (MoonShots)) | - |

**Screen 7 caveat for rows 1, 3, 4 (Wed 22:2x):** their quality cells rest on MSE ratios vs MSB-3. The measured trellis mix
did 1.6-2.5x the damage its MSE ratio predicted (r_eff 2.6-3.2 vs 1.62-1.87, on all three domains). So TRL-2 (row 1,
r 1.85 by MSE) is expected no better than the mix: KILL on chat, code BETWEEN at best. TRL-3 / GPTQ-3 (rows 3-4, r
0.52 / 0.61 by MSE) may also land worse than shown. None of them is ranked until it has its own map arm.
**The trellis family stops (Wed 22:4x, pre-registered test 1):** on the map's own answer rows, the store's whole-expert
output error is only 1.1-1.3x MSB-3's on code and 1.6-1.8x on chat (Fit12's judge). So the inputs do not explain the
miss: the KL damage per unit of output error is 1.4-2.4x higher than for truncation (or the two arms touch different
rows; asked). Rows 1-3 are closed. Row 4 (GPTQ-3, also data-aware) carries the same risk.

Subsumed or dead under today's stack (no re-test):
- Subsumed by QOM at cf 0, which already drops every non-resident rank 2-9 expert on a sure row: dynamic / adaptive
  top-k (DYNK, top-8, ACE, M-09d), UNIQ, expert substitution / cache-aware rerouting.
- Dead on physics (each measured): sketch / low-rank shadow tiers (the expert spectrum is flat: r32 0.920), activation
  sparsity inside experts (0.769 energy at 30 % neurons), expert merging / shared bases (0.20-0.36 % shared energy).
- Out of bounds: static pruning (all 512 experts stay).
- Small or superseded: super-block scales (~5 %; 11-bit already x0.955), salient-group refinement, ASTC / BC texture
  formats, 12-bit scale fields, RARE2 (-4.3 % at 45 GB), POM / LP1 (smaller reads, not fewer; speed KILL).
- Plane-granular caches / LSB-on-miss: their FAST bound is on the map already (3-bit sure rows, t3_all).

## What the sweep says

1. **Code at 24 GB has no measured PASS route under 10 MB per kept token (screen 7, Wed 22:20).** The trellis mix
   (7.2 MB on code) measures 1.45 % code flips in the pre-registered in-frame read (line 1.34: BETWEEN) and KILL on
   chat (9.54 / 10.44 %). Its damage is 1.6-2.5x what its MSE ratio predicted (r_eff 2.6-3.2 vs 1.62-1.87), so
   TRL-2 alone (4.0 MB) is expected no better. The cheapest measured code PASS stays today's FAST stack at ~19.5 MB
   [P] (1.08 %, in-frame). (v1.2 had est. 1.13-1.21 % credit / 1.34-1.41 % strict [G], from MSE ratios.)
2. **Chat is quality-bound at 24 GB, and no single store reaches PASS in the model [G].** The KL budget at the IFEval
   line is 0.0164, and the base alone (QOM + the gated row) uses 0.0093. The two other terms:
   - the sure rows' ranks 0-1 precision (0.0113 at MSB-3);
   - the pool (0.0139 at 171 slots).
   Better codes (TRL-3, GPTQ-3) cut the first; more slots (TRL-2, the mix) cut the second, but at worse precision. The
   best chat stores (TRL-3, GPTQ-3) land BETWEEN on both chat domains at 8.5-14.3 MB (multi KILL in the
   conservative frame).
   **A chat PASS route exists at 24 GB, measured on the map (Wed 22:14): the gate threshold (row 10a, QOM cf 0.5 with
   4-bit sure rows, 147 slots) at 0.71 / 4.30 / 2.73 % flips (lines 1.34 / 5.42 / 3.69) and ~88 MB per kept token on
   chat [P].** That is about half of EXACT's 155-177 MB and ~6-8x today's FAST stack. Code keeps
   cf 0 (19.5 MB, PASS), so the engine would pick the rule per request.
3. **The FAST stack's 3-bit records are a truncation of the EXACT 4-bit codes,** kept nested so EXACT = MSB-3 + LSB. A
   re-quantized 3-bit code has ~half the error at the same or smaller size (GPTQ-3 0.61x, TRL-3 0.50x). FAST does not
   need the nesting, so this is quality at no byte cost.
4. **In the FAST stack, prefill now dominates process bytes for these prompts:** at 24 GB, prefill is 117.9 / 43.1 / 17.0
   MB per kept token vs decode 19.5 / 14.3 / 10.6 (86 / 75 / 62 % of the total) [P]. BYTES10 counts decode only. If
   Kamen's 10 MB counts the prompt too, the next wall is prefill, and FAST 3-bit prefill (x0.74) is only a start.
5. The 11-02 sweep (FAST_TRACK) took only kills "not on quality", which by design filtered out every KLD and error-ratio
   kill. That is why B10 / T3R / TRL-3 / GPTQ-3 were never re-priced as FAST copies. The answer to "why didn't we find
   it earlier" is that filter.

## Why chat stays BETWEEN at 24 GB (the KL model's budget, [G])

v1.2 (verdict frame, measured terms): even at the 45 GB pool, C's form costs 0.0206 of IFEval's 0.0164 budget. A
store there would need r <= ~0.77 (MSE vs MSB-3) to pass. At 24 GB (171 slots, 0.0339) only r ~0 reaches the line,
and that is a 4-bit store, which holds just 126 slots there. So chat PASS at 24 GB needs a different rule, not a
better store. Fable C's gate screens landed 19:38
(Breakthrough ba53ede): 1(b) costs nothing (PASS, null); 1(c), the bonus row gated at cf 0.5 on QOM cf 0.5, costs
+0.12 / +0.20 / +0.28 points but sits at 0.48 / 2.88 / 1.91 % absolute (map frame, 45 GB pool), well under every
line. That rule's bytes at 24 GB are not priced yet. The v1 reasoning below is in the map frame.

The KL budget at each line, minus the fixed base (QOM cf 0 + the gated last row, 4-bit sure rows, measured at the 45 GB
pool): IFEval 0.0164 - 0.0093 = 0.0071; multi 0.0114 - 0.0101 = 0.0013.
- IFEval: with every sure-row expert from one store of error ratio r, PASS needs r x 0.0113 + (r - 1) x 0.0071 +
  (1 - g) x 0.0180 <= 0.0071.
  - TRL-3 (r 0.5) needs the keep gap >= 72 % closed (it gets 36 % at 24 GB).
  - GPTQ-3 (r 0.61) needs >= 83 % (it gets 23 %).
  - A 4-bit store (r 0, every kept expert at 4 bits) needs >= 21 %. But its records are 2.79 MB, so 24 GB holds only 126 slots (worse than 147), and chat's bytes go to ~35-47 MB.
- multi: QOM + the gated last row already use 0.0101 of 0.0114. The gate alone costs multi +1.5 points (3.26 vs
  1.77 %), and it is also what makes chat's bytes small (19.4 vs 84.4 MB at 24 GB). So multi cannot PASS at 24 GB with
  the bonus row gated at cf 0, whatever the store.
- The lever that is left for chat is the gate's threshold: gate the bonus row only when its next draft is likely
  right. Those are Fable C's screens 1(b) / (c) (q1u_0.9_live_bonus9g, qom_0.5_live_bonus5g), already in the
  orchestrator's queue. They price chat's bytes-for-quality trade directly. The two-tier store (row 8) cannot reach the
  budget in this model, so it stays parked.

## Next (cheapest first; v1.3)

- Fri 00:30 (Breakthrough's map slot, after Cloud3's tile arms): C3-R15 and C3-R15b. Masks are built and bit-checked;
  hand-off after Fable C's ~09:00 audit.
- Thu, only if a share slot is free after DENSE-3: drafted 0.3 + bonus gate 0.4 (pre-registered Wed 22:4x).
- EvalSuite same-scores for rows 1-2 of the final list (the adoption path; its queue, not mine).
- Designs, not runs: the 3-bit-prompt map arm (row 5), and a GPTQ-3 KL screen (needs a Hessian capture plan).

(v1.2's Next, kept for the record:)

- The 3-bit tail stays: s7_ref3 PASSED on code (18:4x, 1.04 % at 239 slots). Flag 4 would have cost ~2x bytes
  (W4PRICE 16:3x), so it is not needed.
- Screen 6 PASS (18:0x). Screen 7 done (22:20): the trellis mix is BETWEEN on code and KILL on chat, in-frame
  (data/ledger/s7_read.txt, re-derived by Breakthrough 8a473b0).
- Chat at 24 GB: price the gate-threshold rule (QOM cf 0.5 + the bonus row gated at 0.5: 0.48 / 2.88 / 1.91 % on the
  map) in bytes at 24 GB, then its engine-frame quality.
- MoonShots: re-derive rows 1-4's bytes (the trellis / GPTQ-3 / TRL-3 store prices) before any of them goes to Kamen.
- Screen 7 did not PASS, so the full mix store is not proposed as is; the orchestrator asked for a selective-store
  rescue price (22:3x). TRL-3 / GPTQ-3 rest on MSE ratios that screen 7 shows can miss by ~2x for a data-aware code;
  each needs its own map arm before it is ranked (a GPTQ-3 store needs Hessians for all experts: the capture question).
- A two-tier replay (row 8) and a 3-bit-prompt map arm (row 5): designs, not runs, until the above land.
