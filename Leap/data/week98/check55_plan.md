# CHECK55: plan to an exact 32-row check of ~55 ms and a 16-row check toward ~40 (Leap, Sun 15 Nov 10:5x EET)

> (a quote from the project owner, kept private)
unlocks like 3x the tokens").
- The honest size: 32 rows from 99-102 ms to ~55 (floor 52.0), and 16 rows from 61-64 toward ~40 (floor 38.8).
- Wanted: an ordered plan with ms saved per step and a rough ETA; and whether a second pair of hands helps.

All numbers below are estimates [est] unless marked measured [M]. The profile (step 0) will re-order and re-size them.

## Where the time goes today (from the record)
- **Floors** (pipe_price.txt, 280 GB/s):
  - 16 rows: 10.87 GB, 38.8 ms (dense ~3.4 GB + the union's experts ~7.4 GB);
  - 32 rows: 14.56 GB, 52.0 ms (union 72.5 -> 108.5 per layer: +3.7 GB, +13.2 ms).
- **Engine** (G32T [M]):
  - A16 61-64 ms: ~24 ms over its floor.
  - C32 99-102 ms: ~48 ms over its floor.
  - The shape is a step at 17 rows: 16 -> 24 rows costs +34.7 ms, 24 -> 32 only +3.5-5 ms. So 17-31-row checks cost
    about a 32-row check.
  - Past 16 rows, every op still split 2 x 16 starts a second, padded, latency-bound chunk: the GDN and attention
    cores, the 8-bit router and indexer, the shared expert, the small-N projections, the HC reads.
- **Bound by dispatches and latency, not bytes:**
  - On the slice, the graph build is 0.54 of A16 and 0.62 of C32 (C6 [M]).
  - B1 counted ~1,600 kernels + ~480 glue flushes per check at ~7 us of encode each.
  - The built exact byte levers moved the engine <= 1 % (ISO2 [M]: GDNB, DENSE-N). The check's 24 -> 32-row expert
    bytes were partly hidden (+3.5-5 ms vs +6.6 at the floor).

## Ordered plan
| # | step | 16 rows | 32 rows | ETA |
|---|---|---|---|---|
| 0 | **Profile / attribution** on the exact-45 engine at 16 and 32 rows: CPU (Python, MLX encode, gate service) vs GPU busy vs GPU idle; dispatches per layer by piece; per-class GPU time by paired knock-outs (warm pool). No xctrace. | (fixes the order) | | 1 day; Breakthrough can own it |
| 1 | **One launch per op at 17-32 rows** (kills the step): a 32-row gdn_body (two row tiles in one dispatch, the state in registers); attention core per AR class in one call if MLX's fallback is M-invariant (a 2-minute bit test; else stays 2 x 16 until step 3); 32-row replicas of MLX's 16-row qmm for the 8-bit router / indexer, small-N projections and the shared expert (D2 / mr2 machinery); HC read as one dispatch over both row tiles (mg2-style; hc32's two accumulators lost). Exact by construction, each bit-tested, each with a BigMoE engine arm. | 0 | **-15..-25** -> ~75-85 (R32 ~1.2-1.35) | Sun-Tue |
| 2 | **Dispatch diet** (both lengths): MoE dense side (router + shared + tail) in ~2 dispatches; HC pads in-kernel; GDN / attention glue (pads, concats, gates). ~-800..-1,000 dispatches = -6..-10 ms of encode CPU, -1.5..-2 ms of GPU (B1's count). | -5..-8 | -7..-10 | Tue-Fri |
| 3 | **Attention core as one dispatch per layer** (today ~0.6 ms / layer, 144 SDPA calls per check from MTPLX's 12 head chunks). Bitwise vs today means porting MLX's unfused GEMM -> softmax -> GEMM (uncertain). A new invariant reference (per-row key lists) is simpler and also makes trees exact, but changes REF2-T's bits: Kamen's call. | -4..-5 | -4..-6 | 2-3 days after the choice (4-5 for the port) |
| 4 | **Gate round trips + commit / sync tail** with BigMoE: the gate's per-layer CPU service even at 0 misses; GDNB + lazy commit (built, exact, not in the 45 preset; they pay once 1-3 land); the acceptance read. | -2..-4 | -2..-4 | ~2 days |
| 5 | **Bytes at the floor** once the check is bandwidth-bound: experts at 32 rows and dense via DENSE-N native copies (built, exact) at >= 95 % of 280 GB/s. | -2..-5 | -2..-5 (floor-limited) | 2-3 days |

**Landing [est]:**
- 32 rows: ~60 ms central, ~55 if every step lands at its good end (floor 52).
- 16 rows: ~45 ms central, ~41 at best (floor 38.8).
- About two weeks of build with BigMoE's arms. The first visible drop (step 1, 32 rows) is due by Tue.

## Second pair of hands
**Yes.** Breakthrough can own step 0 without touching Leap's files.
- **The work:** read the engine's loop and v32 read-only, in its own copies and scripts. The output is one
  attribution table vs the floors: ms by class, CPU vs GPU vs idle, dispatches per layer, at 16 and 32 rows.
- **Machine:** a ~47 GB quiet ticket for the engine, at night (after a45 and EvalSuite), or the 12-layer resident
  slice (~23 GB, C6's harness) by day.
- **Optional second piece once the profile names it:** the MoE dense side (router + shared + tail) as a standalone
  exact kernel with its own bit test vs K-73's chain. Leap and BigMoE would integrate it.
- **Not the padding-free 17-31-row path.** The cost past 16 rows is the second chunk's launches, not padding, and
  step 1 makes any 17-32-row length one launch anyway.

## REVISED Sun 11:3x, after Breakthrough's step 0 (f425a3b, ~/Breakthrough/docs/CHECK55_PROFILE.md) and C7
**What changed.**
- The exact check is GPU-bound: GPU busy 94 % / 91 % of the wall at 16 / 32 rows. CPU (build + encode) is ~0.35 of the
  wall and hidden beneath the GPU work. +2 ms of CPU moves the wall only +0.3 / +0.8 ms.
- So removing dispatches saves only their GPU share, ~2 us each: step 2 drops to ~-2 ms.
- My 10:5x central (~45 / ~60) assumed a CPU-bound check and is withdrawn.

**GPU by class at A16** (12-layer slice; x = vs the class's weight floor at 280 GB/s):

| class | ms | x floor |
|---|---:|---:|
| experts | 11.27 | 1.27 |
| dense projections | 2.81 | 2.1 |
| HC | 2.61 | 2.3 |
| head | 1.89 | 1.4 |
| MoE dense side | 1.04-1.23 | 6-7 |
| attention core | 0.88 | - |
| PLE | 0.60 | - |
| GDN core | 0.55 | - |

**The 17-row step** (+14.8 ms on the slice):

| part | ms |
|---|---:|
| experts (union 75 -> 108) | +4.5 |
| dense weights read twice | +3.1 |
| head read twice | +1.9 |
| second-chunk cores | +2.8 |
| a late GPU start (the seams fired only at <= 16 rows) | +2.7 |

**Order now:**
1. **32 rows, step 1 complete:**
   - one-dispatch cores (v32one, C7: 1.574 -> 1.430 x A16);
   - one weight pass for the projections and the head (qmm16c2a, qmv_wide at 32 vectors; bitwise);
   - seams to 32 rows (ref2 seam_hi, opt-in; Breakthrough pass S: -2.0 ms).
   C7b times them. Expected: C32 99-102 -> ~84 ms on the engine; the rest above A16 is the expert union's bytes.
2. **16-row dense side, exact levers:**
   - hc_b (built, ~-1 ms); GDNB (built, ~0 in ISO2);
   - MLX's small-shape kernel (qmv_wide) launched with more threadgroups (per-row bits depend only on k_lanes):
     -1..-1.5;
   - qmm16c's low-parallelism N-2560 shapes with per-group terms in parallel and an ordered sum: -1..-1.5, if the
     compiled arithmetic can be matched;
   - attention's 12 head chunks as 2 calls, if MLX's fallback stays M-invariant: -1..-1.5, bit test first.
3. **Step 2,** the dispatch diet: ~-2 ms of GPU.

**Revised central (today's reference, REF2-T):**
- 16 rows ~55 ms (exact ceiling ~52); 32 rows ~76 ms.
- The rest is pinned by the reference's own kernel arithmetic: the HC read's split-K partitions (2.3x its floor,
  ~10 ms of an engine 16-row check) and MLX's small-shape kernels.

**A re-certified reference built for bandwidth** (HC split-K on more partitions, attention by per-row key lists,
small projections on a parallel kernel; SD == AR stays bitwise) would allow ~45 / ~62. The orchestrator has the
question with Kamen; EvalSuite is pricing the re-certification. The detailed price is on hold until he answers.

**Engine A/Bs booked:** Monday night (Tue 00:30-15:00, WEEK98 SLOTS):
- E1: 16-row levers;
- E2: A16 / C32 split / C32 one-pass with REF2_SEAM_HI=32;
- E3: Breakthrough's engine profile with its hook;
- E4: only if needed, a clean DENSE-N + mr2 re-run.

## UPDATE Sun 17:1x: NewIdeas' lever table, the work split, the 16-row central corrected
NewIdeas' desk pass (~/NewIdeas/research/week98/CHECK55_LEVERS.md, c2620a9) ranks track A (REF2-T) and track B (next
week's reference contract) levers.
- **Two of the 16-row levers above are closed** (screens README, 14:3x):
  - (a) qmv_wide with more threadgroups: KILL (QMVW16: MLX's launch is already near-best).
  - attention's 12 head chunks as 2 calls: parked (ATT-HCHUNK: only 2-head chunks are bitwise at both 16 and 32 rows).
- **So the 16-row central under REF2-T is ~57-58 ms, not ~55.** Left: hc_b (~-1), lever (b) qmm16c N-2560 per-group
  partials (-1..-1.5, to build), step 2 (~-2). NewIdeas' levers 2 (MoE dense side as ~2 exact dispatches, -2.7 est)
  and 3 (HC SLC staging, -1..-1.9 est) could bring it to ~53-55. NewIdeas' transfer rule applies to every lever: a
  standalone or slice gain counts only after a paired engine A/B (ISO2: DENSE-N +0.7 ms on the engine). The 32-row
  figure waits on C7c (tonight).
- **Work split:**
  - lever 3 (HC SLC staging): NewIdeas, microbenches Sun evening (Leap's kernels read-only), then a slice A/B;
  - track-B contract bench (levers 7 + 8): Leap, Mon by day. S4 with the partition size as a parameter; qmm16c /
    qmm16n at a fixed ks per shape; M = 1 / 16 / 32; cold GB/s and the row bit test across M;
  - lever 2 (MoE dense side): Breakthrough's optional piece 2 above; the orchestrator assigns it;
  - DENSE-N and the GDNB re-test: conditional on tonight's engine profile.

## CB1 PRE-REGISTERED (Sun 17:3x, before any run): the reference-contract bench (track B, NewIdeas' levers 6-8)
scripts/week98/cb1.py. Standalone kernels, cold weights (>= 128 MB pools per shape), dependent chains of a check's
calls, forms round-robin, median of 7 rotated rounds, on a quiet ticket. First a smoke under a share lock (bits and every
code path).
- **The set:** 7 four-bit kinds at the real shapes and per-check counts, plus the HC read's down GEMM.
  - GDN qkv | z (16,384 x 2,560, x36); GDN b | a (96, x36), or folded into the in-proj (16,480);
  - o_proj + out_proj (2,560 x 6,144, x48); the QSA qkv_fused (13,312, x12);
  - the shared cat (1,281, x48); shared down (2,560 x 640, x48); lm_head (x1);
  - HC down = S4 (324 x 10,240 bf16, x97).
  - Not in it: the 8-bit router / indexer, HC up, PLE, the experts.
- **The contract per shape:** one ks (the K-chunk), the same at 16 and 32 rows (picked on the 16 + 32 sum), any nt or
  layout; one partition size for S4.
  - 32 rows use a new one-weight-pass kernel with the K split (qmm16c2a_ks).
  - REF2-T's forms are the baseline: qmm16c ks 1, MLX's padded call, S4 at 4,096; v32one's forms at 32 rows.
- **Lines (ms per check over this set):**
  - 16 rows: PASS if >= 4.0 saved; BETWEEN 2.0-4.0; KILL if < 2.0.
  - 32 rows: PASS if >= 6.0 saved; BETWEEN 3.0-6.0; KILL if < 3.0.
  - Bits: every config bitwise across M = 1 / 16 / 32, nt and layout. Any failure is a bug: no verdict until it's fixed.
- **Prediction:** 16 rows -5..-7 ms. The parts:
  - o_proj / out_proj at ks 6: ~115 -> ~45 us, about -3.3;
  - S4 at PS 1,024: ~39 -> ~26 us, -1.2;
  - b | a folded into the in-proj: -0.8;
  - the shared cat: -0.5;
  - the in-proj at ks 2: -0.4.
  32 rows: -6..-10 ms.
- **What it decides:**
  - PASS: the contract's dense side is worth the one re-certification on its own. Next: a slice A/B (the transfer
    rule), then the production kernels and next week's certification plan.
  - KILL: track B rides only on HC + attention by key lists.

## CB1 RESULT (Sun 17:36): PASS on both lines, ALL BITWISE (screens README 'CB1')
- **Per check over the set, standalone:**
  - 16 rows: 18.52 -> 12.54 ms (-5.98);
  - 32 rows: 22.52 -> 16.11 (-6.41).
- **Per-shape contract:**
  - ks 4 for the in-proj, o_proj / out_proj, QSA and lm_head;
  - ks 5 for the shared cat, ks 10 for shared down;
  - b | a folded into the in-proj;
  - S4 at PS 2,048.
- **So the contract's dense side is worth the one re-certification on its own** (the decision rule above), subject to
  the transfer rule.
- **Next:**
  1. a slice A/B, A16 and C32, with an opt-in contract installer over ref2;
  2. a native-layout 32-row twin of qmm16c2a_ks;
  3. next week's re-certification plan, together with the HC / attention levers.

## C8 REFC PRE-REGISTERED (Sun 17:4x, before any run): does CB1's gain carry into a real check? (the transfer rule)
scripts/week98/c8_refc.py, with refc.py (an opt-in installer of the contract's forms over ref2 + v32 + v32one; ck.py
holds the kernels). It runs C7c's slice and harness (12 layers, SB 4, own caches, rotated rounds) with four arms:
A16 / A16c (16 rows, REF2-T / contract) and C32o / C32oc (32 rows, step 1 complete, REF2-T / contract).
- **Bits first:**
  - (b1) one 32-row check == two 16-row checks, every logits row, for the contract and for REF2-T (the control);
  - (b2), info only: 16 one-row checks == one 16-row check.
- **Expected:** CB1's per-call savings times the slice's calls give -1.61 ms per 16-row check and -1.57 per 32-row.
  That is an upper bound: refc runs MLX's layout, while CB1's joint picks took the native layout on 3 shapes.
- **Lines:**
  - (b1) not bitwise under the contract -> a bug, no verdict;
  - T16 = -median(A16c - A16) / 1.61 >= 0.5 -> PASS: the contract arms join Monday night's E2;
  - 0.25-0.5 -> BETWEEN: an E2 arm only if the slot has room;
  - < 0.25 -> KILL: no engine arm, profile first.
- **Prediction:** T16 0.6-0.8, T32 0.6-0.9; (b2) bitwise for both.

## BOF PRE-REGISTERED (Sun 17:5x, before any run): lever (b) as a blocked ordered fold (track A, REF2-T's own bits)
scripts/week98/bof.py (qmm16bof) + bof_bench.py.
- **Correction to lever (b) as first written** ("per-group terms in parallel, then an ordered sum"): it can't keep all
  terms. o_proj would need 16 x 2,560 x 96 fp32 terms per call, 15.7 MB, about twice its weight bytes.
- **BOF instead:**
  - BS simdgroups per 32-column tile each compute one 64-weight group's term: the same MMA, row sums and term
    expression as qmm16c.
  - The terms of a block of BS x RG groups go to threadgroup memory (<= 32 KB).
  - Simdgroup 0 folds them in group order before the next block. That is qmm16c ks 1's fold, so the bits should be
    REF2-T's, if the term compiles the same way: four TERM spellings are tested.
- **Lines:**
  - no TERM variant bitwise on all 5 ks-1 shapes -> KILL;
  - saving per 16-row check over the 5 shapes >= 1.5 ms -> PASS (a ref2 switch, a slice A/B Monday, then E1);
  - 0.5-1.5 BETWEEN; < 0.5 KILL.
- **Prediction:** TERM 0 bitwise; o_proj / out_proj 87 -> 50-60 us, in-proj 118 -> 105-110; total -1.8..-2.5 ms.

## C8 REFC RESULT (Sun 18:02): PASS, SD == AR end to end under the contract (screens README 'C8 REFC v2')
- **The slice:** -1.31 ms per 16-row check, -1.05 per 32-row; transfer T16 0.81, T32 0.67 against CB1's upper bounds.
- **Engine estimate:** roughly -4 ms per check at both sizes, against today's REF2-T central of ~57-58 / ~76.
- **Next:** A16c and C32oc join E2 on the engine (Monday night, my 00:30-02:15 slot), as same-run timing arms. The engine
  path needs a check first: K-73's shared_cat and the head may bypass refc's matmul hook. I'll head-test which contract
  forms fire on the real loop.
- The re-certification itself (a new exact reference) stays Kamen's call, next week.

## BOF RESULT (Sun 18:08): PASS with REF2-T's own bits (screens README 'BOF')
- **The result:** -2.22 ms per 16-row check standalone over the 5 ks-1 shapes. The contract's forms on the same shapes
  give -2.85, so most of the dense gain needs no reference change.
- **Track A moves:** with BOF, the REF2-T central at 16 rows drops by its engine transfer, about -1.5 ms at C8's ratio,
  so ~56-57.
- **Track B's remaining edge on the dense side:** S4's partitions (-1.5 standalone), the shared cat (-0.9), the in-proj
  (-0.4); plus HC up and attention later.

## C9 BOF PRE-REGISTERED (Sun 18:1x, before any run): does BOF carry into a real check?
scripts/week98/c9_bof.py with bofi.py (the opt-in BOF installer; no ref2 edit). The 12-layer slice, A16 vs A16b (BOF on
o_proj / out_proj, QSA, shared down, lm_head), 24 rotated rounds.
- **Bits:** A16b's logits == A16's on every row.
- **Expected:** -0.66 ms per 16-row slice check (bof_bench's per-call savings x the slice's calls).
- **Lines:** logits not bitwise -> a bug, no verdict; T = -median(A16b - A16) / 0.66 >= 0.5 -> PASS (an E1 arm Monday
  night); 0.25-0.5 BETWEEN; < 0.25 KILL.
- **Prediction:** T 0.6-0.9.

## BOF2 PRE-REGISTERED (Sun 18:1x, before any run): BOF's 32-row twin (REF2-T's bits, one weight pass)
bof.qmm16bof2 + bof2_bench.py. Same 5 shapes as BOF; the baseline is REF2-T's 32-row forms (v32one: qmm16c2a ks 1 for
N x K >= 8M, else qmm16c2).
- **Lines:**
  - not bitwise (vs qmm16c ks 1 per half and vs REF2-T's 32-row form, at 32 and 24 rows) -> KILL;
  - saving per 32-row check over the 5 shapes >= 1.5 ms -> PASS (with BOF into E2's C32 arm);
  - 0.5-1.5 BETWEEN; < 0.5 KILL.
- **Prediction:** o_proj / out_proj 112 -> 55-65 us; total -2..-3 ms.

## C10 BOF2-O PRE-REGISTERED (Sun 18:2x, before any run): BOF2 on o_proj / out_proj only, 32 rows
scripts/week98/c10_bof2o.py (bofi.TABLE32). This pick came after BOF2's data (o_proj / out_proj 113.1 -> 73.6 us), so it
is a new test with its own line.
- **Expected:** -0.47 ms per 32-row slice check (12 calls x 39.5 us).
- **Lines:** logits not bitwise -> a bug; T >= 0.5 -> PASS (into E2's C32 arm); 0.25-0.5 BETWEEN; < 0.25 KILL.
- **Prediction:** T 0.6-0.9.

## C9b / C10b PRE-REGISTERED (Sun 18:2x, before any run): C9 / C10 without the cold-read confound
- **What changes:** an untimed check of each round's tokens on a throwaway cache before the timed arms (C9_WARM = 1,
  the default now). C9 (BETWEEN, T 0.46) and C10 (KILL, T -0.05) stand as registered.
- **Lines, the same as C9 / C10:**
  - bits;
  - whole-wall T = -median(paired) / expected (C9b 0.66 ms, C10b 0.47 ms);
  - >= 0.5 PASS (C9b: an E1 arm; C10b: into E2's C32 arm); 0.25-0.5 BETWEEN; < 0.25 KILL.
- **Prediction:** the whole-wall pairs close to C9 / C10's after-build pairs (-0.49 / -0.28 ms), so T ~0.7 / ~0.6.

## C9b / C10b RESULT (Sun 18:32): both PASS (screens README 'C9b + C10b')
- **BOF at 16 rows:** -0.75 ms per slice check (T 1.14).
- **BOF2 on o_proj / out_proj at 32 rows:** -0.50 ms (T 1.06).
- Both are REF2-T's own bits, so no reference change. Engine-scaled about -2.5 / -2 ms [est].
- **Monday:** BOF joins E1 and BOF2-O joins E2's C32 arm, next to the contract arms.
- **Revised REF2-T central [est], subject to the engine A/B:** 16 rows ~57-58 -> ~55-56; 32 rows (C7c tonight, then
  E2) ~76 -> ~74.
