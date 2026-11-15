# CHECK55 levers: a ranked table for Leap (NewIdeas, Sun 15 Nov 2026, 16:2x (`date` 16:21); CPU / desk only)

> **CONTRACT BENCH RESULT (Leap CB1, 58c2439; ~/Leap/data/week98/screens/README.md 'CB1'; recorded 18:38 `date`): PASS on
> both lines, ALL BITWISE.**
> - 376 bit checks across M = 1 / 16 / 32, nt, and both layouts.
> - Standalone, cold, quiet, per check over 7 dense kinds + the HC down GEMM: **16 rows -5.98 ms, 32 rows -6.41 ms**.
> - Chunks by shape (C was NOT one size):
>   - ks 4 for in-proj / o_proj / out_proj / QSA / lm_head: K-chunk 640 at K 2,560, and 1,536 for o_proj at K 6,144;
>   - ks 5 for the shared cat, ks 10 for the shared down;
>   - b | a folded into the in-proj;
>   - S4 at PS 2,048.
> - S4's gain is occupancy (partitions), not bytes: 49 -> 34 us per read in a dependent chain. That agrees with lever 3's
>   in-situ 14 % byte-bound finding.
> - TRANSFER RULE: standalone, so it counts only after the engine A/B that comes with next week's re-certification.

> **LEVER 3 RESULT, Sun 15 Nov 17:37 (quiet ticket 17:36-17:37, pre-registered cceee56; l3_slc_bench.txt / .json):
> NO JUDGEMENT by its rule; KILL in substance.**
> - **STEP 0, the D3-0 reproduction:** S4 33.2 us (D3-0 38.9, inside +-25 %); S4-cached 13.0 us (D3-0 18.7, outside).
>   The pre-registered context gate fails, so line (b) is formally NO JUDGEMENT.
> - **STEP 3, in situ:** the real dependent sequence is evict -> window -> o_proj -> S4. There, S4 costs cold 55.4 us vs
>   bytes-free 47.8 us: only 7.6 us (14 %) of each read is byte-bound, against 52 % in D3-0's independent-call
>   throughput mode.
>   - The ceiling is <= 0.73 ms per 16-row check (96 reads), not the 1.9 ms in this table.
>   - The touch as built is not hidden: touched 77.3 us and late 84.6 us vs cold 55.4. The window costs +25.4 us per
>     touch, about its whole 6.64 MB read (Leap's +5 us line is NOT MET), with bits equal.
> - **Read:**
>   - Even a perfectly hidden touch buys <= 0.73 ms per check. That is under any line worth a better-version search (a
>     smaller-grid touch behind a longer window), so lever 3 is closed by its ceiling, not by one configuration.
>   - STEP 1's throughput-mode gaps for qmm16c (o_proj 36.6 %, in_proj_z 19.8 %, in_proj_qkv 30.7 %) shrink the same way
>     on a dependent path. Staging the projections is not pursued either.
> - **Track A without lever 3:** 16 rows ~57-58 ms (Leap's central) minus lever 2 if it lands (-2.7 est).

> **LEAP'S CORRECTIONS, Sun 15 Nov 17:23 (`date`):**
> - Two of Leap's four 16-row levers were closed at 14:3x (Leap README ATT-HCHUNK, QMVW16):
>   - qmv_wide on more threadgroups: **KILL**. MLX's own launch is already at or near the best (4.12 vs 4.33 ms per
>     16-row check).
>   - attention's 12 head chunks as 2 calls: **PARKED**. Only 2-head chunks are bitwise at both 16 and 32 rows.
> - Leap's track A is therefore hc_b (~-1) + N-2560 per-group partials (-1..-1.5, not built) + step 2 (~-2). **Leap's
>   honest REF2-T central at 16 rows is ~57-58 ms**, unless levers 2 and 3 here land. My ~53-56 counted -3..-4.5 for the
>   three levers, of which only one is left.
> - **Track B's contract bench (7 + 8) is Leap's** (Mon by day): S4's partition size as a parameter; qmm16c / qmm16n at a
>   fixed ks per shape; M 1 / 16 / 32; GB/s cold plus the row bit test across M.
> - **Lever 2 (MoE dense side)** is Breakthrough's optional piece 2. Its owner is the orchestrator's call.
> - **Lever 3's microbench** is NewIdeas' (Leap: YES), after MoonShots' TREE-E2E (~19:30) and before 22:30. Its harness
>   follows Leap's barrier notes:
>   - MLX puts a barrier before any kernel that reads an earlier output, so the touch hides behind ONE kernel per
>     barrier interval;
>   - the eviction step includes o_proj / out_proj's weights;
>   - the stub is a real latency-bound kernel.

> **CORRECTION, Sun 15 Nov 16:25 (`date`), after the orchestrator's 16:3x check.**
> - **Lever 1's premise was wrong.** ISO2 (BM:docs/WED_TESTS.md:2042, Sun 03:07-03:10) was an ENGINE A/B on a GPU-bound
>   A16 check: quiet, 4 configs rotated within each of 12 rounds. neither 64.1 / gdnb 64.3 / dense (REF2_DENSE_NATIVE)
>   64.9 / both 65.0 ms. DENSE-N measured **+0.7 ms**: no gain, not a hidden one.
> - So **lever 1 (DENSE-N) and the GDNB re-test are CONDITIONAL on tonight's full-engine profile** (Breakthrough,
>   deadline Mon 15:00), which decides between two readings: the engine is GPU-bound and standalone gains don't transfer
>   (+1.5 GB of copies, overlap, cache state), or the gate's CPU hides every GPU lever alike. They are out of Monday's
>   daytime kill-tests, and track A is re-totalled without them.
> - **TRANSFER RULE added (the DENSE-N lesson):** a standalone or slice gain counts only after a paired engine A/B.
>   Levers 2, 3 and 7-8 each get a slice A/B (A16, 12 rotated rounds) after their microbench, then the engine arm decides.
> - The SLC size "~24 MB" is an **[est]**: MoonShots IDEAS.md:250, via TRIED "Keep weights in the SLC". Measured instead:
>   Leap's KNEE (4be8cf9) re-read bandwidth by weight-set size: 2-8 MB 2.3-3.2 TB/s; 12 MB 1.1 TB/s; 16-48 MB ~500 GB/s;
>   96 MB+ ~276 GB/s. Lever 3's first bench measures whether one HC read's 6.65 MB stays cached.

The orchestrator's ask, 16:1x: a Goal B research pass for Leap's CHECK55, in two tracks.
- **Track A** runs this week under REF2-T. Each lever keeps every output's arithmetic order, so SD == AR stays
  bitwise.
- **Track B** is next week's bandwidth reference, re-certified ONCE.

Leap decides what gets built; none of Leap's files are touched.

## Sources and pricing method
- **Breakthrough's profile** (~/Breakthrough/docs/CHECK55_PROFILE.md, f425a3b) gives GPU ms by class on the 12-layer
  slice.
  - Engine-scaled: per-layer classes x 4. The head and the single PLE layer are counted once.
  - Floors are the class's weights at 280 GB/s.

  | class (engine-scaled) | A16 ms (floor; x) | C32 ms (x) |
  |---|---|---|
  | dense projections (GDN + attention) | 11.24 (5.48; 2.1x) | 23.8 (4.4x; v32 reads each weight twice) |
  | HC read + write (bf16 mixers) | 10.44 (4.52; 2.3x) | 13.0 (2.9x) |
  | MoE dense side (K-73: router + shared + tail) | 4.16 (0.72; 5.8x) | 6.72 |
  | attention core (no weights) | 3.52 | 7.32 |
  | GDN core (no weights) | 2.20 | 4.44 |
  | head (mixer + lm_head, once) | 1.89 (1.32; 1.4x) | 3.79 |
  | PLE (once) | 0.60 (0.24) | 0.82 |
  | **dense side, total** | **~34** (floor ~12.3) | **~60** |

  Experts are out of scope: they run at 1.23-1.27x their floor, and K90 0b / 0c found no load-side headroom.
- **The record:**
  - Leap's TRIED lines: L1-23 qmm16c (code MMA: raw codes as bf16 via 0x4300 | code, then scale / bias in the
    epilogue); L1-26a / DENSE-N (qmm16n); BP-2a; D3-0; R-09b S4; R-13 OCC; R-13b / c / d; qmm16 tiling sweep; D1-B
    hc_b; GDNB / GDNB-2; R-82; 15a / 15d HC mixer pack; the megakernel KILLs; Rnd16 / "Keep weights in the SLC"; P1
    OVERLAP; K90.
  - Leap's TABLE_350 R-13 OCC (1775).
  - Leap's check55_plan.md 'REVISED Sun 11:3x'.
- **Tags:**
  - already-tried = a TRIED line exists (its verdict follows);
  - published = known outside;
  - untested = a known idea not yet run in this setting;
  - new = not found in TRIED or by search.
  - TRIED was grepped with >= 5 queries per lever: concurren, barrier, overlap, split-k / partition, lossless / pack,
    persistent / megakernel, occupancy, dequant, magic / nibble, SLC, epilogue, split size, reference change.

## The ranked table
Ms are engine-scaled estimates [est] unless measured [M]. C32 numbers assume Leap's step 1 (one weight pass,
one-dispatch cores, seams to 32) has landed, unless the lever is part of it.

| # | lever | track | exact by construction? | 16-row ms | 32-row ms | tag | cheapest kill-test (<= 15 min; bit test; line) |
|---|---|---|---|---|---|---|---|
| 1 | **DENSE-N back in the check now that it is GPU-bound**: qmm16n (fragment-native layout, half the unpack ops) for REF2's 4-bit dense projections | A | yes: bitwise == qmm16c [M] | **-1.9** (x1.20 on 11.2 ms; [M] standalone 8.85 -> 7.39 per check) | -2 if the one-pass 2a kernel gets a native-layout twin; else ~-1 | already-tried: PASS standalone (x1.20), but ENGINE A/B ISO2 +0.7 ms (BM WED_TESTS:2042, GPU-bound). **CONDITIONAL on tonight's engine profile; not a Monday kill-test** | slice A16 + C32, DENSE-N on vs off, 12 rotated rounds, quiet. Bits: logits bitwise. LINE: A16 wall <= -0.40 ms per 12-layer slice (= -1.6 engine). RAM: +1.5 GB unless the native layout REPLACES the stock one for these shapes; check which kernels still read stock |
| 2 | **MoE dense side as ~2 exact dispatches** (router 8-bit GEMV + top-k + shared gate/up/down + tail, K-73's arithmetic verbatim) | A | yes, if K-73's per-output order is kept (bit test) | **-2.7** (5.8x -> ~2x of a 0.72 floor) | **-5** (6.7 -> ~1.7 in one 32-row dispatch) | planned: Leap's step 2 / Breakthrough's piece 2. Profile rule R3 dropped it on share (5-6 %), but in ms it is the largest track-A item after step 1 | Breakthrough's piece-2 kernel alone vs K-73's chain on 3 real layers at 16 and 32 rows. Bits: y bitwise. LINE: <= -0.05 ms per layer at A16 (= -2.4 engine) |
| 3 | **HC SLC STAGING:** while the latency-bound kernels that come after the experts run (MoE tail, shared expert, HC write), a small concurrent "touch" kernel reads the NEXT HC read's mixer weights (~6.65 MB per read; KNEE [M]: re-reads of 2-8 MB run at 2.3-3.2 TB/s; the SLC's ~24 MB is an [est]). The HC read then finds them cached | A | yes: scheduling only, same kernel, same bits | **-1.0..-1.9** (D3-0 [M]: S4 38.9 us cold vs 18.7 us cached; ~24 reads per 12 layers x 20 us = 0.48 ms per 12 layers, so 1.9 engine is the ceiling) | about the same or more (more reads at 32 rows) | **new**, a variant of two KILLs: "Keep weights in SLC" (too small: here only one read's 6.65 MB) and Rnd16 (experts in DRAM-bound decode: here the check's latency-bound windows have slack) | single-kernel bench: [~400 MB streaming kernel = expert-like, evicts the SLC] -> [touch of the HC mixer buffer, concurrent with a latency-bound stub] -> [S4]. Time S4 cold vs touched vs pre-warmed. Bits: S4 output bitwise (same kernel). LINE: touched S4 <= 25 us (>= 70 % of the cached gain) AND the stub +<= 5 us -> engine >= -1 ms. FIRST, the same cold / cached gap for 2-3 qmm16c shapes: if it is < 15 %, staging can't help the projections |
| 4 | Leap's 16-row exact levers: hc_b; GDNB; qmv_wide on more threadgroups; qmm16c N-2560 per-group partials + ordered sum; attention's 12 head chunks as 2 calls | A | yes (each bit-tested) | -0.9..-1.1 / -0.7 / -1..-1.5 / -1..-1.5 / -1..-1.5 (Leap's [est]) | similar | already-tried (hc_b, GDNB built) / Leap's own | Leap's E1. GDNB: ISO2 measured +0.2 ms on the engine, so it is CONDITIONAL on tonight's engine profile |
| 5 | Leap's step 1 for 32 rows: one weight pass (qmm16c2a, qmv_wide at 32 vectors), one-dispatch cores (C7 v32one), seams to 32 | A | yes [M]: C7 bitwise; seams bitwise | 0 | **-10..-14** (weights once) **-10** (cores) **-2.0** [M] (seams) | already-tried / built | Leap's C7b / E2 |
| 6 | **THE REFERENCE CONTRACT** for next week's re-certification (a design rule, not a kernel). For every dense GEMM, the HC read and the head, pin the arithmetic as (a) a FIXED K-chunk C (a multiple of the 64-weight group, e.g. 512), independent of M and of N tiling; (b) inside a chunk, qmm16c's code-MMA order with the per-group scale / bias epilogue; (c) chunk partials summed in increasing chunk order in fp32; (d) one bf16 rounding. Any later kernel that honours (a)-(d) is bitwise, whatever its layout (stock or native), threadgroup map, partition-to-core mapping or prefetch. So there is ONE re-certification, and SD == AR holds by construction because nothing depends on M | B | yes by construction (a new reference, certified once) | enables 7-9 | enables 7-9 | published: fixed split SIZE, not count, for batch invariance (Thinking Machines, "Defeating nondeterminism", 2025; SGLang / vLLM batch-invariant ops); Flash-Decoding split-K. Partly tried: L1-23, R-13d D2, R-13 OCC ("batch-invariant kernels recover ~21 ms but are not bitwise vs shipped"). **New: one contract across dense + HC + head, with C chosen per shape by a sweep** | the bench for 7 and 8 below. LINE: the contract is right if one C per shape reaches >= 80 % of 280 GB/s at M = 16 AND 32, with rows bitwise across M = 1 / 16 / 32, and M = 1 no more than 10 % slower than today's 1-row path (AR, chat, commit) |
| 7 | Dense projections under the contract (native layout + chunk-split for occupancy) | B | yes (contract) | **-4.0..-4.7** (2.1x -> 1.25-1.35x of 5.48) | **-4..-5** (on top of step 1's one pass) | partly tried: R-13 OCC's batch-invariant best was 1.2-1.7x on the big shapes; the tiling sweep got KS 2 / NT 2 to 70 % of the floor on in_proj; R-13d D2 -4.52 [M] standalone | single kernel for in_proj_qkv and o_proj at C in {256, 512, 1024} x M in {1, 16, 32}. GB/s, plus a row bit test across M. LINE as in 6 |
| 8 | HC read under the contract: more K partitions, fixed per shape; then the lossless mixer pack (15d: 12.16 bits per weight, bitwise) once the partitions fix occupancy | B | yes (contract; the pack is lossless) | **-4.5** (2.3x -> 1.3x of 4.52) | **-7.2** (2.9x -> 1.3x) | partly tried: S4 replicates MLX's split-K loop (52 % byte-bound, 171 GB/s); 15d pack PARKED ("bit-identical but slower", 64 GB era, before occupancy was fixed) | S4 with 1x / 2x / 4x partitions at M = 1 / 16 / 32. GB/s, plus the row bit test across M. LINE: >= 85 % of 280 GB/s at 16 and 32 rows. The pack is tested only after that, decode cost vs bytes saved |
| 9 | Attention core by per-row key lists (one call per layer) | B | yes (new reference; it also makes trees exact with no P·V port) | ~-2 [est] (3.5 ms, 112 dispatches per layer) | ~-4..-5 [est] (7.3 ms) | Leap's (check55_plan step 3) | Leap's |
| 10 | Head under the contract (one pass; already 1.4x) | B | yes (contract) | -0.3 | -0.6 (after step 1) | untested in this form | part of 7's bench (lm_head shape) |
| - | lm_head + argmax fused epilogue (deterministic first-max tie-break) | A | yes | ~-0.1 | ~-0.1 | published (fused epilogues) | low: skip unless free |
| - | Not proposed (already KILLed in TRIED, no new angle found): megakernels and grid barriers (barrier 1.97-2.47 us vs a dependent dispatch 1.3-1.6 us); a native per-check loop and ICB / threaded encoding (the check is GPU-bound); MLX's quantized NAX GEMM at 16-row tiles (R-13c: dequant-bound); WIDE16 (R-13b); a second GPU stream (P1 OVERLAP, kappa 0.93); kernel-count cuts alone (Rnd27 / 29: small kernels already overlap big ones); the magic-number nibble->bf16 decode (qmm16c already does it: 0x4300 \| code) | | | | | | |

## Ranking (by ms per unit of Leap's time, then certainty), corrected
- **Monday by day, track A kill-tests:**
  - **3 HC SLC staging:** two 10-min microbenches, then a slice A/B;
  - **2 MoE dense side:** needs a kernel, which Breakthrough's piece 2 could own; then a slice A/B;
  - Leap's own list without GDNB: hc_b, qmv_wide with more threadgroups, qmm16c N-2560 per-group partials, attention's
    12 head chunks as 2 calls.
  - Passes fold into Monday night's E1 (Tue 00:30).
- **Conditional on tonight's engine profile:** 1 DENSE-N and the GDNB re-test.
- **Track A re-totalled** (no DENSE-N, no GDNB; every ms still subject to the transfer rule):
  - 16 rows: ~61-64 -> **~53-56 ms** (hc_b -0.9..-1.1, Leap's three levers -3..-4.5, MoE dense side -2.7, HC staging
    -1.0..-1.9).
  - 32 rows: ~99-102 -> **~68-76 ms** (step 1 -22..-26 [seams -2.0 M], MoE dense side -5, HC staging -1..-2, the 16-row
    levers at 32 rows -2..-4).
- **Track B: get the contract (6) right before the one re-certification.** Its bench (7 + 8, one afternoon, then a
  slice A/B) picks C per shape and proves M-invariance. The dense side then drops a further ~-11 (16 rows) / ~-16
  (32 rows) [est], subject to the same transfer rule. That would reach Leap's ~45 / ~62.

## Caveats
- Every ms here is the slice's class time x 4 against weight floors. The engine's NAX6 experts, the gate and the
  in-check cache state are not on the slice.
- Lever 3 rests on D3-0's cached / cold gap, a standalone measurement, and on the HC mixers not being evicted
  between the touch and the read. Its first bench answers both.
- The contract's M = 1 cost matters for AR, chat 1-2-row checks and the commit replay, which must use the same
  kernels. Its line caps that.
