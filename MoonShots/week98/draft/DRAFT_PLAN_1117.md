# DRAFT LINE plan (MoonShots, co-owner; NewIdeas 07:5x under BREAK98) - Tue 17 Nov 2026 08:1x

## STATUS Tue 11:1x (supersedes the dated plan below)
- The orchestrator (09:4x) set r57's draft = t30b at **Q8 + 32K**, on my Q8 PASS. BigMoE's r57 tree-16 loop (from
  Wed) takes my sf_tree.py --real ops as its spec, with **sync-free** as the default (bitwise, 176 + 120 trees).
- fused2 is OUT for that loop: it was built for the stored head on a chain.
- Real-format pop: **1.11 ms sync-free** vs 1.60 host-synced, i.e. **~16.6 ms per tree-16 block** [K
  draft/sf_bit_real_*.json]. It is GPU-bound: ~0.56 ms of bytes + latency-bound chains [K draft/sf_gprof_he_t8.json].
- Stage fusion pays little at B = 1:
  - fused attention (bitwise per call, <= 512 keys): -0.02-0.04 ms per pop;
  - fused tail: ~0 and not bitwise;
  - mx.compile: -3.5 %, not bitwise.
- Remaining potential ~-0.1-0.15 ms per pop for ~2-3 days [G]. **Recommended: park the kernel line at sync-free**
  (asked the orchestrator / NewIdeas 11:1x). Details: draft/FUSED_TREE_PLAN.md.

Labels: [K] measured (raw data path), [P] derived, [G] estimate. BW = 292.7 GB/s (BET 7's stock fit).

## 0. Correction first: the planned head reads 177.6 MB per level, not ~42-73
- MTP Head's plan is 8-bit Linears + the ~32K screen. That is P2's C0 "as stored": the stored head's quantized Linears
  at their stored 8 bits, plus ScreenedHead(RANK[:32768]).
- Measured parts per level at F = 1 [K ~/MoonShots/week98/p3_m012_stock.log, p3_m012.py's C0 build]:
  - dense 99.7 MB (8-bit Linears + bf16 HC mixes);
  - head 47.2 MB (32,768 lm_head rows at 4 bits);
  - experts 10 x 3.07 = 30.7 MB;
  - total **177.6 MB**.
  - For scale: the same with 4-bit Linears and an 8K screen (C1) is 95.1 MB; 8-bit + 8K is ~142 MB [P].
- **Byte floor: 177.6 MB / 292.7 GB/s = 0.607 ms per level, at zero fixed cost.**

| block | levels | byte floor (f = 0) | today, stock sync-free (0.960 ms per level) [K BET 7] |
|---|---|---|---|
| HE+ tree-32 | 20.8 | 12.6 ms | 20.0 ms (the brief's "19.1") |
| MBPP tree-32 | 14.6 | 8.9 ms | 14.0 ms |
| HE+ tree-16 | 12.6 | 7.6 ms | 12.1 ms |
| chain-32 | 31 | 18.8 ms | 29.8 ms |

- **So <= 4-5 ms per block can't be reached by any kernel work at these bytes.** Even at f = 0 a block fits only
  ~7-8 levels in 4.5 ms.
- What the target needs, at the persistent kernel's f ~0.1 [P]: <= ~34 MB per level for HE+ tree-32, or <= ~75 MB per
  level for HE+ tree-16. That is a 2.4-5x smaller head: MTP Head's lane (bytes x kept), not mine.
- M3 already priced the cheap byte cuts on the 125B: 4-bit Linears x0.92 / x0.90, 8K screen x0.963 / x0.985 (tree-32,
  HE+ / MBPP). Wednesday's Q8 / H32 / H16 run prices 8-bit and the screen sizes.

## 1. What my pieces can do (fixed cost f per level), all measured on THIS level

| step | f, ms per level | ms per level | HE+ tree-32 | HE+ tree-16 | exactness | status |
|---|---|---|---|---|---|---|
| stock, host sync per level (P2's design) | 0.549 [K M1] | ~1.16 | ~24 | ~14.6 | reference | |
| stock, sync-free (async per level) | 0.353 [K BET 7, 400 keys] | 0.960 | 20.0 | 12.1 | bitwise (same kernels) | built (p3_m012 / fl_stock); engine integration only |
| fused2 (17 kernels) | 0.269 [K] | 0.876 | 18.2 | 11.0 | BITWISE vs stock (21 levels x 2 starts at 400 / 1,010 / 2,000 keys) | built, banked (fl_fused.py) |
| persistent level kernel v1 | ~0.12-0.18 [G] | ~0.73-0.79 | ~15.2-16.4 | ~9.2-10.0 | kept-equal (M3 read-out >= 0.99x), bitwise where free | to build |
| persistent block kernel | >= 0.09-0.11 (the (a)-check's measured no-dependency bound) [K fl_overlap] | >= 0.70 | >= 14.5 | >= 8.8 | same | to build |

- **Is f ~0.1 by Friday reachable?** The measured bound is f ~0.09-0.11: the same steps with every dependency removed ran
  0.699 ms per level. A first persistent build realistically lands at f ~0.12-0.18 [G].
  - Grid barriers cost ~2-2.5 us each, no cheaper than a dependent dispatch [K M5]. The win has to come from overlapping
    each step's drain with the next step's first weight loads, held in registers. The cache doesn't keep them: BET 7's
    touch-prefetch KILL.
- **Even at f = 0.1, the draft is ~14.7 ms per HE+ tree-32 block (−27 % vs today). Bytes, not f, set the floor.**

## 2. Plan with dates (GPU: BREAK98 rule; the draft line counts as BREAK98)
- **Tue (today): engine-ready sync-free + fused2.** fused2 is bitwise vs stock and needs no new GPU proof (BET 7's
  quiet bench, a600df5).
  - Check with BigMoE whether its engine draft is host-synced per level. If yes: async per level is −28 % per level
    (0.549 -> 0.311-0.353 f). If no: fused2 is the next −8.7 %.
  - Expected: 20.0 -> 18.2 ms per HE+ tree-32 block. GPU: bit tests under share locks only.
- **Wed: persistent LEVEL kernel v1 build.**
  - One dispatch per level, fused2's 17 bit-matched step bodies, grid barriers between dependent steps.
  - The next step's first tiles are loaded into registers before each barrier.
  - Bit tests (share lock). Q8 / H32 / H16 run 05:24 / 06:24, unchanged.
- **Thu: v1 quiet bench** (<= 10 min, quiet, MEM ~3 GB): f per level at 400 / 1,500 keys, vs stock and fused2,
  interleaved.
  - Line: PASS f <= 0.15 (<= 0.76 ms per level); KILL f >= 0.25 (no better than fused2).
- **Fri: block-persistent form** (31 or 20.8 levels in one dispatch, the frontier chosen on the GPU) if v1 passes.
  - Quiet bench (<= 10 min).
  - The M3 harness's kept read-out of the kernel's drafts: share lock, MEM 12 GB, ~35 min; line >= 0.99x at
    tree-16 / 32, both sets.
- **Sat: hand-off** to BigMoE / Leap for the engine: the kernel, its bit / kept record and the per-level numbers.
- **GPU slots needed:** Thu <= 10 min quiet, Fri <= 10 min quiet + ~35 min share (12 GB). Bit tests: share locks, <= 3 GB,
  any time.

## 3. Risks / what kills it
- Grid barriers need every threadgroup co-resident. The persistent grid must be sized to the GPU's resident capacity.
  If another process holds cores, a spin barrier can stall. Mitigation: a timeout fallback to the discrete fused2 chain.
- Per-step work splitting for small steps (norms, routing top-10, the tail) can stay latency-bound (~0.07 ms per level
  in BET 7's profile). That's the f ~0.12-0.18 v1 guess.
- If MTP Head's head changes format (e.g. the 32K screen grows), the bytes and the floor move with it: 1.44 MB per 1K
  screen rows at 4 bits.
- **The bigger risk is the target itself (section 0):** without a 2.4-5x smaller head, the draft stays >= 12.6 ms per HE+
  tree-32 block.
