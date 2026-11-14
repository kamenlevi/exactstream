# WEEK98 route 3, the cheap check: screen 1 (Leap, Sat 14 Nov 18:04-18:1x; existing per-check data, no run)

The orchestrator 18:0x asked to cut the per-check graph-build overhead for the 16- and 32-row verify, for example with
compiled or cached graphs per row shape or fewer Python syncs. Exact stays bit-exact against NAX6-N. GPU timings wait for
the end of BigMoE's BANK (~20:35) or a quiet lock overnight.

**Result: the 16- and 32-row verify is GPU-bound. The graph build and Metal encode are already hidden behind the GPU at 2
rows and more, so cutting them saves ~0 there.**
- Shipped lossy-45 (L + COH, C 256; BigMoE rlh / rlm / rli_256 checks.json, medians by row count):

  | rows | check wall | bld (build phase) | wt (GPU tail after the build) |
  |---|---|---|---|
  | 1 | 42.2 | 10.1 | 31.8 |
  | 3 | 37.6 | 31.2 | 4.3 |
  | 7 | 47.5 | 38.9 | 6.7 |
  | 10 | 53.9 | 44.0 | 8.9 |
  | 16 | 65.3 | 53.8 | 11.0 |

  - At 1 row this config has no mid-loop eval. The CPU builds for 10.1 ms, then the GPU runs ~31.8 ms after it.
  - At 2 or more rows, SB's mid-loop async_eval overlaps the two. The GPU tail grows with rows (4 -> 11 ms), so the GPU
    is behind the CPU: the wall is GPU time.
  - A fit over 2-16 rows gives check ~30 + 2.47 ms per row. Earlier samples (TABLE_350, B1) showed only ~4 ms of the
    main-thread time is Python. The rest is MLX's graph walk and encode (~7 us per dispatch), hidden at these row counts.
- **24 GB (W24d w13eh / w13fh, C 100, at most 13 rows):** 1 row 54-55 ms (bld 13.7, wt 40.7); 13 rows 75 ms (bld 64,
  wt 10, SSD io 25.9). The fit gives ~33.8 + 3.27 ms per row. GPU- and SSD-bound.
- **CORRECTION to my 17:01 pricing:** "~0.62 ms per layer, CPU-bound graph build" was wrong in its label.
  - The 0.62 ms is the 1-row check's wall per layer where SB overlaps the build, i.e. GPU time. The CPU build is ~0.21 ms
    per layer.
  - The trunk numbers stand as wall-clock. The verdicts are structural (a layer-45 refresh is a verify) and don't move.

**Per layer, before (measured) and the exact 4-bit bandwidth floors (280 GB/s):**

| line | rows | check (ms) | per layer (ms) | floor per check | floor per layer |
|---|---|---|---|---|---|
| 45 | 1 | 42.2 (31.8 GPU; ~32 with SB1) | 0.88 (0.66 GPU) | 15.9 | 0.33 |
| 45 | 16 | 65.3 | 1.36 | 36.5 | 0.76 |
| 45 | 32 | ~90-97 (estimated, unmeasured) | ~1.9-2.0 | 49.1 | 1.02 |
| 24 | 1 | 54-55 (40.7 GPU) | 1.14 (0.85 GPU) | 15.9 + SSD | 0.33 + SSD |
| 24 | 13 | 75 (io 25.9) | 1.56 | ~33 + SSD | ~0.69 + SSD |
| 24 | 32 | not a 24 GB config (13-row cap) | | 49.1 + SSD | |

- **The floors come from bytes per check:** dense 3.44 GB (D32), plus experts at 2.135 MB each x 48 layers x the
  distinct experts per layer (10 / 66.1 / 100.5 at 1 / 16 / 32 rows).
  - A 16-row check is ~20 ms dense (measured 19.8, 11-07) + ~40 ms experts + misses and slack.
  - At 32 rows it is ~22 dense + ~57 experts + ~10.
- **The charter's "~23 ms per 32-row check" is half the exact 4-bit bandwidth floor (49 ms).** It needs lossy byte cuts
  (fewer bits, top-8, union pruning), so it goes on the FAST track with EvalSuite's gate.

**Action:**
1. Close graph caching / compile-for-build-time as a 16 / 32-row lever. It only pays on 1-row checks in configs without
   SB at 1 row, ~10 ms there, and rare on code.
2. The exact cheap check is a GPU program.
   - Experts are ~60-65 % of streaming today (NAX6). A floor-level few-rows kernel would cut ~15-20 ms at 32 rows.
   - Dense is ~20 -> ~13-14 ms at the floor.
   - Together: a 32-row check ~90 -> ~60 ms exact.
   - My earlier rebuild program is the base: STEP 0 PART B, the dispatch diet, D32's 32-row kernels. Each step is a
     GPU screen tonight.
3. A cheaper check does not change whether 32 rows beat 16. A 32-row check costs ~1.39x a 16-row one today (draft
   included) and ~1.32x at the floors. So a deep drafter pays only when kept per 32 >= ~1.35-1.4 x kept per 16.
   - t30b on MBPP: 6.92 / 6.59 = 1.05, so no.
   - 98 % per guess: 23.27 / 13.81 = 1.69, so yes.
   - The drafter decides the 32-row question. The check speed scales both loops.
4. First GPU screen (<= 15 min, after 20:35 or under a quiet lock): measure the one unmeasured "before" number, a real
   32-row check at 45 GB, split into dense / experts / misses. Every 32-row figure above rests on an estimate.
