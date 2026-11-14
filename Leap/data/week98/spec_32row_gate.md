# Draft spec: an exact single-pass 32-row check (Leap for BigMoE, Sat 14 Nov 2026 18:3x; WEEK98 route 3)

The orchestrator 18:2x asked for a draft spec of BigMoE's 32-row check path, with no rush. BigMoE owns the engine. This is
a proposal from reading the 1.2.1 candidate's code (engine/bigmoe/gated.py, stream.py, tier.py, ext/bigmoe_ext.cpp) and
my earlier measurements. Nothing here has been built or run.

## Why
- **The deep drafter needs a 32-row check that costs about 1.4x a 16-row check or less.** With today's two 16-row
  passes it costs 2x. A 98 %-per-guess drafter keeps 23.27 / 13.81 = 1.69x per check, so it pays at ~1.4x but not at 2x
  (data/week98/cheapcheck.md).
- **Today the engine has no exact single-pass check above 16 rows.**
  - The sync-free gate's routing row carries at most 185 ids (gated.py: `assert MAXK <= 185`; ext `xlines_of` caps at 4
    extra id lines), i.e. 18 rows at top-10.
  - Above that, stream.py:77 sends the call to `_sync_call`. At 45 GB that is `tier.prefill_call`, which reads records3
    and rebuilds 4-bit on the CPU for every expert of the call: the prefill's exact 4-bit numerics, not the verify's
    REF2-T tiers (ranks 2-9 at 3 bits).
  - The loop's own comment (r57_screen_rel1.py:107-109) calls this "outside REF2-T's certified 1-16-row verify ... slow
    either way".

## The four pieces
1. **The gate, 320 ids per routing row (exact: a layout change, no arithmetic).**
   - gated.py: `XLINES = min(9, (MAXK - 61 + 30) // 31)` and `assert MAXK <= 340`. ROW = 96 + 32 * PRED_LINES +
     32 * XLINES gains 5 id lines (160 ints) over XLINES 4. The spin region is
     nreg * PAIRS * 32 * (2 + XLINES) ints: x1.8 of today's, ~7-22 MB.
   - ext: `xlines_of` min(4 -> 9). `g_rs` = 32 * (2 + XLINES). The PostIds dispatch (bigmoe_ext.cpp:1212,
     `nt = 192`, "up to 185 ids") -> 352 threads, one per id.
   - The id-line mapping in the post kernel (line ~1101) and the loader's seq scan over `PLINES + XLINES` lines (~1573,
     ~1668) are already generic in XLINES.
   - stream.py:77 needs `GATE_N[0] = 320` for verify arms. The wide-call condition (`indices.size <= C`, or a caller's
     distinct bound `gate_req <= C`) then needs a bound on the call's DISTINCT experts. 320 ids > C 256, and ensure
     aborts past C_main. See the risk below.
2. **The dense side at 32 rows (exact: proven per row in D32, 11-09).**
   - Leap's D32 kernels ('mr2' / 'mg2': one weight read, two row tiles) give per row the 16-row path's bits at every K
     split, independent of padding. Dense per check is 17.96 -> 22.19 ms (R = 1.24).
   - MLX's own 32-row call is NOT row-invariant for 4-bit small-N shapes (shared expert, k / v, in_proj_a / b) or for the
     8-bit router / indexer. Those stay as 2 x 16 calls, as D32 found.
   - Wire this as REF2's 32-row verify path.
3. **The experts at 32 rows.**
   - NAX6 already handles any row count. Its measured slope from 16 to 32 rows is 1.41 ms per added row per check (B3).
   - The union read once is ~100.5 distinct per layer, against 2 x 66.1 for two 16-row passes.
4. **GDN and attention verify at 32 rows.**
   - The per-row GDN state snapshots for rollback double. That is ~1.8 GB per 16-row check today, so measure the 45 GB
     peak.
   - The QSA indexer and attention at 32 rows need the same per-row bitwise check as the dense side.
   - **Measured 21:17 at 24 GB (screen 3, data/week98/cc3/README.md).** With the routed experts stubbed, a 32-row pass
     takes 181 ms, against 23 ms at 13 rows.
     - CORRECTED 21:3x: the cause is the 24 GB config's DENSE-N-RAM (R57_DNR=1), not GDN / attention. Dense weights live in
       the native <= 16-row layout, and a call above 16 rows unpacks the whole weight with MLX ops on every call (ref2
       `_dnr_qmm` -> `_dn_unpack`; the run's counter shows 2592 unpacks = 24 passes x 108 weights).
     - A 24 GB 32-row check therefore also needs ref2's verify path to run 17-32 rows natively: two 16-row calls (per row
       exact) or a 32-row native kernel. The prefill keeps its unpack.
     - exact-45 has no DENSE-N-RAM. Its number (same screen, pending) shows what GDN / attention above 16 rows cost on
       their own.

## Risk: the union can exceed the pool
- 16 rows x top-10 = 160 ids, which is <= C at 45 GB (256) and 32 GB (160), so today's wide gate is always safe there.
  At 32 rows the worst case is 320 distinct, more than any pool.
- **Step 0 DONE (BigMoE 64c9d2d, Sat 21:3x; ~/BigMoE/data/deepdraft/bank/routes_stats.json).**
  - Setup: router top-10 per row on the runtime path (prompt prefill + 9-row verify), 84 files (24 HE+, 30 MBPP, 30
    n1112 val), ~35K windows x 48 layers.
  - Distinct experts per layer per window (answer rows alone run ~1-3 higher than all rows):

    | window | max | p99.9 | p99 | mean |
    |---|---:|---:|---:|---:|
    | 8 | 80 | 77-78 | 70-71 | 45-46 |
    | 16 | 152 | 134-136 | 119-121 | 71-74 |
    | 32 | 242 | 208-210 | 181-185 | 107-110 |

  - The early layers route most diversely. Worst at window 32 (max / p99.9): layer 0 242 / 228-230, layer 1 220 /
    204-206, layer 2 201 / 188-191, layer 7 197 / 186-188, layer 22 195 / 188-189.
- **What it means for the spec:**
  - **The gate's id capacity.** 340 ids covers window 32's max (242) with room. Today's 185 covers window 16 (max 152).
  - **45 GB (C_main >= 256): safe in all ~35K windows.** The max (242, layer 0) is 14 slots under 256. 242 is the
    largest seen, not a bound: an unseen prompt can go higher, up to 320. So keep a guard, which costs nothing when it
    doesn't fire.
    - The gate reports overflow as a flag instead of `ensure` aborting.
    - The host then discards the check (the rollback it already does) and redoes it as 2 x 16.
    - The redo is exact by construction, since the 32-row pass must equal 2 x 16 per row anyway.
    - Its expected cost is ~0: no window crossed 256.
  - **32 GB (C 160, C_main ~175): a fallback for a minority of checks.** At window 32, 10 layers exceed C_main at max
    and 9 at p99.9, but none at the mean, mostly the early layers. A whole-check redo would fire too often there.
    Better: split only the layers that overflow into two 16-row expert calls, or predict overflow per check on the host
    from the previous check's per-layer union.
  - **24 GB (C 100, C_main 94): never single-pass.** At window 32 all 48 layers exceed 94 at max and p99.9, and 40 of 48
    at the mean, so it is 2 x 16 always. The 24 GB line also caps checks at 13 rows (R57_ROWCAP) and has the
    DENSE-N-RAM unpack above 16 rows (piece 4). The 32-row check is a 45 GB feature, maybe 32 GB.
  - **The estimates.**
    - The union mean at 32 rows is 107-110, a little above my ~100.5.
    - New distinct per added row falls from ~3.4 (8-16 rows) to ~2.25 (16-32 rows). That is a factor of ~0.67, where
      cc3_report used 0.72.
    - The 32 / 16 union ratio is 1.50, which supports the misses' x 1.5.

## How to test (exact first, then time)
- **Per piece, bitwise:** 32-row outputs per row == the same rows inside two 16-row calls == the 1-row AR, for every
  dense shape, the experts, GDN (y and both caches) and attention. Run at S 32 / 17 / 31 and with padding.
- **Tokens:** a decode with 32-row checks (for example the deep drafter's 30-deep drafts, or DFlash2 block 32) on 12
  HumanEval + 12 IFEval prompts gives tokens == REF2 rows, and logits hashes == the same prompts' 16-row-check decode on
  every row they share.
- **Time (one quiet screen):** 16-row vs 32-row check at 45 GB, split into dense / experts / misses, against the
  1.35-1.4x line. Screen 3 tonight (cc3_runner.sh) gives the baseline split and a widened-gate estimate.

## Expected size (estimates, 45 GB, before the kernel work)
- Today, 32 rows as two 16-row passes: ~130 ms.
- A widened gate, single pass: ~90-97 ms, which is ~1.39x a 16-row check including the draft. That is right at the line.
- With floor-level kernels (cheapcheck.md action 2): ~60 ms.
