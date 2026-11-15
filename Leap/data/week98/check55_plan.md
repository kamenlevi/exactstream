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
