# D1i for Breakthrough's B12 (TrainLab, 10-22 04:20; streaming added 04:35)

Sub-batched exact prefill, logit-free prefill chunks, and attention decoupled from the MoE chunk. Measured on B12 eager
(load_b12(0.10), HC8, C_m 51, C_l 20), Breakthrough's 'long' prompt (2,617 tokens), MLX cache cap 64 MB, the first call
after load ("cold"), then a repeat call. Footprint = ri_lifetime_max_phys_footprint (bt_b12's figure). Logs: out/*.txt,
out/summary.txt.

## What to take
1. **Logit-free prefill chunks** (driver): `model(chunk, cache=cache, emit_logits=False)` and evaluate the cache
   states. Bit-identical. -0.7 GB at chunk 2048 (13.9 -> 13.2). spec125's prefill already does this.
2. **Grouped tier prefill** (tier.py, `BIGMOE_TIER_PF_GROUP=64`): v3's own exact `_sync_grouped` ported to records3.
   Experts run in ascending-id groups of ~64, each group on its own (row, slot) pairs. The rows are padded to a
   multiple of 64 for BUG1, `read_rebuilt` of group g+1 runs while the GPU computes group g, and the pairs go back to
   their positions before the model's combine. Groups grow past 64 until each keeps the sorted rhs gather (>= 16 rows
   and >= 4 rows per expert); a short tail joins the group before it; otherwise the call keeps the single scratch.
   Same greedy sha as the plain call (long and code prompts, cold and repeat). +25 % prefill at chunk 2048.
3. **Decoupled attention / MoE chunks** (driver, `prefill_decoupled`, `D1I_ATTN=256|512 D1I_MOE=4096`): per layer,
   PLE + HC read + attention + HC write-into-read run in query chunks of 256-512 tokens; then the MoE runs ONCE over up
   to 4,096 tokens, so its experts are read once per layer per 4,096 tokens; then the HC residual write runs per
   query chunk. It uses MTPLX's own DecoderLayer pieces in the same order; it is not valid with the B8 late writes,
   and asserts that. Footprint at the small-chunk level, reads at the big-chunk level.
   Not bit-identical to any plain chunking: the MoE's gather kernel choice follows the rows per group. Outputs are
   fluent; it needs the Eval suite's task check before it replaces chunked prefill.
   `D1I_SPLIT=1` runs the router, shared expert and combine per query chunk; it gains nothing, so keep it at 0.
4. **MTP** (spec125): `d1i_prefill_state()` builds a spec125-compatible `prefilled` state (the decoupled trunk
   prefill, then the MTP head's history appended per query chunk). The cheap fix for the MTP prefill peak is the
   existing `BIGMOE_PF_GROUP=64`: the MTP layer's streamed prefill runs grouped (capped at 0.9 x BIGMOE_MTP_C).

## Not worth taking
- GPU rebuild of records3 (`BIGMOE_TIER_PF_GPU=1`; raw preads into MLX buffers + rebuild4, exact): slower (180 vs
  204 tok/s). Prefill is SSD-bound: ~96 GB read in ~8 s for this prompt at chunk 2048 (2048 + a 569-token tail
  that reads nearly as many experts as a full chunk).
- Warming the pools from the last chunk (`BIGMOE_TIER_WARM=1`; conversion bit-exact vs records3, tests/test_planes.py):
  waited misses over decode tokens 0-16 rose 106 -> 116. This prompt is code and the answer is prose, so the prefill's
  most-used experts are not decode's. The MLX-op conversion also costs ~6 s.

## Numbers (prefill tok/s cold / repeat, peak footprint)
| variant | prefill tok/s | footprint | greedy sha (cold) |
|---|---|---|---|
| chunk 2048, logits (reference) | 162 / 172 | 13.9 GB | 3b63bb9e7a |
| chunk 2048, logit-free | 163 / 173 | 13.2 GB | 3b63bb9e7a |
| + groups of 128 | 175 / 193 | 13.1 GB | 3b63bb9e7a |
| + groups of 64 | 198-204 / 192-206 | 13.1 GB | 3b63bb9e7a |
| + GPU rebuild | 180 / 181 | 13.1 GB | 3b63bb9e7a |
| one 4096 chunk, groups of 64 | 300 / 312 | 13.0 GB | f533e54c0d |
| chunk 512, groups of 64, logit-free | 96 / 96 | 11.3 GB | ea978d1ca2 |
| chunk 256, reference | 50 / 50 | 11.5 GB | 81b8c074c8 |
| **decoupled 512 / 4096** | **272 / 288** | **10.9 GB** | 0c18983e94 |
| **decoupled 256 / 4096** | **279 / 308** | **10.9 GB** | 5ccfb9d726 |
| 32K prompt (37,294 tokens), chunk 2048, groups of 64, logit-free | 215 | 17.6 GB | |
| **32K, decoupled 512 / 4096** | **270** | **12.5 GB** | |
| 32K, decoupled 256 / 4096 (split) | 276 | 12.6 GB | |
| B12 + MTP spec 3:3:0, chunk 256 + BIGMOE_PF_GROUP=64 | (~50) | 11.8 GB | decode 23.0 tok/s, 2.21 tok/pass |
| B12 + MTP, decoupled 256 / 4096 prefilled state | 240 | 12.0 GB | decode 23.0 tok/s, 2.13 tok/pass |

Where the peak sits (MLX active memory, per layer call): at chunk 2048, 11.2 GB is already alive when the MoE starts,
the attention part peaks at 12.0 GB and the MoE part at 12.5 GB. At chunk 256, 9.9 GB is alive at that point. With
the decoupled prefill, 9.6 GB is alive and both parts peak at 10.4 GB. At 32K the binding term moves to the MoE phase
(10.9 GB alive, including ~0.9 GB of KV, + ~1 GB of MoE transient).

## Files
- tier.py.patch: diff vs ~/Breakthrough/v3f/bigmoe_v3f/tier.py (grouped prefill, adaptive groups, GPU-rebuild and
  warming variants off by default, per-layer memory instrumentation in PF).
- bt_b12.py.patch: diff vs ~/Breakthrough/scripts/bt_b12.py. Drop the V3F line, which points at my copy. The rest:
  logit-free chunks, prefill_decoupled + moe_split, d1i_prefill_state for spec125, and the `d1i` cold/repeat task.
- tests/test_planes.py (CPU): the MSB-3 / LSB-1 conversion equals records3 byte for byte.

## 5. Fixed-order expert streaming (04:35, NewIdeas' 04:03 design), `D1I_STREAM=3 D1I_RING=6` = the <= 12 GB DEFAULT (NewIdeas 04:40) with the decoupled prefill; ring 8 if 12.2 GB is allowed; off at long context
Every tier layer's 512 experts are read in id-range groups of 64, in layer order, through a ring of R buffers,
independent of the routing. tier.start_stream3 reads RAW records3 records: one pread per expert in a GIL-free thread
pool, F_NOCACHE. The exact 4-bit pieces are rebuilt on the GPU (rebuild4) as each group runs. After a layer's MoE,
the next layer's first R groups start reading while its attention runs. Same sha as the non-streamed decoupled
prefill (whole id ranges in the scratch change nothing), and decode is unaffected.

| 'long', decoupled 256/4096 | prefill tok/s cold / repeat | footprint | decode tok/s |
|---|---|---|---|
| no streaming | 278 / 298 | 10.9 GB | 18.9 / 21.5 |
| v3, ring 6 | 322 / 352 | 11.8 GB | 18.8 / 21.7 |
| **v3, ring 8** | **350 / 389** | **12.2 GB** | 19.1 / 21.3 |
| v1 (the ext's read_rebuilt on the main thread; it holds the GIL), ring 4 | 249 / 261 | 11.4 GB | 19.5 / 22.3 |
| v2 (4-bit pieces read from the model safetensors, no rebuild), ring 6 / 8 | 377 / 360, 365 / 369 | 11.6 / 12.0 GB | **16.0-16.7** |

- v2 slows the following decode by ~17 % every time (mechanism not found). It touches the model's own safetensors
  files, and v3, which reads records3, does not slow decode. Don't use v2.
- 32K (37K tokens): v3 ring 6 gives 326 tok/s but 13.7 GB (the ring sits on top of the 32K KV and attention state).
  Use no streaming at long context: 12.5 GB at 270 tok/s.
- Read waits are ~0 with v3 at ring 6-8. What is left is per-layer compute and sync: the router readback in
  prefill_call stalls the GPU once per layer, the attention launches go in sequence, and the rebuild adds a little.
  The ring has to hold ~1 GB, the reads of one attention phase at SSD speed, so speed trades against ~0.4 GB per
  2 ring slots.

## 6. Copy-free rebuild kernel (05:00, `D1I_RB2=1` with v3): PASSES the kill line
The v3 rebuild had sliced every piece out of the [G, recsz] byte buffer before rebuild4: a byte-wise strided copy of
~177 MB per group, ~3 s per 2.6K prompt. `_rebuild_gpu2` is a Metal kernel that reads the 3-bit and 1-bit words in
place from the ring (now [G, recsz/4] uint32), with the record stride and piece offsets as template constants and
rebuild4's bit math. Only scales and biases are sliced, 3 MB per piece.

| 'long', decoupled 256/4096, v3 | prefill tok/s cold / repeat | footprint | sha | decode tok/s |
|---|---|---|---|---|
| ring 6, old rebuild (same batch) | 385 / 423 | 11.8 GB | 5ccfb9d726 / 46180ef4c7 | 19.4 / 21.9 |
| **ring 6 + D1I_RB2** | **426 / 474** | **11.8 GB** | 5ccfb9d726 / 46180ef4c7 | 19.5 / 22.6 |
| ring 8 + D1I_RB2 | 433 / 474 | 12.2 GB | same | 19.4 / 22.0 |

The reads are the limit again: ~1 s of read waits, all 512 experts = ~72 GB of records3 per 2.6K at ~12-13 GB/s.
Ring 8 no longer helps. (Run-to-run variance between batches is ~10-15 %: the same ring 6 config measured 322 / 352
in batch h and 385 / 423 in batch j.)

## Against the kill line (2K peak within ~0.5 GB of 11.5 GB at >= 400 tok/s)
- **v3 ring 6 + D1I_RB2: 11.8 GB at 426-474 tok/s, exact, decode unaffected: PASS.**
- (before the kernel) ring 6: 11.8 GB at 322-352 tok/s, 80-88 % of the target
- (before the kernel) ring 8: 12.2 GB at 350-389 tok/s
- The reference was 13.9 GB at 162-172 tok/s.
