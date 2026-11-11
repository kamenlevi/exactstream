# First token: ours vs Strata, phase by phase (Mon 9 Nov 15:3x; JOB 2, Kamen 14:5x via the orchestrator)

> (a quote from the project owner, kept private)
way they do it in our engine?" This is the timeline and the port plan. Strata = MIT, github.com/Niko1221/Strata, read at
commit 146df3c (~/StrataEval/src); nothing of it is run or copied here. Sushi = github.com/beamivalice/sushi (main fa1fd446),
read through its GitHub web pages only, nothing downloaded; its README says MIT but the repo also carries an Apache-2.0
licence file: check before porting anything. Full reading notes with file:line: data/ttft/strata_prefill_map.md.

## Same 12 IFEval prompts (P = 37-101 prompt tokens), 64 GB M5 Pro
| | how the prompt runs | time to first token | SSD read per prompt |
|---|---|---|---|
| Strata, arena 33 GB, P <= 57 (8 prompts) | 2-row decode windows (~38 ms each) | 0.63-1.24 s (prompt_ms) [K] | not logged on Metal |
| Strata, arena 33 GB, P >= 66 (4 prompts) | batched chunk path (~354 ms + 1.57 ms per token) | 0.42-0.49 s (prompt_ms) [K] | not logged; <= ~1 GB inferred |
| ours, 45 GB, chat as shipped (zs + PF-D v4 + read-ahead, exact prefill) | one 4-bit call per layer (PF-D v4) | 1.55-2.59 s, median 2.26 [K] | 4.7-9.4 GB (PF-D's own reads), median 7.9 [K] |
Sources: Strata = ~/StrataEval/h2h/results_arena33/B1.requests.jsonl (prompt_ms, chunk); TTFT adds ~31 ms (its first decode
window). Ours = BM:data/r57/pdz4a_256_runs.json (pfs, pfd), data/pfd/pfd_report.txt. Strata's 0.83 s headline is the median
over both of its paths: on these prompts its long ones are faster than its short ones.

## Ours, phase by phase (median of the 12; seconds)
| phase | median | range |
|---|---|---|
| MoE prefill calls, 48 layers (PF-D v4 tP) | 1.79 | 1.16-2.05 |
| - waiting on SSD reads (tB + tC) | 0.64 | 0.39-0.80 |
| - GPU assemble / rebuild of the 4-bit scratch (tG) | 0.44 | 0.35-0.62 |
| - tM | 0.24 | 0.16-0.28 |
| - next-layer read-ahead prediction, its own GPU sync per layer (tPred) | 0.35 | 0.21-0.40 |
| outside the MoE calls (attention / GDN / dense for P rows, Python, syncs) | 0.45 | 0.39-0.53 |
| first token (pfs) | 2.26 | 1.55-2.59 |
The reads already run near the SSD's speed (9.4 GB in 0.80 s on ifeval/1220): the problem is the BYTES, then the per-layer work.

## Why Strata reads less and waits less
1. Bytes. Our prefill is the exact reference's 4-bit call: every routed expert needs its MSB-3 part AND its LSB-1 part. At
   45 GB the pools hold 440 of 512 MSB parts but only 136 of 512 LSB parts per layer, so most routed experts' LSB parts and
   all absent bases are read. Strata computes the prompt at its own storage precision from an arena holding 72 % of all
   experts as whole file bytes (17,723 of 24,576, filled from a static popularity ranking, data/expert-profile.bin; its own
   curve puts that at ~97.8 % of routing traffic, inferred for prompts), and reads every routed miss exactly
   (strata_prefill_map.md sections 2-3).
2. Per-layer work. Strata stops the GPU once per layer; a C++ thread reads that layer's misses (6 pread threads, F_NOCACHE)
   and releases it; its kernels run on the arena slots in place. Ours runs a Python loop per layer with several GPU syncs
   (routing ids, slot lookups, staging, the assemble pass, the rebuild pass, and a separate sync for the next layer's
   prediction): ~38 ms per layer against Strata's ~7 ms (its chunk path's fixed part / 48).
3. Strata's chunk path queues EVERY non-resident expert of the next layer and drops reads that have not started and are not
   routed when the layer arrives (stream.cpp ~640-711); its window path uses the same predictor as ours (next layer's router
   on this layer's input, top-6 per token). Prompt-streamed experts sit at the eviction end so a prompt does not flush the
   decode cache.

## Ports, in order (credit: Strata, MIT; all gains below are ESTIMATES until measured)
- P1 (lossy, needs EvalSuite's gate): the prompt at the decode's own precision. Exists: PF-D FAST (R57_PFD_FAST=1, 3-bit,
  MSB-only reads) measured 2.17 -> 1.74 s mean, SSD 9.6 -> 4.9 GB per prompt, tokens differ (lossy) [K, pdzfa].
- P2 (lossy, needs a gate; NEW): the prompt in 16-row chunks through the DECODE path: the sync-free gate, kernels on the
  pool slots in place (no assemble), the gate's own look-ahead (no prediction sync), tier precision (no LSB reads for ranks
  2-9). Removes most of reads + assemble + prediction; estimate ~0.3-0.8 s for P 37-101 from the 16-row check cycle
  (~50-70 ms) x P / 16 chunks. Matters more at 24 GB, where the pools are smaller. To build in the W24 screen and measure.
- P3 (exact, no gate): fold the next-layer prediction into the layer's existing sync (up to -0.35 s), and keep
  prompt-streamed experts at the LRU's eviction end (Strata's rule) so the first checks after a prompt keep their cache.
- P4 (exact): an idle SSD keep-awake before a request (Strata reads one page every 100 ms for 60 s after a request); ours
  keeps the SSD awake only during runs (BIGMOE_SSD_KEEPALIVE_US). Small; to measure.
- Not ported: Strata's 2-row windows for short prompts (its own data shows they are slower than its chunk path here).

## Gaps
Strata's bytes per prompt and hit rates are not logged on Metal (its stream report prints only on a normal exit; serve ends
with std::_Exit(0)); its per-request file_mb / experts_streamed fields count CUDA-side reads and are 0 on Metal. Its model
files were deleted in the 11-09 cleanup, so no new Strata timing can run without a 75.8 GB download (the orchestrator's OK).

## Results (Wed 11 Nov; data/w24/w24ft_report.txt, w24ft2_report.txt; docs/WED_TESTS.md W24FT)
- **24 GB, the 12 IFEval prompts:** the card as gated has a median first token of 3.08 s.
  - **P2 v0 at the card's 13-row cap: 1.12 s median** (plain forwards) or 1.18 s (verify scope). Strata's medians there
    are 1.46 s (arm D) and 1.65 s (arm B).
  - P2 is LOSSY (tokens 2/12 vs the exact prefill), peak 23.0 GB, with the decode unchanged.
  - At 16 rows (160 ids > 141 base slots) a chunk leaves the sync-free gate: 1.66 s.
- **P2 is read-bound:** about 2.2 GB per chunk. The next levers are bytes: keep a prompt's experts across its chunks,
  and MSB-only reads for the low ranks.
- **P3 (exact):** the fold keeps tokens and logits hashes (12/12, 634/634). On PF-D at 24 GB it gives 3.32 -> 3.11 s.
  PF-D itself is slower than the card's own prefill at 24 GB and adds 5.7 GB of peak. The 45 GB ABBA runs tonight.
- **P2 at 24 GB: EvalSuite's same-scores gate PASSED (Wed 19:51, EvalSuite 8adeb09).** IFBench -1.33 pt (p 0.80), GPQA
  Diamond's 67 short prompts -2.99 pt (p 0.63); 5 longer-prompt controls token-identical. The clean ABBA timing against
  Strata's 1.46 s runs tonight (docs/PREREG_W24T_2026-11-11.md).
