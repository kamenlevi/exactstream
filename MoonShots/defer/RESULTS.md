# dh_w4 fast mode on BigMoE's v3 engine: RESULT (2026-10-22, turn 1, 11:01-11:36)

**Verdict: do not ship on v3 at 24 GB.** The build works and passes quality, but buys no speed. Half-layer
deferral does what the model said against a no-prefetch baseline (+23 %). BigMoE's deployed JIT prefetch
already takes those same inter-layer SSD windows, and the two do not stack.

## Quality (gating scorer, Breakthrough's v3f decoder, C_m 128 / C_l 51)
All 30 IFEval sequences (6,983 positions; NewIdeas asked for every 4th, all 30 settle the level exactly):

| arm | IFEval KLD | % of target 0.0204 | top-1 |
|---|---|---|---|
| v3 control | 0.01303 | 64 % | 0.950 |
| v3 + dh_w4 | 0.01400 | 69 % | 0.947 |

- Paired difference: +0.00097, with a 95 % CI of [+0.00060, +0.00142] by sequence.
- The dh_w4 mean has a by-sequence 95 % upper bound of 0.0163, so it PASSES by a wide margin.
- The earlier "tight" reading came from the gating subset, which had only 4 IFEval sequences, and those run high.
  defer/gate_cluster.py re-bootstraps that run by sequence.
- Cross-check of my own build with the same scorer (v3d control, each set [0::8]): gsm 0.00304, code 0.00128,
  ifeval 0.01250, multi 0.01093. Breakthrough's v3f control: gsm 0.00298, code 0.00129.

## Correctness of the multi-row build (defer/v3d; teacher-forced logits, S=4 and S=8)
Every check below is BIT-IDENTICAL at both S=4 and S=8:
- dh10 == v3. This is flag mode with nothing late, so the posting, phase split, combine_dh and late HC write change
  nothing.
- force == ref. Here every rank >= 4 is late, against an independent synchronous per-rank split. This covers the
  full-slot phase B, the phase-3 GLU, the late combine and the late write.
- force_sub == ref. This is the same phase 3 restricted to the static pair subset.

## Overhead (fixed-state all-hits verify passes; nothing late, so this is the pure phase-B machinery)

| S | v3 | dh | dh, subset phase 3 | force (all ranks >= 4 late) |
|---|---|---|---|---|
| 4 | 37.93 ms | 39.24 (+1.31, +3.5 %) | 39.27 | 40.22 |
| 8 | 61.43 ms | 63.76 (+2.33, +3.8 %) | 63.35 (+1.92) | 64.19 |

## Decode A/B
Setup: 24 GB, BigMoE b13's deployed v3 env (JIT prefetch on), spec125 adaptive width, 256 tokens. v3 and dh had
2 runs each; d1 and pipe had 1 run each. The verify ratio is taken at matched verify width S.

| prompt | v3 tok/s | dh | d1 (one layer late) | dh + pipelined late reads | verify at matched S: dh / d1 / pipe vs v3 |
|---|---|---|---|---|---|
| code | 34.8 | 34.9 | 34.0 | 33.0 | x0.99 / x1.01 / x1.04 |
| prose | 30.4 | 30.3 | 29.0 | 30.2 | x0.96 / x0.93 / x0.93 |
| edit | 44.6 | 45.6 | 44.7 | 46.0 | x1.00 / x0.98 / x1.01 |

- The edit text is identical across all arms (same sha): dh runs 161 ms/pass vs 156 for v3, and its +2 % tok/s
  comes only from pass boundaries.
- Under deferral the output depends on which experts happen to be cold. dh texts therefore differ from run to
  run, and acceptance differs with them.

**Why (JIT-off pair, code):**
- v3 without JIT: 28.1 tok/s, 115 ms/pass.
- dh without JIT: 34.6 tok/s, 77 ms/pass. That is the model's +23 %, but only against a no-prefetch baseline.
- v3 with JIT is 34.9 tok/s, and dh with JIT is also 34.9.
- Deferral and the JIT prefetcher overlap reads with compute in the same windows. The trace replay
  (stall/defer_sim.py, defer/sim2.py) counted those windows as idle, because prefetch reads are not demand reads.

Per-layer trace (dh, code; defer/trace_tail.py):
- The loader is busy 42.6 of 78.3 ms per pass, about the same SSD time as v3's 42.8 ms.
- Late batch: median 0.50 ms, p90 1.44 ms, on 71 % of layers.
- Phase B(L) -> seen(L+1): at least ~0.2 ms (p10). That is the real GPU tail; the model assumed 0.12.
- On prose, deferral even raises reads per pass from 123 to 167.

SSD batch policy (defer/ssd_batch.c, cold MSB-3 reads):
- With 1 waited read, issuing the late reads 250 µs after the waited ones keeps phase A as fast as sequential
  issue (0.43 ms) and cuts the late batch by 0.15-0.19 ms.
- With 2-3 waited reads, phase A slows 12-19 %.
- As a loader change this is real, but small next to the JIT interplay.

## What is reusable
- **Where prefetch is off or weak, deferral is the substitute that pays (+23 % vs no prefetch on code).** That is
  Breakthrough's small-RAM line, where precise prefetch lost at equal RAM. B12/B8 at 5-10 % could take this
  multi-row phase B for its MTP verify passes; B8 (late MSB-3 ranks 2-3) is today one-row only.
- The pieces are verified bit-exact and ready for that:
  - full-slot phase B (BIGMOE_DEFER_FULL)
  - the multi-row OoO dh path (tier.py `_dh_call`, `install_dh`)
  - combine_dh
  - the static-subset phase 3
  - the pipelined loader variant (build_pipe, BIGMOE_DEFER_PIPE_US / _STEP_US)
- The model lesson: SSD "idle" windows taken from demand-only traces overstate the headroom whenever a prefetcher
  runs. Count prefetch reads as busy.

Files:
- data/defer/summary.txt (the run log) plus per-run txt/json
- data/defer/tfb12/ (IFEval per position)
- data/defer/ssd_batch.txt
- defer/run.py, run_gpu1.sh, gen_report.py, tf_compare.py, trace_tail.py, tf_dense*.py, step_dense.py
