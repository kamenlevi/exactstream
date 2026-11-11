# Speed-of-light GDN layer (NewIdeas task, 2026-10-21)

Question: how fast can one GDN layer of the 125B (Qwen3.8-Flash-Next: HC read/write, gated-delta attention, router, 10
routed experts at the 3-bit tier plan 4433333333 + the shared expert) run as a few fused, bandwidth-shaped Metal kernels
from resident weights, non-exact allowed (fp32 accumulation, any reduction order)? Compare with today's engine interval
(0.89-0.93 ms per layer at M=4) and the byte floor (bytes / ~300 GB/s). The answer decides whether the non-exact line
(frugal, in-RAM speed) gets a from-scratch engine instead of more patches.

## Files
- `sol_layer.py`: the layer as 12 dispatches (shapes from config.json, random weights with sane magnitudes, real byte
  counts and the true dependency chain); 48 chained layers over 6 distinct weight sets (DRAM, not cache); M = 1, 2, 4, 8.
  Also times every dispatch alone as a chain of 48 dependent copies vs 48 independent copies (the dependent-independent
  difference is the most a megakernel could recover). GPU time is measured with the graph built before the clock; the
  Python graph-build time is reported separately. Knobs: `SOL_HC=1|2` (generic vs v2 HC kernels), `SOL_HCD_RB`.
- `persist_pair.py`: the HC read pair as 2 dispatches vs ONE persistent dispatch with a device-atomic grid barrier
  (bounded spin, self-resetting counter, bitwise-equal outputs), G = 20..320 threadgroups; plus the GPU-side cost of a
  tiny dependent hop.
- `ref_gemv.py`: MLX's own GEMVs on the layer's dense shapes (4-bit g64 and bf16), M = 1-8, GB/s.
- `compile_check.py [module]`: offline compile of every generated kernel with `xcrun metal` against MLX's headers (no GPU;
  works during others' quiet locks with `taskpolicy -b`). Modules expose `compile_graph()`.
- Runner: `scripts/sol_runner.sh` (launch while holding a quiet lock; results in `data/day14/sol_layer_*.txt`,
  `persist_pair.txt`, `ref_gemv.txt`, summary lines in `data/day14/summary.txt`).

## The 12 dispatches
| # | kernel | reads | notes |
|---|---|---|---|
| 1 | hc_down (attn) | W_down 324x10240 bf16 (6.6 MB) | v2: threadgroup = 4 simdgroups, one per HC group: one pass gives the group RMS and the partial dots |
| 2 | hc_up (attn) | W_up 10240x320 bf16 (6.6 MB) | v2: 4 lanes per row, 8 rows (2 dims x 4 groups) per simdgroup, mix in registers |
| 3 | gdn_in | W_in 16480x2560 4-bit (23.7 MB) | RMSNorm prologue, causal conv4 over [state; M rows] + silu, per-head l2norm in threadgroup memory |
| 4 | gdn_rec | state 48x128x128 fp32 (3.1 MB r + 3.1 MB w) | one threadgroup per v-head, rows in order over M, gated RMSNorm epilogue |
| 5 | out_proj + HC write | W_out 2560x6144 4-bit (8.9 MB) | stream += inject x out |
| 6-7 | hc_down / hc_up (MLP) | 13.2 MB | as 1-2 |
| 8 | router + shared gate/up | 1796x2560 4-bit (2.6 MB) | logits, shared gate, silu(g) * u |
| 9 | top-10 | 512 logits per row | one simdgroup per row, softmax scores |
| 10-11 | routed experts (3c tier GLU A, B + combine) | ~10 M experts x 2.15 MB (+ LSB for 4-bit ranks) | patches/planes_engine/tierglu.py |
| 12 | shared down + combine + HC write | 2560x640 4-bit (0.9 MB) | |
Dense bytes per layer 68.8 MB (HC mixers 26.4 = 38 %, in_proj 23.7, out_proj 8.9, state 6.3, router/shared 3.5).

## What the literature says (IDEAS.md round 33a)
On M5 Pro a dependent dispatch costs ~1.3-1.6 us at the GPU and an in-kernel grid barrier 2.0-2.5 us (lithos-metal
probes), so persistent kernels don't save sync cost on Apple GPUs; megakernel wins elsewhere come from overlapping the
next op's weight loads with the previous op's tail. Our older 5.4 us/hop figure included Python graph building.

## Results (Wed 10-21 16:04, quiet lock; data/day14/sol_layer_*.txt, persist_pair.txt, ref_gemv.txt)
GPU-side ms per layer, 48 chained layers over 6 weight sets, best of 4 (graph built before the clock):
| variant | M=1 | M=2 | M=4 | M=8 |
|---|---|---|---|---|
| HC v2 (RB 2), 4-bit GEMV RA 4 | **0.426** (216 GB/s) | 0.547 | **0.894** (173 GB/s, 37 distinct experts) | 8.1 (spills) |
| HC v2 (RB 4), RA 8 | 0.434 | 0.575 | 0.952 | 6.3 |
| HC v1 (generic), RA 16 | 0.667 | 0.860 | 1.217 | 9.0 |
| byte floor at 300 GB/s | 0.306 | 0.38 | 0.515 (37 experts) / 0.44 (26, real routing overlap) | 0.69 |

Per dispatch at M=4 (us, dependent chain | independent): hc_down 40 | 28, hc_up 40 | 32, **gdn_in 272 | 241**, rec 40 | 32,
out_proj+HC write 58 | 47, MLP HC read 83 | 64, **router+shared 47 | 28**, topk 8 | 5, tier_glu A+B 307 | 385, fin 23 | 20.
Sum 0.917 dependent, 0.881 independent (M=1: 0.450 vs 0.399): fusing every boundary could save at most 4-12 %.

MLX's own GEMVs (independent, ref_gemv.txt): in_proj 4-bit 88 / 86 / **93** / 180 us at M = 1 / 2 / 4 / 8 (271 / 276 / **254** /
132 GB/s); out_proj 35 / 39 us (M=1 / 4); router+shared 9.5 / 14.4; HC down 26 / 27; HC up 26 / 26. MLX runs each row of x as its
own threadgroup row and lets the cache absorb the weight reuse; my fused 4-bit kernel recomputes the RMSNorm prologue in every
simdgroup (2 passes over M x 2560 values per 4 weight rows) and is ALU-bound from M=4 (and spills at M=8). Fix: normalize once
(tiny kernel), then MLX-style per-row GEMV + conv/l2norm epilogue.

**Confirmed end to end (16:11, v3 = SOL_G4V=mlx: MLX quantized_matmul for the 4-bit GEMVs, the RMSNorm folded into small
epilogues; data/day14/sol_layer_v3_mlx.txt):** M=1 **0.410 ms** (224 GB/s), M=2 0.512, **M=4 0.720 ms** (218 GB/s, 38 distinct
experts), M=8 1.307 (no spill). With real routing overlap (~26 distinct experts for 4 tokens, tier GLU ~-0.09 ms): **~0.63 ms**.

**Composed speed-of-light at M=4** (measured parts; gdn_in and router at MLX speed + small epilogues, HC at MLX speed):
~0.74 ms with 37 distinct experts, ~0.70 with MLX-speed HC, **~0.61 ms with real routing overlap (~26 distinct experts per
layer for 4 tokens)**, vs **0.89-0.93 ms today** (1.45-1.5x) and a 0.44 ms floor (0.48 at the ~270 GB/s kernels reach).
M=1 measured end to end: **0.426 ms** vs 0.306 floor (0.34 at 270 GB/s).

Persistent kernels (persist_pair.txt): the HC read as ONE dispatch with a device-atomic grid barrier is slower than 2 dependent
dispatches at every grid size: best 76-77 vs 64.5 us (M=1), 88-89 vs 78.7 us (M=4); too few threadgroups is far worse (G=20:
145 / 190 us). Bitwise equal, no deadlock up to 320 x 128 threads. GPU-side dependent hop in MLX: 4.6 us (tiny kernels, chain
of 400). CPU side: Python graph build 0.03 ms per layer (~1.5 ms per pass), fully hidden by mx.compile (wall 0.425 = GPU 0.426).

What this means: a from-scratch non-exact engine gains ~1.45-1.5x on the layer stack at M=4 from bandwidth-shaped kernels
(no megakernel, no grid barriers), and more per pass if it also drops the per-layer CPU gates (in-RAM needs none). Remaining
gap to the floor: GEMV efficiency (~250-270 of 300 GB/s), ~12 dependent hops x ~4.6 us, small latency-bound kernels.
