# L1-15 result: multi-row GDN kernel program, 35B (Leap, Sat 2026-10-24 10:51; tuning 11:24-12:41)

## Bottom line (12:41): 35B code 222.6 -> 251.3 tok/s (+12.9 %); with L1-14's truncation 268.4 (+20.6 %)
(data/small35_rolleval.txt: 12 HumanEval+ prompts, block 16, warm, decode tok/s.)

| GDN kernels | rollback replay | truncation | tok/s | tokens per block |
|---|---|---|---|---|
| off (z-lab as is) | lazy | 0 | 222.6 | 10.92 |
| off | separate | 0 | 222.4 | 10.92 |
| on | lazy | 0 | 226.6 | 10.52 |
| **on** | **separate** | 0 | **251.3** | 10.52 |
| off | lazy | 0.3 | 241.6 | 10.11 |
| **on** | **separate** | **0.3** | **268.4** | 10.02 |

- **What it takes** (third_party/sd35 loop copy):
  1. The GDN kernel program in the verify passes, DFL35_GDNK=1: qmm16c projections, exact fused prework and gated norm,
     recurrence with the gates inside.
  2. The rollback replay submitted on its own with mx.async_eval, DFL35_ROLLEVAL=1. z-lab's loop leaves it lazy inside
     the next pass, which cost the kernel-mode pass ~4 ms.
  3. Optionally L1-14's truncation, DFL35_TRUNC=0.3.
- **Exactness, stated precisely:**
  - The GDN mixers are batch-invariant: spec == plain bitwise at 16 and 8 rows.
  - Their only new numerics are qmm16c's matmul, at the fp32-exact floor like MLX's qmv. Every other GDN op is bitwise
    stock MLX 0.32.4.
  - The rest of the 35B pass (attention, experts, router, lm_head) still uses MLX's batch-variant kernels, as z-lab's
    batched verify does. So the whole loop is no more and no less exact than z-lab's: outputs leave plain decode only
    at near-ties.
- **Tokens per block 10.52 vs 10.92:** on text identical in both modes the kernels need the same or fewer blocks, so this
  is the diverging prompts' paths (chance). A larger prompt set should average it out; the tok/s above include it.
- **0.77x of 350 on code.** The rest of the 35B route (L1-13's table) is the experts, attention, lm_head, shared expert
  and drafter, plus whole-model batch invariance for a bit-exact mode.

## Update 12:38: the loop slowdown is the LAZY rollback (test_sustain.txt, small35_prof.txt PROF_MODE 3)
- **Sustained load is not it.** Over 240 back-to-back passes the kernel pass holds 36.7-37.4 ms and stock 44.0-44.5.
- **Fully serialized loop** (the drafter, then the rollback replay, then the target, each finished and timed alone):
  - stock: 212.9 tok/s, target pass 44.1 ms (it loses its overlap);
  - **GDN kernels: 236.5 tok/s, target pass 36.9 ms**, the same as isolated. That is +6 % over stock's normal 222.6.
- **Mechanism:** z-lab's loop leaves the GDN rollback replay lazy, so it is evaluated inside the next verify pass's graph.
  With the kernel program that combined graph runs ~4 ms slower.
- **Fix being measured:** DFL35_ROLLEVAL=1 submits the replay with mx.async_eval right after the rollback, and keeps the
  drafter overlap. Compared over 6 configs, including L1-14's truncation.

## Update 12:35: accuracy fixed, acceptance is chance, rollback is not it (test_qmm_acc.txt, small35_acc.txt, test_rollback.txt)
- **Matmul accuracy vs an fp32-exact reference** (relative RMS). The earlier check used mx.dequantize in bf16, which rounds
  the weights the same way qmm16 does, so it could not see this.

  | path | in_proj | out_proj |
  |---|---|---|
  | floor (exact, rounded to bf16) | 1.66e-3 | 1.66e-3 |
  | MLX 1 row (qmv) | 1.66e-3 | 1.66e-3 |
  | MLX 16 rows | 2.21e-3 | 1.66e-3 |
  | qmm16 (weights rounded to bf16) | 2.21e-3 | 2.21e-3 |
  | qmm16c (code MMA) | 1.66e-3 | 1.66e-3 |

  - qmm16c is now the kernel program's matmul (DFL35_QMM=c, the default). It is at the floor, as fast as qmm16, and
    bitwise spec == plain at 16 and 8 rows.
- **Acceptance is chance, not numerics.** On prompts whose outputs are identical in both modes, the kernels need the same
  or fewer blocks:
  - qmm16c: 165 blocks vs stock's 167, for 2,135 tokens;
  - qmm16: 131 vs 133, for 1,501 tokens.
  The per-block drop came from the prompts whose text diverged.
- **End to end with qmm16c: 222.6 -> 227.3 tok/s (+2.1 %)**, block 48.85 -> 46.12 ms (-5.6 %).
- **The rollback replay is not the cause:** 1.05-1.44 ms stock vs 0.98-1.28 ms with the kernels.
  - With the drafter finished first (DFL35_PROF=2), the target + rollback window is 44.6 ms stock vs 41.6 kernels. The
    pass alone, in bursts, is 44.0 vs 37.0.
  - So the kernel pass runs ~3.4 ms slower inside the loop. Next test: sustained load (GPU clock).

## Update 11:53: where the loop's time goes (data/small35_prof.txt, mlx_opcost.txt)
Phase timers in the loop (12 code prompts, block 16, warm), per block:

| | stock | GDN kernel program |
|---|---|---|
| block | 48.76 ms | 46.36 ms (-2.4) |
| drafter graph build | 1.11 | 1.10 |
| target build: Python graph | ~0.3 | ~0.8 |
| target build: mx.async_eval (blocks on the GPU) | 40.82 | 37.59 |
| wait for drafts | 0.12 | 1.85 (unexplained) |
| wait for target | 6.19 | 4.75 |
| accept + rollback build + yield | 0.21 | 0.24 |
| tokens per block | 11.3 | 10.8 (-4.5 %) |
| tok/s | 223.2 | 225.2 (+0.9 %) |

- **The loop is GPU-bound.** mx.async_eval blocks inside on GPU progress (~41 ms of a 49 ms block). MLX's own CPU cost
  per op is small: ~2 us eager, 3.6 us per custom kernel, 1.2 us compiled. So graph building is not the limit.
- **GPU work per block: ~47.0 -> ~42.3 ms (-4.7)**, against -7.3 ms for the pass alone. The kernel mode also adds ~1.7 ms
  of waiting for the drafts.
- **Tokens per block fell 4.5 %**, which ate most of the 4.9 % shorter block. 6 / 12 outputs left stock's path.
  - Open question: chance (different text) or numerics? qmm16 rounds dequantized weights to bf16 before the matmul, as
    MLX's own NAX qmm does; MLX's qmv applies the scales in fp32.
  - Next check: per-prompt acceptance on the prompts whose outputs are identical in both modes.

## Update 11:40: the pass is 7 ms faster, but the loop is not (data/test_mixer16_split.txt, test_pass16.txt, small35_gdnk.txt)
- **Pre-registered number: 4.23 ms at 16 rows, so the pass line is NOT met (6 % over).** Same method as L1-13: build +
  eval on captured inputs.
  - Python graph building is only ~0.4 ms of it (stock 0.37).
  - Under mx.compile it is 3.92 ms, vs 11.35 ms for compiled stock, with the same bits.
- **Serial, as the model runs it (each layer's input is the previous layer's output):**
  - 30 mixers chained: stock 12.59 ms, kernel program 5.19 ms.
  - The real 16-row verify pass (taps + the loop's GDN capture): **44.0 -> 37.0 ms (-7.3 ms)**. Greedy tokens are the
    same on 16 / 16 rows here.
- **End to end, the DFlash loop did NOT get faster.** 12 code prompts, block 16:
  - No truncation: 221.7 / 222.4 tok/s stock vs 217.5 / 224.0 with the kernels.
  - Truncation 0.3: 241.1 vs 241.2.
  - 6 / 12 outputs are identical to stock: the new rounding moves near-ties, as any batched verify does.
- So the block time is not set by the verify pass's GPU time alone. Next: phase timers in the loop (drafter build,
  target build, waits, rollback) for stock vs kernels, to find where the 7 ms go.
  - Candidates: CPU-side graph building on the critical path; per-S kernel variants (prework / gated norm are templated
    on S) compiling inside the timed loop; GPU idle between phases.

## Update 11:31 (batches 4-5; data/exact_ops_probe.txt, test_qmm16_nt.txt, test_qmm16c.txt, test_mixer16.txt)
- **Best config: 4.23 ms at 16 rows** (stock 11.64), still bitwise spec == plain at 16 and 8 rows. 6 % over the pass
  line. Config: in_proj qmm16 KS 2 / NT 2, out_proj KS 4 / NT 2, beta / g inside the recurrence, 8 value rows per
  thread.
- **Stages (30 layers, S = 16, each timed alone):** qmm16 in_proj 2.04, prework 0.46, recurrence + gates 1.10, gated
  norm 0.36, qmm16 out_proj 1.01 ms.
- **Every non-matmul kernel is now bitwise stock MLX 0.32.4.** The fused mixer equals the eager mixer at S = 1 / 2 / 4
  in outputs and both states. The cause of the earlier rare mismatches: MLX 0.32.4's Sigmoid uses
  metal::precise::exp, and 0.32.2 (which MoonShots' copy follows) used metal::exp.
  - Exhaustive check over all 65,282 non-NaN bf16 inputs: sigmoid, silu, softplus and compute_g (32 heads) have 0
    mismatches.
  - So the only new numerics are qmm16's (the projections).
- **Recurrence with gates inside:** bitwise equal to MLX's gates + mlx_lm's kernel at T = 1 / 8 / 16. At T = 16 it
  takes 1.07 ms with 8 rows per thread, vs 1.41 for MLX's gates + mlx_lm's kernel.
- **Tried and dropped:**
  - qmm16 v2 (64 columns per simdgroup, prefetch): slower.
  - qmm16c ("code MMA": scale and bias per group after a matmul on the raw codes): not faster and slightly less
    accurate. So dequantization ALU is not the bottleneck.
  - n tiles per threadgroup: in_proj 63 -> 70 % of floor; out_proj stays ~50-55 %.

Pre-registration: GDN16_PREREG.md. Data: data/gdn16_profile.txt, qmm_bands35.txt, test_gdn35.txt, test_qmm16.txt,
test_mixer16.txt. Code: scripts/gdn35_exact.py, scripts/qmm16.py.

## Verdict: ALIVE, between the lines (not killed; the 4 ms pass line not met yet)
- **30 GDN mixers at 16 rows: 4.56 ms**, against 11.57 ms for stock MLX (2.5x faster). Kill line > 6 ms; pass <= 4 ms.
  My prediction was 4-5 ms.
- **Spec == plain is bitwise.** One 16-row call equals 16 sequential 1-row calls of the same kernels, in every output,
  the final recurrent state and the conv state, on all 30 layers with real verify rows. 8 rows: also bitwise.
- **Decode gets slower:** 4.52 ms at 1 row vs 3.47 stock (+1.05 ms per token for the 30 mixers). Decode pads to the
  16-row kernel. The spec loop never runs decode; it only defines the reference.
- **New numerics, same accuracy.** vs stock MLX at 16 rows the outputs differ by ~2.5e-3 of their range (rounding order
  only). The kernel's error vs an fp32 reference equals MLX's (0.00696 vs 0.00696).

## What the profile showed (S = 16, 30 layers)
- **The projections were the gap, not the recurrence.** in_proj_qkv 5.22 ms (19 % of its floor), in_proj_z 2.39,
  out_proj 2.38, b / a 0.35. The rest of the chain was 3.3 ms, of which the recurrence was 1.55.
  - Why: MLX's matmul picks a different kernel per (M, N, K). At 16 rows these shapes land on slow paths: qmv_wide
    re-reads the weights per row, and split-K runs at 9-20 % of the floor.
  - Merging qkv | z | b | a into one matmul (N 12352) happens to take MLX's fast NAX path: 2.66 ms.
- **Every non-matmul op was already batch-invariant**: conv + silu, the q/k norms, the recurrence (fp32 state, T at
  run time) and the gated norm, all bitwise S rows == sequential 1-row. Only the matmuls broke spec == plain: 33 % of
  in_proj_qkv's elements differ between 16 rows and 1.

## The kernel program (one set of kernels for every row count)
1. **qmm16:** a new 16-row 4-bit matmul on the M5's neural accelerators (Metal 4 MPP matmul2d, MLX's own fragment
   layout). Decode pads to 16 rows, so a row's bits never depend on the row count.
   - Fused in_proj: 2.14 ms (71 % of the bytes floor). out_proj: 1.0 ms (51 %). MLX at 16 rows: 2.66 / 2.35 ms.
   - Its error vs fp32 equals MLX's. It is bitwise padding-independent at every split.
   - A v2 (64 columns per simdgroup, prefetch) was slower: dropped.
2. **prework35:** conv + silu + q/k rms_norm + scales + new conv state in one dispatch. Adapted from MoonShots'
   gdn_exact.py.
3. MLX's sigmoid / compute_g for beta and g.
4. **recurrence_r4:** mlx_lm's per-row arithmetic, 4 value rows per thread. Bitwise equal to mlx_lm's kernel at every T;
   at T = 16 1.16 ms vs 1.38.
5. **gated_norm35** (rms_norm with weight + silu gate, one dispatch), then qmm16 for out_proj.

## Known gaps
- prework35 is not bitwise to MLX's eager chain in rare elements on large inputs (random x 3: <= 23 of 65k). MLX's
  conv1d and silu alone match fp32 sums, so the difference is in the kernel's bf16 sigmoid expression. It does not
  affect spec == plain (the same kernel runs both). Fix before claiming "non-matmul ops bitwise to stock".
- The rest of the model (attention, experts, shared expert, router, lm_head) is not batch-invariant yet. Whole-model
  spec == plain needs them too. qmm16 serves the dense ones directly. The experts are L1-07 (MoonShots).

## What it buys
- **35B code (L1-13 block 54.5 ms):** GDN 12.2 -> 4.6 ms gives a ~46.9 ms block, ~254 tok/s (x1.16 over 219).
- The same qmm16 kernel can serve the 16-row dense matmuls: attention projections 3.8 ms, shared expert 1.5, lm_head
  1.8; floors 0.55 / 0.25 / 1.0.
- 125B (Fit12, 10:43): GDN is ~10 ms of each 4-row pass at 12 GB (floor ~4-5): +3-5 % exact there, line 2.

## Next, to reach <= 4 ms (needs -0.6 ms)
- Fold beta / g into the recurrence (-0.2 ms).
- A faster recurrence: more rows per thread or wider TGs (1.16 ms; state-traffic floor 0.43).
- Better qmm16 tiling for out_proj (51 % of floor) and in_proj (71 %).
- Then the dense rest of the pass with qmm16, and the whole-model spec == plain check.
